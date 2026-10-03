#!/usr/bin/env python
"""量化"这个演示到底用掉了多少颗果蝇脑"。

三个互相独立的口径，别混为一谈：
  A. **解剖可达性**：从 LC4 出发，几跳之内能碰到多少个神经元。
     （这个数一定很大，所以它几乎不能说明任何问题——列出来是为了不被它骗。）
  B. **功能参与度**：真的跑起来时，有多少个神经元**至少放过一次脉冲**。
     这才是"被用到了"的下界。
  C. **因果必要性**：把某个神经元/某群神经元的输入掐掉，DNp01 的输出会不会变。
     只有这个口径能回答"脑在计算里出了多少力"，而不是"顺便亮了没有"。

用法
    D:/Code/FlyBrain/env/python.exe scripts/loom_utilization.py
    ... --asset spiking_full --ticks 600
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv import looming                      # noqa: E402
from fpv.spiking_brain import SpikingBrain   # noqa: E402

GAIN, TONIC = 3.0, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spiking_circuit")
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--rate", type=float, default=1.0)
    a = ap.parse_args()

    brain = SpikingBrain.from_npz(a.asset, device="cpu")
    brain.gain, brain.tonic = GAIN, TONIC
    g = looming.resolve(brain, looming.LOOM_SENSE + looming.ESCAPE_MOTOR)
    N = brain.N
    lc4, lplc2, dn01 = g["LC4"], g["LPLC2"], g["DNp01"]

    print(f"资产 {a.asset}：N={N:,}  E={len(brain.codes):,}")
    print(f"输入群 LC4={lc4.numel()}  LPLC2={lplc2.numel()}   "
          f"读出 DNp01={dn01.numel()}")
    print(f"输入占全脑比例 {100 * lc4.numel() / N:.4f}%   "
          f"读出占全脑比例 {100 * dn01.numel() / N:.4f}%")

    # ---------------- A 解剖可达性（BFS）
    indptr = brain.indptr.detach().cpu().numpy()
    indices = brain.indices.detach().cpu().numpy()
    seen = np.zeros(N, dtype=bool)
    frontier = lc4.detach().cpu().numpy().copy()
    seen[frontier] = True
    hops = {0: int(frontier.size)}
    for h in range(1, 4):
        nxt = np.unique(np.concatenate(
            [indices[indptr[v]:indptr[v + 1]] for v in frontier])) if frontier.size \
            else np.array([], dtype=np.int64)
        nxt = nxt[~seen[nxt]]
        seen[nxt] = True
        hops[h] = int(nxt.size)
        frontier = nxt
    print(f"\n[A 解剖可达] 从 LC4 出发：1 跳新增 {hops[1]:,}  2 跳 {hops[2]:,}  "
          f"3 跳 {hops[3]:,}   累计可达 {sum(hops.values()):,}/{N:,} "
          f"({100 * sum(hops.values()) / N:.1f}%)")
    print("   ⚠️ 可达 ≠ 被使用。连接组高度连通，这个数天然接近 100%，没有信息量。")

    # ---------------- B 功能参与度
    def run(drive_groups, rate):
        brain.reset()
        if drive_groups:                      # 静息口径：不钳制任何细胞
            idx = torch.cat(drive_groups)
            pr = torch.full((idx.numel(),), rate, device=brain.device)
            clamp = (idx, pr)
        else:
            clamp = None
        ever = torch.zeros(N, dtype=torch.bool, device=brain.device)
        peak = torch.zeros(N, device=brain.device)
        dn_spikes = 0
        for _ in range(a.ticks):
            brain.step(clamp=clamp)
            s = brain.S
            ever |= s
            peak += s.float()
            dn_spikes += int(s[dn01].sum())
        return int(ever.sum()), peak, dn_spikes

    for label, groups, rate in (
            ("静息（不驱动）", [], 0.0),
            (f"LC4 驱动 rate={a.rate}", [lc4], a.rate),
            (f"LC4+LPLC2 驱动 rate={a.rate}", [lc4, lplc2], a.rate)):
        n_ever, peak, dnspk = run(groups, rate)
        active_now = int((peak > 0).sum())
        p90 = float(np.sort(peak.detach().cpu().numpy())[-1]) if active_now else 0
        print(f"\n[B 功能参与] {label}")
        print(f"   {a.ticks} tick 内至少放过 1 次脉冲的神经元：{n_ever:,}/{N:,} "
              f"= {100 * n_ever / N:.2f}%")
        print(f"   平均发放率（活跃者）：{peak.sum().item() / max(n_ever, 1) / a.ticks:.4f}"
              f"   DNp01 累计脉冲 {dnspk}")

    # ---------------- C 因果必要性：逐跳掐掉 LC4 的输出，看 DNp01 还剩多少
    print("\n[C 因果必要] 掐掉 LC4 的**全部出边**，只留它到 DNp01 的直接边：")
    lut = brain.lut.detach().cpu().numpy()
    codes = brain.codes.detach().cpu().numpy()
    dn_np = dn01.detach().cpu().numpy()
    # 每个 LC4 出边里，哪些不是直接进 DNp01 的
    keep_mask = np.ones(len(codes), dtype=bool)
    cut_other = 0
    for v in lc4.detach().cpu().numpy():
        lo, hi = int(indptr[v]), int(indptr[v + 1])
        if hi <= lo:
            continue
        tgt = indices[lo:hi]
        keep_mask[lo:hi] = np.isin(tgt, dn_np)      # 只保留 -> DNp01 的
        cut_other += int((~np.isin(tgt, dn_np)).sum())
    b2 = looming.variant(brain, cut=None)
    import copy
    b2 = copy.copy(brain)
    b2.codes = torch.as_tensor(np.where(keep_mask, codes, 0), dtype=torch.long,
                               device=brain.device)
    b2.lut = brain.lut
    b2.indices = brain.indices
    b2.indptr = brain.indptr
    b2.G = torch.zeros(N, device=brain.device)
    b2.S = torch.zeros(N, dtype=torch.bool, device=brain.device)
    b2._cur = torch.zeros(N, device=brain.device)
    b2.device = brain.device
    idx = lc4
    pr = torch.full((idx.numel(),), a.rate, device=brain.device)
    b2.reset()
    dn_only_direct = 0
    ever2 = torch.zeros(N, dtype=torch.bool, device=brain.device)
    for _ in range(a.ticks):
        b2.step(clamp=(idx, pr))
        dn_only_direct += int(b2.S[dn01].sum())
        ever2 |= b2.S
    print(f"   被掐掉的 LC4 非直接出边：{cut_other:,} 条")
    print(f"   只剩直接边时 DNp01 脉冲：{dn_only_direct}  "
          f"（完整脑 {run([lc4], a.rate)[2]}）")
    print(f"   此时全脑还有 {int(ever2.sum()):,} 个神经元活动 "
          f"({100 * int(ever2.sum()) / N:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
