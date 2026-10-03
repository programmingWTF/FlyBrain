#!/usr/bin/env python
"""阶段 2 核心标定：找「低自发活动 + 高输入区分度」的动力学报工作点。

诊断结论（来自 tune_readout.py）
--------------------------------
在 gain=6, tonic=0.14 下，网络**整体过度活跃**：TARGET 组静息就有 0.87 的
发放率，不同感觉通道引发的下行模式余弦相似度 >0.9 —— 输入信号被自发同步
放电淹没，网络读不出"是哪种刺激"。这正是领域内所有负面结果的共同技术根因。

本脚本扫描 (gain, tonic) 二维网格，对每个点测两个指标：
    1. 全脑静息发放率        —— 越低越好（安静才有信噪比）
    2. 通道区分度           —— 不同感觉输入引发下行模式的平均「1 - 余弦相似度」
                             —— 越高越好（网络能分辨输入）
返回帕累托前沿上的推荐工作点。
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv.spiking_brain import SpikingBrain  # noqa: E402
from tune_readout import READOUT_GROUPS, group_indices  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spiking_circuit")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ticks", type=int, default=150)
    ap.add_argument("--strength", type=float, default=3.0)
    a = ap.parse_args()
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    brain = SpikingBrain.from_npz(a.asset, device=dev)
    groups = {g: group_indices(brain, ns) for g, ns in READOUT_GROUPS.items()}
    channels = ["LC4", "LPLC2", "LC10", "T4", "T5"]
    ch_idx = {c: brain.key_idx(c) for c in channels}

    def run_point(gain, tonic, stim_name=None, strength=0.0, ticks=None):
        ticks = ticks or a.ticks
        brain.gain = gain
        brain.tonic = tonic
        brain.reset()
        for _ in range(15):
            brain.step()
        stim = None
        if stim_name:
            stim = torch.zeros(brain.N, device=brain.device)
            stim[ch_idx[stim_name]] = strength
        pop = 0.0
        grp = {g: 0.0 for g in groups}
        for _ in range(ticks):
            s = brain.step(stim)
            pop += float(s.float().mean())
            for g, ix in groups.items():
                grp[g] += float(s[ix].float().mean())
        return pop / ticks, {g: v / ticks for g, v in grp.items()}

    print(f"asset={a.asset}  N={brain.N:,}  ticks={a.ticks}  strength={a.strength}")
    print(f"扫描 gain x tonic，测「静息活动」与「通道区分度」\n")

    gains = [1.0, 2.0, 3.0, 4.0, 6.0]
    tonics = [0.0, 0.05, 0.10, 0.14, 0.20]
    hdr = f"{'gain':>5}{'tonic':>7}{'静息%':>9}{'LC4-ESCAPE':>12}{'区分度':>9}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for g, tc in itertools.product(gains, tonics):
        rest_pop, _ = run_point(g, tc, None, 0.0)
        pats = {}
        for c in channels:
            _, grp = run_point(g, tc, c, a.strength)
            pats[c] = np.array([grp[k] for k in groups])
        # 区分度：1 - 平均两两余弦
        cs = []
        for i, j in itertools.combinations(range(len(channels)), 2):
            u, v = pats[channels[i]], pats[channels[j]]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu > 1e-9 and nv > 1e-9:
                cs.append(float(u @ v / (nu * nv)))
        div = 1.0 - (np.mean(cs) if cs else 1.0)
        _, grp4 = run_point(g, tc, "LC4", a.strength)
        resp = grp4["ESCAPE"]
        rows.append((g, tc, rest_pop * 100, resp, div))
        print(f"{g:>5.1f}{tc:>7.2f}{rest_pop*100:>9.3f}{resp:>12.4f}{div:>9.4f}")

    # 推荐：静息 < 2% 且区分度最高
    cand = [r for r in rows if r[2] < 2.0]
    if cand:
        best = max(cand, key=lambda r: r[4])
        print(f"\n★ 推荐工作点：gain={best[0]}  tonic={best[1]}  "
              f"（静息 {best[2]:.3f}%  区分度 {best[4]:.4f}  LC4->ESCAPE {best[3]:.3f}）")
    else:
        best = min(rows, key=lambda r: r[2])
        print(f"\n★ 最低静息点：gain={best[0]} tonic={best[1]} 静息 {best[2]:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
