#!/usr/bin/env python
"""加分方案的前置验证：T4/T5 方向通道到底连不连到逃逸下行神经元。

想法是给苍蝇补上**方向选择性运动输入**（T4a/b/c/d、T5a/b/c/d 四个方向各一组），
这样它才能从视网膜光流里知道自己在**下坠**（垂直流）而不只是"有东西来了"。
但如果 T4/T5 根本传不到 DNp01/DNp02/DNp04/DNp11，这条路就不该做。

所以这里只做一件事：逐个群钳制成固定发放率，量每个逃逸 DN 的响应，
并且同时跑一个 shuffled 对照——如果打乱拓扑后一样响，那"响应"就不是
真实接线给的，而是电流泄漏之类的假象。

用法
    D:/Code/FlyBrain/env/python.exe scripts/loom_t4t5_probe.py
    ... --asset spiking_full --rate 1.0 --ticks 400
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
# T4/T5 四个亚型 = 四个方向（前->后 / 后->前 / 背->腹 / 腹->背）
MOTION = [f"T4{c}" for c in "abcd"] + [f"T5{c}" for c in "abcd"]
OTHER_SENSE = ["LC4", "LPLC2", "LC10a", "LPLC1"]
DN = ["DNp01", "DNp02", "DNp03", "DNp04", "DNp05", "DNp06", "DNp11"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spiking_full")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--warm", type=int, default=200)
    a = ap.parse_args()

    brain = SpikingBrain.from_npz(a.asset, device="cpu")
    brain.gain, brain.tonic = GAIN, TONIC
    meta = pd.read_feather(ROOT.parent / "data" / "fafb_783_meta.feather"
                           if (ROOT.parent / "data" / "fafb_783_meta.feather").exists()
                           else ROOT / "data" / "fafb_783_meta.feather")
    meta["cell_type"] = meta["cell_type"].fillna("")
    have = set(meta["cell_type"])
    names = [n for n in MOTION + OTHER_SENSE + DN if n in have]
    missing = [n for n in MOTION if n not in have]
    if missing:
        print(f"! 这些方向亚型在元数据里不存在：{missing}")
    g = looming.resolve(brain, names)
    dns = [n for n in DN if n in g]
    print(f"资产 {a.asset}  N={brain.N:,}  E={len(brain.codes):,}")
    print("群大小：" + "  ".join(f"{n}={g[n].numel()}" for n in names))

    shuf = looming.variant(brain, shuffle_seed=11)

    def measure(br, grp_name):
        idx = g[grp_name]
        pr = torch.full((idx.numel(),), a.rate, device=br.device)
        br.reset()
        base = {n: 0 for n in dns}
        dur = {n: 0 for n in dns}
        for i in range(a.warm + a.ticks):
            br.step(clamp=(idx, pr))
            tgt = base if i < a.warm else dur
            for n in dns:
                tgt[n] += int(br.S[g[n]].sum())
        return ({n: base[n] / (g[n].numel() * a.warm) for n in dns},
                {n: dur[n] / (g[n].numel() * a.ticks) for n in dns})

    print(f"\n驱动 rate={a.rate}，{a.ticks} tick；表内是 DN 群发放率(脉冲/tick)")
    print(f"{'输入群':<10}" + "".join(f"{n:>10}" for n in dns) + "     对照(shuffled)")
    rows = []
    for src in [n for n in MOTION + OTHER_SENSE if n in g]:
        b, d = measure(brain, src)
        _, ds = measure(shuf, src)
        line = f"{src:<10}" + "".join(f"{d[n]:10.4f}" for n in dns)
        flag = "  " + " ".join(f"{n}:{ds[n]:.4f}" for n in dns
                               if ds[n] > 1e-4)
        print(line + "     " + (flag or "全 0"))
        rows.append(dict(src=src, **{f"{n}": d[n] for n in dns},
                         **{f"{n}_shuf": ds[n] for n in dns}))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / "loom_t4t5_probe.csv", index=False)

    print("\n=== 判定 ===")
    for src in [n for n in MOTION if n in g]:
        r = df[df.src == src].iloc[0]
        best = max(dns, key=lambda n: r[n])
        print(f"  {src}: 最强下游 {best} = {r[best]:.4f}/tick "
              f"({r[best] * 50:.2f} Hz)；shuffled 同位 = {r[f'{best}_shuf']:.4f}  "
              + ("→ 真实接线带来的" if r[best] > 10 * max(r[f'{best}_shuf'], 1e-6)
                 else "→ 与打乱图无显著差别，别接"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
