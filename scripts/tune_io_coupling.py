#!/usr/bin/env python
"""阶段 2.2 调参：找到「感觉注入强度 -> 运动神经元响应」的可用工作点。

为什么需要这一步
----------------
刚才的冒烟测试发现：LC4 刺激后 DNp01 几乎不发放（0.01 spike/tick）。
原因不是通路断了（实测 LC4 -> DNp01 只有 1 跳、651 个突触前伙伴），
而是**单跳电流越不过阈值**：LC4 即使全发放，传过来的加权电流
仍小于 (threshold - tonic) 的缺口。

这对应一个核心的「感觉编码」标定问题，pinme.dev 也是这么解决的：
把输入映射到一个**足够强**的注入电流区间。

本脚本扫描：
    - 注入强度 stim_strength ∈ {0.1 .. 8.0}
    - 全局增益 gain ∈ {1, 3, 6, 12}
对每个组合测 DNp01 在 100 tick 里的发放率，找到「静息安静 + 刺激可靠响应」的窗口。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv.spiking_brain import SpikingBrain  # noqa: E402


def measure(brain, stim_idx, strength, ticks=100, warmup=20) -> tuple[float, float]:
    """返回 (静息 DNp01 发放率, 刺激后 DNp01 发放率)。"""
    dn = brain.key_idx("DNp01")
    brain.reset()
    for _ in range(warmup):
        brain.step()
    base = sum(int(brain.S[dn].sum()) for _ in range(ticks)) / ticks

    brain.reset()
    for _ in range(warmup):
        brain.step()
    stim = torch.zeros(brain.N, device=brain.device)
    if stim_idx is not None and strength > 0:
        stim[stim_idx] = strength
    hot = sum(int(brain.S[dn].sum()) for _ in range(ticks)) / ticks
    return base, hot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spiking_circuit")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ticks", type=int, default=100)
    a = ap.parse_args()
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    brain = SpikingBrain.from_npz(a.asset, device=dev)
    lc4 = brain.key_idx("LC4")
    print(f"asset={a.asset}  device={dev}  N={brain.N:,}  LC4={lc4.numel()}\n")

    gains = [1.0, 3.0, 6.0, 12.0, 20.0]
    strengths = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    print(f"{'gain':>6} | " + " ".join(f"{s:>7.2f}" for s in strengths))
    print("-" * (9 + 8 * len(strengths)))
    results = {}
    for g in gains:
        brain.gain = g
        row = []
        for s in strengths:
            base, hot = measure(brain, lc4, s, ticks=a.ticks)
            row.append(hot)
            results[(g, s)] = (base, hot)
        print(f"{g:>6.1f} | " + " ".join(f"{v:>7.2f}" for v in row))

    print(f"\n（表内是 DNp01 在 {a.ticks} tick 内的平均发放数/tick；静息基线见下）")
    for g in gains:
        base, _ = results[(g, 0.0)]
        print(f"  gain={g:>5.1f}  静息 DNp01 = {base:.3f}")

    # 挑一个推荐工作点：静息低、刺激响应明显
    best = None
    for (g, s), (base, hot) in results.items():
        if s == 0:
            continue
        score = hot - base * 3          # 奖励响应、惩罚静息噪声
        if best is None or score > best[0]:
            best = (score, g, s, base, hot)
    if best:
        _, g, s, base, hot = best
        print(f"\n推荐工作点：gain={g}  注入强度={s}  "
              f"（静息 {base:.2f} -> 刺激 {hot:.2f}  spike/tick）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
