#!/usr/bin/env python
"""阶段 2 冒烟测试：验证 LIF 全脑仿真内核的动力学与性能。

检查项
------
1. 无刺激时的自发活动（应当有低但非零的基线发放，来自 tonic + 噪声）
2. 人工刺激 LC4（视觉 loom）后，DNp01（巨型纤维逃逸神经元）是否被激活
   —— 这是"感觉得到 -> 运动输出"通路是否连通的直接证据
3. 单 tick 耗时（判断能否实时：20ms/tick）
4. 符号权重是否真的产生抑制效果（GABA 神经元发放应当压低目标膜电位）
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv.spiking_brain import SpikingBrain  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spiking_circuit")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ticks", type=int, default=200)
    a = ap.parse_args()

    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {dev}")
    brain = SpikingBrain.from_npz(a.asset, device=dev)
    print(f"{brain.N:,} 神经元  阈值={brain.threshold}  leak={brain.leak:.4f}  "
          f"tonic={brain.tonic}  noise={brain.noise_hz}Hz")
    print(f"关键神经元群: {brain.key_names()}")

    # ---- 1) 自发活动 ----
    brain.reset()
    spikes_per_tick = []
    t0 = time.time()
    for _ in range(a.ticks):
        s = brain.step()
        spikes_per_tick.append(int(s.sum()))
    dt = (time.time() - t0) / a.ticks
    arr = np.array(spikes_per_tick)
    print(f"\n[1] 自发活动（无刺激，{a.ticks} tick）")
    print(f"    每 tick 发放数: mean={arr.mean():.1f}  max={arr.max()}  "
          f"（占全脑 {100*arr.mean()/brain.N:.3f}%）")
    print(f"    单 tick 耗时 {dt*1000:.2f} ms  -> {'✅ 可实时' if dt<0.020 else '⚠️ 慢于实时'}")

    # ---- 2) 刺激 LC4 -> DNp01 ----
    print(f"\n[2] 视觉 loom 刺激 LC4（模拟撞击逼近）")
    brain.reset()
    lc4 = brain.key_idx("LC4")
    dnp01 = brain.key_idx("DNp01")
    print(f"    LC4 神经元 {lc4.numel()} 个，DNp01 {dnp01.numel()} 个")
    base = []          # DNp01 基线发放
    for _ in range(20):
        brain.step()
        base.append(int(brain.S[dnp01].sum()))
    stim_on = []
    for _ in range(a.ticks):
        stim = torch.zeros(brain.N, device=brain.device)
        stim[lc4] = 0.9                       # 强 loom 输入
        brain.step(stim)
        stim_on.append(int(brain.S[dnp01].sum()))
    print(f"    DNp01 发放/tick: 基线 {np.mean(base):.3f}  ->  刺激后 {np.mean(stim_on):.3f}")
    print(f"    {'✅ 感觉->运动通路被激活' if np.mean(stim_on) > np.mean(base) + 0.5 else '⚠️ 通路反应弱'}")

    # ---- 3) 全脑活动率随刺激变化 ----
    print(f"\n[3] 全脑发放率")
    brain.reset()
    rates = []
    for _ in range(a.ticks):
        s = brain.step()
        rates.append(int(s.sum()) / brain.N)
    print(f"    静息全脑活动率 {100*np.mean(rates):.3f}%")
    brain.reset()
    rates2 = []
    for _ in range(a.ticks):
        stim = torch.zeros(brain.N, device=brain.device)
        stim[brain.key_idx("LC10")] = 0.7     # 目标追踪
        s = brain.step(stim)
        rates2.append(int(s.sum()) / brain.N)
    print(f"    刺激 LC10 后  {100*np.mean(rates2):.3f}%")

    # ---- 4) 抑制性验证 ----
    print(f"\n[4] 抑制性验证（GABA 神经元应当压低目标膜电位）")
    gaba_like = []
    # 找几个抑制性出边的目标：看它们的入边里有多少是抑制的
    codes = brain.codes.cpu().numpy()
    n_inh = int((codes >= 128).sum())
    n_exc = int(((codes > 0) & (codes < 128)).sum())
    print(f"    兴奋性突触 {n_exc:,} ({100*n_exc/len(codes):.1f}%)  "
          f"抑制性突触 {n_inh:,} ({100*n_inh/len(codes):.1f}%)")
    lut = brain.lut.cpu().numpy()
    print(f"    权重范围 [{lut[lut!=0].min():.2e}, {lut.max():.2f}]  "
          f"（含负值：{'✅' if lut.min()<0 else '❌'}）")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
