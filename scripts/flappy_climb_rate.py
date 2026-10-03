#!/usr/bin/env python
"""量"可持续爬升率"：决定 bidi 能不能追上突然变高的缺口。

诊断脚本 flappy_trace_death.py 发现：早死局的死法是**缺口突然变高 180px**，
而鸟的可持续爬升率只有 ~136 px/s（需要 ~144 px/s 以上才追得上）。
瓶颈是：vy_gate 让上冲期间的地面驱动归零，于是每次拍翅后必须**再落回来
重新积分**才能拍下一次 —— 一个 0.4s 周期只买 49px。

这个脚本直接量"从静止、缺口在上方时，鸟的 y(t) 爬升曲线"，并比较：
  · 驱动增益（模拟"更亮的刺激"）
  · 是否在上冲期间保持驱动（即关掉 vy_gate 的上冲抑制）

用法
    D:/Code/FlyBrain/env/python.exe scripts/flappy_climb_rate.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import flappy_bench as fb   # noqa: E402


def run(h, gain, vent_gate_up, seed, ticks=2500):
    spec = dict(s50=15.0, s50size=30.0, gain=gain, width=0.5, baseline=0.25,
                ground_aware=1, elev_scale=1.0, pipe_w=1.0, span=0.6, base=0.6,
                gap_margin=22.0, vy_gate=1, vent_gate_up=vent_gate_up)
    return h.play("bidi", spec, seed=seed, max_ticks=ticks, policy="brain", trace=True)


def main() -> int:
    h = fb.Harness("spiking_full")
    print("每个配置跑 40 局（seed 1000..1039），量分数与'持续爬升段'的斜率\n")
    hdr = f"{'配置':<34}{'均分':>8}{'中位':>6}{'最高':>6}{'拍翅/局':>9}{'存活':>8}"
    print(hdr)
    print("-" * len(hdr))
    for gain in (1.0, 2.0, 4.0):
        for vgu in (0, 1):
            sc, fl, tk = [], [], []
            for k in range(40):
                r = run(h, gain, vgu, 1000 + k)
                sc.append(r["score"]); fl.append(r["flaps"]); tk.append(r["ticks"])
            sc = np.array(sc)
            tag = f"gain={gain}  上冲保持驱动={'开' if vgu else '关'}"
            print(f"{tag:<34}{sc.mean():>8.2f}{np.median(sc):>6.0f}{sc.max():>6.0f}"
                  f"{np.mean(fl):>9.1f}{np.mean(tk):>8.0f}")

    # ---- 直接量爬升曲线：找一个"缺口在上方"的早死局，把 y(t) 打印出来
    print("\n爬升曲线（seed=1004，缺口在 y=154，鸟从 y≈370 起步）")
    for gain in (1.0, 2.0, 4.0):
        r = run(h, gain, 0, 1004, ticks=400)
        tr = r["trace"]
        ys = [e["y"] for e in tr if 95 <= e["tick"] <= 160]
        if len(ys) < 10:
            continue
        # 用滑窗斜率估计爬升率（px/s = px/tick * 50）
        arr = np.array(ys)
        k = min(20, len(arr) // 2)
        slope = (arr[:k].mean() - arr[-k:].mean()) / max(len(arr) - k, 1) * 50
        print(f"  gain={gain}: 分数 {r['score']:<3} 拍翅 {r['flaps']:<3} "
              f"y(t) 从 {arr[0]:.0f} -> {arr[-1]:.0f}（{len(arr)} tick）"
              f"  平均爬升 ≈ {slope:.0f} px/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
