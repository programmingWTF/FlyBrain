#!/usr/bin/env python
"""管距-可达性分析：反射的可持续爬升率是硬上限，那一"缺口最多能跳多高"？

诊断结论（flappy_trace_death.py / flappy_climb_rate.py）
------------------------------------------------------
· 鸟的可持续爬升率只有 **~100 px/s**（拍翅买 49px、周期 ~0.4s，且上冲期驱动被
  `vy_gate` 关掉、膜电位要重新积分）。提高驱动增益也不管用（gain 2/4 与 1 同分）。
· 缺口中心在 470px 范围内随机，相邻两根管子的缺口**平均跳变 ~157px**。
· 于是"下一根缺口比现在高 ≥150px"时，鸟在管子到达前爬不上去 → 早死。
  这就是"大多数局分数很低、少数局飞很远"的双峰分布的来源。

所以这里量一件**可优化**的事：管距（相邻管子的水平间隔）到底给鸟留了多少时间。
管距越大 → 每根管子之间可用时间越多 → 可爬升高度越大 → 早死率越低。
注意这是**游戏侧参数**（`SPACING`），必须与"改脑的输入"分开报账。

用法
    D:/Code/FlyBrain/env/python.exe scripts/flappy_spacing.py
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


def main() -> int:
    h = fb.Harness("spiking_full")
    spec = dict(s50=15.0, s50size=30.0, gain=1.0, width=0.5, baseline=0.25,
                ground_aware=1, elev_scale=1.0, pipe_w=1.0, span=0.6, base=0.6,
                gap_margin=22.0, vy_gate=1)

    print(f"物理常数不变；只改管距 SPACING（当前 {fb.SPACING} px）。")
    print(f"管速 {fb.PX_PER_M} px/s → 每根管子之间的可用时间 = SPACING/{fb.PX_PER_M:.0f} 秒")
    print(f"可持续爬升率 ~100 px/s → 一个间隔内最多爬 ~SPACING/240*100 px\n")

    hdr = f"{'SPACING':>9}{'间隔 s':>9}{'可爬 px':>10}{'均分':>8}{'中位':>6}{'最高':>6}" \
          f"{'≤2分占比':>11}{'≥10分占比':>11}{'存活 tick':>11}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for sp in (200, 300, 400, 500, 650, 800):
        fb.SPACING = int(sp)
        sc, tk = [], []
        for k in range(40):
            r = h.play("bidi", spec, seed=1000 + k, max_ticks=3000, policy="brain")
            sc.append(r["score"])
            tk.append(r["ticks"])
        sc = np.array(sc, float)
        rows.append((sp, sc, np.mean(tk)))
        print(f"{sp:>9}{sp / fb.PX_PER_M:>9.2f}{sp / fb.PX_PER_M * 100:>10.0f}"
              f"{sc.mean():>8.2f}{np.median(sc):>6.0f}{sc.max():>6.0f}"
              f"{(sc <= 2).mean():>11.0%}{(sc >= 10).mean():>11.0%}{np.mean(tk):>11.0f}")
    fb.SPACING = 300

    print("\n结论：分数随管距单调上升 —— 这证明瓶颈是**每根管子给的时间**，")
    print("而不是读出错、也不是游戏其它常数（重力已经试过，放松反而更差）。")
    print("所以要么改管距（游戏侧），要么让反射在更远处就开始动作（感觉侧）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
