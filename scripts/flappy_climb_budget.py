#!/usr/bin/env python
"""验证"没爬够"的算术：鸟进管时的实际高度 vs 它一个拍翅周期能净爬多少。

诊断（flappy_death_site.py）量到：撞管死亡 62 局**全部**是"偏低没爬够"，
dev = y − 缺口中心 中位 +77（安全带是 ±73）—— 差一点点。
而死亡集中在第 2 根管子。

假设：可解性约束只限制"相邻**缺口中心**的跳变 ≤ max_climb"，
但鸟习惯在**比缺口中心高约 70px** 的位置进管，所以
  · 约束说"要爬 40px"
  · 落地成"实际要爬 40 + 70 = 110px"
  · 而它一个拍翅周期只净赚 ~20px（爬 49px 再落回来 29px）
于是差一点点。这个脚本把这三个数摆在一起核对，并扫 max_climb。

用法
    D:/Code/FlyBrain/env/python.exe scripts/flappy_climb_budget.py --games 80
"""
from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spiking_full")
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--max-ticks", type=int, default=5000)
    a = ap.parse_args()

    h = fb.Harness(a.asset)

    # ---- 先量"鸟进管时比缺口中心高多少"（过管瞬间的 y 与缺口中心差）
    spec0 = dict(s50=15.0, s50size=30.0, gain=1.0, width=0.5, baseline=0.25,
                 ground_aware=1, elev_scale=1.0, pipe_w=1.0, span=0.6, base=0.6,
                 gap_margin=18.0, vy_gate=1, max_climb=40.0, dors_scale=0.35)
    offs = []
    for k in range(40):
        r = h.play("bidi", spec0, seed=1000 + k, max_ticks=a.max_ticks,
                   policy="brain", trace=True)
        tr = r.get("trace") or []
        prev = 0
        for e in tr:
            if e["score"] > prev:
                prev = e["score"]
                offs.append(e["y"] - e["gap"])       # <0 = 比缺口中心高
    offs = np.array(offs, float)
    print(f"过管瞬间 y − 缺口中心：n={len(offs)}  均值 {offs.mean():+.0f}  "
          f"中位 {np.median(offs):+.0f}  （负=比中心高）")
    print(f"  → 它习惯在比缺口中心高约 {abs(np.median(offs)):.0f}px 的位置进管")

    # ---- 一个拍翅周期的净爬升（理论）
    rise = fb.FLAP_V ** 2 / (2 * fb.GRAV)            # 一次拍翅的弹道上升高度
    print(f"\n一次拍翅的弹道上升 = FLAP_V²/(2·GRAV) = {rise:.0f}px")
    print(f"  但上冲期地面驱动被 vy_gate 关掉、膜电位要重新积分 ~8 tick，")
    print(f"  实测可持续爬升率 ~100 px/s × 0.4s 周期 ≈ 40px/周期；")
    print(f"  扣掉周期内的回落，**净爬升约 20px/周期**（这是硬上限）。")

    print(f"\n所以约束的算术是：")
    print(f"  '缺口中心要爬 X px'  →  实际要爬 X + {abs(np.median(offs)):.0f} px")
    print(f"  一个管子间隔 1.25s 能净爬 ≈ {1.25 * 100:.0f}px（乐观）")

    # ---- 扫 max_climb
    print(f"\nmax_climb 扫描（{a.games} 局/点，dors_scale=0.35）：")
    print(f"{'max_climb':>10}{'均分':>8}{'中位':>6}{'最高':>6}{'≤2分':>8}{'≥20分':>8}"
          f"{'撞管':>6}{'存活满':>7}")
    for mc in (40, 30, 22, 15, 8):
        spec = dict(spec0, max_climb=float(mc))
        scores, causes = [], {}
        for k in range(a.games):
            r = h.play("bidi", spec, seed=1000 + k, max_ticks=a.max_ticks,
                       policy="brain")
            scores.append(r["score"])
            key = r["cause"] or "存活到上限"
            causes[key] = causes.get(key, 0) + 1
        sc = np.array(scores, float)
        print(f"{mc:>10}{sc.mean():>8.2f}{np.median(sc):>6.0f}{sc.max():>6.0f}"
              f"{(sc <= 2).mean():>8.0%}{(sc >= 20).mean():>8.0%}"
              f"{causes.get('撞上管子', 0):>6}{causes.get('存活到上限', 0):>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
