#!/usr/bin/env python
"""死亡点分布：区分"刚起步就撞管"和"飞了很久才撞管"。

用户两次提到"刚起步就撞管子也不少"，所以要把它量化，不能凭感觉。
判据：
  · 死亡时已过管子数 score → 死在第 score+1 根
  · 死亡时刻（秒）与"第一根管子进入碰撞区"的时刻对比
  · 死亡时鸟相对缺口的偏差（偏高/偏低）

用法
    D:/Code/FlyBrain/env/python.exe scripts/flappy_death_site.py --games 100
"""
from __future__ import annotations

import argparse
import pathlib
import statistics as st
import sys
from collections import Counter

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
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--max-ticks", type=int, default=5000)
    ap.add_argument("--dors-scale", type=float, default=0.35)
    a = ap.parse_args()

    h = fb.Harness(a.asset)
    spec = dict(s50=15.0, s50size=30.0, gain=1.0, width=0.5, baseline=0.25,
                ground_aware=1, elev_scale=1.0, pipe_w=1.0, span=0.6, base=0.6,
                gap_margin=18.0, vy_gate=1, max_climb=40.0, dors_scale=a.dors_scale)
    rows = []
    for k in range(a.games):
        r = h.play("bidi", spec, seed=1000 + k, max_ticks=a.max_ticks,
                   policy="brain", trace=True)
        rows.append(r)

    sc = np.array([r["score"] for r in rows], float)
    print(f"n={a.games}  dors_scale={a.dors_scale}  max_ticks={a.max_ticks}")
    print(f"均分 {sc.mean():.2f}  中位 {np.median(sc):.0f}  最高 {sc.max():.0f}  "
          f"≤2 分 {(sc <= 2).mean():.0%}")

    # ---- 死在第几根管子
    print("\n死在第几根管子（= 已过管数 + 1）：")
    idx = Counter(r["score"] + 1 for r in rows if r["cause"])
    for k in sorted(idx):
        bar = "#" * idx[k]
        print(f"  第 {k:>3} 根 : {bar} ({idx[k]})")

    # ---- 起步就撞？第一根管子进入碰撞区的时刻 ≈ 1.2s + 0.75s = 1.95s
    t_contact1 = fb.WARM_S + (fb.BIRD_X + 0.75 * fb.PX_PER_M - fb.PIPE_W - fb.BIRD_X) \
        / fb.PX_PER_M
    # 第一根管子的左缘到达鸟 x：初始 x = birdX + 0.75*PX_PER_M - PIPE_W
    # 接触到鸟身右缘 (birdX+11) 需要再走 (PIPE_W - 11) px
    t_hit1 = fb.WARM_S + (0.75 * fb.PX_PER_M - fb.PIPE_W + fb.PIPE_W - fb.BIRD_R) / fb.PX_PER_M
    print(f"\n第一根管子约在 t≈{t_hit1:.2f}s 进入碰撞区（开局悬停 {fb.WARM_S}s + 飞行）")
    early = [r for r in rows if r["cause"] and r["ticks"] * fb.TICK_S <= t_hit1 + 0.25]
    print(f"  **在它之前/刚到时就被判死**的局数：{len(early)} / {a.games}")
    print("  （这类是『没飞过第一根管子』）")

    # ---- 死亡时的偏差
    devs, ticks = [], []
    for r in rows:
        tr = r.get("trace") or []
        if not tr or r["cause"] != "撞上管子":
            continue
        last = tr[-1]
        devs.append(last["dev"])          # y - 缺口中心：>0 偏低（没爬够）
        ticks.append(last["t"])
    if devs:
        devs = np.array(devs)
        print(f"\n撞管死亡 n={len(devs)}：")
        print(f"  偏差 dev=y−缺口中心: 均值 {devs.mean():+.1f}  中位 {np.median(devs):+.1f}")
        print(f"  偏低（没爬够）{(devs > 0).sum()} 局   偏高（爬过头）{(devs < 0).sum()} 局")
        print(f"  死亡时刻: 均值 {np.mean(ticks):.2f}s  中位 {np.median(ticks):.2f}s  "
              f"最早 {min(ticks):.2f}s")
        late = [t for t in ticks if t > t_hit1 + 0.5]
        print(f"  其中 t > {t_hit1 + 0.5:.2f}s（明确飞过了第一根）：{len(late)} 局")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
