#!/usr/bin/env python
"""失败点诊断：分数为什么是双峰的（大多数局 0~2 分，偶尔 26 分）。

均分 7.2 但中位数只有 2，说明它不是"稳定飞 7 根"，而是"多数早死 + 少数长飞"。
要提分就得先知道**死在第几根管子**、以及**死的时候鸟在缺口的哪一侧、偏多少**。
这个脚本把逐局轨迹拆开统计，并把"下一根管子的缺口位置"和"鸟当时的高度"
对上，回答一个具体问题：**bird 是够不着缺口，还是够着了但走错了方向？**

用法
    D:/Code/FlyBrain/env/python.exe scripts/flappy_failures.py --games 60
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
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--max-ticks", type=int, default=2500)
    ap.add_argument("--mode", default="bidi")
    ap.add_argument("--gap-margin", type=float, default=22.0)
    ap.add_argument("--vy-gate", type=int, default=1)
    a = ap.parse_args()

    h = fb.Harness(a.asset)
    proj_name, policy = fb.MODES[a.mode]
    spec = dict(s50=15.0, s50size=30.0, gain=1.0, width=0.5, baseline=0.25,
                ground_aware=1, elev_scale=1.0, pipe_w=1.0, span=0.6, base=0.6,
                gap_margin=a.gap_margin, vy_gate=a.vy_gate)

    rows = []
    for k in range(a.games):
        r = h.play(proj_name, spec, seed=1000 + k, max_ticks=a.max_ticks,
                   policy=policy, trace=True)
        rows.append(r)

    sc = np.array([r["score"] for r in rows], float)
    print(f"模式 {a.mode}  死区 {a.gap_margin}  vy_gate {a.vy_gate}  n={len(rows)}")
    print(f"均分 {sc.mean():.2f}  中位 {np.median(sc):.0f}  最高 {sc.max():.0f}  "
          f"分布 {dict(sorted(Counter(sc.astype(int)).items()))}")
    print(f"0 分局占比 {(sc == 0).mean():.0%}   <=2 分局占比 {(sc <= 2).mean():.0%}   "
          f">=10 分局占比 {(sc >= 10).mean():.0%}")

    print("\n死因：", dict(Counter(r["cause"] or "存活到上限" for r in rows)))

    # ---- 每次死亡发生在"第 N 根管子"上
    print(f"\n{'死亡时的管子序号（应过而没过的那根）':<40}")
    idx = Counter()
    for r in rows:
        if r["cause"]:
            idx[r["score"] + 1] += 1          # 已过 score 根，死在第 score+1 根
    for k in sorted(idx):
        print(f"  第 {k:>2} 根管子 : {'#' * idx[k]} ({idx[k]})")

    # ---- 死亡瞬间：鸟相对缺口的偏差
    print("\n死亡瞬间（撞管）的偏差：dev = y − 缺口中心（正=鸟偏低，负=鸟偏高）")
    devs, d_fronts = [], []
    for r in rows:
        tr = r.get("trace") or []
        if not tr or not r["cause"]:
            continue
        last = tr[-1]
        if r["cause"] == "撞上管子":
            devs.append(last["dev"])
            d_fronts.append(last["d_front"])
    if devs:
        devs = np.array(devs)
        print(f"  n={len(devs)}  dev 均值 {devs.mean():+.1f} px  "
              f"中位 {np.median(devs):+.1f}  范围 [{devs.min():+.0f}, {devs.max():+.0f}]")
        print(f"  鸟偏低（dev>0，没爬够）: {(devs > 0).sum()} 局   "
              f"鸟偏高（dev<0，爬过头）: {(devs < 0).sum()} 局")
        print(f"  |dev| < 84（半个缺口高，本该过得去却撞了）: "
              f"{int((np.abs(devs) < 84).sum())} 局")
        print(f"  死亡时管子前缘距离（px）: 均值 {np.mean(d_fronts):.0f}  "
              f"中位 {np.median(d_fronts):.0f}")

    # ---- 每根管子到达时，鸟的高度是否在可达范围内
    print("\n每根管子到达时，鸟的 y 与缺口中心的关系（只看实际通过的那些）")
    passes = []
    for r in rows:
        tr = r.get("trace") or []
        # 用 score 增加的时刻近似"过管瞬间"
        prev = 0
        for e in tr:
            if e["score"] > prev:
                prev = e["score"]
                passes.append((e["y"], e["gap"], e["y"] - e["gap"]))
    if passes:
        p = np.array(passes)
        print(f"  共 {len(p)} 次过管：|y − gap| 均值 {np.abs(p[:, 2]).mean():.1f} px  "
              f"中位 {np.median(np.abs(p[:, 2])):.1f}  最大 {np.abs(p[:, 2]).max():.0f}")
        print(f"  其中 |偏差| < 73（缺口半高减鸟半径，安全余量内）: "
              f"{int((np.abs(p[:, 2]) < 73).mean() * 100)}%")

    # ---- 长飞局 vs 早死局：第一根管子怎么处理的
    print("\n长飞局（≥10 分）与早死局（≤2 分）的对比")
    good = [r for r in rows if r["score"] >= 10]
    bad = [r for r in rows if r["score"] <= 2]
    for tag, grp in (("长飞局", good), ("早死局", bad)):
        if not grp:
            continue
        fl = [r["flaps"] for r in grp]
        tk = [r["ticks"] for r in grp]
        print(f"  {tag} n={len(grp):<3} 拍翅/局 {np.mean(fl):5.1f}  存活 {np.mean(tk):6.0f} tick"
              f"  第一次拍翅 t={np.mean([r['t_first_flap'] or 0 for r in grp]):.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
