#!/usr/bin/env python
"""逐局打印"早死局"的轨迹关键段：看鸟在最后一根管子前到底卡在哪。

用法
    D:/Code/FlyBrain/env/python.exe scripts/flappy_trace_death.py --games 60 --show 3
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
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--max-ticks", type=int, default=2500)
    ap.add_argument("--min-score", type=int, default=1)
    ap.add_argument("--max-score", type=int, default=2)
    a = ap.parse_args()

    h = fb.Harness("spiking_full")
    spec = dict(s50=15.0, s50size=30.0, gain=1.0, width=0.5, baseline=0.25,
                ground_aware=1, elev_scale=1.0, pipe_w=1.0, span=0.6, base=0.6,
                gap_margin=22.0, vy_gate=1)

    shown = 0
    for k in range(a.games):
        r = h.play("bidi", spec, seed=1000 + k, max_ticks=a.max_ticks,
                   policy="brain", trace=True)
        if not (a.min_score <= r["score"] <= a.max_score) or shown >= a.show:
            continue
        shown += 1
        tr = r["trace"]
        print("=" * 118)
        print(f"seed={1000 + k}  分数 {r['score']}  死因 {r['cause']}  "
              f"死亡 tick {len(tr)}  拍翅 {r['flaps']}")
        # 只打印最后一次"过管"之后的 80 tick（那一段决定生死）
        last_pass = 0
        for i, e in enumerate(tr):
            if e["score"] > 0 and (i == 0 or tr[i - 1]["score"] < e["score"]):
                last_pass = i
        seg = tr[max(0, last_pass - 10):]
        print(f"  最后一次过管在 tick {last_pass}（分数 {tr[last_pass]['score'] if tr else 0}）"
              f"，之后 {len(seg)} tick：")
        print(f"  {'tick':>5}{'y':>8}{'gap':>8}{'dev':>7}{'vy':>8}{'d_front':>9}"
              f"{'ratio':>7}{'spk':>4}{'flap':>6}")
        step = max(1, len(seg) // 40)
        for e in seg[::step]:
            print(f"  {e['tick']:>5}{e['y']:>8.0f}{e['gap']:>8.0f}{e['dev']:>7.0f}"
                  f"{e['vy']:>8.0f}{e['d_front']:>9.0f}{e['ratio']:>7.2f}"
                  f"{e['spk']:>4}{'  拍' if e['flap'] else '   -':>6}")
    if not shown:
        print(f"没有落在 [{a.min_score}, {a.max_score}] 分区间的一局")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
