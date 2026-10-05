"""调关卡生成的"随机性 vs 可玩性"平衡。

背景：原来的生成器**冻死成常数**（200 根管子 top 全是 205、标准差 0）。
修成有界随机游走后，缺口真的铺满了全量程，但均分从 ~40 掉到 ~18 ——
太容易死，而演示需要鸟能连飞一段时间（否则画面一直重开，观感很差）。

这个脚本用**服务端真实对局**扫几组参数，报均分/中位/最高/拍翅，
以及"生成器的随机性指标"，让选择有据可依。

用法: python scripts/tune_levels.py [--games 12] [--ticks 8000]
"""
from __future__ import annotations

import argparse
import pathlib
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "demo"))

from server import FIRST_GAP_EXTRA, Session, GameWorld   # noqa: E402


def generator_stats(*, max_climb, gap_lo, gap_hi, jitter, max_drop, n=8000, seed=11):
    """生成器本身的统计（不跑鸟）：量程覆盖、独特性、跳变分布。"""
    w = GameWorld(max_climb=max_climb, first_gap_extra=0.0, seed=seed,
                  gap_lo=gap_lo, gap_hi=gap_hi, climb_jitter=jitter,
                  max_drop=max_drop)
    w.pipes.append({"x": w.G_W + 30.0, "top": 250.0, "passed": False})
    w.spawned += 1
    tops = [250.0]
    up_bad = dn_bad = 0
    for _ in range(n):
        prev_c = w.pipes[-1]["top"] + w.GAP / 2
        t = w._next_gap_top()
        c = t + w.GAP / 2
        if c - prev_c > w.max_climb + 1e-6:
            up_bad += 1
        if prev_c - c > w.max_drop + 1e-6:
            dn_bad += 1
        w.pipes.append({"x": w.G_W + 30.0, "top": float(t), "passed": False})
        w.spawned += 1
        tops.append(t)
    # 跳变用"生成时的真实配对"算，避免 off-by-one（我自己踩过）
    jumps = []
    for i in range(1, len(tops) - 1):
        jumps.append(tops[i + 1] - tops[i])
    rise = [-d for d in jumps if d < 0]
    fall = [d for d in jumps if d > 0]
    return dict(
        std=st.pstdev(tops), uniq=len(set(round(t, 2) for t in tops)), n=len(tops),
        lo=min(tops), hi=max(tops), mean=st.mean(tops),
        rise_pct=100 * len(rise) / max(len(jumps), 1),
        max_rise=(max(rise) if rise else 0), max_fall=(max(fall) if fall else 0),
        up_bad=up_bad, dn_bad=dn_bad,
    )


def play(sess, *, games, ticks, max_climb, gap_lo, gap_hi, jitter, max_drop, seed0=0):
    rows = []
    for k in range(games):
        with sess.game_lock:
            sess.game = GameWorld(max_climb=max_climb,
                                  first_gap_extra=FIRST_GAP_EXTRA,
                                  seed=seed0 + k, gap_lo=gap_lo, gap_hi=gap_hi,
                                  climb_jitter=jitter, max_drop=max_drop)
            sess.brain.set_seed(7)
            sess.reset()
            n = 0
            while n < ticks and not sess.game.dead:
                sess._game_tick()
                n += 1
            rows.append((sess.game.score, n))
    sc = [r[0] for r in rows]
    return dict(mean=st.mean(sc), med=st.median(sc), mx=max(sc), mn=min(sc),
                ticks=st.mean(r[1] for r in rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--ticks", type=int, default=8000)
    a = ap.parse_args()

    # (说明, max_climb, gap_lo, gap_hi, jitter, max_drop)
    combos = [
        ("旧行为(冻死)      ", 40.0, 70.0, 286.0, 0.0, 1e9),
        ("全量程(当前)      ", 110.0, 70.0, 286.0, 18.0, 250.0),
        ("全量程+小抖动     ", 110.0, 70.0, 286.0, 10.0, 250.0),
        ("收窄上界 240      ", 110.0, 70.0, 240.0, 14.0, 220.0),
        ("收窄上界 250      ", 110.0, 70.0, 250.0, 16.0, 230.0),
        ("收窄 + 限下坠 180 ", 110.0, 70.0, 250.0, 16.0, 180.0),
        ("居中(上界 200)    ", 100.0, 80.0, 200.0, 14.0, 160.0),
    ]
    print(f"载入脑…")
    sess = Session("spiking_full", "cpu", start_loop=False)
    print(f"  {'方案':<20}{'均分':>7}{'中位':>6}{'最高':>6}{'存活tick':>9}   "
          f"{'std':>6}{'不同值':>7}{'上移%':>6}{'最大升':>7}{'最大降':>7}  违规")
    for tag, mc, glo, ghi, jit, mdrop in combos:
        g = generator_stats(max_climb=mc, gap_lo=glo, gap_hi=ghi, jitter=jit,
                            max_drop=mdrop)
        p = play(sess, games=a.games, ticks=a.ticks, max_climb=mc, gap_lo=glo,
                 gap_hi=ghi, jitter=jit, max_drop=mdrop)
        print(f"  {tag:<20}{p['mean']:7.1f}{p['med']:6.0f}{p['mx']:6d}{p['ticks']:9.0f}   "
              f"{g['std']:6.1f}{g['uniq']:7d}{g['rise_pct']:5.0f}%{g['max_rise']:7.0f}"
              f"{g['max_fall']:7.0f}  {g['up_bad']}/{g['dn_bad']}")
    sess.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
