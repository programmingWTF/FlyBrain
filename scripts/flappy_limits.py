#!/usr/bin/env python
"""两件必须分开算账的事：读出窗口（脑侧）与游戏物理（游戏侧）。

已知（ESCAPE.md §6）：腹侧 LPLC2 地面反射停在均分 2.23、最高 7，
而 `lookahead` 外部规划器在**同一套物理**上能活过 1500 tick、过 21 根管子。
所以问题在反射本身。这个脚本把两个可能的原因分开量：

  1. **读出窗口**：现在拍翅判据是"最近 5 tick 的 DNp01 脉冲 >= 1"，
     即 100ms 的观察窗。窗口越长，脉冲越容易被看到，但拍翅会延迟。
     DNp01 满驱动时才 ~0.45 脉冲/tick，这个窗口可能把反射卡死。
  2. **游戏物理**：重力/拍翅冲量是外部常数。如果把它们调到更容易，
     反射能拿多少？——这条**不是**"改脑的输入"，必须单独报，
     而且要明确说这是改游戏，不是模型的能力。

用法
    D:/Code/FlyBrain/env/python.exe scripts/flappy_limits.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import flappy_bench as fb   # noqa: E402


def run(h: fb.Harness, mode: str, games: int, ticks: int, **kw) -> dict:
    proj_name, policy = fb.MODES[mode]
    spec = dict(s50=15.0, s50size=30.0, gain=1.0, width=0.5, baseline=0.25,
                ground_aware=1, elev_scale=1.0, pipe_w=1.0, span=0.6, base=0.6)
    spec.update(kw)
    rows = [h.play(proj_name, spec, seed=1000 + k, max_ticks=ticks, policy=policy)
            for k in range(games)]
    sc = np.array([r["score"] for r in rows], float)
    return dict(mean=sc.mean(), sd=sc.std(), mx=sc.max(), n=len(rows),
                ticks=np.mean([r["ticks"] for r in rows]),
                flaps=np.mean([r["flaps"] for r in rows]),
                pass_rate=float((sc >= 1).mean()))


def main() -> int:
    h = fb.Harness("spiking_full")
    print(f"资产 spiking_full  N={h.base.N:,}  载入 {h.load_time:.1f}s\n")

    print("=" * 100)
    print("实验 A：读出窗口（脑侧）—— 拍翅判据 = 最近 N tick 的 DNp01 脉冲")
    print(f"{'窗口 tick':>10}{'窗口 ms':>10}{'均分':>8}{'标准差':>8}{'最高':>6}"
          f"{'存活 tick':>11}{'拍翅/局':>9}{'过管率':>8}")
    rows_a = []
    for win in (1, 2, 3, 5, 8):
        fb.SPIKE_WINDOW = win
        r = run(h, "ground_lock", games=24, ticks=1200)
        rows_a.append((win, r))
        print(f"{win:>10}{win * 20:>10}{r['mean']:>8.2f}{r['sd']:>8.2f}{r['mx']:>6.0f}"
              f"{r['ticks']:>11.0f}{r['flaps']:>9.2f}{r['pass_rate']:>8.0%}")
    fb.SPIKE_WINDOW = 5
    best = max(rows_a, key=lambda t: t[1]["mean"])
    print(f"→ 最佳窗口 {best[0]} tick（{best[0] * 20} ms），均分 {best[1]['mean']:.2f}"
          f"；当前默认 5 tick 是 {dict(rows_a)[5]['mean']:.2f}")

    print("\n" + "=" * 100)
    print("实验 B：游戏物理（游戏侧！不是模型能力）—— 只改重力/拍翅冲量，投射不变")
    print(f"{'GRAV':>7}{'FLAP_V':>9}{'均分':>8}{'最高':>6}{'存活 tick':>11}{'拍翅/局':>9}")
    rows_b = []
    g0, f0 = fb.GRAV, fb.FLAP_V
    for grav, flap in ((1180, -340), (900, -300), (700, -280), (500, -260), (1400, -380)):
        fb.GRAV, fb.FLAP_V = float(grav), float(flap)
        r = run(h, "ground_lock", games=24, ticks=1500)
        rows_b.append((grav, flap, r))
        print(f"{grav:>7}{flap:>9}{r['mean']:>8.2f}{r['mx']:>6.0f}{r['ticks']:>11.0f}"
              f"{r['flaps']:>9.2f}")
    fb.GRAV, fb.FLAP_V = g0, f0
    print("→ 注意：这一栏是**改游戏**，不是**改脑的输入**。原常数 "
          f"GRAV={g0} FLAP_V={f0} 保持不动，报告里必须分开写。")

    print("\n" + "=" * 100)
    print("实验 C：外部规划器在同一套物理下的上限（证明世界不是瓶颈）")
    for look in (40, 80, 160):
        h.oracle_look = look
        r = run(h, "lookahead", games=6, ticks=1500)
        print(f"  lookahead look={look:<5} 均分 {r['mean']:.1f}  最高 {r['mx']:.0f}  "
              f"存活 {r['ticks']:.0f} tick")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
