#!/usr/bin/env python
"""为什么死区只到 ~22px？把 bidi 在 margin=22 与 margin=35 下的轨迹并排看。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import flappy_bench as fb   # noqa: E402


def main() -> int:
    h = fb.Harness("spiking_full")
    for margin in (22.0, 35.0):
        spec = dict(s50=15.0, s50size=30.0, gain=1.0, width=0.5, baseline=0.25,
                    ground_aware=1, elev_scale=1.0, pipe_w=1.0, span=0.6, base=0.6,
                    gap_margin=margin, vy_gate=1, max_climb=80.0)
        r = h.play("bidi", spec, seed=1000, max_ticks=3000, policy="brain", trace=True)
        print("=" * 112)
        print(f"margin={margin}  分数 {r['score']}  死因 {r['cause']}  "
              f"拍翅 {r['flaps']}  存活 {r['ticks']} tick")
        tr = r["trace"]
        seg = tr[58:min(len(tr), 58 + 40)]
        print(f"  {'tick':>5}{'y':>8}{'gap':>8}{'dev':>7}{'vy':>8}{'d_front':>9}"
              f"{'ratio':>7}{'spk':>4}{'flap':>6}")
        for e in seg:
            print(f"  {e['tick']:>5}{e['y']:>8.0f}{e['gap']:>8.0f}{e['dev']:>7.0f}"
                  f"{e['vy']:>8.0f}{e['d_front']:>9.0f}{e['ratio']:>7.2f}"
                  f"{e['spk']:>4}{'  拍' if e['flap'] else '   -':>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
