#!/usr/bin/env python
"""端到端验证：用**页面自己的算法**（关卡生成 + 双向几何）跑分，确认与评测台一致。

为什么需要这个脚本
------------------
改了 `demo/app.js`（关卡生成的 maxClimb 约束 + 双向投射的几何计算）之后，光看
`scripts/flappy_bench.py` 是不够的 —— 那是**另一份实现**。两边一旦漂移，
页面上的数字就和报告对不上，实验也就不可复现了。

所以这里把 app.js 里的两个函数**逐行照抄**到 Python 来跑：
  · `spawnPipe()` 的可解性约束（上一根缺口中心 + maxClimb）
  · `bidiBody()` 的几何（等效半宽 0.55、方向由缺口高低决定、vy_gate）
然后对比 `flappy_bench.py --modes bidi` 的结果。两边应当落在同一个量级。

用法
    D:/Code/FlyBrain/env/python.exe scripts/flappy_page_parity.py --games 40
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


class PageWorld(fb.World):
    """完全照抄 `demo/app.js` 的关卡生成与 Flappy 物理。

    与 `fb.World` 的**唯一**差别就是 `_next_gap_top`：这里用 app.js 的
    `spawnPipe()` 那段可解性约束。物理与碰撞一行不改。
    """

    def __init__(self, rng, max_climb=40.0):
        self._mc = max_climb
        super().__init__(rng, gap_top=fb.NATURAL_GAP_TOP, solvable=True,
                         max_climb=max_climb)

    def _next_gap_top(self) -> float:
        """= app.js 的 spawnPipe()：lo=70, hi=G.H-GAP-150=402"""
        lo = 70.0
        hi = fb.G_H - fb.GAP - 150.0
        v = lo + self.rng.random() * (hi - lo)          # Math.random() 的默认值
        last = self.pipes[-1] if self.pipes else None
        if last is not None:
            prev_c = last["top"] + fb.GAP / 2.0
            top_max = min(hi, prev_c + self._mc - fb.GAP / 2.0)
            v = (lo + self.rng.random() * (top_max - lo)) if top_max > lo else lo
        return float(v)


def page_geometry(w: fb.World, y: float, vy: float, gap_c: float, margin: float):
    """= app.js 的 bidiBody()：返回 (up, theta_deg, amp) 或 None（不驱动）。"""
    up = gap_c < y - margin
    if not (up or gap_c > y + margin):
        return None
    h_px = max((fb.GROUND_Y - y) if up else y, 1.0)
    dist = max(h_px / fb.PX_PER_M, 0.02)
    theta = np.degrees(2 * np.arctan2(0.55, dist))
    x = max(theta, 0.0) ** 3.0
    amp = x / (x + 30.0 ** 3.0)
    return bool(up), float(theta), float(amp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spiking_full")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--max-ticks", type=int, default=5000)
    ap.add_argument("--max-climb", type=float, default=40.0)
    ap.add_argument("--gap-margin", type=float, default=18.0)
    a = ap.parse_args()

    h = fb.Harness(a.asset)
    dn01 = h.dn01_idx()
    wmap = h.wmap
    recent: list[int] = []
    scores = []
    ok_ticks = 0

    for k in range(a.games):
        rng = np.random.default_rng(1000 + k)
        w = PageWorld(rng, max_climb=a.max_climb)
        h.base.reset()
        recent.clear()
        cooldown_left = 0.0
        for tick in range(a.max_ticks):
            # ---- 页面算法：算几何 → 送驱动 → 脑推进
            g = page_geometry(w, w.y, w.vy, w.gap_center(), a.gap_margin)
            if g is not None and (g[0] and w.vy > 0 or (not g[0]) and w.vy < 0):
                up, _theta, amp = g
                parts_i, parts_p = [], []
                for gname in ("LC4", "LPLC2"):
                    u = h.u[gname]
                    m = (u < 0.5) if up else (u >= 0.5)
                    parts_i.append(h.gidx[gname][m])
                    parts_p.append(np.full(int(m.sum()), amp))
                import torch
                cidx = torch.as_tensor(np.concatenate(parts_i), dtype=torch.long)
                pv = torch.as_tensor(np.concatenate(parts_p), dtype=torch.float32)
                h.base.step(clamp=(cidx, pv))
            else:
                h.base.step()
            spk = int(h.base.S[dn01].sum())
            recent.append(spk)
            if len(recent) > fb.SPIKE_WINDOW:
                recent.pop(0)
            want = sum(recent) >= 1
            if want and cooldown_left <= 0 and w.t >= fb.WARM_S:
                cooldown_left = h.cooldown
            else:
                want = False
            cooldown_left -= fb.TICK_S
            w.step(want)
            if w.dead:
                break
        scores.append(w.score)
        ok_ticks += tick + 1

    sc = np.array(scores, float)
    n = len(sc)
    print(f"页面算法（app.js 逐行照抄）n={n}  max_climb={a.max_climb} "
          f"死区={a.gap_margin}")
    print(f"  均分 {sc.mean():.2f} ± {sc.std():.2f}  中位 {np.median(sc):.0f}  "
          f"最高 {sc.max():.0f}  过管率 {(sc >= 1).mean():.0%}")
    print(f"  ≤2 分占比 {(sc <= 2).mean():.0%}   ≥20 分占比 {(sc >= 20).mean():.0%}")

    # ---- 对照：评测台同一配置
    spec = dict(s50=15.0, s50size=30.0, gain=1.0, width=0.5, baseline=0.25,
                ground_aware=1, elev_scale=1.0, pipe_w=1.0, span=0.6, base=0.6,
                gap_margin=a.gap_margin, vy_gate=1, max_climb=a.max_climb)
    ref = [h.play("bidi", spec, seed=1000 + k, max_ticks=a.max_ticks, policy="brain")
           for k in range(n)]
    rs = np.array([r["score"] for r in ref], float)
    print(f"\n评测台（flappy_bench --modes bidi）n={n}")
    print(f"  均分 {rs.mean():.2f} ± {rs.std():.2f}  中位 {np.median(rs):.0f}  "
          f"最高 {rs.max():.0f}  过管率 {(rs >= 1).mean():.0%}")
    print(f"  ≤2 分占比 {(rs <= 2).mean():.0%}   ≥20 分占比 {(rs >= 20).mean():.0%}")

    ratio = sc.mean() / max(rs.mean(), 1e-9)
    print(f"\n页面/评测台 均分比 = {ratio:.2f}")
    if 0.6 <= ratio <= 1.6:
        print("→ 两边在同一个量级，页面与报告的数字可以对上。")
    else:
        print("→ ⚠️ 两边差距明显，app.js 与 bench 的实现已经漂移，需要对齐。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
