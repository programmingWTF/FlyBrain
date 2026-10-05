"""为桌面上的算法说明文档生成配图（真实数据，不是示意图）。

产出：
  · 20ms 闭环示意图（SVG）
  · LC4/LPLC2/DNp01 的真实发放栅格 + 触发时刻（SVG）
  · 缺口生成器的分布对照（修前冻死 vs 修后随机）（SVG）

数据全部来自**正在跑的服务端**（/api/game/state）或本地直接跑仿真，
不用手写的假数。

用法: python scripts/make_doc_figures.py --out <目录>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "demo"))


# ---------------------------------------------------------------- 采集真实发放
def collect_spikes(asset: str, ticks: int = 220, seed: int = 7):
    """直接跑服务端会话，逐 tick 记录几个关键群的发放（真实数据）。

    返回 dict: per-tick 的 LC4/LPLC2/DNp01 发放细胞索引（相对该群的序号）
    以及 y/vy/gap，供画栅格与轨迹。
    """
    from server import BIDI, GROUND_Y, FIRST_GAP_EXTRA, MAX_CLIMB, Session, GameWorld
    import torch

    sess = Session(asset, "cpu", start_loop=False)
    rows = []
    with sess.game_lock:
        sess.game = GameWorld(max_climb=MAX_CLIMB, first_gap_extra=FIRST_GAP_EXTRA,
                              seed=seed)
        sess.brain.set_seed(seed)
        sess.reset()
        lc4 = sess.g["LC4"]; lplc2 = sess.g["LPLC2"]; dn01 = sess.g["DNp01"]
        for k in range(ticks):
            w = sess.game
            gap_c = w.gap_center()
            body = dict(BIDI)
            body.update(y=w.y, vy=w.vy, gap=gap_c, ground_y=GROUND_Y)
            idx, pv, _ = sess._bidi_plan(body)
            clamp = (torch.cat(idx), torch.cat(pv)) if idx else None
            sess.brain.step(clamp=clamp)
            S = sess.brain.S
            rows.append(dict(
                tick=k, y=round(w.y, 1), vy=round(w.vy, 1),
                gap=round(gap_c, 1),
                branch=sess.last_plan.get("branch", ""),
                amp=round(float(sess.last_plan.get("amp", 0.0)), 4),
                lc4=[int(v) for v in torch.nonzero(S[lc4], as_tuple=False).flatten().tolist()],
                lplc2=[int(v) for v in torch.nonzero(S[lplc2], as_tuple=False).flatten().tolist()],
                dn01=[int(v) for v in torch.nonzero(S[dn01], as_tuple=False).flatten().tolist()],
            ))
            dn = int(S[dn01].sum())
            sess.dn_tick.append(dn)
            while len(sess.dn_tick) > 5:
                sess.dn_tick.popleft()
            flap = sum(sess.dn_tick) >= 1
            w.cooldown = max(0.0, w.cooldown - 0.02)
            if flap and w.cooldown > 0:
                flap = False
            elif flap:
                w.cooldown = 0.14
            w.step(flap)
            rows[-1]["flap"] = bool(flap)
            if w.dead:
                break
        n_lc4, n_lplc2 = len(lc4), len(lplc2)
    sess.close()
    return rows, dict(n_lc4=n_lc4, n_lplc2=n_lplc2)


# ---------------------------------------------------------------- 画栅格
C_LC4 = "#c2410c"      # 角速度通道
C_LPLC2 = "#7c3aed"    # 角大小通道
C_DN = "#be123c"       # 巨纤维（指令）
C_INK = "#16181d"
C_RULE = "#d9d5cc"
C_PAPER = "#fbfaf7"


def fig_raster(rows, meta, *, w=980, h=520) -> str:
    """LC4 / LPLC2 / DNp01 的逐 tick 发放栅格 + 鸟的 y 轨迹 + 拍翅时刻。

    LC4/LPLC2 是**均匀降采样**后的代表细胞（每群约 26 个）：全画会有 7000+ 个圆点、
    SVG 涨到 550 KB，而读者要看的结构（哪些 tick 在发放、和拍翅的对应关系）
    降采样后一样清楚。DNp01 只有 2 个细胞，保持全分辨率 —— 它是指令读出，
    每一点都算数。
    """
    T = len(rows)
    LANE_ROWS = 26               # 每条通道画多少个代表细胞
    L, R, TOP = 26, 22, 54
    lane_h = 138      # 含 22px 标签区 + 116px 点区
    pw = w - L - R
    x = lambda k: L + (k / max(T - 1, 1)) * pw

    def pick(idxs, k):
        """把细胞序号均匀映射到 k 行（降采样）。"""
        if not idxs:
            return []
        step = max(len(idxs) / k, 1.0)
        return sorted({int(i / step) for i in idxs})

    def lane(k, key, label, n, color, sub):
        y0 = TOP + k * lane_h
        # 标签放在**条带上方**（左栏太窄，挤进去会压到点上，也没法容纳长说明）
        out = [f'<text x="{L}" y="{y0}" class="lbl" fill="{color}">{label}</text>',
               f'<text x="{L + 62}" y="{y0}" class="sub">{sub}</text>',
               f'<line x1="{L}" y1="{y0 + lane_h - 12}" x2="{w - R}" y2="{y0 + lane_h - 12}" '
               f'stroke="{C_RULE}" stroke-width="1"/>']
        span = lane_h - 46
        always_full = (label == "DNp01")
        for r in rows:
            carr = r[key]
            cells = carr if always_full else pick(carr, LANE_ROWS)
            for cell in cells:
                fy = (cell / max(n - 1, 1)) if n > 1 else 0.0
                cy = y0 + 24 + fy * span      # +24 = 让开标签区
                out.append(f'<circle cx="{x(r["tick"]):.1f}" cy="{cy:.1f}" r="1.7" '
                           f'fill="{color}" opacity="0.9"/>')
        return "\n".join(out), y0 + lane_h - 12

    a, ya = lane(0, "lc4", "LC4", meta["n_lc4"], C_LC4,
                 f'角速度通道 · {meta["n_lc4"]} 细胞（图中 {LANE_ROWS} 个代表）')
    b, yb = lane(1, "lplc2", "LPLC2", meta["n_lplc2"], C_LPLC2,
                 f'角大小通道 · {meta["n_lplc2"]} 细胞（图中 {LANE_ROWS} 个代表）')
    c, yc = lane(2, "dn01", "DNp01", 2, C_DN, "巨纤维 · 2 细胞（指令读出，全画）")

    # 鸟的 y 轨迹，叠在一条独立基线上（右侧刻度）
    base = yc + 58
    span_y = 74
    ys = [r["y"] for r in rows]
    lo, hi = min(ys), max(ys)
    fy = lambda v: base + span_y - (v - lo) / max(hi - lo, 1e-6) * span_y
    path = " ".join(("M" if i == 0 else "L") + f"{x(r['tick']):.1f} {fy(r['y']):.1f}"
                    for i, r in enumerate(rows))
    flaps = [r for r in rows if r.get("flap")]
    flap_marks = "\n".join(
        f'<line x1="{x(r["tick"]):.1f}" y1="{base - 4}" x2="{x(r["tick"]):.1f}" '
        f'y2="{base + span_y + 6}" stroke="{C_DN}" stroke-width="1" opacity="0.5"/>'
        for r in flaps)

    # 20ms 刻度：每 5 tick（100ms）一条竖线
    grid = []
    for k in range(0, T, 5):
        grid.append(f'<line x1="{x(k):.1f}" y1="{TOP - 6}" x2="{x(k):.1f}" y2="{base + span_y + 10}" '
                    f'stroke="{C_RULE}" stroke-width="0.6" opacity="0.8"/>')
        grid.append(f'<text x="{x(k):.1f}" y="{base + span_y + 26}" class="tick">{k*20}ms</text>')

    return f'''<svg viewBox="0 0 {w} {h + 26}" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="LC4、LPLC2 与 DNp01 的逐 tick 真实发放栅格，以及鸟的高度轨迹与拍翅时刻">
  <rect width="{w}" height="{h + 26}" fill="{C_PAPER}"/>
  {"".join(grid)}
  {a}
  {b}
  {c}
  <text x="{L}" y="{base - 6}" class="lbl">鸟的 y</text>
  <text x="{L + 62}" y="{base - 6}" class="sub">高度轨迹 · {lo:.0f}–{hi:.0f} px</text>
  <line x1="{L}" y1="{base + span_y}" x2="{w - R}" y2="{base + span_y}" stroke="{C_RULE}"/>
  <path d="{path}" fill="none" stroke="{C_INK}" stroke-width="1.6" opacity="0.85"/>
  {flap_marks}
  <text x="{w - R}" y="{base + span_y + 26}" class="tick" text-anchor="end">竖线 = 一次拍翅 · 每格 20 ms</text>
</svg>'''


def fig_loop(w=980, h=250) -> str:
    """20ms 闭环：几何 → 驱动 → 脑 → 读出 → 动作 → 物理。"""
    boxes = [
        ("视觉几何", "缺口中心 / 离地高度"),
        ("逼近投射", "θ → Naka-Rushton"),
        ("LC4 · LPLC2", "半视野整半招募"),
        ("DNp01", "漏积分，需持续驱动"),
        ("拍翅", "vy = −340"),
        ("物理", "y += vy·dt"),
    ]
    bw, bh, gap = 138, 66, 22
    total = len(boxes) * bw + (len(boxes) - 1) * gap
    x0 = (w - total) / 2
    out = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" '
           f'aria-label="20 毫秒闭环：从视觉几何到物理推进的六个环节，再回到视觉几何">',
           f'<rect width="{w}" height="{h}" fill="{C_PAPER}"/>']
    cy = 96
    for i, (t, s) in enumerate(boxes):
        bx = x0 + i * (bw + gap)
        col = C_LC4 if "LC4" in t else C_LPLC2 if "LPLC2" in t else C_DN if "DNp" in t else C_INK
        out.append(f'<rect x="{bx:.1f}" y="{cy - bh/2}" width="{bw}" height="{bh}" rx="2" '
                   f'fill="#fff" stroke="{col}" stroke-width="{1.6 if col != C_INK else 1.1}"/>')
        out.append(f'<text x="{bx + bw/2:.1f}" y="{cy - 6}" class="bt" fill="{col}" '
                   f'text-anchor="middle">{t}</text>')
        out.append(f'<text x="{bx + bw/2:.1f}" y="{cy + 14}" class="bs" text-anchor="middle">{s}</text>')
        if i:
            ax = bx - gap / 2
            out.append(f'<path d="M{ax - 5:.1f} {cy} l6 -4 v8 z" fill="{C_RULE}"/>')
            out.append(f'<line x1="{ax - gap + 2:.1f}" y1="{cy}" x2="{ax - 4:.1f}" y2="{cy}" '
                       f'stroke="{C_RULE}" stroke-width="1.2"/>')
    # 回环箭头
    yb = cy + bh / 2
    out.append(f'<path d="M{x0 + total - bw/2:.1f} {yb} V{yb + 42} H{x0 + bw/2:.1f} V{yb}" '
               f'fill="none" stroke="{C_RULE}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    out.append(f'<path d="M{x0 + bw/2 - 5:.1f} {yb + 8} l5 5 l5 -5" fill="none" '
               f'stroke="{C_RULE}" stroke-width="1.2"/>')
    out.append(f'<text x="{w/2}" y="{yb + 36}" class="tick" text-anchor="middle">'
               f'一个 tick = 20 ms · 服务端固定 50 Hz · 决策与物理严格 1:1</text>')
    out.append(f'<text x="{w/2}" y="{cy - bh/2 - 26}" class="bt2" text-anchor="middle">'
               f'控制回路：每 20 毫秒闭合一次</text>')
    out.append('</svg>')
    return "\n".join(out)


def fig_gapgen(*, w=980, h=300) -> str:
    """缺口生成的分布对照：修前冻死成常数，修后铺满量程。"""
    import numpy as np
    from server import GameWorld

    def sample(**kw):
        w0 = GameWorld(first_gap_extra=0.0, seed=11, **kw)
        w0.pipes.append({"x": w0.G_W + 30.0, "top": 250.0, "passed": False})
        w0.spawned += 1
        out = []
        for _ in range(4000):
            t = w0._next_gap_top()
            w0.pipes.append({"x": w0.G_W + 30.0, "top": float(t), "passed": False})
            w0.spawned += 1
            out.append(t)
        return np.asarray(out, dtype=float)

    old = sample(max_climb=40.0, gap_hi=286.0, climb_jitter=0.0, max_drop=1e9)
    new = sample(max_climb=110.0, gap_hi=250.0, climb_jitter=16.0, max_drop=230.0)

    L, R, TOP = 26, 24, 54
    rows_h = 96
    pw = w - L - R
    GLO, GHI = 70.0, 250.0        # = GameWorld 的 GAP_LO / GAP_HI
    bins = np.linspace(GLO, GHI, 25)

    def panel(data, k, title, note, color):
        y0 = TOP + k * rows_h
        hist, _ = np.histogram(data, bins=bins)
        hist = hist / hist.max()
        out = [f'<text x="{L}" y="{y0 + 2}" class="lbl" fill="{color}">{title}</text>',
               f'<text x="{L + 76}" y="{y0 + 2}" class="sub">{note}</text>']
        for i, v in enumerate(hist):
            bx = L + (bins[i] - GLO) / (GHI - GLO) * pw
            bwid = pw / (len(bins) - 1) - 1.4
            hh = v * 32
            out.append(f'<rect x="{bx:.1f}" y="{y0 + 52 - hh:.1f}" width="{bwid:.1f}" '
                       f'height="{hh:.1f}" fill="{color}" opacity="0.72"/>')
        out.append(f'<line x1="{L}" y1="{y0 + 52}" x2="{w - R}" y2="{y0 + 52}" stroke="{C_RULE}"/>')
        out.append(f'<text x="{L}" y="{y0 + 70}" class="tick">top = {GLO:.0f}（贴顶）</text>')
        out.append(f'<text x="{w - R}" y="{y0 + 70}" class="tick" text-anchor="end">'
                   f'top = {GHI:.0f}（贴近地面）</text>')
        return "\n".join(out)

    return f'''<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="缺口生成分布对照：修前所有管子落在同一个位置，修后铺满整个量程">
  <rect width="{w}" height="{h}" fill="{C_PAPER}"/>
  {panel(old, 0, "修前", "全部落在同一个值上（标准差 0.0）", "#9ca3af")}
  {panel(new, 1, "修后", "铺满量程、标准差约 50 px", C_LPLC2)}
  <text x="{w/2}" y="{h - 8}" class="tick" text-anchor="middle">缺口 top 的分布（每根管子一次，4000 根，横轴 = 真实可行量程）</text>
</svg>'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/doc_figs")
    ap.add_argument("--asset", default="spiking_full")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("采集真实发放…")
    rows, meta = collect_spikes(a.asset, ticks=220, seed=7)
    print(f"  {len(rows)} tick；LC4={meta['n_lc4']} LPLC2={meta['n_lplc2']}")
    n_flap = sum(1 for r in rows if r.get("flap"))
    n_dn = sum(len(r["dn01"]) for r in rows)
    print(f"  DNp01 发放总次数 {n_dn}；拍翅 {n_flap} 次")

    (out / "raster.svg").write_text(fig_raster(rows, meta), encoding="utf-8")
    (out / "loop.svg").write_text(fig_loop(), encoding="utf-8")
    (out / "gapgen.svg").write_text(fig_gapgen(), encoding="utf-8")
    (out / "spikes.json").write_text(json.dumps(
        dict(meta=meta, n_flap=n_flap, n_dn=n_dn, ticks=len(rows),
             dn01_fired_ticks=[r["tick"] for r in rows if r["dn01"]],
             flaps=[r["tick"] for r in rows if r.get("flap")],
             y=[r["y"] for r in rows]),
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out}/raster.svg  loop.svg  gapgen.svg  spikes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
