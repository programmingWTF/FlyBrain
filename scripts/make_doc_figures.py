"""为桌面上的算法教材生成配图 —— 全部取自**真实仿真数据**，不是示意图。

产出（内联进 HTML，单文件离线可看）：
  loop.svg      20ms 闭环：几何 → 投射 → 感觉群 → 指令 → 动作 → 物理
  raster.svg    LC4 / LPLC2 / DNp01 逐 tick 发放栅格 + 鸟的高度轨迹 + 拍翅时刻
  gapgen.svg    缺口生成的分布对照（修前冻死成常数 vs 修后铺满量程）
  membrane.svg  DNp01 膜电位轨迹（讲清"漏积分 + 阈值"到底长什么样）

用法: python scripts/make_doc_figures.py --out output/doc_figs
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "demo"))


# ---------------------------------------------------------------- 语义配色
C_LC4 = "#c2410c"      # 角速度通道
C_LPLC2 = "#7c3aed"    # 角大小通道
C_DN = "#be123c"       # 巨纤维（指令读出）
C_PHYS = "#1d4ed8"     # 物理
C_INK = "#14161b"
C_RULE = "#d7d4cd"
C_PAPER = "#fbfaf7"

SVG_OPEN = ('<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            'role="img" aria-label="{aria}" '
            'font-family="IBM Plex Mono, Consolas, monospace">')


# ---------------------------------------------------------------- 采集真实发放
def collect_spikes(asset: str, ticks: int = 220, seed: int = 7):
    """直接跑服务端会话，逐 tick 记录真实发放。"""
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
            body = dict(BIDI)
            body.update(y=w.y, vy=w.vy, gap=w.gap_center(), ground_y=GROUND_Y)
            idx, pv, _ = sess._bidi_plan(body)
            clamp = (torch.cat(idx), torch.cat(pv)) if idx else None
            sess.brain.step(clamp=clamp)
            S = sess.brain.S
            rows.append(dict(
                tick=k, y=round(w.y, 1), vy=round(w.vy, 1),
                gap=round(w.gap_center(), 1),
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


def membrane_trace(*, n=40, drive=0.1048, leak=0.818731, gain=3.0, thr=1.0):
    """按 LIF 公式**逐步算**出 DNp01 的膜电位轨迹（讲"漏积分 + 阈值"用）。

    这不是从仿真里抓的，而是照公式手算的 —— 所以它同时是一个可核对的算例：
    读者拿纸笔能复现每一个点。恒定驱动 0.1048 下应当是"每 5 个 tick 发一次"。
    """
    g, trace, fires = 0.0, [], []
    for t in range(n):
        g = leak * g + gain * drive
        if g >= thr:
            trace.append(round(g, 4))
            fires.append(t)
            g = 0.0
        else:
            trace.append(round(g, 4))
    return trace, fires


# ---------------------------------------------------------------- 图 1：闭环
def fig_loop(w=960, h=268) -> str:
    boxes = [
        ("1", "视觉几何", "缺口中心 / 离地高度"),
        ("2", "逼近投射", "θ → Naka-Rushton"),
        ("3", "LC4 · LPLC2", "半视野整半招募"),
        ("4", "DNp01", "漏积分，需持续驱动"),
        ("5", "拍翅", "vy = −340"),
        ("6", "物理", "y += vy·dt"),
    ]
    bw, bh, gap = 132, 72, 20
    total = len(boxes) * bw + (len(boxes) - 1) * gap
    x0 = (w - total) / 2
    cy = 108
    o = [SVG_OPEN.format(w=w, h=h,
                         aria="20 毫秒闭环：从视觉几何到物理推进，再回到视觉几何"),
         f'<rect width="{w}" height="{h}" fill="{C_PAPER}"/>']
    for i, (num, t, s) in enumerate(boxes):
        bx = x0 + i * (bw + gap)
        col = (C_LC4 if "LC4" in t else C_LPLC2 if "LPLC2" in t
               else C_DN if "DNp" in t else C_PHYS if t == "物理" else C_INK)
        strong = col != C_INK
        o.append(f'<rect x="{bx:.1f}" y="{cy-bh/2}" width="{bw}" height="{bh}" rx="2" '
                 f'fill="#fff" stroke="{col}" stroke-width="{1.7 if strong else 1.1}"/>')
        o.append(f'<text x="{bx+9:.1f}" y="{cy-16}" font-size="10" fill="{col}" '
                 f'opacity="0.75">{num}</text>')
        o.append(f'<text x="{bx+bw/2:.1f}" y="{cy-1}" font-size="14" font-weight="600" '
                 f'fill="{col}" text-anchor="middle">{t}</text>')
        o.append(f'<text x="{bx+bw/2:.1f}" y="{cy+16}" font-size="10.5" fill="#6b6b6b" '
                 f'text-anchor="middle">{s}</text>')
        if i:
            ax = bx - gap / 2
            o.append(f'<path d="M{ax-4:.1f} {cy} l6 -4.5 v9 z" fill="{C_RULE}"/>')
            o.append(f'<line x1="{ax-gap+3:.1f}" y1="{cy}" x2="{ax-3:.1f}" y2="{cy}" '
                     f'stroke="{C_RULE}" stroke-width="1.3"/>')
    yb = cy + bh / 2
    o.append(f'<path d="M{x0+total-bw/2:.1f} {yb} V{yb+40} H{x0+bw/2:.1f} V{yb}" fill="none" '
             f'stroke="{C_RULE}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    o.append(f'<path d="M{x0+bw/2-5:.1f} {yb+10} l5 5.5 l5 -5.5" fill="none" '
             f'stroke="{C_RULE}" stroke-width="1.2"/>')
    o.append(f'<text x="{w/2}" y="{yb+34}" font-size="11" fill="#6b6b6b" text-anchor="middle">'
             f'一个 tick = 20 ms　·　服务端固定 50 Hz　·　决策与物理严格 1:1</text>')
    o.append(f'<text x="{w/2}" y="{cy-bh/2-24}" font-size="13" font-weight="600" '
             f'fill="{C_INK}" text-anchor="middle">控制回路：每 20 毫秒闭合一次</text>')
    o.append('</svg>')
    return "\n".join(o)


# ---------------------------------------------------------------- 图 2：栅格
def fig_raster(rows, meta, *, w=960, h=520) -> str:
    T = len(rows)
    LANE_ROWS = 26
    L, R, TOP = 30, 24, 54
    lane_h = 138
    pw = w - L - R
    x = lambda k: L + (k / max(T - 1, 1)) * pw

    def pick(idxs, k):
        if not idxs:
            return []
        step = max(len(idxs) / k, 1.0)
        return sorted({int(i / step) for i in idxs})

    def lane(k, key, label, n, color, sub):
        y0 = TOP + k * lane_h
        out = [f'<text x="{L}" y="{y0}" font-size="13" font-weight="600" '
               f'fill="{color}">{label}</text>',
               f'<text x="{L+64}" y="{y0}" font-size="10.5" fill="#6b6b6b">{sub}</text>',
               f'<line x1="{L}" y1="{y0+lane_h-12}" x2="{w-R}" y2="{y0+lane_h-12}" '
               f'stroke="{C_RULE}" stroke-width="1"/>']
        span = lane_h - 46
        full = (label == "DNp01")
        for r in rows:
            cells = r[key] if full else pick(r[key], LANE_ROWS)
            for cell in cells:
                fy = (cell / max(n - 1, 1)) if n > 1 else 0.0
                out.append(f'<circle cx="{x(r["tick"]):.1f}" cy="{y0+24+fy*span:.1f}" '
                           f'r="1.75" fill="{color}" opacity="0.9"/>')
        return "\n".join(out), y0 + lane_h - 12

    a, ya = lane(0, "lc4", "LC4", meta["n_lc4"], C_LC4,
                 f'角速度通道 · {meta["n_lc4"]} 细胞（图中 {LANE_ROWS} 个代表）')
    b, yb = lane(1, "lplc2", "LPLC2", meta["n_lplc2"], C_LPLC2,
                 f'角大小通道 · {meta["n_lplc2"]} 细胞（图中 {LANE_ROWS} 个代表）')
    c, yc = lane(2, "dn01", "DNp01", 2, C_DN, "巨纤维 · 2 细胞（指令读出，全画）")

    base = yc + 54
    span_y = 70
    ys = [r["y"] for r in rows]
    lo, hi = min(ys), max(ys)
    fy = lambda v: base + span_y - (v - lo) / max(hi - lo, 1e-6) * span_y
    path = " ".join(("M" if i == 0 else "L") + f"{x(r['tick']):.1f} {fy(r['y']):.1f}"
                    for i, r in enumerate(rows))
    flaps = [r for r in rows if r.get("flap")]
    marks = "\n".join(
        f'<line x1="{x(r["tick"]):.1f}" y1="{base-6}" x2="{x(r["tick"]):.1f}" '
        f'y2="{base+span_y+8}" stroke="{C_DN}" stroke-width="1" opacity="0.45"/>'
        for r in flaps)
    grid = []
    for k in range(0, T, 5):
        grid.append(f'<line x1="{x(k):.1f}" y1="{TOP-18}" x2="{x(k):.1f}" '
                    f'y2="{base+span_y+12}" stroke="{C_RULE}" stroke-width="0.6" opacity="0.75"/>')
        grid.append(f'<text x="{x(k):.1f}" y="{base+span_y+26}" font-size="9.5" '
                    f'fill="#8a8a8a" text-anchor="middle">{k*20}</text>')
    grid.append(f'<text x="{w-R}" y="{base+span_y+26}" font-size="9.5" fill="#8a8a8a" '
                f'text-anchor="end">ms</text>')
    return (SVG_OPEN.format(w=w, h=h + 30,
                            aria="LC4、LPLC2 与 DNp01 的逐 tick 真实发放栅格，"
                                 "以及鸟的高度轨迹与拍翅时刻") + "\n" +
            f'<rect width="{w}" height="{h+30}" fill="{C_PAPER}"/>\n' +
            "\n".join(grid) + "\n" + a + "\n" + b + "\n" + c + "\n" +
            f'<text x="{L}" y="{base-8}" font-size="13" font-weight="600" fill="{C_INK}">'
            f'鸟的 y</text>\n'
            f'<text x="{L+64}" y="{base-8}" font-size="10.5" fill="#6b6b6b">'
            f'高度轨迹 · {lo:.0f}–{hi:.0f} px</text>\n'
            f'<line x1="{L}" y1="{base+span_y}" x2="{w-R}" y2="{base+span_y}" stroke="{C_RULE}"/>\n'
            f'<path d="{path}" fill="none" stroke="{C_INK}" stroke-width="1.7" opacity="0.85"/>\n'
            f'{marks}\n'
            f'<text x="{w-R}" y="{base+span_y+44}" font-size="10" fill="{C_DN}" '
            f'text-anchor="end">红竖线 = 一次拍翅</text>\n'
            f'<text x="{L}" y="{base+span_y+44}" font-size="10" fill="#6b6b6b">'
            f'横轴 = 仿真时间（毫秒）· 每格 20 ms</text>\n</svg>')


# ---------------------------------------------------------------- 图 3：分布
def fig_gapgen(*, w=960, h=310) -> str:
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

    L, R, TOP = 30, 24, 54
    rows_h = 100
    pw = w - L - R
    GLO, GHI = 70.0, 250.0
    bins = np.linspace(GLO, GHI, 25)

    def panel(data, k, title, note, color):
        y0 = TOP + k * rows_h
        hist, _ = np.histogram(data, bins=bins)
        hist = hist / max(hist.max(), 1)
        out = [f'<text x="{L}" y="{y0}" font-size="13" font-weight="600" '
               f'fill="{color}">{title}</text>',
               f'<text x="{L+72}" y="{y0}" font-size="10.5" fill="#6b6b6b">{note}</text>']
        for i, v in enumerate(hist):
            bx = L + (bins[i] - GLO) / (GHI - GLO) * pw
            bwid = pw / (len(bins) - 1) - 1.5
            hh = v * 34
            out.append(f'<rect x="{bx:.1f}" y="{y0+56-hh:.1f}" width="{bwid:.1f}" '
                       f'height="{hh:.1f}" fill="{color}" opacity="0.72"/>')
        out.append(f'<line x1="{L}" y1="{y0+56}" x2="{w-R}" y2="{y0+56}" stroke="{C_RULE}"/>')
        out.append(f'<text x="{L}" y="{y0+72}" font-size="9.5" fill="#8a8a8a">'
                   f'top = {GLO:.0f}（贴顶）</text>')
        out.append(f'<text x="{w-R}" y="{y0+72}" font-size="9.5" fill="#8a8a8a" '
                   f'text-anchor="end">top = {GHI:.0f}（贴近地面）</text>')
        return "\n".join(out)

    return (SVG_OPEN.format(w=w, h=h,
                            aria="缺口生成分布对照：修前所有管子落在同一个位置，"
                                 "修后铺满整个量程") + "\n" +
            f'<rect width="{w}" height="{h}" fill="{C_PAPER}"/>\n' +
            panel(old, 0, "修前", "全部落在同一个值上（标准差 0.0）", "#9ca3af") + "\n" +
            panel(new, 1, "修后", "铺满量程、标准差约 50 px", C_LPLC2) + "\n" +
            f'<text x="{w/2}" y="{h-8}" font-size="10" fill="#6b6b6b" text-anchor="middle">'
            f'缺口 top 的分布（每根管子一次，各 4000 根）· 横轴 = 真实可行量程</text>\n</svg>')


# ---------------------------------------------------------------- 图 4：膜电位
def fig_membrane(*, w=960, h=306) -> str:
    """LIF 的膜电位轨迹：恒定驱动 0.1048 → 每 5 个 tick 发放一次。"""
    drive = 0.1048
    trace, fires = membrane_trace(drive=drive)
    L, R, TOP, BOT = 58, 158, 52, 230
    pw, ph = w - L - R, BOT - TOP
    n = len(trace)
    xn = lambda i: L + (i + 0.5) / n * pw
    yv = lambda v: BOT - (v / 1.6) * ph

    o = [SVG_OPEN.format(w=w, h=h,
                         aria="DNp01 在恒定驱动 0.1048 下的膜电位轨迹："
                              "每 5 个 tick 泄漏累积到阈值后发放并归零"),
         f'<rect width="{w}" height="{h}" fill="{C_PAPER}"/>']
    for v in (0.0, 0.4, 0.8, 1.2, 1.6):
        o.append(f'<line x1="{L}" y1="{yv(v):.1f}" x2="{L+pw}" y2="{yv(v):.1f}" '
                 f'stroke="{C_RULE}" stroke-width="0.6" opacity="0.8"/>')
        o.append(f'<text x="{L-8}" y="{yv(v)+3.5:.1f}" font-size="9.5" fill="#8a8a8a" '
                 f'text-anchor="end">{v:.1f}</text>')
    o.append(f'<line x1="{L}" y1="{yv(1.0):.1f}" x2="{L+pw}" y2="{yv(1.0):.1f}" '
             f'stroke="{C_DN}" stroke-width="1.4" stroke-dasharray="5 3"/>')
    o.append(f'<text x="{L+pw+6}" y="{yv(1.0)+4:.1f}" font-size="10.5" fill="{C_DN}">'
             f'threshold = 1.0</text>')
    # 渐近值 = gain·C / (1 − leak)，**不是** gain·C —— 漏积分会把输入放大 1/(1−leak) 倍。
    # 我第一版就写成了 gain·C = 0.3144，比正确值 1.7338 小了 5.5 倍。
    ss = 3.0 * drive / (1.0 - 0.818731)
    o.append(f'<line x1="{L}" y1="{yv(ss):.1f}" x2="{L+pw}" y2="{yv(ss):.1f}" '
             f'stroke="{C_PHYS}" stroke-width="0.9" stroke-dasharray="2 4" opacity="0.75"/>')
    o.append(f'<text x="{L+pw+6}" y="{yv(ss)+4:.1f}" font-size="10" fill="{C_PHYS}">'
             f'渐近值 {ss:.3f}</text>')
    pts = " ".join(f"{xn(i):.1f} {yv(v):.1f}" for i, v in enumerate(trace))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{C_INK}" stroke-width="1.8"/>')
    for i, v in enumerate(trace):
        o.append(f'<circle cx="{xn(i):.1f}" cy="{yv(v):.1f}" r="2.1" fill="{C_INK}"/>')
    for i in fires:
        o.append(f'<line x1="{xn(i):.1f}" y1="{TOP-6}" x2="{xn(i):.1f}" y2="{BOT}" '
                 f'stroke="{C_DN}" stroke-width="1.1" opacity="0.5"/>')
        o.append(f'<circle cx="{xn(i):.1f}" cy="{yv(1.0):.1f}" r="4.4" fill="none" '
                 f'stroke="{C_DN}" stroke-width="1.8"/>')
    first = fires[0] if fires else 0
    # 不在图内重复标"第 N 个 tick 发放" —— 顶部空间太窄会被裁掉，
    # 而且 figcaption 与底部说明已经讲清楚了。
    o.append(f'<line x1="{L}" y1="{BOT}" x2="{L+pw}" y2="{BOT}" stroke="{C_RULE}"/>')
    o.append(f'<text x="{L}" y="{BOT+26}" font-size="10" fill="#6b6b6b">tick 0</text>')
    o.append(f'<text x="{L+pw}" y="{BOT+26}" font-size="10" fill="#6b6b6b" '
             f'text-anchor="end">tick {n-1}</text>')
    o.append(f'<text x="{L}" y="{TOP-34}" font-size="13" font-weight="600" fill="{C_INK}">'
             f'DNp01 膜电位</text>')
    o.append(f'<text x="{L+108}" y="{TOP-34}" font-size="10.5" fill="#6b6b6b">'
             f'恒定驱动 0.1048 = amp 0.97 × 腹侧权重 0.1080</text>')
    o.append(f'<text x="{L}" y="{h-12}" font-size="10.5" fill="#6b6b6b">'
             f'每个 tick：先漏掉 18.127%，再加 3 × 0.1048 = 0.3144 → '
             f'第 {first+1} 个 tick 越过 1.0 并归零，之后周期复现'
             f'（间隔 {fires[1]-fires[0]} tick = {(fires[1]-fires[0])*20} ms）</text>')
    o.append('</svg>')
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/doc_figs")
    ap.add_argument("--asset", default="spiking_full")
    ap.add_argument("--ticks", type=int, default=220)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"采集真实发放（{a.ticks} tick）…")
    rows, meta = collect_spikes(a.asset, ticks=a.ticks, seed=7)
    n_flap = sum(1 for r in rows if r.get("flap"))
    n_dn = sum(len(r["dn01"]) for r in rows)
    print(f"  LC4={meta['n_lc4']} LPLC2={meta['n_lplc2']}；"
          f"DNp01 发放 {n_dn} 次；拍翅 {n_flap} 次")

    trace, fires = membrane_trace()
    gap = (fires[1] - fires[0]) if len(fires) > 1 else 0
    print(f"  膜电位算例：驱动 0.1048 → 第 {fires[0]+1} 个 tick 首次发放，"
          f"此后每 {gap} tick 一次；"
          f"渐近值 {3.0*0.1048/(1.0-0.818731):.4f}（= gain·C/(1−leak)）")

    (out / "raster.svg").write_text(fig_raster(rows, meta), encoding="utf-8")
    (out / "loop.svg").write_text(fig_loop(), encoding="utf-8")
    (out / "gapgen.svg").write_text(fig_gapgen(), encoding="utf-8")
    (out / "membrane.svg").write_text(fig_membrane(), encoding="utf-8")
    (out / "spikes.json").write_text(json.dumps(dict(
        meta=meta, n_flap=n_flap, n_dn=n_dn, ticks=len(rows),
        dn01_fired_ticks=[r["tick"] for r in rows if r["dn01"]],
        flaps=[r["tick"] for r in rows if r.get("flap")],
        y=[r["y"] for r in rows],
        mem_fires=fires, mem_trace=trace, mem_gap=gap,
    ), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out}/ raster.svg  loop.svg  gapgen.svg  membrane.svg  spikes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
