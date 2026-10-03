#!/usr/bin/env python
"""腹侧 LPLC2 到底能不能点亮 DNp01 —— 把"优先沿这条走"这句话变成数字。

背景（SCORE_PROMPT.md 说这是最有希望的一条）
-------------------------------------------
ESCAPE.md §3.7/§3.8 只做到"LPLC2 整群被驱动时能触发 DNp01"，以及
`loom_lateral_readout.py` 报过 LPLC2_lo 的有效驱动/阈值 = 1.21、LPLC2_hi = 0.43。
但那个 lo/hi 是在**全体 LPLC2** 上按视野轴中位数劈开的，而且没有报：
  · 每只细胞到 DNp01 的实际权重分布（决定阈值的是加权覆盖率，不是细胞数）
  · 只用腹侧那一半时，Σw·p 到底是多少、离 0.0604 有多远
  · 与**精确视野定位**（高斯窗）的关系

本脚本把这四件事一次量清，并直接给出 Flappy 要用的那个数：**腹侧半视野能提供的
最大有效驱动 Σw·p**。如果它本身 < need=0.0604，那"改成腹侧 LPLC2 为主"这条
在算术上就不可能成立，必须先说清楚，而不是到游戏里瞎调。

用法
    D:/Code/FlyBrain/env/python.exe scripts/loom_ventral_probe.py --asset spiking_full
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
try:                                     # Windows 控制台默认 GBK，中文表头会变成乱码
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from fpv import looming                      # noqa: E402
from fpv.spiking_brain import SpikingBrain   # noqa: E402

GAIN, TONIC = 3.0, 0.0
TICKS = 60


# ------------------------------------------------------------------ 坐标 / 视野轴
def build_coords(brain) -> np.ndarray:
    mf = json.loads((REPO / "data" / "brain" / "manifest.json").read_text(encoding="utf-8"))
    xs = mf["bbox"]
    ctr = np.array([(xs[0] + xs[3]) / 2, (xs[1] + xs[4]) / 2, (xs[2] + xs[5]) / 2])
    half = np.array([(xs[3] - xs[0]) / 2, (xs[4] - xs[1]) / 2, (xs[5] - xs[2]) / 2])
    row_of = {str(n["id"]): i for i, n in enumerate(mf["neurons"])}
    secs = mf["coarse"]["sections"]
    id_of = np.asarray(brain.node_ids, np.int64)
    out = np.zeros((brain.N, 3), dtype=np.float32)
    for k, nid in enumerate(id_of):
        j = row_of.get(str(int(nid)))
        if j is not None and j < len(secs) and secs[j].get("bbox"):
            b = secs[j]["bbox"]
            out[k] = (np.array([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2,
                                (b[2] + b[5]) / 2]) - ctr) / half
    return out


def field_coord(coords, idx, side):
    """群的视野坐标 u∈[0,1]（**半球轴先剔除**，见 ESCAPE.md 的"视野轴"坑）。

    返回 (info, u)。u 在每个半球内部各自归一化，所以它是"每只眼睛视野的高低"，
    不是"左脑/右脑"。
    """
    pts = coords[idx]
    sd = np.asarray(side)
    hemi = int(np.argmax([pts[:, a].std() for a in range(3)]))
    cand = [a for a in range(3) if a != hemi]
    axis = int(max(cand, key=lambda a: np.mean([pts[sd == s, a].std()
                                                for s in ("left", "right")
                                                if (sd == s).sum() > 3])))
    u = np.zeros(len(idx), dtype=np.float64)
    for s in ("left", "right"):
        m = sd == s
        if m.sum() < 2:
            continue
        v = pts[m, axis]
        u[m] = (v - v.min()) / max(v.max() - v.min(), 1e-9)
    info = {"hemi_axis": "xyz"[hemi], "field_axis": "xyz"[axis],
            "std_per_axis": [round(float(v), 4) for v in pts.std(0)]}
    return info, u


def side_of_group(brain, idx):
    import pandas as pd
    meta = pd.read_feather(ROOT / "data" / "fafb_783_meta.feather")
    meta["id"] = meta["fafb_783_id"].astype("int64")
    meta["side"] = meta["side"].fillna("?")
    ser = pd.Series(meta["side"].to_numpy(), index=meta["id"].to_numpy())
    ids = np.asarray(brain.node_ids, np.int64)[idx]
    return ser.reindex(pd.Index(ids)).fillna("?").to_numpy()


# ------------------------------------------------------------------ 权重
def weights_to(brain, group_idx, targets) -> np.ndarray:
    """每个细胞 -> targets 的带符号突触权重和（除以 target 数 = 每只 DN 拿到的量）。"""
    lut = brain.lut.detach().cpu().numpy()
    indptr = brain.indptr.detach().cpu().numpy()
    indices = brain.indices.detach().cpu().numpy()
    codes = brain.codes.detach().cpu().numpy()
    dn = targets.detach().cpu().numpy()
    w = np.zeros(len(group_idx), dtype=np.float64)
    for k, s in enumerate(group_idx):
        lo, hi = int(indptr[s]), int(indptr[s + 1])
        if hi <= lo:
            continue
        sel = np.isin(indices[lo:hi], dn)
        if sel.any():
            w[k] = float(lut[codes[lo:hi][sel]].sum())
    return w / max(len(dn), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spiking_full")
    ap.add_argument("--ticks", type=int, default=TICKS)
    a = ap.parse_args()

    brain = SpikingBrain.from_npz(a.asset, device="cpu")
    brain.gain, brain.tonic = GAIN, TONIC
    need = (1 - brain.leak) * brain.threshold / brain.gain
    g = looming.resolve(brain, ["LC4", "LPLC2", "LPLC1", "DNp01", "DNp04"])
    coords = build_coords(brain)

    print(f"资产 {a.asset}: N={brain.N:,}  E={len(brain.codes):,}  "
          f"gain={brain.gain} tonic={brain.tonic}")
    print(f"DNp01 每 tick 需要净输入 need = (1-leak)*thr/gain = {need:.5f}\n")

    info_by = {}
    cells = {}
    for nm in ("LC4", "LPLC2"):
        idx = g[nm].detach().cpu().numpy()
        info, u = field_coord(coords, idx, side_of_group(brain, idx))
        w = weights_to(brain, idx, g["DNp01"])
        cells[nm] = dict(idx=idx, u=u, w=w)
        info_by[nm] = info
        print(f"{nm:<6} n={len(idx):<5} 半球轴={info['hemi_axis']} "
              f"视野轴={info['field_axis']} std/轴={info['std_per_axis']}")
        print(f"       -> DNp01 权重: 总 {w.sum():.4f}  均值 {w.mean():.5f} "
              f"中位 {np.median(w):.5f} 最大 {w.max():.5f}  非零 {int((w > 0).sum())}")
        # 权重最大的那批细胞落在视野的哪里？如果强突触集中在背侧，
        # "腹侧通道"这条在算术上就是错的，必须现在发现。
        o = np.argsort(-w)[:max(3, len(w) // 10)]
        print(f"       top10% 强突触细胞的视野 u: 均值 {u[o].mean():.3f} "
              f"范围 [{u[o].min():.3f}, {u[o].max():.3f}]  "
              f"（u=0 一端 / u=1 另一端）")
        print(f"       u<0.5 那半的权重和 {w[u < 0.5].sum():.4f} / "
              f"u>=0.5 那半 {w[u >= 0.5].sum():.4f}")

    # ---------------------------------------------------------------- 静态钳制扫描
    print(f"\n静态钳制 {a.ticks} tick（先 40 tick 稳定，后 20 tick 计数）：")
    print(f"{'条件':<34}{'Σw·p':>9}{'/need':>8}{'DNp01脉冲':>11}{'DNp04脉冲':>11}")
    lc4 = cells["LC4"]["idx"]
    lp = cells["LPLC2"]["idx"]
    dn01, dn04 = g["DNp01"], g["DNp04"]

    def run(groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[float, int, int]:
        """groups: [(该群的资产索引, 该群逐细胞到 DNp01 的权重, 逐细胞发放率)]。

        ⚠️ 权重必须跟着索引一起取子集 —— 第一版只对索引取子集、权重还是全群，
        形状不匹配（104 vs 47）当场报错。决定 eff 的是 Σ w_i·p_i，两者必须对齐。
        """
        cidx = torch.as_tensor(np.concatenate([ix for ix, _, _ in groups]),
                               dtype=torch.long, device=brain.device)
        pv = torch.as_tensor(np.concatenate([p for _, _, p in groups]),
                             dtype=torch.float32, device=brain.device)
        brain.reset()
        s01 = s04 = 0
        for i in range(a.ticks):
            brain.step(clamp=(cidx, pv))
            if i >= a.ticks - 20:
                s01 += int(brain.S[dn01].sum())
                s04 += int(brain.S[dn04].sum())
        eff = float(sum(float((w * p).sum()) for _, w, p in groups))
        return eff, s01, s04

    cases: list[tuple[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]] = []
    for amp in (0.25, 0.5, 0.75, 1.0):
        cases.append((f"LC4 整群 amp={amp}",
                      [(lc4, cells["LC4"]["w"], np.full(len(lc4), amp))]))
        cases.append((f"LPLC2 整群 amp={amp}",
                      [(lp, cells["LPLC2"]["w"], np.full(len(lp), amp))]))
    for amp in (0.5, 0.75, 1.0):
        cases.append((f"LC4+LPLC2 整群 amp={amp}",
                      [(lc4, cells["LC4"]["w"], np.full(len(lc4), amp)),
                       (lp, cells["LPLC2"]["w"], np.full(len(lp), amp))]))
    for nm in ("LC4", "LPLC2"):
        u, ix, w = cells[nm]["u"], cells[nm]["idx"], cells[nm]["w"]
        for tag, m in (("lo(u<.5)", u < 0.5), ("hi(u>=.5)", u >= 0.5),
                       ("lo 25%", u < 0.25), ("hi 25%", u >= 0.75)):
            cases.append((f"{nm} {tag} amp=1",
                          [(ix[m], w[m], np.ones(int(m.sum())))]))
    # 精确视野定位（高斯窗，与 demo/server.py 同一套公式）
    for nm in ("LC4", "LPLC2"):
        u, ix, w = cells[nm]["u"], cells[nm]["idx"], cells[nm]["w"]
        for width in (0.5, 0.7, 1.0):
            sh = np.exp(-((u - 0.5) / max(width / 2.355, 1e-6)) ** 2 / 2.0)
            cases.append((f"{nm} 高斯窗 w={width} amp=1", [(ix, w, sh)]))

    rows = []
    for name, groups in cases:
        eff, s01, s04 = run(groups)
        rows.append(dict(case=name, eff=eff, ratio=eff / need, dn01=s01, dn04=s04))
        print(f"{name:<34}{eff:9.4f}{eff / need:8.2f}{s01:11d}{s04:11d}")

    import pandas as pd
    df = pd.DataFrame(rows)
    out = ROOT / "output" / "loom_ventral_probe.csv"
    df.to_csv(out, index=False)
    print(f"\n→ {out}")

    # ---------------------------------------------------------------- 结论行
    vent = df[df.case.str.contains("lo")]
    if len(vent):
        best = vent.loc[vent.ratio.idxmax()]
        print(f"\n腹侧/低 u 半视野能给出的最大 Σw·p/need = {best.ratio:.2f}"
              f"（{best.case}）")
    print("判据：Σw·p/need >= 1 才可能让 DNp01 发放（ESCAPE.md §3.8 的 cliff）。")

    # ---------------------------------------------------------------- 腹侧半群专项
    # 这是 SCORE_PROMPT.md 点名要优先走的那条路。它能不能成立只看一件事：
    # **只驱动腹侧那一半时，Σw·p 有没有越过 need**。稳态比是静态的，
    # 但 DNp01 是靠膜电位积分（leak=0.819/tick）累积的，所以还要看
    # 持续驱动下多少 tick 能攒够 —— 这决定 Flappy 里"来不来得及拍翅"。
    print("\n" + "=" * 78)
    print("腹侧半群专项：只驱动 LPLC2 的 u<0.5（低 y = 腹侧）那一半")
    w, u, ix = cells["LPLC2"]["w"], cells["LPLC2"]["u"], cells["LPLC2"]["idx"]
    m = u < 0.5
    print(f"  细胞 {int(m.sum())}/{len(ix)}   权重和 {w[m].sum():.4f} "
          f"（全群 {w.sum():.4f} 的 {w[m].sum() / w.sum():.1%}）")
    print(f"{'amp':>6}{'Σw·p':>9}{'/need':>8}"
          f"{' 每10tick DNp01脉冲（0-10 10-20 ... 50-60）':>44}{'总':>6}")
    for amp in (0.3, 0.5, 0.7, 0.85, 1.0):
        cidx = torch.as_tensor(ix[m], dtype=torch.long, device=brain.device)
        pv = torch.full((int(m.sum()),), float(amp), device=brain.device)
        brain.reset()
        bins = []
        for b in range(6):
            s = 0
            for _ in range(10):
                brain.step(clamp=(cidx, pv))
                s += int(brain.S[dn01].sum())
            bins.append(s)
        print(f"{amp:6.2f}{w[m].sum() * amp:9.4f}{w[m].sum() * amp / need:8.2f}"
              f"{'  ' + ' '.join(f'{v:4d}' for v in bins):>44}{sum(bins):6d}")
    print("→ 若 1.0 档也要攒好几个 tick 才发放，那 Flappy 里就必须让腹侧通道"
          "**提前**（威胁还远时）就开始驱动，否则来不及。")

    # ---------------------------------------------------------------- 触发曲线
    # Flappy 里到底有没有"够大"的刺激？必须把它变成一个可比的数：
    # **要让 DNp01 稳定发放，需要多大的角尺寸 θ / 多快的角扩张 dθ/dt。**
    # 然后把游戏里各候选威胁的几何量代进去，看谁够得着。这一步不做，
    # 后面所有"接哪条通道"的讨论都是猜。
    print("\n" + "=" * 78)
    print("触发曲线：把 LPLC2 整群按 Naka-Rushton 钳制成 p(θ) 或 p(dθ)，"
          "看什么量级才让 DNp01 发放")
    print(f"{'刺激':<26}{'Σw·p':>9}{'/need':>8}{'60tick DNp01':>14}{'每tick率':>10}")
    w_lp = cells["LPLC2"]["w"]
    curves = []
    for tag, smax, s50 in (("角尺寸 θ", 180.0, 30.0), ("角尺寸 θ (s50=15)", 180.0, 15.0),
                           ("角扩张 dθ/dt", 600.0, 30.0),
                           ("角扩张 dθ/dt (s50=15)", 600.0, 15.0)):
        for frac in (0.25, 0.4, 0.55, 0.7, 0.85, 1.0):
            s = frac * smax
            p = s ** 3 / (s ** 3 + s50 ** 3)
            cidx = torch.as_tensor(lp, dtype=torch.long, device=brain.device)
            pv = torch.full((len(lp),), float(p), device=brain.device)
            brain.reset()
            tot = 0
            for _ in range(60):
                brain.step(clamp=(cidx, pv))
                tot += int(brain.S[dn01].sum())
            eff = float(w_lp.sum() * p)
            curves.append(dict(stim=tag, param=s, p=p, eff=eff, ratio=eff / need,
                               spikes=tot, rate=tot / 60.0))
            print(f"{tag + f' = {s:.0f}':<26}{eff:9.4f}{eff / need:8.2f}"
                  f"{tot:14d}{tot / 60.0:10.2f}")
    import pandas as pd
    pd.DataFrame(curves).to_csv(ROOT / "output" / "loom_trigger_curve.csv", index=False)
    print(f"→ {ROOT / 'output' / 'loom_trigger_curve.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
