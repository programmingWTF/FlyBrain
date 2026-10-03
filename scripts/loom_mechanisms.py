#!/usr/bin/env python
"""视觉逼近逃避反射 第 3 步：**连接组在这条通路上做了什么**（两个机制实验）。

实验 A：前馈抑制的作用
---------------------
把全部负权（GABA 能突触）置 0，重测 **DNp01 自己**的激发阈值 r50 与增益。
判据必须按细胞分开：上一版把 DNp01+DNp04 混在一起算"有没有发放"，
于是 r50 被更容易发的 DNp04 污染成了 0.10 —— 那是 DNp04 的阈值，不是 DNp01 的。

实验 B：侧向分离（同侧感觉 -> 同侧下行胞体？还是跨中线？）
--------------------------------------------------------
DNp01 只有 2 个细胞（左/右各 1），它的轴突在腹神经索里越中线。
把刺激只打给左侧 LC4 / 只打右侧 LC4 / 双侧同打，比较左、右 DNp01 各自的发放率。
判据是**同一试次内两侧之差**（配对），不看绝对值。

用法
----
    D:/Code/FlyBrain/env/python.exe scripts/loom_mechanisms.py --inhibition
    D:/Code/FlyBrain/env/python.exe scripts/loom_mechanisms.py --lateral
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv import looming                      # noqa: E402
from fpv.spiking_brain import SpikingBrain   # noqa: E402

OUT = ROOT / "output"
GAIN, TONIC = 3.0, 0.0


def allside(gs, nm):
    """某个类型群所有侧别拼在一起（resolve_side 返回的是 {side: 索引} 字典）。"""
    return torch.cat(list(gs[nm].values()))


def named_watch(gs, specs):
    """[("DNp01","left"), ...] -> [(名字, 索引张量)]。"""
    out = []
    for nm, sd in specs:
        t = gs[nm].get(sd)
        if t is not None:
            out.append((f"{nm}_{sd}", t))
    return out


def relay(brain, inp_idx, watch_named, r, *, ticks=400, pre=200, seed=0):
    """恒定发放率 r 驱动 inp_idx；返回 {名字: 该群各细胞的发放率数组} 与首发时刻。"""
    idx = torch.cat([t for _, t in watch_named])
    sizes = [t.numel() for _, t in watch_named]
    p = np.concatenate([np.zeros(pre), np.full(ticks, float(r))])
    spikes = looming.run_relay(brain, [(inp_idx, p)], idx, seed=seed)
    post = spikes[pre:]
    res, sl = {}, 0
    for (nm, _), k in zip(watch_named, sizes):
        cells = post[:, sl:sl + k]
        res[nm] = cells.sum(axis=0) / ticks
        sl += k
    base = spikes[:pre].sum()
    first = np.flatnonzero(post.sum(axis=1) > 0)
    return res, (int(first[0]) if first.size else None), int(base)


def logistic_r50(xs, pf):
    xs, pf = np.asarray(xs, float), np.asarray(pf, float)
    if pf.min() > 0.05 or pf.max() < 0.95:
        return None
    from scipy.optimize import curve_fit
    f = lambda x, r50, k: 1.0 / (1.0 + np.exp(-k * (x - r50)))
    try:
        popt, _ = curve_fit(f, xs, pf, p0=[0.7, 10.0], maxfev=20000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return None


def sweep(brain, inp, watch_named, rates, reps, label, key="DNp01"):
    rows = []
    for r in rates:
        for rep in range(reps):
            res, lat, base = relay(brain, inp, watch_named, r, seed=500 + rep)
            row = dict(graph=label, r_in=float(r), rep=rep, first_tick=lat,
                       base_spk=base)
            for nm, v in res.items():
                row[nm] = float(np.mean(v))
                row[f"{nm}_any"] = bool(np.any(v > 0))
            row[f"fired_{key}"] = any(row[nm + "_any"] for nm in res
                                      if nm.startswith(key))
            rows.append(row)
    return rows


def inhibition_test(brain, gs, rates, reps):
    print("\n" + "=" * 96)
    print("实验 A：关掉前馈抑制（全部负权 -> 0）后，**DNp01 自己**的阈值与增益怎么变")
    print("=" * 96)
    watch = named_watch(gs, [("DNp01", "left"), ("DNp01", "right"),
                            ("DNp04", "left"), ("DNp04", "right")])
    fits = {}
    frames = []
    for lab, br in (("real", brain),
                    ("no_inhibition", looming.variant(brain, block_inhibition=True))):
        df = pd.DataFrame(sweep(br, allside(gs, "LC4"), watch, rates, reps, lab))
        frames.append(df.assign(leg=lab))
        pf = df.groupby("r_in")["fired_DNp01"].mean()
        fit = logistic_r50(pf.index.to_numpy(), pf.to_numpy())
        fits[lab] = fit
        print(f"\n--- {lab} ---")
        tab = pd.concat([df.groupby("r_in")["fired_DNp01"].mean().rename("P_fire(DNp01)"),
                         df.groupby("r_in")["DNp01_left"].mean().rename("DNp01_L率"),
                         df.groupby("r_in")["DNp01_right"].mean().rename("DNp01_R率"),
                         df.groupby("r_in")["DNp04_left"].mean().rename("DNp04_L率"),
                         df.groupby("r_in")["DNp04_right"].mean().rename("DNp04_R率")],
                        axis=1)
        print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"  DNp01 的 r50 = {fit[0]:.3f}（斜率 {fit[1]:.1f}）"
              f"  ->  每只 LC4 {fit[0]*50:.1f} Hz" if fit else "  r50: 未跨过 0.5")
    pd.concat(frames).to_csv(OUT / "loom_mech_inhibition.csv", index=False)
    if fits.get("real") and fits.get("no_inhibition"):
        a, b = fits["real"][0], fits["no_inhibition"][0]
        msg = ("抑制在**降低**阈值（去掉抑制反而更难触发）" if b > a + 0.02 else
               "抑制在**抬高**阈值（去掉抑制更容易触发 -> 抑制是逃逸的门控）"
               if b < a - 0.02 else "抑制对阈值几乎没有影响")
        print(f"\n  >>> Δr50 = {b-a:+.3f}（相对 {100*(b-a)/a:+.1f}%）：{msg}")
    allf = pd.concat(frames)
    for leg in ("real", "no_inhibition"):
        s = allf[allf.leg == leg].groupby("r_in")[["DNp01_left", "DNp01_right",
                                                   "DNp04_left", "DNp04_right"]].mean().sum(1)
        print(f"  {leg:<15} 4 个 DN 细胞的发放率之和 随 r_in: " +
              "  ".join(f"{r:.1f}->{v:.3f}" for r, v in s.items()))


def lateral_test(brain, gs, rates, reps):
    print("\n" + "=" * 96)
    print("实验 B：单侧 LC4 驱动 -> DNp01 / DNp04 的左、右细胞谁发放")
    print("=" * 96)
    watch = named_watch(gs, [("DNp01", "left"), ("DNp01", "right"),
                            ("DNp04", "left"), ("DNp04", "right")])
    names = [n for n, _ in watch]
    L, R = gs["LC4"].get("left"), gs["LC4"].get("right")
    rows = []
    for lab, inp in (("LC4_left", L), ("LC4_right", R),
                     ("LC4_both", allside(gs, "LC4"))):
        rows += sweep(brain, inp, watch, rates, reps, lab)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "loom_mech_lateral.csv", index=False)
    m = df.groupby(["graph", "r_in"])[names].mean()
    print("\n每个条件下各 DN 细胞的平均发放率（脉冲/tick）：")
    print(m.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n侧向分离判据（同侧驱动细胞 vs 对侧细胞，逐试次配对 + Wilcoxon）：")
    from scipy.stats import wilcoxon
    for lab, ipsi, contra in (("LC4_left", "left", "right"),
                              ("LC4_right", "right", "left")):
        sub = df[df.graph == lab]
        for dn in ("DNp01", "DNp04"):
            i = sub[f"{dn}_{ipsi}"].to_numpy()
            c = sub[f"{dn}_{contra}"].to_numpy()
            d = i - c
            st, pv = wilcoxon(d) if np.any(d != 0) else (0.0, 1.0)
            print(f"  {lab} -> {dn}: 同侧(={ipsi}) {i.mean():.4f} vs 对侧(={contra}) "
                  f"{c.mean():.4f}   比值 {(i.mean()+1e-9)/(c.mean()+1e-9):8.1f}  "
                  f"Wilcoxon p={pv:.3g}  "
                  f"{'同侧胞体占优' if i.mean() > c.mean() else '对侧胞体占优'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--asset", default="spiking_circuit")
    ap.add_argument("--inhibition", action="store_true")
    ap.add_argument("--lateral", action="store_true")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--rates", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    a = ap.parse_args()
    if not (a.inhibition or a.lateral):
        a.inhibition = a.lateral = True

    brain = SpikingBrain.from_npz(a.asset, device=a.device)
    brain.gain, brain.tonic = GAIN, TONIC
    gs = looming.resolve_side(brain, ["LC4", "LPLC2", "DNp01", "DNp04"])
    print(f"资产 {a.asset}  N={brain.N:,}")
    print("侧别：" + str({k: {s: int(v.numel()) for s, v in d.items()}
                          for k, d in gs.items()}))
    rates = [float(x) for x in a.rates.split(",")]
    if a.inhibition:
        inhibition_test(brain, gs, rates, a.reps)
    if a.lateral:
        lateral_test(brain, gs, rates, a.reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
