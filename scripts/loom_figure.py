#!/usr/bin/env python
"""把逼近逃避反射的四组结果画成一张图（output/loom_escape.png）。

面板
  A 传递函数：DNp01 发放率 vs 钳制的 LC4 群发放率，四条曲线 =
             真实拓扑 / 切直接边 / 打乱拓扑 / 驱动非逼近感觉群 LC10a
  B 激发概率 P_fire 的阈值曲线（真实 vs 关掉前馈抑制）——抑制是门控
  C 侧向分离：只驱动左 LC4 / 只驱动右 LC4 时，DNp01 左右细胞各自的发放率
  D 逼近刺激下的触发点：θ_esc 与 τ_esc 随逼近速度 v

依赖前三个脚本的输出 CSV；缺哪个面板就跳过哪个。
    D:/Code/FlyBrain/env/python.exe scripts/loom_figure.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

# 中文标签在 matplotlib 默认字体下会成方框，直接用英文标注图内文字
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 150})

HAVE = [p for p in ("loom_transfer.csv", "loom_mech_lateral.csv",
                    "loom_reflex_agg.csv", "loom_mech_inhibition.csv")
        if (OUT / p).exists()]
print("可用数据：" + ", ".join(HAVE) if HAVE else "没有可画的 CSV")


def panel_a(ax):
    df = pd.read_csv(OUT / "loom_transfer.csv")
    style = {"real": ("#c62828", "o", "real connectome"),
             "cut_lc4_dn01": ("#1565c0", "s", "cut LC4->DNp01 direct edges"),
             "shuffled": ("#2e7d32", "^", "targets shuffled (topology destroyed)"),
             "ctrl_lc10a": ("#6a1b9a", "D", "drive LC10a instead (non-loom)")}
    for lab, (c, mk, name) in style.items():
        sub = df[df.graph == lab]
        if not len(sub):
            continue
        m = sub.groupby("r_in").agg(rate=("dn_dur", "mean"),
                                    sd=("dn_dur", "std")).reset_index()
        ax.errorbar(m["r_in"], m["rate"] * 50, yerr=np.nan_to_num(m["sd"]) * 50,
                    marker=mk, color=c, label=name, ms=4, lw=1.4)
    ax.set_xlabel("clamped LC4 population rate (spikes/tick = Hz/50)")
    ax.set_ylabel("DNp01 firing rate (Hz)")
    ax.set_title("A  Relay: LC4 rate -> DNp01 (frozen connectome)")
    ax.legend(fontsize=7, loc="upper left")


def panel_b(ax):
    df = pd.read_csv(OUT / "loom_mech_inhibition.csv")
    for lab, name, c in (("real", "with inhibition", "#c62828"),
                         ("no_inhibition", "inhibition blocked", "#1565c0")):
        sub = df[df.graph == lab]
        if not len(sub):
            continue
        pf = sub.groupby("r_in")["fired_DNp01"].mean()
        ax.plot(pf.index, pf.values, "o-", color=c, ms=4, label=name)
        k = np.flatnonzero(pf.values >= 0.5)
        if k.size:
            i = int(k[0])
            if i == 0:
                r50 = float(pf.index[0])
            else:
                x0, x1 = float(pf.index[i - 1]), float(pf.index[i])
                y0, y1 = float(pf.values[i - 1]), float(pf.values[i])
                r50 = x0 + (0.5 - y0) / max(y1 - y0, 1e-9) * (x1 - x0)
            ax.axvline(r50, color=c, ls=":", lw=1)
            ax.annotate(f"r50≈{r50:.2f}", (r50, 0.05), color=c, fontsize=7,
                        xytext=(r50 + 0.02, 0.05 + 0.13 * (1 if c == "#c62828" else 0)))
    ax.set_xlabel("clamped LC4 rate")
    ax.set_ylabel("P(DNp01 fires | trial)")
    ax.set_title("B  Threshold is gated by feedforward inhibition")
    ax.legend(fontsize=7, loc="lower right")


def panel_c(ax):
    df = pd.read_csv(OUT / "loom_mech_lateral.csv")
    sub = df[df["r_in"] >= 0.6]
    conds = [c for c in ("LC4_left", "LC4_right") if (sub.graph == c).any()]
    if not conds:
        ax.text(0.5, 0.5, "no lateral trials", ha="center")
        return
    m = sub.groupby("graph")[["DNp01_left", "DNp01_right"]].mean().loc[conds]
    x = np.arange(len(conds))
    ax.bar(x - 0.19, m["DNp01_left"] * 50, 0.36, color="#c62828",
           label="DNp01 left cell")
    ax.bar(x + 0.19, m["DNp01_right"] * 50, 0.36, color="#1565c0",
           label="DNp01 right cell")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("LC4_", "drive LC4 ") for c in conds])
    ax.set_ylabel("DNp01 rate (Hz), mean over rate>=0.6")
    ax.set_title("C  Side is preserved: left LC4 -> only left DNp01")
    ax.legend(fontsize=7)
    for i, (a, b) in enumerate(zip(m["DNp01_left"] * 50, m["DNp01_right"] * 50)):
        ax.text(i - 0.19, a + 0.05, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.19, b + 0.05, f"{b:.2f}", ha="center", fontsize=7)


def panel_d(ax):
    df = pd.read_csv(OUT / "loom_reflex_agg.csv")
    sub = df[df.p_fire >= 0.5].copy()      # 对照组 P_fire=0，自然被排除
    # theta>90° 意味着"盘子已经铺满视野"，不是可用的逃逸触发点
    n_drop = int((sub.theta_deg > 90).sum())
    sub = sub[sub.theta_deg <= 90]
    if not len(sub):
        ax.text(0.5, 0.5, "no reflex trials recorded", ha="center")
        return
    # 文献基准：Fotowat 2009 翼抬起 49±4°、起飞 54±5°；de Vries 2012 计算阈值 67.6°
    for y, lab, c in ((49, "wing elevation 49±4° (Fotowat 2009)", "#555555"),
                      (54, "takeoff 54±5° (Fotowat 2009)", "#555555"),
                      (67.6, "comp. threshold 67.6° (de Vries 2012)", "#888888")):
        ax.axhline(y, color=c, ls="--", lw=0.9)
        ax.text(1.02, y, lab, transform=ax.get_yaxis_transform(), fontsize=6.5,
                va="center", color=c)
    palette = ["#c62828", "#ef6c00", "#1565c0", "#6a1b9a", "#00838f", "#33691e",
               "#ad1457", "#4527a0"]
    labels = sorted(sub.graph + " s50=" + sub.sens.astype(str))
    for i, lab in enumerate(labels):
        s = sub[(sub.graph + " s50=" + sub.sens.astype(str)) == lab].sort_values("speed_ms")
        if len(s) < 2:
            continue
        ax.plot(s["speed_ms"], s["theta_deg"], "o-", ms=4, lw=1.3,
                color=palette[i % len(palette)], label=lab)
    ax.set_xscale("log")
    xs = sorted(df.speed_ms.unique())
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("approach speed v (m/s)")
    ax.set_ylabel("escape angle theta_esc (deg)")
    ax.set_ylim(0, 95)
    ax.set_title("D  Looming -> DNp01 trigger angle vs literature critical size"
                 + (f"  [{n_drop} late points dropped]" if n_drop else ""),
                 fontsize=9)
    ax.legend(fontsize=6, loc="upper left", ncol=2)


def main() -> int:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fns = {"loom_transfer.csv": (axes[0, 0], panel_a),
           "loom_mech_inhibition.csv": (axes[0, 1], panel_b),
           "loom_mech_lateral.csv": (axes[1, 0], panel_c),
           "loom_reflex_agg.csv": (axes[1, 1], panel_d)}
    for name, (ax, fn) in fns.items():
        if (OUT / name).exists():
            try:
                fn(ax)
            except Exception as e:                      # 一个面板坏不掉整张图
                ax.text(0.5, 0.5, f"{name}\n{type(e).__name__}: {e}", ha="center")
                ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, f"{name} missing", ha="center")
            ax.set_axis_off()
    fig.suptitle("Looming escape reflex in the frozen FlyWire connectome "
                 "(59,548 neurons / 2.45M synapses, LIF)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = OUT / "loom_escape.png"
    fig.savefig(p, dpi=150)
    print(f"✓ {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
