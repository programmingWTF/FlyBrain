#!/usr/bin/env python
"""三方对照汇总：把 runs/ 下所有 summary.json 聚成表格 + 学习曲线图。

用法：
    python scripts/aggregate.py                    # 汇总 runs/*/summary.json
    python scripts/aggregate.py --pattern "s*_s*"  # 只汇总匹配的 tag

输出：
    终端表格（每个 arch 的 mean±std、最高分、用时）
    output/compare.png   学习曲线 + 最终分数箱线
    output/compare.csv   明细
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "output"

ARCH_LABEL = {
    "mlp": "① MLP 基线（稠密，无结构）",
    "rewired": "② 随机稀疏图（同边数/同度分布）",
    "connectome": "③ 真实 FlyWire 接线",
}
ORDER = ["mlp", "rewired", "connectome"]


def load_all(pattern: str | None = None):
    rows = []
    for p in sorted(RUNS.glob("*/summary.json")):
        tag = p.parent.name
        if pattern and not pathlib.Path(tag).match(pattern):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  跳过 {tag}: {e}")
            continue
        d["_dir"] = str(p.parent)
        rows.append(d)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--min-steps", type=int, default=0, help="只统计步数 >= 该值的 run")
    a = ap.parse_args()

    rows = [r for r in load_all(a.pattern) if r.get("steps", 0) >= a.min_steps]
    if not rows:
        print("没有找到 summary.json；先跑 scripts/train.py")
        return 1

    groups = defaultdict(list)
    for r in rows:
        groups[r["arch"]].append(r)

    print(f"共 {len(rows)} 个 run\n")
    hdr = f"{'配置':34s} {'run':>3s} {'步数':>9s} {'最终均分':>16s} {'最高':>6s} {'训练近期':>9s} {'用时min':>8s}"
    print(hdr)
    print("-" * len(hdr))
    summary_rows = []
    for arch in ORDER + [k for k in groups if k not in ORDER]:
        if arch not in groups:
            continue
        rs = groups[arch]
        means = [r["final_eval"]["mean"] for r in rs]
        maxs = [r["final_eval"]["max"] for r in rs]
        recent = [r.get("train_mean_last100", 0.0) for r in rs]
        mins = [r.get("wall_seconds", 0) / 60 for r in rs]
        print(f"{ARCH_LABEL.get(arch, arch):34s} {len(rs):3d} {rs[0]['steps']:9,d} "
              f"{sum(means)/len(means):8.2f} ± {_std(means):5.2f} {max(maxs):6d} "
              f"{sum(recent)/len(recent):9.2f} {sum(mins)/len(mins):8.1f}")
        summary_rows.append({
            "arch": arch, "runs": len(rs), "mean": sum(means)/len(means), "std": _std(means),
            "max": max(maxs), "train_recent": sum(recent)/len(recent),
            "wall_min": sum(mins)/len(mins),
        })

    print("\n=== 关键对照 ===")
    base = next((s for s in summary_rows if s["arch"] == "mlp"), None)
    rew = next((s for s in summary_rows if s["arch"] == "rewired"), None)
    conn = next((s for s in summary_rows if s["arch"] == "connectome"), None)
    if base and rew:
        print(f"  稀疏结构贡献 (①->②)：{rew['mean'] - base['mean']:+.2f}")
    if rew and conn:
        print(f"  真实接线贡献 (②->③)：{conn['mean'] - rew['mean']:+.2f}")
    if base and conn:
        print(f"  连接组 vs MLP (①->③)：{conn['mean'] - base['mean']:+.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    import csv
    with (OUT / "compare.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "arch", "seed", "steps", "eval_mean", "eval_max", "eval_min",
                    "train_mean_last100", "wall_seconds"])
        for r in rows:
            w.writerow([r["tag"], r["arch"], r.get("seed"), r["steps"],
                        r["final_eval"]["mean"], r["final_eval"]["max"],
                        r["final_eval"]["min"], r.get("train_mean_last100"),
                        r.get("wall_seconds")])
    print(f"\n明细 -> {OUT/'compare.csv'}")

    try:
        _plot(rows, groups, OUT / "compare.png")
        print(f"图   -> {OUT/'compare.png'}")
    except Exception as e:
        print(f"画图跳过（{type(e).__name__}: {e}）")
    return 0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _plot(rows, groups, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for arch in ORDER:
        if arch not in groups:
            continue
        for r in groups[arch]:
            ev = r.get("evals") or []
            if not ev:
                continue
            xs = [e["step"] for e in ev]
            ys = [e["mean"] for e in ev]
            ax.plot(xs, ys, marker="o", ms=3, lw=1.2, alpha=0.8,
                    label=f"{arch} seed{r.get('seed')}")
    ax.set_xlabel("env steps")
    ax.set_ylabel("greedy eval mean score")
    ax.set_title("学习曲线（贪心评测）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    labels, data = [], []
    for arch in ORDER:
        if arch not in groups:
            continue
        labels.append(arch)
        data.append([r["final_eval"]["mean"] for r in groups[arch]])
    if data:
        ax2.boxplot(data, labels=labels)
        for i, d in enumerate(data, start=1):
            ax2.scatter([i] * len(d), d, zorder=3, s=24)
    ax2.set_ylabel("final greedy eval mean score")
    ax2.set_title("最终分数（每个 run 一个点）")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
