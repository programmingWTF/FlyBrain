#!/usr/bin/env python
"""按**实验分组**汇总（而不是把所有 connectome 混在一起）。

分组规则（对应 overnight.py 的 JOBS 清单）：
  主对照 n4000 : mlp_s{0,1,2} / rewired_s{0,1,2} / connectome_s{0,1,2}
  复现  n3000  : rep_n3000_{mlp,rewired,connectome}_s{10,11}
  权重消融     : ab_n3000_{norm,count,binary}

为什么要分开：n4000 与 n3000 是**两张不同的子图**，混在一起平均会把
「子图规模的影响」和「拓扑的影响」混为一谈。

同时报告均值、中位数、最大最小值——DQN 在这个任务上是**双峰分布**
（要么学会永久飞、分数极高；要么 0 分），只看均值会被离群值主导。

用法：python scripts/report_by_group.py
"""
from __future__ import annotations

import json
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

GROUPS = {
    "主对照 (n4000, 4000节点/356k边)": {
        "① MLP 基线": ["mlp_s0", "mlp_s1", "mlp_s2"],
        "② 随机图": ["rewired_s0", "rewired_s1", "rewired_s2"],
        "③ 真实接线": ["connectome_s0", "connectome_s1", "connectome_s2"],
    },
    "复现 (n3000, 3000节点/210k边)": {
        "① MLP 基线": ["rep_n3000_mlp_s10", "rep_n3000_mlp_s11"],
        "② 随机图": ["rep_n3000_rewired_s10", "rep_n3000_rewired_s11"],
        "③ 真实接线": ["rep_n3000_connectome_s10", "rep_n3000_connectome_s11"],
    },
    "权重消融 (n3000, 同一张图)": {
        "norm (数据自带归一化)": ["ab_n3000_norm"],
        "count (log1p 原始突触数)": ["ab_n3000_count"],
        "binary (只剩拓扑)": ["ab_n3000_binary"],
    },
}


def load(tag: str):
    p = RUNS / tag / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize(tags: list[str]):
    vals, maxs, missing = [], [], []
    for t in tags:
        d = load(t)
        if d is None:
            missing.append(t)
            continue
        vals.append(float(d["final_eval"]["mean"]))
        maxs.append(int(d["final_eval"]["max"]))
    return vals, maxs, missing


def fmt(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{st.mean(vals):.1f} ± {st.pstdev(vals):.1f}"


def main() -> int:
    print("=" * 78)
    print("FlyWire 连接组 × FlappyBird —— 1,000,000 env-step 结果汇总")
    print("=" * 78)

    for gname, arms in GROUPS.items():
        print(f"\n### {gname}")
        hdr = f"{'组':<26}{'n':>3}{'最终均分':>18}{'中位数':>10}{'最高':>8}  明细"
        print(hdr)
        print("-" * len(hdr))
        means = {}
        for aname, tags in arms.items():
            vals, maxs, missing = summarize(tags)
            if not vals:
                print(f"{aname:<26}{0:>3}{'（无数据）':>18}")
                continue
            means[aname] = st.mean(vals)
            med = st.median(vals)
            det = ", ".join(f"{v:.0f}" for v in vals)
            warn = f"  缺:{missing}" if missing else ""
            print(f"{aname:<26}{len(vals):>3}{fmt(vals):>18}{med:>10.0f}{max(maxs):>8}  [{det}]{warn}")
        keys = list(means)
        if len(keys) == 3:
            a, b, c = keys
            print(f"\n  稀疏结构贡献 (①→②)：{means[b]-means[a]:+.2f}")
            print(f"  真实接线贡献 (②→③)：{means[c]-means[b]:+.2f}")
            print(f"  连接组 vs MLP (①→③)：{means[c]-means[a]:+.2f}")

    print("\n" + "=" * 78)
    print("全部 18 个 run 的逐条明细")
    print("=" * 78)
    print(f"{'tag':<28}{'arch':<12}{'seed':>5}{'均分':>10}{'最高':>8}{'用时min':>9}")
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        j = load(d.name)
        if j is None:
            print(f"{d.name:<28}{'(未完成)':<12}")
            continue
        print(f"{d.name:<28}{j['arch']:<12}{j.get('seed',''):>5}"
              f"{j['final_eval']['mean']:>10.1f}{j['final_eval']['max']:>8}"
              f"{j['wall_seconds']/60:>9.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
