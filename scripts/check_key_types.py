#!/usr/bin/env python
"""查验任务相关的关键细胞类型在 FlyWire v783 里的存在情况与数量。

FlappyBird 的 18 维状态说到底就是「到最近管道的水平距离 / 缺口中心相对高度 / 垂直速度」。
果蝇里对应的真实神经元：
  - LC4 / LPLC2 : 逼近(loom)/碰撞检测，投射到巨型纤维
  - T4 / T5     : 运动方向检测（ON/OFF 通路），视叶输出主力
  - LC10        : 目标追踪
  - DNp01 等     : 下行巨型纤维（逃逸反应）
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
META = ROOT / "data" / "fafb_783_meta.feather"

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

WANT = ["LC4", "LPLC2", "LPLC1", "LC10", "LC9", "LC11", "LC15", "LC17", "LC21",
        "T4", "T5", "T2", "T3", "DNp01", "DNp", "DNa", "DNb", "DNc", "DNd", "DNe",
        "DNg", "DNae", "MDN", "VPN"]


def main() -> int:
    m = pd.read_feather(META)
    m["ct"] = m["cell_type"].fillna("")
    m["cc"] = m["cell_class"].fillna("")
    m["sc"] = m["cell_sub_class"].fillna("")

    print("=== 在 cell_type 里精确/前缀匹配关键类型 ===")
    for w in WANT:
        hit = m[m["ct"].str.fullmatch(w) | m["ct"].str.startswith(w + "-", na=False)]
        # 也统计所有以 w 开头的
        pref = m[m["ct"].str.startswith(w, na=False)]
        print(f"  {w:8s} 精确/短横线 {len(hit):6,}   前缀 {len(pref):6,}")

    print("\n=== 所有 cell_type 以 DN/LC/LPLC/T4/T5 开头的（top 40） ===")
    pat = m["ct"].str.match(r"^(DN|LC|LPLC|T4|T5|LLPC)")
    print(m.loc[pat, "ct"].value_counts().head(40).to_string())

    print("\n=== descending 超类里 cell_type 分布（top 25） ===")
    dn = m[m["sc"].str.contains("descending", case=False, na=False) | m["cc"].eq("descending_neuron")]
    print(f"descending 总数 {len(dn):,}")
    print(dn["ct"].value_counts().head(25).to_string())

    print("\n=== LC4 / LPLC2 详细 ===")
    for w in ("LC4", "LPLC2", "LC10a"):
        s = m[m["ct"].str.contains(w, na=False)]
        print(f"\n[{w}] {len(s)} 个")
        if len(s):
            print(s[["fafb_783_id", "side", "region", "super_class", "cell_class", "cell_type"]]
                  .head(10).to_string())

    print("\n=== T4/T5 各类数量（视叶输出主力） ===")
    t = m[m["ct"].str.match(r"^T[45][a-d]?$")]
    print(t["ct"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
