#!/usr/bin/env python
"""为 FlappyBird 控制任务挑选「运动输出」与「感觉输入」候选神经元。

关键问题：flow=='efferent' 只有 194 个，太少了。需要找到：
  - 下行神经元（descending）：把信号从脑送到 VNC，是"动作输出"的天然读出点
  - 运动神经元（motor）：真正的肌肉驱动
  - 与视觉/机械感觉相关的输入类
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
META = ROOT / "data" / "fafb_783_meta.feather"

pd.set_option("display.width", 200)


def main() -> int:
    m = pd.read_feather(META)
    m["id"] = m["fafb_783_id"].astype("int64")

    print("=== flow 全分布 ===")
    print(m["flow"].value_counts(dropna=False).to_string())

    print("\n=== super_class 全分布 ===")
    print(m["super_class"].value_counts(dropna=False).to_string())

    print("\n=== cell_class 全部（按数量） ===")
    print(m["cell_class"].value_counts(dropna=False).to_string())

    print("\n=== 含 'descending'/'motor'/'efferent' 的行 ===")
    pat = "descend|motor|efferent|command"
    hit = m[
        m["cell_class"].str.contains(pat, case=False, na=False)
        | m["cell_sub_class"].str.contains(pat, case=False, na=False)
        | m["super_class"].str.contains(pat, case=False, na=False)
        | m["cell_type"].str.contains(pat, case=False, na=False)
        | m["cell_function"].str.contains(pat, case=False, na=False)
    ]
    print(f"命中 {len(hit)} 个")
    print(hit["super_class"].value_counts(dropna=False).to_string())
    print(hit["cell_class"].value_counts(dropna=False).to_string())

    print("\n=== 感觉侧：视觉 / 机械感觉 ===")
    for col in ("cell_class", "cell_sub_class", "super_class"):
        v = m[col].value_counts(dropna=False)
        sub = v[v.index.astype(str).str.contains("visual|photo|mechano|sensory|optic|descending|ascending", case=False, na=False)]
        if len(sub):
            print(f"\n[{col}]")
            print(sub.to_string())

    print("\n=== body_part_effector 分布 ===")
    print(m["body_part_effector"].value_counts(dropna=False).head(20).to_string())

    print("\n=== cell_function 分布 ===")
    print(m["cell_function"].value_counts(dropna=False).head(25).to_string())

    print("\n=== 神经递质分布（对激励/抑制建模有用） ===")
    nt = m["neurotransmitter_predicted"].astype(str)
    print(nt.value_counts(dropna=False).head(20).to_string())

    print("\n=== 推荐的运动读出候选（下行 + 运动 + efferent） ===")
    cand = m[
        (m["flow"] == "efferent")
        | m["cell_class"].str.contains("descending|motor", case=False, na=False)
        | m["super_class"].str.contains("descending|motor", case=False, na=False)
        | m["cell_sub_class"].str.contains("descending|motor", case=False, na=False)
    ]
    print(f"候选 {len(cand)} 个")
    print(cand[["id", "region", "flow", "super_class", "cell_class", "cell_sub_class", "cell_type"]]
          .head(40).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
