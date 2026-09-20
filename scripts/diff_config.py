#!/usr/bin/env python
"""逐字段对比「原项目 baseline_config」与「我的 graph_config」，找出训练差异的来源。"""
from __future__ import annotations

import dataclasses
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv.graph_config import graph_config  # noqa: E402
from fpv.vendor_flappyrl.config import baseline_config  # noqa: E402

theirs = baseline_config()
mine = graph_config()

print(f"{'字段':<24}{'原项目 baseline':>18}{'我的 graph_config':>20}")
print("-" * 64)
diffs = []
for f in dataclasses.fields(theirs):
    a = getattr(theirs, f.name)
    b = getattr(mine, f.name, "<缺失>")
    flag = "" if a == b else "   <-- 不同"
    if a != b:
        diffs.append(f.name)
    print(f"{f.name:<24}{str(a):>18}{str(b):>20}{flag}")

print()
print(f"相同 {len(dataclasses.fields(theirs)) - len(diffs)} 项，不同 {len(diffs)} 项：{diffs}")
