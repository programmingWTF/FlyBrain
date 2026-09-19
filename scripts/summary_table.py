#!/usr/bin/env python
"""打印所有候选子图的规模对照表。"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

hdr = ("配置", "节点", "边", "感觉", "读出", "可达%", "中央脑", "视投射", "均出度", "孤立")
print(f"{hdr[0]:34s} {hdr[1]:>6s} {hdr[2]:>9s} {hdr[3]:>5s} {hdr[4]:>5s} "
      f"{hdr[5]:>6s} {hdr[6]:>7s} {hdr[7]:>7s} {hdr[8]:>7s} {hdr[9]:>4s}")

for p in sorted(DATA.glob("sg_*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    sc = d["super_class_counts"]
    reach = d["motor_reachable_from_sensory"] / max(1, d["n_motor"]) * 100
    print(f"{p.stem:34s} {d['n_nodes']:6d} {d['n_edges']:9,d} {d['n_sensory']:5d} "
          f"{d['n_motor']:5d} {reach:6.1f} {sc.get('central_brain_intrinsic',0):7d} "
          f"{sc.get('visual_projection',0):7d} {d['out_degree_mean']:7.1f} {d['isolated']:4d}")

print("\n基线 MLP (RainbowNet 躯干, 18->256->256) 可学参数 ≈ 200k；开 Noisy 再 +100k")
