#!/usr/bin/env python
"""把仿真的"哪些下行神经元响应 LC4"和**实验测得的 LC4 靶神经元集合**做统计学比对。

对照的文献事实（必须写清楚来源，不许含糊）
------------------------------------------
Dombrovski M, Peek MY, Park J-Y, ... Namiki S, Zipursky SL, Card GM.
**Synaptic gradients transform object location to action.** Nature 2023.
doi:10.1038/s41586-022-05562-8
  - LC4 与 **9 个下行神经元(DN)** 型接触；文中点名的有 **GF、DNp02、DNp04、DNp06、DNp11**；
    单个 LC4 对每个靶的突触数从 1 到 75 不等。
  - 光遗传激活：GF -> 短跳逃逸率 >90%；DNp04 或 DNp11 -> 长跳 15~40%。
  - DNp02 全细胞记录：0->30°、500°/s 扩张刺激下，前部视野 vs 后部视野
    44 vs 13 个脉冲（分析窗 = 刺激起始后 150 ms）。

我们要问的是一个可证伪的问题：
    **冻结连接组里"被 LC4 驱动的下行神经元"，是否显著就是实验认定的那批 LC4 靶？**
用 Fisher 精确检验（单尾，"富集"方向），在两种资产上都做一遍：
  截断图 59,548 神经元 / 472 个 DN；全脑图 144,837 神经元 / 1301 个 DN。

⚠️ DNp01 与 GF 是否同一个细胞，本次**未能在文献里确证**，所以两种口径都报：
   保守口径（不含 DNp01）和含 DNp01 的口径。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

# Dombrovski et al. 2023 点名的 LC4 靶 DN 型（保守：不含身份未确证的 GF/DNp01）
LIT_NAMED = ["DNp02", "DNp04", "DNp06", "DNp11"]
LIT_PLUS_GF = LIT_NAMED + ["DNp01"]
RESP_DELTA = 0.001          # 与 loom_transfer.py 的判据一致


def test(csv, label):
    p = OUT / csv
    if not p.exists():
        print(f"  跳过 {csv}（不存在）")
        return
    df = pd.read_csv(p)
    n = len(df)
    resp = set(df.loc[df["delta"] > RESP_DELTA, "cell_type"])
    print(f"\n=== {label}：{n} 个下行神经元，其中 {len(resp)} 个型响应 LC4 ===")
    print(f"  响应型：{sorted(resp)}")
    for names, tag in ((LIT_NAMED, "保守（文献点名的 4 型）"),
                       (LIT_PLUS_GF, "含 DNp01（若 DNp01=GF）")):
        tgt_cells = df[df["cell_type"].isin(names)]
        K = len(tgt_cells)
        hit_cells = tgt_cells[tgt_cells["delta"] > RESP_DELTA]
        hit_types = sorted(set(hit_cells["cell_type"]))
        # 2x2: [响应且是靶, 响应但非靶; 不响应但是靶, 都不]
        a = len(hit_cells)
        b = int((df["delta"] > RESP_DELTA).sum()) - a
        c = K - a
        d = n - a - b - c
        or_, pv = fisher_exact([[a, b], [c, d]], alternative="greater")
        print(f"  {tag}: 靶细胞 {K} 个 / 命中 {a} 个 -> 命中的型 {hit_types}")
        print(f"    Fisher 单尾 p = {pv:.3g}   OR = {or_:.1f}"
              f"   （{len(hit_types)}/{len(names)} 个文献靶型被仿真独立复现）")


def main() -> int:
    print("文献基准：Dombrovski et al., Nature 2023, doi:10.1038/s41586-022-05562-8")
    test("loom_dn_rank_real.csv", "截断资产 spiking_circuit (59,548 神经元)")
    test("loom_dn_rank_real_spiking_full.csv", "全脑资产 spiking_full (144,837 神经元)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
