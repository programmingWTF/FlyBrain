#!/usr/bin/env python
"""探查 FlyWire 连接组数据（边表 + 元数据）的结构，为剪子图做准备。

只读、不改数据。输出列名、行数、度分布、细胞类型分布等。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EDGE = DATA / "fafb_783_simple_edgelist.feather"
META = DATA / "fafb_783_meta.feather"


def show(name: str, df: pd.DataFrame, n: int = 5) -> None:
    print(f"\n=== {name} ===")
    print(f"shape = {df.shape}")
    print("columns:", list(df.columns))
    print(df.head(n).to_string())
    print("dtypes:", dict(df.dtypes.astype(str)))


def main() -> int:
    for p in (EDGE, META):
        if not p.exists():
            print(f"缺少 {p}")
            return 1

    edges = pd.read_feather(EDGE)
    show("edge list", edges)

    meta = pd.read_feather(META)
    show("meta", meta)

    # 找出可能的 pre/post/weight 列
    def guess(cands):
        for c in df_cols:
            for k in cands:
                if k.lower() in c.lower():
                    return c
        return None

    df_cols = list(edges.columns)
    pre = guess(["pre", "source", "upstream", "from"])
    post = guess(["post", "target", "downstream", "to"])
    wcol = guess(["weight", "syn", "count", "n_syn"])
    print(f"\n猜测列：pre={pre}  post={post}  weight={wcol}")
    if pre is None or post is None:
        print("!! 分不清 pre/post 列，需要人工看列名")
        return 2

    n_e = len(edges)
    n_pre = edges[pre].nunique()
    n_post = edges[post].nunique()
    print(f"\n边数 {n_e:,}  唯一 pre {n_pre:,}  post {n_post:,}")
    if wcol:
        w = edges[wcol].to_numpy()
        print(f"权重列 '{wcol}': min={w.min()} max={w.max()} mean={w.mean():.2f} "
              f"median={np.median(w):.0f} p99={np.percentile(w, 99):.0f}")

    # 度分布
    indeg = edges.groupby(post).size()
    outdeg = edges.groupby(pre).size()
    for nm, d in (("入度", indeg), ("出度", outdeg)):
        print(f"{nm}: mean={d.mean():.1f} median={d.median():.0f} "
              f"p90={d.quantile(0.9):.0f} p99={d.quantile(0.99):.0f} max={d.max():,}")

    # meta 里的分类列
    obj = [c for c in meta.columns if meta[c].dtype == object]
    for c in obj[:6]:
        vc = meta[c].value_counts(dropna=False)
        print(f"\nmeta['{c}'] 取值 {meta[c].nunique()} 种，top 12：")
        print(vc.head(12).to_string())

    # id 对齐检查
    idcol = None
    for c in meta.columns:
        if "root" in c.lower() or c.lower() in ("id", "bodyid", "body_id"):
            idcol = c
            break
    print(f"\nmeta 的 id 列 = {idcol}")
    if idcol:
        a = set(meta[idcol].astype(np.int64))
        b = set(edges[pre].astype(np.int64)) | set(edges[post].astype(np.int64))
        print(f"meta id 数 {len(a):,}   边表出现过的 id 数 {len(b):,}   交集 {len(a & b):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
