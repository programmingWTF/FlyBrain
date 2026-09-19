#!/usr/bin/env python
"""验证 LC4/LPLC2(LPLC) → 下行神经元 是否「直接」相连，以及子图的真实计算深度。

这一条很关键：如果碰撞检测神经元直接接到下行神经元，
那么「感觉得到 → 动作输出」只隔 1 层，图策略的表达能力会非常受限。
"""
from __future__ import annotations

import pathlib
import sys
from collections import deque

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

COLLISION = ["LC4", "LPLC2", "LPLC1", "LC9", "LC11", "LC15", "LC17", "LC21"]
MOTION = ["T4", "T5"]


def main() -> int:
    m = pd.read_feather(DATA / "fafb_783_meta.feather")
    e = pd.read_feather(DATA / "fafb_783_simple_edgelist.feather")
    m["id"] = m["fafb_783_id"].astype("int64")
    m["ct"] = m["cell_type"].fillna("")

    ids = np.sort(m["id"].unique())
    idx = pd.Series(np.arange(len(ids)), index=ids)
    n = len(ids)

    pre = idx.reindex(e["pre"].astype("int64")).to_numpy()
    post = idx.reindex(e["post"].astype("int64")).to_numpy()
    ok = ~(np.isnan(pre) | np.isnan(post))
    pre, post = pre[ok].astype(np.int64), post[ok].astype(np.int64)
    cnt = e["count"].to_numpy()[ok]

    mm = m.set_index("id").reindex(ids).reset_index()
    mm["ct"] = mm["cell_type"].fillna("")

    def mask_of(pats):
        out = np.zeros(n, dtype=bool)
        for p in pats:
            out |= mm["ct"].str.startswith(p, na=False).to_numpy()
        return out

    coll = mask_of(COLLISION)
    mot = mask_of(MOTION)
    dn = (mm["super_class"].eq("descending").to_numpy()
          | mm["cell_class"].eq("descending_neuron").to_numpy())

    print(f"碰撞检测 {coll.sum():,}  T4/T5 {mot.sum():,}  下行 {dn.sum():,}")

    # 直接边统计（在子集之间）
    def direct(a_mask, b_mask, label):
        sel = a_mask[pre] & b_mask[post]
        k = int(sel.sum())
        src_n = int(a_mask.sum())
        dst_n = int(b_mask.sum())
        # 有多少 b 至少收到一条来自 a 的边
        covered = len(np.unique(post[sel])) if k else 0
        w = cnt[sel]
        print(f"  {label}: 直接边 {k:,}   覆盖目标 {covered}/{dst_n} "
              f"({100*covered/max(1,dst_n):.1f}%)   突触数 中位 {np.median(w) if k else 0:.0f} "
              f"最大 {w.max() if k else 0}")
        return k

    print("\n=== 直接连边检查 ===")
    c2dn = direct(coll, dn, "碰撞检测 -> 下行")
    t2dn = direct(mot, dn, "T4/T5 -> 下行")
    c2t = direct(coll, mot, "碰撞检测 -> T4/T5")
    dn2dn = direct(dn, dn, "下行 -> 下行（层内）")

    # 平均最短路径：从碰撞检测到下行
    A = sp.csr_matrix((np.ones(len(pre), np.float32), (pre, post)), shape=(n, n))
    Af = (A > 0)
    src_nodes = np.nonzero(coll)[0]
    tgt = np.nonzero(dn)[0]
    tgt_set = set(tgt.tolist())

    print("\n=== 从碰撞检测出发的 BFS 深度分布（到最近的下行神经元）===")
    dist = np.full(n, -1, dtype=np.int32)
    dq = deque()
    for s in src_nodes:
        if dist[s] < 0:
            dist[s] = 0
            dq.append(s)
    found = {}
    while dq:
        v = dq.popleft()
        if dist[v] > 8:
            continue
        if v in tgt_set and dist[v] > 0:
            found.setdefault(dist[v], 0)
            found[dist[v]] += 1
        start, end = Af.indptr[v], Af.indptr[v + 1]
        for w in Af.indices[start:end]:
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                dq.append(w)
    for d in sorted(found):
        print(f"  距离 {d} 跳：{found[d]:,} 个下行神经元")
    tot = sum(found.values())
    print(f"  可达下行总数 {tot}/{len(tgt)} ({100*tot/len(tgt):.1f}%)")

    # 子图深度：加载已生成的子图，算 sens->motor 的最短路
    for name in ("sg_collision_3000", "sg_flight_4000"):
        p = DATA / f"{name}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        src, dst = d["src"], d["dst"]
        ns = int(d["node_ids"].shape[0])
        is_s, is_mo = d["is_sensory"], d["is_motor"]
        As = sp.csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(ns, ns))
        Asf = As > 0
        dist2 = np.full(ns, -1, dtype=np.int32)
        dq = deque()
        for s in np.nonzero(is_s)[0]:
            dist2[s] = 0
            dq.append(s)
        depth_hist = {}
        while dq:
            v = dq.popleft()
            if is_mo[v] and dist2[v] > 0:
                depth_hist[dist2[v]] = depth_hist.get(dist2[v], 0) + 1
            st, en = Asf.indptr[v], Asf.indptr[v + 1]
            for w in Asf.indices[st:en]:
                if dist2[w] < 0:
                    dist2[w] = dist2[v] + 1
                    dq.append(w)
        print(f"\n=== 子图 {name}: 感觉->读出 最短跳数分布 ===")
        for k in sorted(depth_hist)[:8]:
            print(f"  {k} 跳：{depth_hist[k]:,}")
        print(f"  读出覆盖 {sum(depth_hist.values())}/{int(is_mo.sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
