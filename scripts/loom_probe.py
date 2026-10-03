#!/usr/bin/env python
"""视觉逃避反射 第 0 步：通路审计（在做任何仿真之前必须先过这一关）。

要回答的问题（全部是"能不能做这个实验"的前提，不是结果）：
  A. 类型学审计：`cell_type.startswith("LC4")` 到底匹配了哪些真实类型？
     LC4 在 FlyWire 全脑里应该只有个位数细胞；若匹配到 207 个，说明吸进了
     LC40/LC41/LC42 等其它类型 —— 那"注入 LC4"这个说法就不成立。
  B. 原始边表里 DNp01 的突触前伙伴到底是谁（按类型 + 突触数排序）。
  C. **冻结资产**里 LC4 -> DNp01 的直接边：条数、解码权重、总和，
     以及这点电流能不能越过 LIF 阈值（给出定量判据）。
  D. 冻结资产里 LC4 -> DNp01 的最短跳数；若不连通，实验前提就断了。

用法
    D:/Code/FlyBrain/env/python.exe scripts/loom_probe.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import deque

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from fpv.spiking_brain import SpikingBrain, encode_lut   # noqa: E402

# 我们关心的类型名（key_types.json 里的键）及其在 extract_key_types.py 中的前缀
PREFIX = {
    "LC4": ["LC4"],
    "LPLC2": ["LPLC2"],
    "LPLC1": ["LPLC1"],
    "DNp01": ["DNp01"],
    "DNp04": ["DNp04"],
}


def sec(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ------------------------------------------------------------------ A 类型学
def audit_types() -> dict[str, list[int]]:
    sec("A. 类型学审计：前缀匹配到底吸进了什么")
    m = pd.read_feather(DATA / "fafb_783_meta.feather")
    m["id"] = m["fafb_783_id"].astype("int64")
    m["cell_type"] = m["cell_type"].fillna("")
    m["side"] = m["side"].fillna("?")
    print(f"meta 行数 {len(m):,}   唯一 id {m['id'].nunique():,}")
    exact_ids: dict[str, list[int]] = {}
    for name, pats in PREFIX.items():
        for pat in pats:
            pref = m["cell_type"].str.startswith(pat, na=False)
            ex = m["cell_type"] == pat
            print(f"\n  [{name}] 前缀 {pat!r}: 前缀匹配 {int(pref.sum()):,}  "
                  f"精确匹配 {int(ex.sum()):,}")
            vc = m.loc[pref, "cell_type"].value_counts()
            for ct, c in vc.head(12).items():
                mark = "  <== 精确" if ct == pat else ""
                print(f"      {ct:<18}{c:>5,}{mark}")
            if len(vc) > 12:
                print(f"      ...（共 {len(vc)} 种 cell_type）")
            if ex.any():
                sub = m.loc[ex]
                print(f"      精确 {pat} 的 side: "
                      f"{sub['side'].value_counts().to_dict()}   "
                      f"递质: {sub['neurotransmitter_predicted'].fillna('unknown').value_counts().to_dict()}")
                print(f"      id: {sub['id'].tolist()}")
                exact_ids[name] = sub["id"].astype("int64").tolist()
    return exact_ids


# ------------------------------------------------------- B 原始边表：DNp01 上游
def raw_partners(exact_ids: dict[str, list[int]]) -> None:
    sec("B. 原始边表：DNp01 的突触前伙伴（按突触数）")
    m = pd.read_feather(DATA / "fafb_783_meta.feather")
    m["id"] = m["fafb_783_id"].astype("int64")
    m["cell_type"] = m["cell_type"].fillna("")
    ct_of = pd.Series(m["cell_type"].to_numpy(), index=m["id"].to_numpy())
    e = pd.read_feather(DATA / "fafb_783_simple_edgelist.feather")
    e["pre"] = e["pre"].astype("int64")
    e["post"] = e["post"].astype("int64")
    print(f"边数 {len(e):,}")

    dn = set(exact_ids.get("DNp01", []))
    if not dn:
        print("  ! 没有精确的 DNp01 id，跳过")
        return
    sub = e[e["post"].isin(dn)]
    print(f"\n  DNp01(id={sorted(dn)}) 入边 {len(sub):,} 条，"
          f"伙伴 {sub['pre'].nunique():,} 个")
    g = sub.groupby("pre").agg(syn=("count", "sum"), w=("norm", "sum")).sort_values(
        "syn", ascending=False)
    g["cell_type"] = ct_of.reindex(g.index).fillna("?")
    print(f"  突触数分位: {np.percentile(g['syn'], [50, 75, 90, 99, 100]).round(1)}")
    print("\n  Top 25 上游：")
    print(f"    {'cell_type':<20}{'syn':>6}{'norm_sum':>11}   pre_id")
    for pid, r in g.head(25).iterrows():
        print(f"    {r['cell_type']:<20}{int(r['syn']):>6}{r['w']:>11.4f}   {pid}")
    # 视觉类型在上游里的占比
    vis = g[g["cell_type"].str.match(r"^(a?LC|LCx|LPLC|T4|T5|Mi|Lm|M)", na=False)]
    print(f"\n  上游里的视觉类（LC*/LPLC*/T4/T5 等）：{len(vis)} 个伙伴 / "
          f"{int(vis['syn'].sum()):,} 突触 / norm 合计 {vis['w'].sum():.4f}")
    print(f"  全上游 norm 合计 {g['w'].sum():.4f}")
    lc4 = exact_ids.get("LC4", [])
    if lc4:
        s4 = g.loc[g.index.isin(lc4)]
        if len(s4):
            print(f"\n  精确 LC4 (id={lc4}) -> DNp01: 边 {len(s4)} 条, "
                  f"突触 {int(s4['syn'].sum())}, norm {s4['w'].sum():.4f}")
        else:
            print(f"\n  精确 LC4 (id={lc4}) -> DNp01: 原始边表里【没有直接边】")


# ------------------------------------------------- C/D 冻结资产：边权 + 跳数
def frozen(brain: SpikingBrain, exact_ids: dict[str, list[int]]) -> None:
    sec("C. 冻结资产：LC4 -> DNp01 的直接边与可驱动性")
    lut = brain.lut.detach().cpu().numpy()
    ids = brain.node_ids                     # 子图索引 -> fafb id
    pos = pd.Series(np.arange(len(ids)), index=ids)

    def sub_idx(name: str) -> np.ndarray:
        want = exact_ids.get(name, [])
        hit = pos.reindex(pd.Index(want, dtype=np.int64)).to_numpy()
        return hit[~np.isnan(hit)].astype(np.int64)

    lc4 = sub_idx("LC4")
    dn = sub_idx("DNp01")
    print(f"资产内精确 LC4: {len(lc4)}/{len(exact_ids.get('LC4', []))} 个")
    print(f"资产内精确 DNp01: {len(dn)}/{len(exact_ids.get('DNp01', []))} 个")
    print(f"key_neurons['LC4'] 用了 {len(brain.key_idx('LC4'))} 个（前缀匹配）")
    print(f"key_neurons['DNp01'] 用了 {len(brain.key_idx('DNp01'))} 个")

    dn_set = set(dn.tolist())
    rows = []
    for s in lc4:
        lo, hi = int(brain.indptr[s]), int(brain.indptr[s + 1])
        if hi <= lo:
            continue
        tgt = brain.indices[lo:hi].detach().cpu().numpy()
        code = brain.codes[lo:hi].detach().cpu().numpy()
        sel = np.isin(tgt, dn)
        if sel.any():
            w = lut[code[sel]]
            for t, ww in zip(tgt[sel], w):
                rows.append((s, int(t), float(ww)))
    direct = pd.DataFrame(rows, columns=["pre", "post", "w"])
    print(f"\n  直接边 LC4 -> DNp01: {len(direct)} 条")
    if len(direct):
        print(direct.groupby("post")["w"].agg(["count", "sum", "max"]).to_string())

    # 也报告：用现有（前缀）LC4 群时有多少直接边 —— 解释文档里"1 跳"的说法从哪来
    key_lc4 = brain.key_idx("LC4").detach().cpu().numpy()
    cnt_all, sum_all = 0, 0.0
    for s in key_lc4:
        lo, hi = int(brain.indptr[s]), int(brain.indptr[s + 1])
        if hi <= lo:
            continue
        tgt = brain.indices[lo:hi].detach().cpu().numpy()
        code = brain.codes[lo:hi].detach().cpu().numpy()
        sel = np.isin(tgt, dn)
        if sel.any():
            cnt_all += int(sel.sum())
            sum_all += float(lut[code[sel]].sum())
    print(f"  （对照）前缀 LC4 群 {len(key_lc4)} 个 -> DNp01 直接边 {cnt_all} 条, "
          f"权重合计 {sum_all:.4f}")

    # --- 可驱动性定量判据：这些电流能不能让 DNp01 越过阈值 ---
    sec("C'. LIF 可驱动性：DNp01 需要的输入 vs 实际能拿到的")
    leak, gain, thr = brain.leak, brain.gain, brain.threshold
    print(f"  leak={leak:.5f}  gain={gain}  threshold={thr}  tonic={brain.tonic}")
    print(f"  稳态 G = gain*C/(1-leak)，故每 tick 期望突触输入需 >= "
          f"(1-leak)*thr/gain = {(1 - leak) * thr / gain:.5f}（tonic=0 时）")
    # DNp01 的全部入边（需要转置 CSR）
    import scipy.sparse as sp
    n = brain.N
    indptr = brain.indptr.detach().cpu().numpy()
    indices = brain.indices.detach().cpu().numpy()
    codes = brain.codes.detach().cpu().numpy()
    w_all = lut[codes]
    A = sp.csr_matrix((w_all, (np.repeat(np.arange(n), np.diff(indptr)), indices)),
                      shape=(n, n))
    AT = A.T.tocsr()
    for label, target in (("精确 DNp01", dn), ("前缀 DNp01", dn),
                          ("DNp04", sub_idx("DNp04"))):
        if len(target) == 0:
            continue
        for t in target[:2]:
            lo, hi = int(AT.indptr[t]), int(AT.indptr[t + 1])
            pre = AT.indices[lo:hi]
            win = AT.data[lo:hi]
            need = (1 - leak) * thr / gain
            print(f"\n  神经元 #{t}（{label}）入边 {len(pre):,} 条, "
                  f"权重合计 {win.sum():.3f}, 需要 >= {need:.4f}")
            print(f"    最大单边权 {win.max():.4f}   第2大 {np.sort(win)[-2]:.4f}")
            # 要多少突触前伙伴同时发放才能触发
            sw = np.sort(win)[::-1]
            cum = np.cumsum(sw)
            k = int(np.searchsorted(cum, need) + 1) if cum.size else -1
            print(f"    若伙伴每 tick 都发放：最少需 {k} 个伙伴（取最强若干条）")
            print(f"    若伙伴发放率 p：需 p*Σw >= {need:.4f} -> Σw(全部)={win.sum():.3f}")
        break

    # DNp01 上游的类型构成（冻结资产内）
    sec("D. 冻结资产里 DNp01 的上游类型 + LC4->DNp01 最短跳数")
    meta = pd.read_feather(DATA / "fafb_783_meta.feather")
    meta["id"] = meta["fafb_783_id"].astype("int64")
    meta["cell_type"] = meta["cell_type"].fillna("")
    ct_of = pd.Series(meta["cell_type"].to_numpy(), index=meta["id"].to_numpy())
    for t in dn:
        lo, hi = int(AT.indptr[t]), int(AT.indptr[t + 1])
        pre = AT.indices[lo:hi]
        win = AT.data[lo:hi]
        s4 = np.isin(pre, lc4)
        skl = np.isin(pre, key_lc4)
        print(f"\n  DNp01 #{t}: 上游 {len(pre)} 个, 权重合计 {win.sum():.3f}")
        print(f"    其中精确 LC4: {int(s4.sum())} 个 (权重 {win[s4].sum():.4f})")
        print(f"    其中前缀 LC4 群: {int(skl.sum())} 个 (权重 {win[skl].sum():.4f})")
        ptypes = ct_of.reindex(pd.Index(ids[pre])).fillna("?")
        topw = pd.DataFrame({"ct": ptypes.to_numpy(), "w": win}).groupby("ct")["w"] \
            .agg(["count", "sum"]).sort_values("sum", ascending=False)
        print("    上游类型按权重合计 Top 15:")
        print(topw.head(15).to_string(float_format=lambda x: f"{x:.4f}"))

    # BFS：从 LC4 群到 DNp01 的最短跳数
    from collections import defaultdict
    dist = {int(s): 0 for s in lc4} if len(lc4) else {int(s): 0 for s in key_lc4}
    dq = deque(dist)
    reach = {}
    while dq:
        v = dq.popleft()
        if dist[v] >= 8:
            continue
        lo, hi = int(brain.indptr[v]), int(brain.indptr[v + 1])
        for u in brain.indices[lo:hi].detach().cpu().numpy():
            u = int(u)
            if u not in dist:
                dist[u] = dist[v] + 1
                if u in dn_set:
                    reach[u] = dist[u]
                dq.append(u)
    src_name = "精确 LC4" if len(lc4) else "前缀 LC4 群"
    print(f"\n  BFS（{src_name} 出发）到 DNp01 的最短跳数: {reach or '不可达'}")
    print(f"  可达神经元总数: {len(dist):,} / {brain.N:,}")


def main() -> int:
    exact = audit_types()
    raw_partners(exact)
    brain = SpikingBrain.from_npz("spiking_circuit", device="cpu")
    print(f"\n冻结资产: N={brain.N:,}  E={len(brain.codes):,}")
    frozen(brain, exact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
