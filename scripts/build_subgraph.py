#!/usr/bin/env python
"""从 FlyWire v783 全连接组剪出 FlappyBird 用的「任务子图」。

设计
----
- 输入种子（sensory）：把 18 维游戏状态喂进去的神经元。
  选运动/碰撞敏感的视觉投射神经元：LC4 / LPLC2 / T4 / T5 / LC10 / lobula_plate_tangential …
- 输出读出（motor）：descending_neuron（1301 个），大脑→VNC 的真实指令通道。
- 子图 = 种子周围 BFS 半径 r 内的神经元；同时报告「种子→读出」是否存在路径。

输出
----
- data/subgraph.npz   压缩稀疏结构的 CSR 数组（pre/post 索引 + 权重）
- data/subgraph.json  元数据：节点 id 列表、类别、IO 分组、连通性报告
- 终端打印连通性与规模报告，供人工确认后再进入训练

用法
----
    python scripts/build_subgraph.py --radius 2
    python scripts/build_subgraph.py --radius 3 --max-nodes 40000
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EDGE = DATA / "fafb_783_simple_edgelist.feather"
META = DATA / "fafb_783_meta.feather"

# 感觉输入候选：运动/碰撞敏感的视觉通路（Flight 相关）
SENSORY_SETS = {
    # 碰撞/逼近检测（loom）：投射到巨型纤维，生物学上最贴近「要不要躲」
    "collision": ["LC4", "LPLC2", "LPLC1", "LC9", "LC11", "LC15", "LC17", "LC21"],
    # 运动方向检测：视叶输出主力
    "motion": ["T4", "T5", "T2", "T3"],
    # 目标追踪
    "target": ["LC10"],
    # 以上全部
    "flight": None,
}

# 明确排除的非神经元
EXCLUDE_SUPER = {"glia", "trachea"}

# 视叶内部神经元：占全脑 54%（photoreceptor/lamina/medulla/transmedullary…），
# 两跳之内就能把整个脑吞进来。本任务把种子落在**视叶输出侧**（T4/T5/LPLC2/LC4 等），
# 所以视叶内部神经元只当「输入来源」，不参与 BFS 扩张。
OPTIC_LOBE_SUPER = {"optic_lobe_intrinsic"}


def norm_key(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def pick_sensory(meta: pd.DataFrame, mode: str) -> np.ndarray:
    """按 cell_type 前缀匹配运动敏感视觉神经元。mode 见 SENSORY_SETS。"""
    if mode == "flight":
        pats = [p for grp in SENSORY_SETS.values() if grp for p in grp]
    else:
        pats = SENSORY_SETS[mode]
    ct = meta["cell_type"].fillna("")
    hit = np.zeros(len(meta), dtype=bool)
    for p in pats:
        hit |= ct.str.startswith(p, na=False).to_numpy()
    return hit


def bfs(adj: sp.csr_matrix, seeds: np.ndarray, radius: int) -> np.ndarray:
    """从 seeds 出发在 adj 上 BFS radius 跳，返回访问到的节点掩码。"""
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    seen[seeds] = True
    frontier = seeds
    indptr, indices = adj.indptr, adj.indices
    for _ in range(radius):
        nxt = []
        for v in frontier:
            a, b = indptr[v], indptr[v + 1]
            if b > a:
                nxt.append(indices[a:b])
        if not nxt:
            break
        cand = np.concatenate(nxt)
        cand = np.unique(cand)
        cand = cand[~seen[cand]]
        if cand.size == 0:
            break
        seen[cand] = True
        frontier = cand
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=2, help="种子/读出的 BFS 半径")
    ap.add_argument("--max-nodes", type=int, default=60000, help="节点数上限（超出则收窄）")
    ap.add_argument("--min-weight", type=int, default=1, help="丢弃突触数少于该值的边")
    ap.add_argument("--sensory", default="flight", choices=sorted(SENSORY_SETS),
                    help="感觉种子集合：collision/motion/target/flight")
    ap.add_argument("--sensory-max", type=int, default=0,
                    help="感觉种子数上限；超出则按「到运动读出的突触总权重」取前 N（0=不限）")
    ap.add_argument("--motor-class", default="descending",
                    help="运动读出：descending | descending+motor | efferent_only")
    ap.add_argument("--out", default="subgraph", help="输出文件名前缀（data/<out>.npz）")
    a = ap.parse_args()

    t0 = time.time()
    print("载入数据 ...")
    m = pd.read_feather(META)
    e = pd.read_feather(EDGE)
    print(f"  meta {m.shape}  edges {e.shape}  ({time.time()-t0:.1f}s)")

    # ---- 全局 id -> 索引 ----
    m = m.copy()
    m["id"] = m["fafb_783_id"].astype("int64")
    ids = np.sort(m["id"].unique())
    idx = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids)
    n_all = len(ids)
    print(f"  唯一神经元 {n_all:,}")

    if a.min_weight > 1:
        e = e[e["count"] >= a.min_weight]
        print(f"  按 min-weight>={a.min_weight} 过滤后剩余边 {len(e):,}")

    pre = idx.reindex(e["pre"].astype("int64")).to_numpy()
    post = idx.reindex(e["post"].astype("int64")).to_numpy()
    ok = ~(np.isnan(pre) | np.isnan(post))
    pre, post = pre[ok].astype(np.int64), post[ok].astype(np.int64)
    cnt = e["count"].to_numpy()[ok].astype(np.float32)
    norm = e["norm"].to_numpy()[ok].astype(np.float32)
    print(f"  成功映射的边 {len(pre):,}")

    adj_f = sp.csr_matrix((np.ones(len(pre), np.float32), (pre, post)), shape=(n_all, n_all))
    adj_b = sp.csr_matrix((np.ones(len(pre), np.float32), (post, pre)), shape=(n_all, n_all))

    # ---- 选 IO ----
    m = m.set_index("id").reindex(ids).reset_index()
    m["super_class"] = m["super_class"].fillna("")
    m["cell_class"] = m["cell_class"].fillna("")
    m["cell_sub_class"] = m["cell_sub_class"].fillna("")
    m["cell_type"] = m["cell_type"].fillna("")

    is_glia = m["super_class"].isin(EXCLUDE_SUPER).to_numpy()
    sens = pick_sensory(m, a.sensory) & ~is_glia
    if a.motor_class == "efferent_only":
        motor = m["flow"].eq("efferent").to_numpy() & ~is_glia
    elif a.motor_class == "descending+motor":
        motor = (
            m["super_class"].isin(["descending", "motor"]).to_numpy()
            | m["cell_class"].isin(["descending_neuron", "motor_neuron"]).to_numpy()
        ) & ~is_glia
    else:  # descending
        motor = (
            m["super_class"].eq("descending").to_numpy()
            | m["cell_class"].eq("descending_neuron").to_numpy()
        ) & ~is_glia

    # 感觉种子太多会挤掉中间层（1374 个种子喂 18 维输入本身也是浪费）。
    # 按「到运动读出的突触总权重」挑最强的一批，保留最有信息量的输入通道。
    if a.sensory_max and sens.sum() > a.sensory_max:
        sel = sens[pre] & motor[post]
        strength = np.zeros(n_all, dtype=np.float64)
        np.add.at(strength, pre[sel], cnt[sel].astype(np.float64))
        cand = np.nonzero(sens)[0]
        order = cand[np.argsort(-strength[cand])]
        keep_s = np.zeros(n_all, dtype=bool)
        keep_s[order[: a.sensory_max]] = True
        print(f"  感觉种子从 {sens.sum():,} 收到上限 {a.sensory_max:,}"
              f"（按到运动读出的突触总权重取 top）")
        sens = keep_s

    print(f"\n感觉种子候选 {sens.sum():,} 个；运动读出候选 {motor.sum():,} 个")
    print("  感觉种子里 top cell_type:")
    print(m.loc[sens, "cell_type"].value_counts().head(12).to_string())

    # ---- BFS 子图（用 scipy 稀疏布尔乘做可达性，比 python 循环快得多） ----
    # 「死胡同」节点：视叶内部。扩张时穿过它，但不把它纳入子图。
    blocked = is_glia | m["super_class"].isin(OPTIC_LOBE_SUPER).to_numpy()
    pool = ~blocked
    print(f"\n不参与扩张的节点（胶质/气管/视叶内部）：{blocked.sum():,}")

    Af = (adj_f > 0).astype(np.int8)
    Ab = (adj_b > 0).astype(np.int8)
    # 只允许落在 pool 里的节点被保留；但为了穿过视叶，邻接矩阵保留全部
    keepmask = pool.copy()

    def reach(adj, seeds_mask, r):
        """从种子出发 r 跳可达（含种子），返回布尔掩码。"""
        v = seeds_mask.astype(np.int8)
        out = v.astype(bool)
        for _ in range(r):
            v = ((adj.T @ v) > 0).astype(np.int8) if False else ((v @ adj) > 0).astype(np.int8)
            out |= v.astype(bool)
        return out

    print(f"BFS radius={a.radius} ...")
    sens0 = np.nonzero(sens)[0]
    mot0 = np.nonzero(motor)[0]
    r_out_s = reach(Af, np.isin(np.arange(n_all), sens0).astype(np.int8), a.radius)
    r_in_s = reach(Ab, np.isin(np.arange(n_all), sens0).astype(np.int8), a.radius)
    r_out_m = reach(Af, np.isin(np.arange(n_all), mot0).astype(np.int8), a.radius)
    r_in_m = reach(Ab, np.isin(np.arange(n_all), mot0).astype(np.int8), a.radius)
    keep = (r_out_s | r_in_s | r_out_m | r_in_m) & keepmask
    print(f"  并集后节点 {keep.sum():,}")

    if keep.sum() > a.max_nodes:
        print(f"  超过上限 {a.max_nodes:,}，按「离种子/读出的跳数」收紧 ...")
        core = (sens | motor) & keep
        # 从核心出发逐跳生长，直到达到上限
        sel = core.copy()
        frontier = core.copy()
        for hop in range(1, 20):
            if sel.sum() >= a.max_nodes:
                break
            v = frontier.astype(np.int8)
            nxt = (((v @ Af) > 0) | ((v @ Ab) > 0))
            nxt &= keep & ~sel
            # 只加剩余额度
            room = a.max_nodes - int(sel.sum())
            if room <= 0:
                break
            cand = np.nonzero(nxt)[0]
            if cand.size > room:
                # 同跳内按度数（越中心越优先）排序
                deg = np.asarray(Af[cand].sum(1)).ravel() + np.asarray(Ab[cand].sum(1)).ravel()
                cand = cand[np.argsort(-deg)][:room]
            sel[cand] = True
            frontier = np.zeros(n_all, dtype=bool)
            frontier[cand] = True
            print(f"    hop {hop}: +{cand.size:,}  -> 共 {sel.sum():,}")
        keep = sel
        print(f"  收紧后节点 {keep.sum():,}")

    # ---- 诱导子图 ----
    in_keep = keep[pre] & keep[post]
    sp_, po_ = pre[in_keep], post[in_keep]
    w_, nm_ = cnt[in_keep], norm[in_keep]
    print(f"\n子图：节点 {keep.sum():,}  边 {len(sp_):,}")

    # 重编号
    new_id = np.full(n_all, -1, dtype=np.int64)
    new_id[keep] = np.arange(keep.sum(), dtype=np.int64)
    src, dst = new_id[sp_], new_id[po_]

    # ---- 连通性报告 ----
    n_sub = int(keep.sum())
    A = sp.csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(n_sub, n_sub))
    sens_sub = np.nonzero(keep & sens)[0]
    mot_sub = np.nonzero(keep & motor)[0]
    # 重映射到子图编号
    sens_sub = new_id[sens_sub]
    mot_sub = new_id[mot_sub]
    print(f"  子图内 感觉种子 {len(sens_sub):,}  运动读出 {len(mot_sub):,}")

    # 从感觉种子出发的可达性（布尔稀疏乘，多跳迭代到收敛）
    v = np.zeros(n_sub, dtype=np.int8)
    v[sens_sub] = 1
    B = (A > 0).astype(np.int8)
    reach = v.astype(bool)
    for _ in range(30):
        v2 = ((v @ B) > 0).astype(np.int8)
        new = v2 & ~v
        if new.sum() == 0:
            break
        v = v | v2
        reach |= v.astype(bool)
    hit_motor = int(reach[mot_sub].sum()) if len(mot_sub) else 0
    print(f"  ★ 从感觉种子可达的运动神经元：{hit_motor}/{len(mot_sub)}"
          f"  ({100.0*hit_motor/max(1,len(mot_sub)):.1f}%)")

    deg_in = np.bincount(dst, minlength=n_sub)
    deg_out = np.bincount(src, minlength=n_sub)
    print(f"  度分布：出度 mean={deg_out.mean():.1f} max={deg_out.max():,}   "
          f"入度 mean={deg_in.mean():.1f} max={deg_in.max():,}")
    iso = int(((deg_in + deg_out) == 0).sum())
    print(f"  孤立节点 {iso:,}")

    # 类别构成
    print("\n  子图 super_class 构成：")
    print(m.loc[keep, "super_class"].value_counts().head(12).to_string())

    # ---- 保存 ----
    is_sens_sub = np.zeros(n_sub, dtype=bool)
    is_sens_sub[sens_sub] = True
    is_mot_sub = np.zeros(n_sub, dtype=bool)
    is_mot_sub[mot_sub] = True

    out_npz = DATA / f"{a.out}.npz"
    np.savez_compressed(
        out_npz,
        src=src.astype(np.int32),
        dst=dst.astype(np.int32),
        count=w_.astype(np.float32),
        norm=nm_.astype(np.float32),
        node_ids=ids[keep].astype(np.int64),
        is_sensory=is_sens_sub,
        is_motor=is_mot_sub,
    )
    meta_out = {
        "sensory_set": a.sensory,
        "radius": a.radius,
        "min_weight": a.min_weight,
        "n_nodes": n_sub,
        "n_edges": int(len(src)),
        "n_sensory": int(len(sens_sub)),
        "n_motor": int(len(mot_sub)),
        "motor_reachable_from_sensory": hit_motor,
        "nnz_ratio": float(len(src) / max(1, n_sub * n_sub)),
        "out_degree_mean": float(deg_out.mean()),
        "out_degree_max": int(deg_out.max()),
        "isolated": iso,
        "super_class_counts": m.loc[keep, "super_class"].value_counts().to_dict(),
        "cell_class_counts": m.loc[keep, "cell_class"].value_counts().head(40).to_dict(),
    }
    out_json = DATA / f"{a.out}.json"
    out_json.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2))
    print(f"\n✓ 写出 {out_npz}  和  {out_json}   ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
