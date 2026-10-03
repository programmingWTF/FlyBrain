#!/usr/bin/env python
"""阶段 1.3：构建「脉冲连接组」紧凑二进制资产。

对齐 pinme.dev 的 FLYW 格式思路，但数据源是 FlyWire v783 全脑：
    - CSR（压缩稀疏行）布局：每个神经元的**出边**连续存放
    - 边权：log 量化的突触强度（unsigned byte 0..255）
    - **符号**：来自神经递质预测（GABA=抑制 -> 负号），存在代码的 bit7
      （pinme.dev 就是这么做的：code<128 兴奋 / code>=128 抑制）

为什么必须是 CSR 而不是 COO
--------------------------
LIF 脉冲仿真每个 tick 都要做「按源神经元聚集突触电流」，
CSR 可以按神经元顺序遍历、内存连续；COO 会退化成随机访问。
pinme.dev 的 FLYW 也是 CSR。

输出的 .npz 可以直接被 numpy/torch 加载；同时导出二进制 .bin 供网页端使用。

用法
----
    python scripts/build_spiking_asset.py                       # 全脑
    python scripts/build_spiking_asset.py --min-count 3         # 过滤弱突触
    python scripts/build_spiking_asset.py --max-nodes 30000     # 只保留关键回路
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EDGE = DATA / "fafb_783_simple_edgelist.feather"
META = DATA / "fafb_783_meta.feather"
NT = DATA / "nt_index.npz"
KEYS = DATA / "key_types.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="spiking_flywire")
    ap.add_argument("--min-count", type=int, default=1, help="丢弃突触数小于该值的边")
    ap.add_argument("--max-nodes", type=int, default=0, help="0=全脑；>0 只保留关键回路 r 跳内")
    ap.add_argument("--radius", type=int, default=3, help="--max-nodes 时的 BFS 半径")
    ap.add_argument("--ln-min", type=float, default=-11.694705,
                    help="log 量化下界（对齐 pinme.dev；权重范围 exp(ln_min)..1）")
    a = ap.parse_args()

    t0 = time.time()
    print("载入 ...")
    e = pd.read_feather(EDGE)
    m = pd.read_feather(META)
    nt = np.load(NT)
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    print(f"  边 {len(e):,}  神经元 {len(m):,}  ({time.time()-t0:.1f}s)")

    # ---- 全局 id -> 索引（0..N-1）----
    m = m.copy()
    m["id"] = m["fafb_783_id"].astype("int64")
    ids = np.sort(m["id"].unique())
    N = len(ids)
    id_to_i = pd.Series(np.arange(N, dtype=np.int64), index=ids)

    pre_i = id_to_i.reindex(e["pre"].astype("int64")).to_numpy()
    post_i = id_to_i.reindex(e["post"].astype("int64")).to_numpy()
    ok = ~(np.isnan(pre_i) | np.isnan(post_i))
    pre_i = pre_i[ok].astype(np.int64)
    post_i = post_i[ok].astype(np.int64)
    cnt = e["count"].to_numpy()[ok].astype(np.float32)
    norm = e["norm"].to_numpy()[ok].astype(np.float32)
    print(f"  成功映射 {len(pre_i):,} 条边")

    if a.min_count > 1:
        keep = cnt >= a.min_count
        pre_i, post_i, cnt, norm = pre_i[keep], post_i[keep], cnt[keep], norm[keep]
        print(f"  min-count>={a.min_count} 后 {len(pre_i):,} 条边")

    # ---- 可选：只保留关键回路周围的子图 ----
    if a.max_nodes:
        import scipy.sparse as sp
        sens_ids = [x for v in keys["sensory"].values() for x in v]
        mot_ids = [x for v in keys["motor"].values() for x in v]
        seed_ids = np.array(sorted(set(sens_ids) | set(mot_ids)), dtype=np.int64)
        seed_i = id_to_i.reindex(seed_ids).dropna().to_numpy(np.int64)
        adj = sp.csr_matrix((np.ones(len(pre_i), np.int8), (pre_i, post_i)), shape=(N, N))
        ab = adj.T.tocsr()
        v = np.zeros(N, dtype=np.int8); v[seed_i] = 1
        keepmask = v.astype(bool)
        for _ in range(a.radius):
            v = (((v @ adj) > 0) | ((v @ ab) > 0)).astype(np.int8)
            keepmask |= v.astype(bool)
        print(f"  BFS r={a.radius} 命中 {keepmask.sum():,} 神经元")
        if keepmask.sum() > a.max_nodes:
            # 只保留关键神经元 + 在其附近的节点
            core = np.zeros(N, dtype=bool); core[seed_i] = True
            print(f"  超过 {a.max_nodes:,}，收窄到关键神经元邻域")
            keepmask = core
            v = core.astype(np.int8)
            for hop in range(a.radius):
                v = (((v @ adj) > 0) | ((v @ ab) > 0)).astype(np.int8)
                keepmask |= v.astype(bool)
                if keepmask.sum() >= a.max_nodes:
                    break
        in_keep = keepmask[pre_i] & keepmask[post_i]
        pre_i, post_i, cnt, norm = (pre_i[in_keep], post_i[in_keep],
                                    cnt[in_keep], norm[in_keep])
        # 重编号
        newi = np.full(N, -1, dtype=np.int64)
        newi[keepmask] = np.arange(keepmask.sum())
        pre_i, post_i = newi[pre_i], newi[post_i]
        ids = ids[keepmask]
        N = int(keepmask.sum())
        print(f"  子图：{N:,} 神经元 / {len(pre_i):,} 边")

    # ---- 符号：按突触前神经元的神经递质 ----
    # nt_index.npz 的 node_ids 与 ids 是同一套排序，先建映射
    nt_ids = nt["node_ids"].astype(np.int64)
    nt_sign = nt["nt_sign"].astype(np.int8)          # (n_nt,)
    nt_index = nt["nt_index"].astype(np.int16)       # (N_total,)
    id_to_ntpos = pd.Series(np.arange(len(nt_ids)), index=nt_ids)
    pos = id_to_ntpos.reindex(ids).to_numpy()
    if np.isnan(pos).any():
        print(f"  ! {int(np.isnan(pos).sum())} 个神经元在递质表里找不到，按未知处理")
        pre_sign_all = np.zeros(N, dtype=np.int8)
        valid = ~np.isnan(pos)
        pre_sign_all[valid] = nt_sign[nt_index[pos[valid].astype(np.int64)]]
    else:
        pre_sign_all = nt_sign[nt_index[pos.astype(np.int64)]]
    edge_sign = pre_sign_all[pre_i]                   # (E,) +1/-1/0

    # ---- 权重量化 + 符号 -> unsigned byte ----
    # ⚠️ 教训（踩过的坑）：不能把 pinme.dev 的公式直接照搬。
    #    pinme.dev 的输入本身就是「已归一化到 0..1 的 log 权重」，
    #    而我们的 norm 是「该突触占目标总输入的占比」，典型值只有 0.01~0.1。
    #    直接套 127*exp(ln_min*(1-w)) 会得到 <1 的值 -> 全部四舍五入到同一档
    #    -> 整张图退化成 binary 图（实测 99.9999% 的边变成 code=1）。
    #
    # 正解：先在该边集内做**log 域线性归一化到 [0,1]**，再量化到 1..127。
    #    这样既保留弱突触的相对差异（log 压缩重尾），又用满整个动态范围。
    w_raw = np.maximum(norm.astype(np.float64), 1e-9)
    lw = np.log(w_raw)
    lo, hi = float(lw.min()), float(lw.max())
    if hi - lo < 1e-12:
        wn = np.ones_like(lw)
    else:
        wn = (lw - lo) / (hi - lo)                     # 0..1
    mag = np.round(1.0 + 126.0 * wn).astype(np.int64)  # 1..127，保留动态范围
    mag = np.clip(mag, 1, 127)
    mag[edge_sign == 0] = 0                            # 未知符号 -> 权重 0
    code = np.where(edge_sign < 0, mag + 128, mag).astype(np.uint8)
    print(f"  权重量化：log 域 [{lo:.3f}, {hi:.3f}] -> code 1..127"
          f"  （唯一 code 数 {len(np.unique(code))}）")

    # 供仿真内核使用的真实权重（未量化），避免量化误差影响动力学标定
    # 实际仿真用 code 解码即可；这里额外存一份精确权重供对照诊断
    w_decode = np.exp(lo + (mag - 1) / 126.0 * (hi - lo))

    n_exc = int((edge_sign > 0).sum())
    n_inh = int((edge_sign < 0).sum())
    n_unk = int((edge_sign == 0).sum())
    print(f"  突触符号：兴奋 {n_exc:,} ({100*n_exc/len(code):.1f}%)  "
          f"抑制 {n_inh:,} ({100*n_inh/len(code):.1f}%)  未知 {n_unk:,}")

    # ---- 构建 CSR（按 pre 排序，行 = 源神经元）----
    order = np.argsort(pre_i, kind="stable")
    pre_s, post_s, code_s = pre_i[order], post_i[order], code[order]
    # indptr: 长度 N+1，indptr[i]..indptr[i+1] 是神经元 i 的出边
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(indptr, pre_s + 1, 1)
    np.cumsum(indptr, out=indptr)

    out_deg = np.diff(indptr)
    print(f"  CSR: nnz={len(post_s):,}  出度 mean={out_deg.mean():.1f} max={out_deg.max():,}")
    print(f"  孤立神经元（出度 0）: {int((out_deg == 0).sum()):,}")

    # ---- 关键神经元在子图里的新编号 ----
    key_out = {"sensory": {}, "motor": {}}
    newi_all = pd.Series(np.arange(N), index=ids)
    for group in ("sensory", "motor"):
        for name, lst in keys[group].items():
            if not lst:
                continue
            hit = newi_all.reindex(pd.Index(lst, dtype=np.int64)).dropna().to_numpy(np.int64)
            if len(hit):
                key_out[group][name] = sorted(int(x) for x in hit)

    # ---- 保存 ----
    npz = DATA / f"{a.out}.npz"
    np.savez_compressed(
        npz,
        node_ids=ids.astype(np.int64),
        indptr=indptr.astype(np.int64),
        indices=post_s.astype(np.int32),
        codes=code_s.astype(np.uint8),
    )
    meta_out = {
        "source": "FlyWire FAFB v783",
        "n_neurons": int(N),
        "n_synapses": int(len(post_s)),
        "n_excitatory": n_exc,
        "n_inhibitory": n_inh,
        "n_unknown_sign": n_unk,
        "out_degree_mean": float(out_deg.mean()),
        "out_degree_max": int(out_deg.max()),
        "isolated": int((out_deg == 0).sum()),
        "ln_min": a.ln_min,
        "weight_log_range": [lo, hi],   # code 解码用：w = exp(lo + (mag-1)/126*(hi-lo))
        "min_count": a.min_count,
        "radius": a.radius if a.max_nodes else None,
        "key_neurons": key_out,
        "dynamics": {"dt": 0.02, "tau": 0.1, "gain": 3.0, "tonic": 0.14,
                     "noise_hz": 1.2, "noise_amp": 0.22, "threshold": 1.0},
    }
    (DATA / f"{a.out}.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 网页端用的紧凑二进制（对齐 pinme.dev 的 FLYW 布局）
    binp = DATA / f"{a.out}.bin"
    with binp.open("wb") as f:
        f.write(b"FLY8")
        f.write(np.array([N, len(post_s)], dtype=np.int64).tobytes())
        f.write(np.array([a.ln_min], dtype=np.float64).tobytes())
        f.write(indptr.astype(np.uint32).tobytes())      # (N+1) uint32
        f.write(post_s.astype(np.uint32).tobytes())      # nnz uint32
        f.write(code_s.astype(np.uint8).tobytes())       # nnz uint8

    print(f"\n✓ {npz}  ({npz.stat().st_size/1e6:.1f} MB)")
    print(f"✓ {binp}  ({binp.stat().st_size/1e6:.1f} MB)")
    print(f"✓ {DATA/f'{a.out}.json'}")
    print(f"  关键感觉 {sum(len(v) for v in key_out['sensory'].values())}  "
          f"关键运动 {sum(len(v) for v in key_out['motor'].values())}")
    print(f"总用时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
