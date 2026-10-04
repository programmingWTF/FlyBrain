#!/usr/bin/env python
"""把"解剖坐标"从 51.5 MB 的 manifest.json 里**预计算**出来，存成一个很小的 .npz。

为什么需要这一步
----------------
`demo/server.py` 启动时要给每个神经元一个解剖坐标，用来
  ① 把右侧 3D 脑图按**真实解剖位置**摆出来（不是螺旋点云）
  ② 认 LC4 的视野轴（`_retino_groups`，供"真实视野"模式用）

原来这两件事都靠启动时读 `data/brain/manifest.json`（**51.5 MB**）。
那个文件是全脑渲染器编译出来的，绝大部分是粗几何的偏移索引 ——
对 demo 来说只需要里面的包围盒中心。

所以这里把它**烤成 1.7 MB 的 float32 数组**，之后服务器再也不需要 manifest。
好处：仓库自包含（不用塞 51.5 MB），启动也更快。

用法
----
    python scripts/bake_coords.py \
        --manifest /path/to/data/brain/manifest.json \
        --asset spiking_full \
        --out data/coords.npz

`--asset` 决定"点名点亮"的那批关键神经元（LC4/LPLC2/DNp01/...）的坐标，
必须与 demo 实际加载的资产一致。

产物 data/coords.npz 里有两项：
    all   float32 (N, 3)   资产里**每个**神经元的归一化坐标（没有几何的填 0）
    lit   float32 (M, 3)   关键神经元（LIT_TYPES）的坐标，顺序与 server.py 的 lit 列表一致
    lit_node_id  int64 (M,)  对应的 fafb id，便于核对
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

#: 与 demo/server.py 的 LIT_TYPES **必须保持一致**。这里不 import server.py，
#: 因为它会连带 import torch（本脚本刻意保持轻量，只要 numpy）。
LIT_TYPES = ["LC4", "LPLC2", "DNp01", "DNp04", "LC10a", "DNp02", "DNp11"]
#: 需要**逐侧**拆分的类型（对齐 demo/server.py 里 resolve_side 的调用）
SIDE_TYPES = ["DNp01", "DNp02", "DNp04", "DNp11"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=pathlib.Path,
                    help="全脑 manifest.json（来自全脑渲染项目，约 51.5 MB）")
    ap.add_argument("--asset", default="spiking_full",
                    help="资产名（读 data/<asset>.npz 拿 node_ids 与关键群）")
    ap.add_argument("--data-dir", type=pathlib.Path, default=None,
                    help="资产目录，默认 <仓库根>/data")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="输出路径，默认 <仓库根>/data/coords.npz")
    a = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent      # 仓库根
    data = a.data_dir or (root / "data")
    out = a.out or (data / "coords.npz")

    if not a.manifest.is_file():
        print(f"找不到 manifest：{a.manifest}", file=sys.stderr)
        return 1
    npz = data / f"{a.asset}.npz"
    if not npz.is_file():
        print(f"找不到资产：{npz}", file=sys.stderr)
        return 1

    print(f"读 manifest …（{a.manifest.stat().st_size / 1e6:.1f} MB，这一步慢是正常的）")
    mf = json.loads(a.manifest.read_text(encoding="utf-8"))
    xs = mf["bbox"]
    ctr = np.array([(xs[0] + xs[3]) / 2, (xs[1] + xs[4]) / 2, (xs[2] + xs[5]) / 2])
    half = np.array([(xs[3] - xs[0]) / 2, (xs[4] - xs[1]) / 2, (xs[5] - xs[2]) / 2])
    secs = mf["coarse"]["sections"]

    with np.load(npz) as d:
        node_ids = np.asarray(d["node_ids"], dtype=np.int64)
    #: 资产里没有 key_* 群，所以关键群要从细胞类型表推（与运行期
    #  `fpv.looming.resolve` 用的是同一份表、同一个判据）。
    meta_f = data / "fafb_783_meta.feather"
    key: dict[str, np.ndarray] = {}
    m = None
    pos_of = {int(v): k for k, v in enumerate(node_ids)}
    if meta_f.is_file():
        import pandas as pd
        m = pd.read_feather(meta_f)
        m = m.assign(id=m["fafb_783_id"].astype("int64"),
                     ct=m["cell_type"].fillna(""))
        for nm in LIT_TYPES:
            ids = m.loc[m["ct"] == nm, "id"].to_numpy(np.int64)
            idx = [pos_of[int(i)] for i in ids if int(i) in pos_of]
            if idx:
                key[nm] = np.asarray(sorted(idx), dtype=np.int64)
    else:
        print(f"  [警告] 没有 {meta_f.name}，无法确定关键群")

    N = len(node_ids)
    print(f"资产 {a.asset}: {N:,} 个神经元；manifest 里 {len(mf['neurons']):,} 个")

    # fafb id -> manifest 里的粗几何包围盒中心
    coord_of: dict[str, list[float]] = {}
    for i, n in enumerate(mf["neurons"]):
        if i >= len(secs):
            break
        b = secs[i].get("bbox")
        if not b:
            continue
        c = np.array([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2])
        coord_of[str(n["id"])] = ((c - ctr) / half).tolist()
    print(f"  manifest 里带几何的：{len(coord_of):,}")

    arr = np.zeros((N, 3), dtype=np.float32)
    hit = 0
    for j, nid in enumerate(node_ids):
        v = coord_of.get(str(int(nid)))
        if v is not None:
            arr[j] = v
            hit += 1
    print(f"  资产神经元命中坐标：{hit:,}/{N:,}（{hit / N:.1%}）")

    # 关键神经元：**顺序必须与 server.py 的 _build_lit 一致**
    #（它按 LIT_TYPES 的顺序 append，前端按同一顺序点亮）
    if key:
        names = sorted(key)
        print(f"  资产自带关键群：{names}")
        want: dict[str, dict] = {}
        for nm in LIT_TYPES:
            idx = key.get(nm)
            if idx is None:
                print(f"    [警告] 资产里没有 {nm}")
                continue
            for i in idx:
                want[str(int(node_ids[int(i)]))] = {"asset": int(i), "type": nm}
        lit_ids, lit_xyz, lit_assets = [], [], []
        for nid, d in want.items():
            v = coord_of.get(nid)
            if v is None:
                continue                       # 与 server.py 一致：没坐标的不点亮
            lit_ids.append(int(nid)); lit_xyz.append(v); lit_assets.append(d["asset"])
        lit = np.asarray(lit_xyz, dtype=np.float32).reshape(-1, 3)
        print(f"  关键神经元可点亮：{len(lit_ids):,}")
    else:
        print("  [警告] 资产里没有 key_* 群，lit 置空")
        lit = np.zeros((0, 3), dtype=np.float32)
        lit_ids = []
        lit_assets = []

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        all=arr,
        lit=lit,
        lit_node_id=np.asarray(lit_ids, dtype=np.int64),
        lit_asset=np.asarray(lit_assets, dtype=np.int64),
    )
    # 关键群的**资产索引**也一起存（服务器优先用它，这样连 cell_type 都不需要）
    for nm, idx in key.items():
        payload[f"key_{nm}"] = np.asarray(idx, dtype=np.int64)
    # 逐侧索引也存（`fpv.looming.resolve_side` 优先读它，于是 1.4 MB 的 meta 也不必入库）
    if meta_f.is_file():
        sub = m.loc[m["ct"].isin(SIDE_TYPES), ["ct", "id", "side"]]
        n_side = 0
        for (nm, sd), grp in sub.groupby(["ct", "side"], sort=True):
            idx = [pos_of[int(i)] for i in grp["id"].to_numpy(np.int64) if int(i) in pos_of]
            if idx:
                payload[f"side_{nm}_{sd}"] = np.asarray(sorted(idx), dtype=np.int64)
                n_side += 1
        print(f"  逐侧索引：{n_side} 组")
    np.savez_compressed(out, **payload)
    print(f"\n写出 {out}（{out.stat().st_size / 1e6:.2f} MB）")
    print(f"  含：all {arr.shape}、lit {lit.shape}、关键群 {sorted(key)}")
    print("现在服务器会优先用它，不再需要 manifest.json。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
