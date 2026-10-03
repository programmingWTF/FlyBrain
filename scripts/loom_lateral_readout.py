#!/usr/bin/env python
"""转向读出的前置验证：DN 的左右不对称里到底有没有可用的方向信号。

Flappy 现在只能"拍/不拍"（DNp01 二值），所以要加分就得把输出变成**连续的方向**。
候选是 DNp02 / DNp11 —— Dombrovski(Nature 2023) 说逃逸方向由 LC4 前部→DNp02、
后部→DNp11 的**反平行突触梯度**编码。那前提是：驱动某一侧/某一段 LC4 时，
这些 DN 的左右（或前后）响应确实不对称。

这里就量这个：分别只驱动
  · 左侧 LC4 / 右侧 LC4
  · 视野上半 LC4 / 下半 LC4（按解剖视野轴，数据自选）
  · 左侧 LPLC2 / 右侧 LPLC2
然后看每个 DN 的**左/右发放率之比**，并和 shuffled 对照比 ——
如果打乱拓扑后同样不对称，那这个不对称就不能用作方向信号。

用法
    D:/Code/FlyBrain/env/python.exe scripts/loom_lateral_readout.py --asset spiking_full
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv import looming                      # noqa: E402
from fpv.spiking_brain import SpikingBrain   # noqa: E402

GAIN, TONIC = 3.0, 0.0
DN = ["DNp01", "DNp02", "DNp03", "DNp04", "DNp05", "DNp11"]
RATE, TICKS, WARM = 1.0, 400, 200


def field_axis(brain, coords, group, side_of_group):
    """该群的**视野坐标**轴与归一化 u。

    ⚠️ 第一版用"全体包围盒中心方差最大的轴"，结果选中了 x 轴——而 x 完美区分
    左右半球（分侧 AUC=0/1），所以那是**半球轴**，不是视野轴。
    现在改成：先剔掉半球轴，再在**同侧内部**取方差最大的轴（LC4/LPLC2/LPLC1
    同侧内 y 轴 std≈0.09 最大，x 只有 0.02~0.04）。u 也在每个半球内部各自归一化，
    这样 hi/lo 才是"每只眼睛视野的上/下"，而不是"左脑/右脑"。
    """
    idx = group.detach().cpu().numpy()
    pts = coords[idx]
    sd = np.asarray(side_of_group)
    hemi = np.argmax([pts[:, a].std() for a in range(3)])     # 半球轴
    cand = [a for a in range(3) if a != hemi]
    per_side = np.zeros(len(idx))
    axis = max(cand, key=lambda a: np.mean([pts[sd == s, a].std()
                                            for s in ("left", "right")
                                            if (sd == s).sum() > 3]))
    for s in ("left", "right"):
        m = sd == s
        if m.sum() < 2:
            continue
        v = pts[m, axis]
        per_side[m] = (v - v.min()) / max(v.max() - v.min(), 1e-9)
    return {"hemi_axis": "xyz"[int(hemi)], "field_axis": "xyz"[axis]}, per_side


def build_coords(brain):
    mf = json.loads((REPO / "data" / "brain" / "manifest.json").read_text(encoding="utf-8"))
    xs = mf["bbox"]
    ctr = np.array([(xs[0] + xs[3]) / 2, (xs[1] + xs[4]) / 2, (xs[2] + xs[5]) / 2])
    half = np.array([(xs[3] - xs[0]) / 2, (xs[4] - xs[1]) / 2, (xs[5] - xs[2]) / 2])
    row_of = {str(n["id"]): i for i, n in enumerate(mf["neurons"])}
    secs = mf["coarse"]["sections"]
    id_of = np.asarray(brain.node_ids, np.int64)
    out = np.zeros((brain.N, 3), dtype=np.float32)
    for k, nid in enumerate(id_of):
        j = row_of.get(str(int(nid)))
        if j is not None and j < len(secs) and secs[j].get("bbox"):
            b = secs[j]["bbox"]
            out[k] = (np.array([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2,
                                (b[2] + b[5]) / 2]) - ctr) / half
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spiking_full")
    a = ap.parse_args()
    brain = SpikingBrain.from_npz(a.asset, device="cpu")
    brain.gain, brain.tonic = GAIN, TONIC
    g = looming.resolve(brain, ["LC4", "LPLC2"] + DN)
    sd = looming.resolve_side(brain, ["LC4", "LPLC2"] + DN)
    coords = build_coords(brain)

    # 视野上下半（按**同侧内**最大方差轴，见 field_axis 的说明）
    meta = pd.read_feather(ROOT / "data" / "fafb_783_meta.feather")
    meta["id"] = meta["fafb_783_id"].astype("int64")
    meta["side"] = meta["side"].fillna("?")
    side_of = pd.Series(meta["side"].to_numpy(), index=meta["id"].to_numpy())
    ids = np.asarray(brain.node_ids, np.int64)
    halves = {}
    for nm in ("LC4", "LPLC2"):
        info, u = field_axis(brain, coords, g[nm],
                             side_of.reindex(pd.Index(ids[g[nm].cpu().numpy()]))
                             .fillna("?").to_numpy())
        idx = g[nm].detach().cpu().numpy()
        halves[f"{nm}_hi"] = torch.as_tensor(idx[u >= 0.5], device=brain.device)
        halves[f"{nm}_lo"] = torch.as_tensor(idx[u < 0.5], device=brain.device)
        print(f"{nm}  {info}  上/下半 = "
              f"{len(halves[nm + '_hi'])}/{len(halves[nm + '_lo'])}")

    dns_lr = {}
    for nm in DN:
        for side in ("left", "right"):
            if side in sd[nm]:
                dns_lr[f"{nm}_{side}"] = sd[nm][side]
    print("读出：" + "  ".join(dns_lr))

    conds = {}
    for nm in ("LC4", "LPLC2"):
        if "left" in sd[nm]:
            conds[f"{nm}_L"] = sd[nm]["left"]
            conds[f"{nm}_R"] = sd[nm]["right"]
        conds[f"{nm}_hi"] = halves[f"{nm}_hi"]
        conds[f"{nm}_lo"] = halves[f"{nm}_lo"]

    def measure(br, idx):
        pr = torch.full((idx.numel(),), RATE, device=br.device)
        br.reset()
        base = {k: 0 for k in dns_lr}
        dur = {k: 0 for k in dns_lr}
        for i in range(WARM + TICKS):
            br.step(clamp=(idx, pr))
            t = base if i < WARM else dur
            for k, v in dns_lr.items():
                t[k] += int(br.S[v].sum())
        return {k: dur[k] / (dns_lr[k].numel() * TICKS) for k in dns_lr}, \
               {k: base[k] / (dns_lr[k].numel() * WARM) for k in dns_lr}

    shuf = looming.variant(brain, shuffle_seed=11)
    print(f"\n驱动 rate={RATE}  {TICKS} tick。表：左/右发放率与不对称指数 ASI=(L-R)/(L+R)")
    print(f"{'条件':<12}{'DN':<8}{'左':>9}{'右':>9}{'ASI':>8}{'ASI乱':>9}{'可用?':>7}")
    rows = []
    for cname, idx in conds.items():
        d, _b = measure(brain, idx)
        ds, _ = measure(shuf, idx)
        for nm in DN:
            lk, rk = f"{nm}_left", f"{nm}_right"
            if lk not in d or rk not in d:
                continue
            l, r = d[lk], d[rk]
            sl, sr = ds[lk], ds[rk]
            asi = (l - r) / (l + r) if (l + r) > 1e-6 else 0.0
            asis = (sl - sr) / (sl + sr) if (sl + sr) > 1e-6 else 0.0
            usable = (l + r) > 5e-3 and abs(asi) > 0.25 and abs(asi) > 2 * abs(asis)
            print(f"{cname:<12}{nm:<8}{l:9.4f}{r:9.4f}{asi:8.2f}{asis:9.2f}"
                  f"{'  ✓' if usable else '   -':>7}")
            rows.append(dict(cond=cname, dn=nm, left=l, right=r, asi=asi,
                             asi_shuffled=asis, usable=usable))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / "loom_lateral_readout.csv", index=False)
    n_ok = int(df.usable.sum())
    print(f"\n可用于转向的（条件 × DN）组合：{n_ok} / {len(df)}")
    if n_ok:
        print("最佳几个（|ASI| 最大）：")
        print(df.reindex(df.asi.abs().sort_values(ascending=False).index)
              .head(8)[["cond", "dn", "left", "right", "asi", "asi_shuffled"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("→ 左右不对称里没有可用方向信号；转向得另找读出（试 DN 群体编码或前后轴）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
