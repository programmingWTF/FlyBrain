#!/usr/bin/env python
"""量化"真实视野"对照：威胁覆盖多少 LC4 才会触发 DNp01 逃逸指令。

背景（为什么要做这个）
--------------------
前面所有实验都是把 **104 个 LC4 一起**钳制成同一个发放率（scripts/loom_transfer.py）。
但真实视网膜上，一个威胁只落在视野的一小块，只该驱动偏好位置与之重叠的那批 LC4
（LC4 是柱状、retinotopic 排布的）。所以那个 r50≈0.58 的阈值隐含了一个
**不现实的假设：整个 LC4 群同时接近最大发放**。

本脚本把这个假设拆掉，量两件事：
  1. **经验 cliff**：视野覆盖率（高斯窗宽度）要多大，DNp01 才发放；
  2. **加权覆盖率**：按每个 LC4 到 DNp01 的**实际突触权重**加权的覆盖率
     —— 决定阈值的是"被驱动的细胞里有多少条强突触"，不是细胞个数占比。

用法
    D:/Code/FlyBrain/env/python.exe scripts/loom_retino_coverage.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv import looming                      # noqa: E402
from fpv.spiking_brain import SpikingBrain   # noqa: E402

GAIN, TONIC = 3.0, 0.0
TICKS = 40


def gaussian(u, center, width):
    return np.exp(-((u - center) / max(width / 2.355, 1e-6)) ** 2 / 2.0)


def main() -> int:
    brain = SpikingBrain.from_npz("spiking_full", device="cpu")
    brain.gain, brain.tonic = GAIN, TONIC
    g = looming.resolve(brain, ["LC4", "LPLC2", "DNp01"])

    # --- 每个 LC4 到 DNp01 的实际权重（两只细胞各自的）
    lut = brain.lut.detach().cpu().numpy()
    indptr = brain.indptr.detach().cpu().numpy()
    indices = brain.indices.detach().cpu().numpy()
    codes = brain.codes.detach().cpu().numpy()
    lc4 = g["LC4"].detach().cpu().numpy()
    lplc2 = g["LPLC2"].detach().cpu().numpy()
    dn = g["DNp01"].detach().cpu().numpy()

    def w_to_dn(group):
        w = np.zeros(len(group))
        for k, s in enumerate(group):
            lo, hi = int(indptr[s]), int(indptr[s + 1])
            if hi <= lo:
                continue
            tgt = indices[lo:hi]
            sel = np.isin(tgt, dn)
            if sel.any():
                w[k] = float(lut[codes[lo:hi][sel]].sum())
        return w

    w_lc4, w_lplc2 = w_to_dn(lc4), w_to_dn(lplc2)
    print(f"LC4  -> DNp01 总权重 {w_lc4.sum():.4f}  (每只细胞均值 "
          f"{w_lc4.mean():.5f}, 最大 {w_lc4.max():.5f})")
    print(f"LPLC2-> DNp01 总权重 {w_lplc2.sum():.4f}  (均值 {w_lplc2.mean():.5f})")

    # --- 视野轴：用包围盒中心在 LC4 群体上方差最大的解剖轴（与 demo 同法）
    import json
    mf = json.loads((REPO / "data" / "brain" / "manifest.json").read_text(encoding="utf-8"))
    xs = mf["bbox"]
    ctr = np.array([(xs[0] + xs[3]) / 2, (xs[1] + xs[4]) / 2, (xs[2] + xs[5]) / 2])
    half = np.array([(xs[3] - xs[0]) / 2, (xs[4] - xs[1]) / 2, (xs[5] - xs[2]) / 2])
    pos_of = {str(int(i)): k for k, i in enumerate(np.asarray(brain.node_ids, np.int64))}
    secs = mf["coarse"]["sections"]
    id_of = np.asarray(brain.node_ids, np.int64)
    # 一次建索引，别在循环里线性扫 13.9 万条
    row_of = {str(nn["id"]): i for i, nn in enumerate(mf["neurons"])}

    def axis_coord(group):
        pts = np.zeros((len(group), 3), dtype=np.float32)
        for k, a in enumerate(group):
            j = row_of.get(str(int(id_of[a])))
            if j is not None and j < len(secs) and secs[j].get("bbox"):
                b = secs[j]["bbox"]
                pts[k] = (np.array([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2,
                                    (b[2] + b[5]) / 2]) - ctr) / half
        ok = np.abs(pts).sum(1) > 1e-6
        axis = int(np.argmax(pts[ok].std(0)))
        v = pts[:, axis]
        return (v - v[ok].min()) / max(v[ok].max() - v[ok].min(), 1e-9)

    u_lc4, u_lplc2 = axis_coord(lc4), axis_coord(lplc2)

    need = (1 - brain.leak) * brain.threshold / brain.gain
    print(f"\nDNp01 每 tick 需要的净输入 = (1-leak)*thr/gain = {need:.4f}\n")
    print(f"{'width':>7}{'覆盖(计数)':>12}{'覆盖(加权)':>12}"
          f"{'Σw·p 双通道':>14}{'DNp01脉冲/40tick':>18}{'预测':>6}")
    for width in (0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 2.0):
        s4 = gaussian(u_lc4, 0.5, width)
        s2 = gaussian(u_lplc2, 0.5, width)
        cov = float(gaussian(np.linspace(0, 1, 101), 0.5, width).mean())
        wcov = float((w_lc4 * s4).sum() / max(w_lc4.sum(), 1e-9))
        eff = float((w_lc4 * s4).sum() + (w_lplc2 * s2).sum())
        cidx = torch.cat([g["LC4"], g["LPLC2"]])
        pvec = torch.as_tensor(np.concatenate([s4, s2]), dtype=torch.float32,
                               device=brain.device)
        brain.reset()
        spk = 0
        for _ in range(TICKS):
            brain.step(clamp=(cidx, pvec))
            spk += int(brain.S[g["DNp01"]].sum())      # 全程累加，别只读最后一个 tick
        print(f"{width:7.2f}{cov:12.3f}{wcov:12.3f}{eff:14.4f}{spk:11d}"
              f"{'YES' if eff >= need else '   -':>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
