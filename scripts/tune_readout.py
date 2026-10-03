#!/usr/bin/env python
"""阶段 2.3：标定「感觉编码 -> 下行神经元群体读出」的工作点。

为什么不用单个 DNp01
--------------------
DNp01 在 FlyWire 里只有 2 个神经元（左右各一），发放率是 0/1/2 的整数，
分辨率太低、太脆弱。pinme.dev 用的是**命名下行神经元群**（TARGET_DN、
LOOM_DN、ESCAPE_DN 等），取群体的发放比例作为读出特征。

本脚本：
1. 按 pinme.dev 的分组思路，把 16 个命名 DN 类型归成几"组"（目标追踪 / 逃逸 / 其他）
2. 扫描注入强度，看每组群体发放率随刺激的变化 —— 找到编码窗口
3. 检查不同感觉通道（LC4 / LC10 / T4）是否引发**不同**的下行响应模式
   （如果全都一样，说明网络把输入混在一起了，读不出区分性）
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv.spiking_brain import SpikingBrain  # noqa: E402

# pinme.dev 的分组思路（LOOM / TARGET / ESCAPE / 其他）
READOUT_GROUPS = {
    "ESCAPE": ["DNp01", "DNp04", "DNg40"],
    "TARGET": ["DNae002", "DNae001", "DNg111", "DNge109", "DNb01"],
    "LOOM": ["DNa07", "DNp04", "DNg40", "DNp06"],
    "OTHER": ["DNp07", "DNp09", "DNp11", "DNa02", "DNg13", "DNge103", "DNp54"],
}


def group_indices(brain, names) -> torch.Tensor:
    idx = []
    for n in names:
        try:
            idx.append(brain.key_idx(n))
        except KeyError:
            pass
    return torch.cat(idx) if idx else torch.empty(0, dtype=torch.long)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spiking_circuit")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gain", type=float, default=6.0)
    ap.add_argument("--ticks", type=int, default=200)
    a = ap.parse_args()
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    brain = SpikingBrain.from_npz(a.asset, device=dev)
    brain.gain = a.gain
    print(f"asset={a.asset}  N={brain.N:,}  gain={a.gain}\n")

    groups = {g: group_indices(brain, ns) for g, ns in READOUT_GROUPS.items()}
    for g, ix in groups.items():
        print(f"  {g:<8} {ix.numel():>3} 神经元")
    print()

    def measure(stim_name, strength, ticks):
        """返回各组在 ticks 内的平均群体发放率（spike/neuron/tick）。"""
        brain.reset()
        for _ in range(20):
            brain.step()
        stim = torch.zeros(brain.N, device=brain.device)
        if stim_name:
            stim[brain.key_idx(stim_name)] = strength
        acc = {g: 0.0 for g in groups}
        for _ in range(ticks):
            s = brain.step(stim if stim_name else None)
            for g, ix in groups.items():
                acc[g] += float(s[ix].float().mean())
        return {g: v / ticks for g, v in acc.items()}

    print("=" * 88)
    print("注入强度扫描：各下行神经元群的群体发放率")
    print("=" * 88)
    print(f"{'刺激':<10}{'强度':>6} | " + " ".join(f"{g:>10}" for g in groups))
    print("-" * 88)
    r = measure(None, 0, a.ticks)
    print(f"{'(静息)':<10}{0:>6.1f} | " + " ".join(f"{r[g]:>10.4f}" for g in groups))

    for ch, strengths in [("LC4", [0.5, 1.0, 2.0, 4.0]), ("LC10", [1.0, 2.0, 4.0]),
                          ("T4", [1.0, 2.0, 4.0])]:
        for s in strengths:
            r = measure(ch, s, a.ticks)
            print(f"{ch:<10}{s:>6.1f} | " + " ".join(f"{r[g]:>10.4f}" for g in groups))
        print()

    print("=" * 88)
    print("通道区分性检验：不同感觉输入 -> 下行群体的响应模式是否不同")
    print("=" * 88)
    patterns = {}
    for ch in ["LC4", "LPLC2", "LC10", "T4", "T5"]:
        r = measure(ch, 2.0, a.ticks)
        patterns[ch] = np.array([r[g] for g in groups])
        print(f"  {ch:<8} " + " ".join(f"{g}={r[g]:.4f}" for g in groups))
    ks = list(patterns)
    print("\n  模式两两余弦相似度（越接近 1 越说明网络读不出区分）：")
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            u, v = patterns[ks[i]], patterns[ks[j]]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            cos = float(u @ v / (nu * nv + 1e-12)) if nu > 0 and nv > 0 else float("nan")
            print(f"    {ks[i]:<7} vs {ks[j]:<7} cos={cos:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
