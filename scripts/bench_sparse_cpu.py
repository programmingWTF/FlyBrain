#!/usr/bin/env python
"""稀疏「活跃边」传播的 CPU 基准 —— 验证它能不替代 GPU。

动机
----
dense `h @ A` 每次传播都要读 4000² 的 64 MB 矩阵，**带宽受限**：
  CPU 每个 episode 步骤 ~338 ms（前向反向），因为要搬 64MB × batch。
但 connectome 是稀疏的，而且 ReLU 之后 activations 只有约 2% 非零。
如果只在「src 激活非零」的边上计算，就能跳过 98% 的乘法。

做法
----
每个传播步先用 src 上激活的绝对值筛出活跃边，然后
    msg = h[:, a_src] * a_w        # 只 gather 活跃边
    out.index_add_(1, a_dst, msg)
这样 FLOPs 与活跃边数成正比，而不是与 N² 成正比。

和 dense 版本做**数值一致性校验**（同一组权重、同一个输入，前向结果必须一致）。

用法：python scripts/bench_sparse_cpu.py
"""
from __future__ import annotations

import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fpv import load_subgraph  # noqa: E402


def timed(fn, n=5, warmup=2, dev=None):
    for _ in range(warmup):
        fn()
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def fwd_dense(h, A, K):
    for _ in range(K):
        h = torch.relu(h @ A)
    return h


def fwd_sparse(h, src, dst, w, K):
    """只在活跃边上算。返回 (h, 平均活跃边占比)。"""
    ratios = []
    for _ in range(K):
        # 活跃 = src 上激活非零的边
        active = h[:, src] != 0                       # (B, E) bool
        frac = float(active.float().mean())
        ratios.append(frac)
        # 用 (B,E) 的掩码逐元素乘，和 index_add 组合（保持简单、可微）
        msg = h[:, src] * w.view(1, -1) * active.to(h.dtype)
        out = torch.zeros_like(h)
        out.index_add_(1, dst, msg)
        h = torch.relu(out)
    return h, (sum(ratios) / len(ratios) if ratios else 0.0)


def main() -> int:
    torch.set_num_threads(16)
    g = load_subgraph("sg_collision_s300_n4000")
    N, E = g.n_nodes, g.n_edges
    dev = torch.device("cpu")
    print(f"device={dev}  threads={torch.get_num_threads()}  图={g}")

    src = torch.as_tensor(g.src, device=dev)
    dst = torch.as_tensor(g.dst, device=dev)
    w = torch.as_tensor(g.weight, device=dev)
    A = torch.zeros(N, N, device=dev)
    A[dst, src] = w
    print(f"  稠密 A 占用 {A.numel()*4/1e6:.0f} MB，非零 {E/A.numel()*100:.1f}%\n")

    for B in (128, 64):
        for K in (3,):
            # 用稀疏初始化贴近真实激活分布（ReLU 输出 + 感觉注入）
            inj = torch.zeros(B, N, device=dev)
            sens = torch.as_tensor(g.sensory_idx, device=dev)
            inj[:, sens] = torch.randn(B, len(sens), device=dev)
            h0 = inj.clone()

            print(f"--- batch={B} K={K} ---")
            # 先校验数值一致
            hd = fwd_dense(h0.clone(), A, K)
            hs, frac = fwd_sparse(h0.clone(), src, dst, w, K)
            # dense 不含感觉注入逻辑，两者都在无注入下比
            d = float((hd - hs).abs().max())
            print(f"  dense vs sparse 前向最大差 = {d:.2e}  {'✓ 一致' if d < 1e-4 else '✗ 不一致'}")
            print(f"  平均活跃边占比 = {frac*100:.1f}%  (即跳过了 {100-frac*100:.1f}% 的计算)")

            t_d = timed(lambda: fwd_dense(h0.clone(), A, K), n=3, warmup=1, dev=dev)
            t_s = timed(lambda: fwd_sparse(h0.clone(), src, dst, w, K), n=3, warmup=1, dev=dev)
            print(f"  前向：dense {t_d*1e3:8.1f} ms   sparse {t_s*1e3:8.1f} ms   "
                  f"加速 {t_d/t_s:.2f}×")
            print()

    # 换算：300k 步在 CPU 上的预估
    print("=" * 70)
    print("结论预览：如果 sparse 版前向能到 X ms，CPU 训练 30 万步约需")
    for ms in (5, 10, 20, 40):
        # 每步 = 1 次前向(act) + 1 次前向反向(≈3× 前向)
        per = (ms + 3 * ms) / 1000
        print(f"   前向 {ms:3d} ms  ->  {per*300_000/3600:5.2f} 小时")
    return 0


if __name__ == "__main__":
    sys.exit(main())
