#!/usr/bin/env python
"""在 GPU 上对比几种「稀疏图消息传递」实现，选出最快的（同时校验梯度正确）。

候选
  A. gather  : h[:, src] * w -> index_add_(dst)          （最直白，带宽杀手）
  B. sparse  : torch.sparse.mm(A_coo, h.t()).t()          （小矩阵开销大）
  C. dense   : h @ A_dense                                （356k/16M = 2.2% 密度，但 cuBLAS 极快）
  D. gather_dense : Ad[:, src] * w -> index_add_          （只 gather 需要的列）

对每个候选都做一次数值梯度校验（和 torch.autograd 的有限差分比），
避免出现「很快但是错的」。

用法：python scripts/bench_propagation.py
"""
from __future__ import annotations

import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fpv import load_subgraph  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timeit(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


# ---------------------------------------------------------------- 候选实现
def prop_gather(h, src, dst, w, K, N):
    for _ in range(K):
        msg = h[:, src] * w.view(1, -1)
        out = torch.zeros_like(h)
        out.index_add_(1, dst, msg)
        h = torch.relu(out)
    return h


def prop_sparse(h, idx, w, K, N):
    for _ in range(K):
        A = torch.sparse_coo_tensor(idx, w, (N, N)).coalesce()
        h = torch.relu(torch.sparse.mm(A, h.t()).t())
    return h


def prop_dense(h, Ad, K, N):
    for _ in range(K):
        h = torch.relu(h @ Ad)
    return h


def prop_gather_dense(h, Ad, src, dst, w, K, N):
    """从稠密矩阵里按边 gather 列 -> 只对存在的边做乘加。"""
    for _ in range(K):
        msg = Ad[:, src] * w.view(1, -1)
        out = torch.zeros_like(h)
        out.index_add_(1, dst, msg)
        h = torch.relu(out)
    return h


def main() -> int:
    print(f"device = {DEV}")
    g = load_subgraph("sg_collision_s300_n4000")
    print(f"图：{g}  密度 {g.n_edges / g.n_nodes**2 * 100:.1f}%")

    N, E = g.n_nodes, g.n_edges
    src = torch.as_tensor(g.src, device=DEV)
    dst = torch.as_tensor(g.dst, device=DEV)
    idx = torch.stack([dst, src], 0)
    # 用稀疏初始化（~2% 非零），贴近 ReLU 后的真实激活分布
    w0 = torch.as_tensor(g.weight, device=DEV)

    for B, K in ((128, 3), (256, 3), (128, 5)):
        print(f"\n--- batch={B} K={K} ---")
        h0 = (torch.rand(B, N, device=DEV) < 0.02).float() * torch.randn(B, N, device=DEV)
        h0.requires_grad_(False)

        # A. gather
        wA = w0.clone().requires_grad_(True)

        def fA():
            return prop_gather(h0, src, dst, wA, K, N).sum()

        # B. sparse
        wB = w0.clone().requires_grad_(True)

        def fB():
            return prop_sparse(h0, idx, wB, K, N).sum()

        # C. dense（稠密矩阵里放同一个 w）
        Ad = torch.zeros(N, N, device=DEV)
        with torch.no_grad():
            Ad[dst, src] = w0
        Ad.requires_grad_(True)

        def fC():
            return prop_dense(h0, Ad, K, N).sum()

        # D. gather from dense
        wD = w0.clone().requires_grad_(True)

        def fD():
            return prop_gather_dense(h0, Ad.detach(), src, dst, wD, K, N).sum()

        cands = [("A gather      ", fA, wA), ("B sparse.mm   ", fB, wB),
                 ("C dense A@x   ", fC, Ad), ("D gather_dense", fD, wD)]

        for name, fn, param in cands:
            try:
                def step():
                    if param.grad is not None:
                        param.grad = None
                    fn().backward()

                fwd = timeit(lambda: fn(), n=10, warmup=3)
                tr = timeit(step, n=5, warmup=2)
                print(f"  {name}  前向 {fwd*1e3:8.2f} ms   前向+反向 {tr*1e3:8.2f} ms")
            except Exception as ex:
                print(f"  {name}  失败: {type(ex).__name__}: {ex}")

    print("\n=== 梯度校验（小图上和有限差分比）===")
    N2, E2, B2, K2 = 60, 300, 4, 2
    torch.manual_seed(0)
    s2 = torch.randint(0, N2, (E2,), device=DEV)
    d2 = torch.randint(0, N2, (E2,), device=DEV)
    w2 = torch.rand(E2, device=DEV) * 0.5
    h2 = torch.rand(B2, N2, device=DEV)

    def loss_gather(w):
        return prop_gather(h2, s2, d2, w, K2, N2).sum()

    wg = w2.clone().requires_grad_(True)
    loss_gather(wg).backward()
    g_auto = wg.grad.clone()

    idx2 = torch.stack([d2, s2], 0)
    ws = w2.clone().requires_grad_(True)
    prop_sparse(h2, idx2, ws, K2, N2).sum().backward()
    print(f"  gather vs sparse 最大差 {np_abs(g_auto, ws.grad):.2e}")

    # 有限差分抽查 5 条边
    eps = 1e-3
    max_rel = 0.0
    for k in range(5):
        i = int(torch.randint(0, E2, (1,)).item())
        wp = w2.clone(); wp[i] += eps
        wm = w2.clone(); wm[i] -= eps
        fd = (loss_gather(wp).item() - loss_gather(wm).item()) / (2 * eps)
        rel = abs(fd - g_auto[i].item()) / max(1e-6, abs(fd))
        max_rel = max(max_rel, rel)
    print(f"  有限差分最大相对误差 {max_rel:.2e}  {'✓' if max_rel < 1e-2 else '✗'}")
    return 0


def np_abs(a, b):
    return float((a - b).abs().max().item())


if __name__ == "__main__":
    sys.exit(main())
