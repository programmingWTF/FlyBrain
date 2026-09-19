#!/usr/bin/env python
"""前向/反向耗时基准 —— 决定这套图网络能不能在合理时间内训完。

对每个 (arch, batch, steps) 组合测：纯前向、以及 forward+backward。
并给出「训练 30 万步需要多久」的外推。

用法：python scripts/bench_forward.py
"""
from __future__ import annotations

import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv import graph_config, load_subgraph  # noqa: E402
from fpv.vendor_flappyrl.networks import ConnectomeNet, RainbowNet  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timeit(fn, n=5, warmup=2):
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


def main() -> int:
    print(f"device = {DEV}")
    g = load_subgraph("sg_collision_s300_n4000")
    print(f"图：{g}")

    rows = []
    for steps in (1, 2, 3, 5):
        cfg = graph_config(net_arch="connectome", graph_steps=steps)
        net = ConnectomeNet(
            cfg, g.src, g.dst, g.weight, g.n_nodes, g.sensory_idx, g.motor_idx
        ).to(DEV)
        for bs in (32, 128, 256):
            x = torch.randn(bs, 18, device=DEV)
            net.eval()
            with torch.no_grad():
                fwd = timeit(lambda: net(x), n=5, warmup=2)
            net.train()
            opt = torch.optim.Adam(net.parameters(), lr=1e-4)

            def step():
                opt.zero_grad(set_to_none=True)
                out = net(x)
                out.sum().backward()
                opt.step()

            tr = timeit(step, n=5, warmup=2)
            # 训练循环里每个 env-step 约有 1 次前向(act) + 1 次 forward/backward
            per_env_step = fwd + tr
            eta_h = per_env_step * 300_000 / 3600
            rows.append((steps, bs, fwd * 1e3, tr * 1e3, per_env_step * 1e3, eta_h))
            print(f"  K={steps} batch={bs:4d}  前向 {fwd*1e3:7.2f} ms  "
                  f"训练步 {tr*1e3:7.2f} ms  => 30万步约 {eta_h:5.2f} 小时")

    print("\n=== MLP 基线对照 ===")
    cm = graph_config(net_arch="mlp")
    net = RainbowNet(cm).to(DEV)
    for bs in (128, 256):
        x = torch.randn(bs, 18, device=DEV)
        net.eval()
        with torch.no_grad():
            fwd = timeit(lambda: net(x), n=10, warmup=3)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)

        def step():
            opt.zero_grad(set_to_none=True)
            net(x).sum().backward()
            opt.step()

        tr = timeit(step, n=10, warmup=3)
        print(f"  MLP  batch={bs:4d}  前向 {fwd*1e3:7.3f} ms  训练步 {tr*1e3:7.3f} ms")

    best = min(rows, key=lambda r: r[5])
    print(f"\n最快配置：K={best[0]} batch={best[1]} -> 30 万步约 {best[5]:.2f} 小时")
    return 0


if __name__ == "__main__":
    sys.exit(main())
