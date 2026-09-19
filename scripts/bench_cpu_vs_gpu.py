#!/usr/bin/env python
"""CPU vs GPU：这个任务到底需不需要 GPU？

同一个进程里（所以 CPU/GPU 对比不受并发训练干扰）分别测：
  1. 基线 MLP 的前向 + 前向反向
  2. ConnectomeNet(dense A@x) 的前向 + 前向反向，K=3
  3. 连接组稀疏版（只在活跃边上算）的前向，看 CPU 上是否反而更划算

然后给出「30 万步需要多久」的 CPU/GPU 对比结论。

用法：python scripts/bench_cpu_vs_gpu.py
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

HAS_CUDA = torch.cuda.is_available()


def timeit(fn, n=8, warmup=3, device=None):
    for _ in range(warmup):
        fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def bench_net(name, build, dev, bs=128, n=8):
    net = build(dev)
    x = torch.randn(bs, 18, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    net.eval()
    with torch.no_grad():
        fwd = timeit(lambda: net(x), n=n, warmup=3, device=dev)
    net.train()

    def step():
        opt.zero_grad(set_to_none=True)
        net(x).sum().backward()
        opt.step()

    tr = timeit(step, n=n, warmup=3, device=dev)
    print(f"    {name:30s} 前向 {fwd*1e3:8.2f} ms   前向+反向 {tr*1e3:8.2f} ms")
    return fwd, tr


def main() -> int:
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"cuda={HAS_CUDA}")
    g = load_subgraph("sg_collision_s300_n4000")
    N = g.n_nodes
    print(f"图：{g}  ->  稠密 A 需 {N*N*4/1e6:.0f} MB\n")

    devices = [torch.device("cpu")]
    if HAS_CUDA:
        devices.append(torch.device("cuda"))

    results = {}
    for dev in devices:
        print(f"--- device = {dev} ---")
        cfgs = {}
        # 基线 MLP
        cm = graph_config(net_arch="mlp")
        f1, t1 = bench_net("MLP 基线 (18->256->256)", lambda d: RainbowNet(cm).to(d), dev)
        cfgs["MLP"] = (f1, t1)

        # 连接组 K=3
        cg = graph_config(net_arch="connectome", graph_steps=3)
        f3, t3 = bench_net(
            "Connectome K=3 (dense A@x)",
            lambda d: ConnectomeNet(cg, g.src, g.dst, g.weight, g.n_nodes,
                                    g.sensory_idx, g.motor_idx).to(d), dev)
        cfgs["ConnK3"] = (f3, t3)

        # 连接组 K=1
        cg1 = graph_config(net_arch="connectome", graph_steps=1)
        f1k, t1k = bench_net(
            "Connectome K=1 (dense A@x)",
            lambda d: ConnectomeNet(cg1, g.src, g.dst, g.weight, g.n_nodes,
                                    g.sensory_idx, g.motor_idx).to(d), dev)
        cfgs["ConnK1"] = (f1k, t1k)
        results[dev.type] = cfgs
        print()

    # 换算 30 万步
    STEPS = 300_000
    print("=" * 74)
    print(f"换算：每步 ≈ 1 次前向(act) + 1 次前向反向(learn)  ->  {STEPS:,} 步需要多久")
    print("=" * 74)
    for key in ("MLP", "ConnK3", "ConnK1"):
        line = f"  {key:8s}"
        for d in ("cpu", "cuda"):
            if d in results:
                f, t = results[d][key]
                hrs = (f + t) * STEPS / 3600
                line += f"   {d}: {hrs:6.2f} h"
        if "cpu" in results and "cuda" in results:
            fc, tc = results["cpu"][key]
            fg, tg = results["cuda"][key]
            line += f"   (GPU 快 {(fc+tc)/(fg+tg):.1f}×)"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
