#!/usr/bin/env python
"""冒烟测试：确认接入没有破坏任何东西。

检查项
  1. vendored 环境能跑（FlappySim 出 18 维状态 / 2 动作）
  2. MLP 基线 (RainbowNet) 输出形状与契约正确
  3. ConnectomeNet 输出形状与契约**完全一致**，且前向能跑
  4. 两者的参数量对比（公平性前提）
  5. 端到端：RainbowDQN(net_arch='connectome') 能存/取经验并做一步 learn()
  6. 稀疏图是否真的比 MLP 快/慢（顺手测个前向耗时）

用法：python scripts/smoke_test.py
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv import env_config, graph_config, load_subgraph  # noqa: E402
from fpv.vendor_flappyrl.agent import RainbowDQN  # noqa: E402
from fpv.vendor_flappyrl.networks import ConnectomeNet, RainbowNet  # noqa: E402
from fpv.vendor_flappyrl.sim import FlappySim  # noqa: E402

ok = True


def check(cond: bool, msg: str) -> None:
    global ok
    print(f"  {'✓' if cond else '✗'} {msg}")
    if not cond:
        ok = False


def main() -> int:
    print("=" * 72)
    print("1) 环境")
    env = FlappySim(env_config(n_envs=4))
    s = env.reset_all()
    print(f"   状态 shape {s.shape}  dtype {s.dtype}")
    check(s.shape == (4, 18), "FlappySim 出 (n_envs, 18) 状态")
    s2, r, d, info = env.step(np.array([0, 1, 0, 1]))
    check(s2.shape == (4, 18) and r.shape == (4,), "step() 返回形状正确")

    print("\n2) MLP 基线 RainbowNet")
    cfg_mlp = graph_config(net_arch="mlp")
    net_mlp = RainbowNet(cfg_mlp)
    x = torch.randn(8, cfg_mlp.state_dim)
    q = net_mlp(x)
    eq = net_mlp.expected_q(x)
    check(q.shape == (8, cfg_mlp.action_dim), f"forward -> {tuple(q.shape)}")
    check(eq.shape == (8, cfg_mlp.action_dim), f"expected_q -> {tuple(eq.shape)}")
    n_mlp = sum(p.numel() for p in net_mlp.parameters())
    print(f"   可学参数 {n_mlp:,}")

    print("\n3) ConnectomeNet")
    g = load_subgraph("sg_collision_s300_n4000")
    print(f"   {g}")
    cfg_g = graph_config(net_arch="connectome", graph_steps=3)
    net_g = ConnectomeNet(
        cfg_g, src=g.src, dst=g.dst, weight=g.weight, n_nodes=g.n_nodes,
        sensory_idx=g.sensory_idx, motor_idx=g.motor_idx,
    )
    qg = net_g(x)
    eqg = net_g.expected_q(x)
    check(qg.shape == q.shape, f"forward 形状一致 {tuple(qg.shape)}")
    check(eqg.shape == eq.shape, f"expected_q 形状一致 {tuple(eqg.shape)}")
    check(torch.isfinite(qg).all().item(), "输出全为有限值（没有 NaN/Inf）")
    n_g_dense = sum(p.numel() for p in net_g.parameters())
    pc = g.param_count(cfg_g.hidden)
    print(f"   稠密槽位 {n_g_dense:,}（含 4000² 矩阵），但**有效可学参数** = {pc['total']:,}")
    print(f"     其中 边 {pc['edges']:,} + 输入 {pc['w_in']:,} + 读出 {pc['readout']:,}")
    print(f"   MLP 基线有效参数 {n_mlp:,}  ->  比值 {pc['total']/n_mlp:.2f}×")

    print("\n4) C51 分布模式下的契约")
    cfg_d = graph_config(net_arch="connectome", distributional=True, num_atoms=51)
    net_d = ConnectomeNet(
        cfg_d, src=g.src, dst=g.dst, weight=g.weight, n_nodes=g.n_nodes,
        sensory_idx=g.sensory_idx, motor_idx=g.motor_idx,
    )
    dd = net_d(x)
    check(dd.shape == (8, 2, 51), f"distributional forward -> {tuple(dd.shape)}")
    check(net_d.expected_q(x).shape == (8, 2), "distributional expected_q -> (8,2)")

    print("\n5) 前向耗时（CPU，batch=128）")
    xb = torch.randn(128, 18)
    for nm, net in (("MLP", net_mlp), ("Connectome", net_g)):
        with torch.no_grad():
            net(xb)
            t0 = time.perf_counter()
            for _ in range(10):
                net(xb)
            dt = (time.perf_counter() - t0) / 10
        print(f"   {nm:11s} {dt*1000:7.2f} ms / forward")

    print("\n6) 端到端：agent 能收集 + learn")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   device = {dev}")
    for arch in ("mlp", "connectome"):
        c = graph_config(net_arch=arch)
        ag = RainbowDQN(c, env_config(n_envs=4), dev, graph=(g if arch == "connectome" else None))
        e = FlappySim(env_config(n_envs=4))
        st = e.reset_all()
        # 至少要攒够 batch_size 条 transition，learn() 才会真的更新
        n_steps = max(c.batch_size * 2, c.learning_starts + c.batch_size)
        for _ in range(n_steps):
            a = ag.act(st, training=True)
            ns, rw, dn, _ = e.step(a)
            ag.store_transition(st, a, rw, ns, dn)
            st = ns
        buf_n = len(ag.buffer) if hasattr(ag.buffer, "__len__") else "?"
        loss = ag.learn()
        check(loss is not None and np.isfinite(loss),
              f"[{arch}] 收集 {n_steps} 步(缓冲 {buf_n}) 后 learn() 返回有限 loss = {loss}")

        # 再连做几步，确认能持续更新、且图网络的非边位置保持为 0
        for _ in range(20):
            ag.learn()
        if arch == "connectome":
            with torch.no_grad():
                A = ag.online_net.A
                off = A[~ag.online_net.edge_mask]
                check(float(off.abs().max()) == 0.0,
                      f"非边位置严格为 0（max |off-edge| = {float(off.abs().max()):.1e}）")
                on = A[ag.online_net.edge_mask]
                check(float(on.abs().max()) > 0,
                      f"边位置上权重非平凡（max = {float(on.abs().max()):.4f}）")
        p = ROOT / "checkpoints" / f"smoke_{arch}.pt"
        ag.save(str(p))
        check(p.exists(), f"[{arch}] checkpoint 落盘 {p.name}")

    print("\n" + "=" * 72)
    print("冒烟测试通过 ✅" if ok else "冒烟测试失败 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
