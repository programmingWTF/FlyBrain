#!/usr/bin/env python
"""训练循环耗时拆解 —— 回答「时间到底花在哪」。

对同一套 train.py 的循环，逐段计时：
    env        : FlappySim.step（纯 numpy，CPU，逐环境 python 循环）
    act        : 选动作（一次 GPU 前向 + 传回 CPU）
    store      : 写进回放缓冲（n-step 拼接 + SumTree）
    learn      : 采样 + forward + backward + optimizer.step + mask
    anneal/quat: eps/beta 调度等零碎

同时报告「每秒多少 env-step」与「GPU 前向本身占多少」，
从而区分「GPU 算得慢」和「GPU 在等 CPU」。

用法：python scripts/profile_train.py --arch connectome --steps 600
      python scripts/profile_train.py --arch mlp --steps 600
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv import env_config, graph_config, load_subgraph  # noqa: E402
from fpv.rewire import rewire_degree_preserving  # noqa: E402
from fpv.vendor_flappyrl.agent import RainbowDQN  # noqa: E402
from fpv.vendor_flappyrl.sim import FlappySim  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="connectome", choices=["mlp", "connectome", "rewired"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--graph-steps", type=int, default=3)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = None
    if a.arch != "mlp":
        g = load_subgraph("sg_collision_s300_n4000")
        if a.arch == "rewired":
            g = rewire_degree_preserving(g, seed=0)

    cfg = graph_config(net_arch=a.arch, graph_steps=a.graph_steps, batch_size=a.batch_size)
    ec = env_config(n_envs=a.n_envs, seed=0)
    agent = RainbowDQN(cfg, ec, dev, graph=g)
    sim = FlappySim(ec)
    st = sim.reset_all()

    # 预热（让 cudnn/cublas 选好算法、回放缓冲填满）
    for _ in range(max(50, cfg.learning_starts // (a.n_envs * 4)) + 60):
        acts = agent.act(st, training=True)
        st2, rw, dn, _ = sim.step(acts)
        agent.store_transition(st, acts, rw, st2, dn)
        st = st2
        agent.learn()

    acc = {k: 0.0 for k in ("env", "act", "store", "learn")}
    n_learn = 0
    if dev.type == "cuda":
        torch.cuda.synchronize()

    t_all0 = time.perf_counter()
    for _ in range(a.steps):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        acts = agent.act(st, training=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        st2, rw, dn, info = sim.step(acts)
        t2 = time.perf_counter()

        agent.store_transition(st, acts, rw, st2, dn)
        t3 = time.perf_counter()

        loss = agent.learn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        acc["act"] += t1 - t0
        acc["env"] += t2 - t1
        acc["store"] += t3 - t2
        acc["learn"] += t4 - t3
        if loss is not None:
            n_learn += 1
        st = st2

    total = time.perf_counter() - t_all0
    print(f"\n=== arch={a.arch}  n_envs={a.n_envs}  batch={a.batch_size}  "
          f"K={a.graph_steps}  device={dev} ===")
    print(f"总计 {a.steps} 个循环 = {total:.2f}s   -> {a.steps/total:.1f} loop/s, "
          f"{a.steps*a.n_envs/total:.0f} env-step/s")
    print(f"（其中 {n_learn} 次 learn() 真的做了梯度更新）\n")
    for k in ("env", "act", "store", "learn"):
        pct = acc[k] / total * 100
        print(f"  {k:6s} {acc[k]*1000:8.1f} ms  {pct:5.1f}%   "
              f"每次循环 {acc[k]/a.steps*1000:6.3f} ms")
    print(f"\n  纯 GPU 前向(act) 只占 {acc['act']/total*100:4.1f}%  -> "
          f"GPU 大部分时间在等 CPU（python 循环 + numpy 仿真 + 回放采样）")

    # 单独测一次 learn 里的前向/反向纯 GPU 耗时作对照
    if dev.type == "cuda":
        bs = a.batch_size
        xb = torch.randn(bs, 18, device=dev)
        net = agent.online_net
        opt = agent.optimizer
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            net(xb).sum().backward()
            opt.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            opt.zero_grad(set_to_none=True)
            net(xb).sum().backward()
            opt.step()
        torch.cuda.synchronize()
        pure = (time.perf_counter() - t0) / 30
        print(f"  纯 GPU「前向+反向+optimizer」={pure*1000:.2f} ms；"
              f"而 learn() 实际 {acc['learn']/max(1,n_learn)*1000:.2f} ms "
              f"-> 差额是回放采样/搬运/掩码等 CPU 开销")
    return 0


if __name__ == "__main__":
    sys.exit(main())
