#!/usr/bin/env python
"""test_resume.py 的子进程：从检查点恢复并继续训练到 1500 步，把结果写成 json。

单独起进程是故意的 —— 这样才真正验证「进程重启后能从存档续上」，
而不是在同一个进程里自欺欺人。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv import env_config, graph_config, load_subgraph, resume as resume_mod  # noqa: E402
from fpv.vendor_flappyrl.agent import RainbowDQN  # noqa: E402
from fpv.vendor_flappyrl.sim import FlappySim  # noqa: E402


def main() -> int:
    ck = pathlib.Path(sys.argv[1])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = load_subgraph("sg_collision_s200_n3000")
    cfg = graph_config(net_arch="connectome", graph_steps=3, batch_size=128,
                       learning_starts=200, buffer_size=20_000)
    ec = env_config(n_envs=8, seed=0)

    agent = RainbowDQN(cfg, ec, dev, graph=g)
    payload = resume_mod.load(agent, ck, device=dev)
    start_step = int(payload.get("step", 0))
    eps_at_start = float(agent.eps)

    sim = FlappySim(ec)
    st = sim.reset_all()
    for step in range(start_step, 1500):
        acts = agent.act(st, training=True)
        st2, rw, dn, info = sim.step(acts)
        agent.store_transition(st, acts, rw, st2, dn)
        st = st2
        agent.learn()
        agent.anneal(step + 1, 1500)

    with torch.no_grad():
        tot = sum(float(p.sum()) for p in agent.online_net.parameters())
        off = agent.online_net.A[~agent.online_net.edge_mask]
        off_max = float(off.abs().max()) if off.numel() else 0.0

    out = {
        "start_step": start_step,
        "end_step": 1500,
        "buffer_len": len(agent.buffer),
        "eps_at_start": eps_at_start,
        "fingerprint_sum": tot,
        "off_edge_max": off_max,
    }
    (ROOT / "checkpoints" / "_resume_child.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"  子进程：{start_step} -> 1500，回放 {len(agent.buffer):,}，"
          f"指纹 {tot:.4f}，非边max {off_max:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
