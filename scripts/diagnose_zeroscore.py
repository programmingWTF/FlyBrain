#!/usr/bin/env python
"""用「已知能打 303 分」的模型来定位问题到底出在环境/评测，还是训练循环。

原项目提供 checkpoints/best.pt（baseline_long @ 975k 步，均分 303.67 / 最高 991）。
把它加载进**我这份 vendored 网络**，再用**我的 evaluate()** 跑：

  - 如果也能打出高分 -> 环境与评测都是对的，问题一定在训练循环
  - 如果也是 0 分     -> 环境或评测有问题（状态编码/动作顺序/重置逻辑）

用法：python scripts/diagnose_zeroscore.py
"""
from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv import env_config, graph_config  # noqa: E402
from fpv.vendor_flappyrl.networks import RainbowNet  # noqa: E402
from fpv.vendor_flappyrl.sim import FlappySim  # noqa: E402

REF = pathlib.Path(r"D:\Code\DQN\checkpoints\best.pt")


def evaluate(net, n_envs=8, episodes=30, seed=12345, greedy=True, max_steps=200_000):
    ec = env_config(n_envs=n_envs, seed=seed, auto_reset=True)
    sim = FlappySim(ec)
    st = sim.reset_all()
    scores = []
    steps = 0
    while len(scores) < episodes and steps < max_steps:
        with torch.no_grad():
            q = net.expected_q(torch.as_tensor(st, dtype=torch.float32))
        a = q.argmax(1).numpy().astype("int64")
        st, r, done, info = sim.step(a)
        steps += 1
        for i in range(n_envs):
            if done[i]:
                scores.append(int(info["terminal_scores"][i]))
    return scores[:episodes]


def main() -> int:
    if not REF.exists():
        print(f"找不到参考模型 {REF}")
        return 1

    ckpt = torch.load(REF, map_location="cpu", weights_only=False)
    print(f"参考模型：{REF}")
    print(f"  文件 {REF.stat().st_size/1e6:.1f} MB")
    meta = ckpt.get("meta", {})
    for k in ("state_dim", "action_dim", "hidden", "distributional", "num_atoms"):
        print(f"   meta.{k} = {meta.get(k)}")

    cfg = graph_config(net_arch="mlp")
    print(f"\n我的 cfg: state_dim={cfg.state_dim} action_dim={cfg.action_dim} "
          f"hidden={cfg.hidden} dist={cfg.distributional} atoms={cfg.num_atoms} "
          f"dueling={cfg.dueling} noisy={cfg.noisy}")

    net = RainbowNet(cfg)
    missing, unexpected = net.load_state_dict(ckpt["online"], strict=False)
    print(f"\n加载 state_dict：missing={list(missing)}  unexpected={list(unexpected)}")
    net.eval()

    print("\n用我的 evaluate() 跑 30 局（贪心）...")
    sc = evaluate(net, n_envs=8, episodes=30)
    import statistics
    print(f"  局数 {len(sc)}  均分 {statistics.mean(sc):.2f}  最高 {max(sc)}  最低 {min(sc)}")
    print(f"  明细 {sc}")
    if statistics.mean(sc) > 20:
        print("\n=> 环境与评测都正常。问题出在训练循环（不是状态/动作/重置）。")
    else:
        print("\n=> 环境或评测有问题！参考模型在这里也拿不到分。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
