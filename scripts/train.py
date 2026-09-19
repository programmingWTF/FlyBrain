#!/usr/bin/env python
"""训练入口 —— 环境/回放/agent 全部复用 vendored 的 flappyrl（与原项目逐行等价），
只在网络躯干上有差别。

    python scripts/train.py --arch mlp        --steps 300000 --seed 0
    python scripts/train.py --arch connectome --steps 300000 --seed 0
    python scripts/train.py --arch rewired    --steps 300000 --seed 0

产物：
    runs/<tag>/log.csv        逐步训练日志
    runs/<tag>/summary.json   训练结束时用**未截断**的贪心评测（30 局）结果
    checkpoints/<tag>.pt      最终权重
"""
from __future__ import annotations

import argparse
import csv
import json
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


def build_graph(arch: str, base_name: str, weight_mode: str, seed: int):
    """arch -> SubgraphData（mlp 用不到图，返回 None）。"""
    if arch == "mlp":
        return None
    g = load_subgraph(base_name, weight_mode=weight_mode)
    if arch == "connectome":
        return g
    if arch == "rewired":
        return rewire_degree_preserving(g, seed=seed)
    raise ValueError(f"未知 arch: {arch!r}")


def evaluate(agent: RainbowDQN, n_envs: int = 8, episodes: int = 30, seed: int = 12345):
    """贪心评测：跑到 30 局结束为止（不设全局步数上限，避免原项目踩过的截断坑）。"""
    ec = env_config(n_envs=n_envs, seed=seed, auto_reset=True)
    sim = FlappySim(ec)
    st = sim.reset_all()
    scores: list[int] = []
    steps = 0
    max_steps = 2_000_000
    while len(scores) < episodes and steps < max_steps:
        a = agent.act(st, training=False)
        st, r, done, info = sim.step(a)
        steps += 1
        ts = info["terminal_scores"]
        for i in range(n_envs):
            if done[i]:
                scores.append(int(ts[i]))
    scores = scores[:episodes]
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "episodes": int(arr.size),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "max": int(arr.max()) if arr.size else 0,
        "min": int(arr.min()) if arr.size else 0,
        "std": float(arr.std()) if arr.size else 0.0,
        "scores": [int(x) for x in arr],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mlp", choices=["mlp", "connectome", "rewired"])
    ap.add_argument("--graph", default="sg_collision_s300_n4000", help="子图文件名前缀")
    ap.add_argument("--weight-mode", default="norm", choices=["norm", "count", "binary"])
    ap.add_argument("--steps", type=int, default=300_000, help="总 env-steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--graph-steps", type=int, default=3, help="图传播步数 K")
    ap.add_argument("--graph-gain", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-eval", action="store_true", help="跳过中途评测（只训）")
    a = ap.parse_args()

    tag = a.tag or f"{a.arch}_s{a.seed}"
    run_dir = ROOT / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ROOT / "checkpoints" / f"{tag}.pt"

    dev = torch.device(a.device if a.device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    g = build_graph(a.arch, a.graph, a.weight_mode, a.seed)
    cfg = graph_config(net_arch=a.arch, graph_steps=a.graph_steps, graph_gain=a.graph_gain)
    ec = env_config(n_envs=a.n_envs, seed=a.seed)

    print(f"=== {tag} ===")
    print(f"  device={dev}  steps={a.steps:,}  n_envs={a.n_envs}  seed={a.seed}")
    print(f"  arch={a.arch}  graph_steps={a.graph_steps}  weight_mode={a.weight_mode}")
    if g is not None:
        pc = g.param_count(cfg.hidden)
        print(f"  {g}")
        print(f"  有效可学参数 {pc['total']:,}（边 {pc['edges']:,} / 输入 {pc['w_in']:,} / 读出 {pc['readout']:,}）")

    agent = RainbowDQN(cfg, ec, dev, graph=g)
    sim = FlappySim(ec)
    st = sim.reset_all()

    log_path = run_dir / "log.csv"
    log_f = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    writer.writerow(["step", "loss", "eps", "mean_recent_score", "wall_s"])

    t0 = time.time()
    episode_scores: list[int] = []
    losses: list[float] = []
    next_eval = a.eval_every
    evals: list[dict] = []
    step = 0

    while step < a.steps:
        actions = agent.act(st, training=True)
        st2, rewards, dones, info = sim.step(actions)
        agent.store_transition(st, actions, rewards, st2, dones)
        st = st2

        loss = agent.learn()
        if loss is not None:
            losses.append(loss)
        ts = info["terminal_scores"]
        for i in range(a.n_envs):
            if dones[i]:
                episode_scores.append(int(ts[i]))

        step += 1
        agent.anneal(step, a.steps)

        if step % 500 == 0:
            recent = float(np.mean(episode_scores[-100:])) if episode_scores else 0.0
            writer.writerow([step, f"{np.mean(losses[-100:]):.5f}" if losses else "",
                             f"{agent.eps:.4f}", f"{recent:.2f}", f"{time.time()-t0:.1f}"])
            log_f.flush()

        if not a.no_eval and step >= next_eval:
            next_eval += a.eval_every
            ev = evaluate(agent, n_envs=a.n_envs, episodes=a.eval_episodes, seed=999 + step)
            evals.append({"step": step, **{k: v for k, v in ev.items() if k != "scores"}})
            print(f"  [{step:7,d}] 训练近期均分 {np.mean(episode_scores[-100:]) if episode_scores else 0:6.1f} "
                  f"| 贪心评测 {ev['mean']:6.2f} (max {ev['max']}, n={ev['episodes']}) "
                  f"| {time.time()-t0:6.0f}s", flush=True)

    log_f.close()
    agent.save(str(ckpt))

    print("\n最终评测（贪心，30 局）...")
    final = evaluate(agent, n_envs=a.n_envs, episodes=30, seed=20240919)
    summary = {
        "tag": tag,
        "arch": a.arch,
        "graph": a.graph if g is not None else None,
        "weight_mode": a.weight_mode,
        "graph_steps": a.graph_steps,
        "seed": a.seed,
        "steps": a.steps,
        "n_envs": a.n_envs,
        "device": str(dev),
        "wall_seconds": round(time.time() - t0, 1),
        "train_episodes": len(episode_scores),
        "train_mean_score": float(np.mean(episode_scores)) if episode_scores else 0.0,
        "train_mean_last100": float(np.mean(episode_scores[-100:])) if episode_scores else 0.0,
        "final_eval": final,
        "evals": evals,
        "effective_params": (g.param_count(cfg.hidden) if g is not None else None),
        "checkpoint": str(ckpt.relative_to(ROOT)),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"  最终：均分 {final['mean']:.2f}  最高 {final['max']}  最低 {final['min']}  "
          f"({final['episodes']} 局)")
    print(f"  用时 {summary['wall_seconds']}s -> {run_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
