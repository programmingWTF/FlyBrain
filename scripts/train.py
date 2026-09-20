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
from fpv import resume as resume_mod  # noqa: E402
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
    # ⚠️ 尺度陷阱：原项目的 target_update_freq / eps_decay_steps 都是 **env-step** 尺度，
    # 而本脚本的 step 是「循环数」（8 个并行环境）。所以直接用原值会慢 8 倍。
    # 这里按「梯度更新次数」解释，默认值 = 原项目值 / n_envs，保证与原项目行为一致。
    ap.add_argument("--target-update-freq", type=int, default=None,
                    help="每多少次梯度更新同步 target 网络（默认 = 原项目 1000/n_envs）")
    ap.add_argument("--eps-decay-steps", type=int, default=None,
                    help="eps 退火步数（默认 = 原项目 50000/n_envs）")
    ap.add_argument("--learning-starts", type=int, default=None,
                    help="多少条 transition 后开始学习（默认 2000，与原项目一致）")
    ap.add_argument("--grad-steps", type=int, default=None,
                    help="每次循环做几次梯度更新（默认 1）")
    ap.add_argument("--loop-semantics", action="store_true",
                    help="按「循环数」记账（旧行为，仅作对照/调试）。"
                         "默认是 env-step 语义，与原项目 train.py 完全一致 —— "
                         "这一点至关重要：用循环数记账会让 anneal 的时钟慢 n_envs 倍，"
                         "eps 长期停在 0.2~0.9，本任务直接学不动（实测 0 分 vs 152 分）。")
    ap.add_argument("--save-every", type=int, default=25_000,
                    help="每多少步存一次完整检查点（0=只在结束时存）")
    ap.add_argument("--resume", default=None,
                    help="从检查点续训；传 'auto' 则自动找 checkpoints/<tag>.pt")
    ap.add_argument("--keep-last", type=int, default=2, help="保留最近几个滚动检查点")
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
    n = max(1, a.n_envs)
    if a.grad_steps is None:
        a.grad_steps = 1

    # ---- 记账尺度（重要）----
    # 默认按 **env-step** 记账：global_step += n_envs，与原项目 train.py 完全一致。
    # target_update_freq / eps_decay_steps 都是这个尺度下的值，直接照用即可。
    #
    # 曾经默认按「循环数」记账（step += 1），结果 anneal 的时钟慢了 n_envs 倍：
    # eps 长时间停在 0.2~0.9，FlappyBird 这种需要精确时机的任务直接学不动，
    # 12 个 run 全部 0 分。改用 env-step 后同一配置立刻到 152 分。
    # --loop-semantics 保留旧行为仅作对照/调试。
    if a.loop_semantics or a.target_update_freq is not None or a.eps_decay_steps is not None:
        cfg_tuf = a.target_update_freq if a.target_update_freq is not None else max(1, 1000 // n)
        cfg_eds = (a.eps_decay_steps if a.eps_decay_steps is not None
                   else max(1, 50_000 // n))
        scale_note = f"循环语义（每循环 +1）；env-step 尺度超参已除以 {n}"
        total_units = a.steps
    else:
        cfg_tuf = 1000             # 原项目同值、同尺度
        cfg_eds = 50_000
        scale_note = f"env-step 语义（每循环 +{n}，与原项目一致）"
        total_units = a.steps * n

    overrides = dict(target_update_freq=cfg_tuf, eps_decay_steps=cfg_eds)
    if a.learning_starts is not None:
        overrides["learning_starts"] = a.learning_starts
    cfg = graph_config(net_arch=a.arch, graph_steps=a.graph_steps, graph_gain=a.graph_gain,
                       **overrides)
    ec = env_config(n_envs=a.n_envs, seed=a.seed)

    print(f"=== {tag} ===")
    print(f"  device={dev}  {a.steps:,} 步 x {n} env = {total_units:,} env-step  "
          f"(seed={a.seed})")
    print(f"  arch={a.arch}  graph_steps={a.graph_steps}  weight_mode={a.weight_mode}")
    print(f"  {scale_note}")
    print(f"  target_update_freq={cfg.target_update_freq}  eps_decay_steps={cfg.eps_decay_steps}"
          f"  learning_starts={cfg.learning_starts}  grad_steps={a.grad_steps}")
    if g is not None:
        pc = g.param_count(cfg.hidden)
        print(f"  {g}")
        print(f"  有效可学参数 {pc['total']:,}（边 {pc['edges']:,} / 输入 {pc['w_in']:,} / 读出 {pc['readout']:,}）")

    agent = RainbowDQN(cfg, ec, dev, graph=g)
    sim = FlappySim(ec)
    st = sim.reset_all()

    log_path = run_dir / "log.csv"
    # 续训时追加而不是覆盖，免得把中断前的曲线抹掉
    append_log = bool(a.resume) and log_path.exists()
    log_f = log_path.open("a" if append_log else "w", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    if not append_log:
        writer.writerow(["step", "loss", "eps", "mean_recent_score", "wall_s"])

    t0 = time.time()
    episode_scores: list[int] = []
    losses: list[float] = []
    next_eval = a.eval_every
    evals: list[dict] = []
    loop = 0

    # 滚动检查点的辅助：保留最近 keep_last 个
    roll_dir = run_dir / "ckpt"
    roll_dir.mkdir(parents=True, exist_ok=True)

    def save_ckpt(tag: str) -> pathlib.Path:
        extra = {"episode_scores": episode_scores[-2000:], "losses": losses[-2000:],
                 "evals": evals, "arch": a.arch, "graph": a.graph,
                 "weight_mode": a.weight_mode, "graph_steps": a.graph_steps}
        return resume_mod.save(agent, ckpt, loop=loop, cfg=cfg, extra=extra)

    def prune_rolls() -> None:
        files = sorted(roll_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
        for old in files[: max(0, len(files) - a.keep_last)]:
            try:
                old.unlink()
            except OSError:
                pass

    # ================= 训练主循环 =================
    # 单位约定（这一块踩过坑，务必守住）：
    #   loop  每转一圈 +1，**只用来控制循环与存/读检查点**
    #   unit  记账单位：env-step 语义下 = loop*n（与原项目一致），loop 语义下 = loop
    #   所有调度（anneal / target 同步 / 评测 / 日志）一律用 unit
    # 之前把 loop 当成 unit 存进检查点、又拿 --steps（env-step）去比，
    # 导致续训后进度条错乱、且会多跑 n 倍。
    loops = max(1, a.steps // n) if not a.loop_semantics else a.steps
    if a.loop_semantics:
        loops = a.steps
    loop = 0

    def unit_of(lp: int) -> int:
        return lp if a.loop_semantics else lp * n

    # ---- 续训 ----
    if a.resume:
        ckpt_in = ckpt if a.resume == "auto" else pathlib.Path(a.resume)
        print(f"从 {ckpt_in} 续训 ...")
        payload = resume_mod.load(agent, ckpt_in, device=dev)
        loop = int(payload.get("loop", payload.get("step", 0)))
        ex = payload.get("extra") or {}
        episode_scores = list(ex.get("episode_scores", []))
        losses = list(ex.get("losses", []))
        evals = list(ex.get("evals", []))
        print(f"  从 loop={loop:,}（= {unit_of(loop):,} env-step）继续，"
              f"已有 {len(episode_scores)} 局记录，eps={agent.eps:.3f}")
        next_eval = (unit_of(loop) // a.eval_every + 1) * a.eval_every

    print(f"  计划 {loops:,} 个循环 -> {unit_of(loops):,} env-step")

    # 滚动检查点的辅助：保留最近 keep_last 个
    roll_dir = run_dir / "ckpt"
    roll_dir.mkdir(parents=True, exist_ok=True)

    while loop < loops:
        actions = agent.act(st, training=True)
        st2, rewards, dones, info = sim.step(actions)
        agent.store_transition(st, actions, rewards, st2, dones)
        st = st2

        loss = None
        for _ in range(a.grad_steps):
            l = agent.learn()
            if l is not None:
                loss = l
                losses.append(l)
        ts = info["terminal_scores"]
        for i in range(a.n_envs):
            if dones[i]:
                episode_scores.append(int(ts[i]))

        loop += 1
        unit = unit_of(loop)
        agent.anneal(unit, unit_of(loops))

        # target 网络同步：按 env-step 周期（与原项目对齐）
        if (not a.loop_semantics) and unit % cfg.target_update_freq == 0:
            agent.sync_target()

        if loop % 500 == 0:
            recent = float(np.mean(episode_scores[-100:])) if episode_scores else 0.0
            writer.writerow([unit, f"{np.mean(losses[-100:]):.5f}" if losses else "",
                             f"{agent.eps:.4f}", f"{recent:.2f}", f"{time.time()-t0:.1f}"])
            log_f.flush()

        # 定期存滚动检查点（先写临时文件再改名，避免读到写了一半的文件）
        if a.save_every and unit % a.save_every == 0 and loop < loops:
            roll = roll_dir / f"step_{unit:09d}.pt"
            tmp = roll.with_suffix(".pt.tmp")
            resume_mod.save(agent, tmp, loop=loop, cfg=cfg,
                            extra={"episode_scores": episode_scores[-2000:],
                                   "losses": losses[-2000:], "evals": evals})
            tmp.replace(roll)
            prune_rolls()
            print(f"  [{unit:9,d}] 已存档 -> {roll.name}", flush=True)

        if not a.no_eval and unit >= next_eval:
            next_eval += a.eval_every
            ev = evaluate(agent, n_envs=a.n_envs, episodes=a.eval_episodes, seed=999 + unit)
            evals.append({"step": unit, **{k: v for k, v in ev.items() if k != "scores"}})
            print(f"  [{unit:9,d}] 训练近期均分 "
                  f"{np.mean(episode_scores[-100:]) if episode_scores else 0:6.1f} "
                  f"| 贪心评测 {ev['mean']:6.2f} (max {ev['max']}, n={ev['episodes']}) "
                  f"| {time.time()-t0:6.0f}s", flush=True)

    log_f.close()
    # 结束时的权威检查点：完整训练态，可直接 --resume 续跑
    save_ckpt("final")
    print(f"已保存完整训练态 -> {ckpt}（loop={loop:,} = {unit_of(loop):,} env-step）")

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
