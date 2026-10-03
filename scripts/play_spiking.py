#!/usr/bin/env python
"""阶段 3.1：端到端 —— 冻结脉冲脑玩 FlappyBird。

用法
----
    # 随机读出（未训练基线）
    python scripts/play_spiking.py --episodes 20

    # 训练读出（进化策略，只训练几百个参数）
    python scripts/play_spiking.py --train --generations 60

    # 对照：度分布保持重连的随机脑
    python scripts/play_spiking.py --brain spiking_circuit_rewired --episodes 20
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

# 深调用栈下的保护：torch 的若干上下文管理器在 Py3.11 + torch 2.x 上会
# 因 `_contextlib` 反复构造而触发 RecursionError，抬一下上限即可根除。
sys.setrecursionlimit(10000)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv.spiking_brain import SpikingBrain  # noqa: E402
from fpv.spiking_controller import SpikingController  # noqa: E402
from fpv.vendor_flappyrl.config import AgentConfig, EnvConfig  # noqa: E402
from fpv.vendor_flappyrl.networks import RainbowNet  # noqa: E402
from fpv.vendor_flappyrl.sim import FlappySim  # noqa: E402


def load_teacher(tag: str, device) -> RainbowNet:
    """载入已训练好的 MLP 教师（会玩，均分 368 / 最高 1657）。

    ⚠️ 教训：早期版本用"鸟在缺口下方就 flap"的手写规则当教师，
    那条规则自己只能活 121 帧 / 0 分，蒸馏它等于蒸馏一个笨蛋，
    所以读出永远学不会。必须用真会玩的 checkpoint/mlp_s0.pt。
    """
    cfg = AgentConfig(double=True, dueling=True, noisy=False, per=False,
                      n_step=1, distributional=False)
    net = RainbowNet(cfg).to(device)
    p = torch.load(ROOT / "checkpoints" / f"{tag}.pt", map_location=device,
                   weights_only=False)
    net.load_state_dict(p["online"])
    net.eval()
    return net


def teacher_action(teacher: RainbowNet, state_np, device) -> int:
    s = torch.as_tensor(state_np, dtype=torch.float32, device=device).view(1, -1)
    with torch.no_grad():
        return int(torch.argmax(teacher(s), dim=1).item())


def rollout(ctrl: SpikingController, sim: FlappySim, cfg: EnvConfig,
            max_ticks: int = 3000, decide_every: int = 1, seed: int = 0):
    """跑一局，返回 (score, ticks)。

    脑**每个游戏帧都推进 1 tick**（膜电位跨帧累积），每 decide_every 帧读取
    一次下行群体发放率并做一次 flap 决策。这样信号才有时间从感觉神经元传播
    到下行神经元（之前每个决策点只跑 2 tick，特征恒为 0）。
    """
    ctrl.reset()
    st = sim.reset_all()
    scores = []
    step = 0
    action = 0
    info = {"terminal_scores": np.zeros(cfg.n_envs, dtype=np.int32)}
    while step < max_ticks:
        state = torch.as_tensor(st[0], dtype=torch.float32, device=ctrl.brain.device)
        ctrl.advance(state, n_ticks=1)                # 脑持续运行
        if step % decide_every == 0:
            feat = ctrl.read_features()
            action = ctrl.act(feat)
        a = np.zeros(cfg.n_envs, dtype=np.int64)
        a[0] = action
        st, r, done, info = sim.step(a)
        step += 1
        if done[0]:
            scores.append(int(info["terminal_scores"][0]))
            break
    if not scores:
        scores.append(int(info["terminal_scores"][0]) if info["terminal_scores"].size else 0)
    return scores[0], step


def evaluate(ctrl, cfg, episodes, decide_every, max_ticks=3000, seed=0, verbose=False):
    sim = FlappySim(EnvConfig(n_envs=1, seed=seed, auto_reset=True,
                              width=cfg.width, height=cfg.height, pipe_gap=cfg.pipe_gap,
                              pipe_speed=cfg.pipe_speed, pipe_spawn_dist=cfg.pipe_spawn_dist))
    sc = []
    for e in range(episodes):
        s, t = rollout(ctrl, sim, cfg, max_ticks=max_ticks, decide_every=decide_every,
                       seed=seed + e)
        sc.append(s)
        if verbose:
            print(f"    ep{e}: score={s} ticks={t}")
    return np.array(sc, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="spiking_circuit")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--decide-every", type=int, default=1,
                    help="每多少个游戏帧读一次脑并决策（脑每帧都推进）")
    ap.add_argument("--max-ticks", type=int, default=3000)
    ap.add_argument("--readout", default="groups", choices=["groups", "wide"])
    ap.add_argument("--n-wide", type=int, default=512, help="宽读出的群体数")
    ap.add_argument("--head-dim", type=int, default=0,
                    help="读出隐层维度；0 = 纯线性读出（参数最少，ES 搜得动）")
    ap.add_argument("--wide-per", type=int, default=64, help="每个宽群体多少神经元")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--teacher", default="mlp_s0", help="蒸馏用的教师 checkpoint 名")
    ap.add_argument("--dagger-rounds", type=int, default=0,
                    help="DAgger 在线纠偏轮数（修行为克隆的误差累积）")
    ap.add_argument("--dagger-eps", type=int, default=8,
                    help="每轮 DAgger 跑几局采集")
    ap.add_argument("--pretrain-eps", type=int, default=60,
                    help="教师跑多少局来收集蒸馏样本（教师均分 368，一局几千帧）")
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--pop", type=int, default=24, help="ES 种群大小")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    tag = a.tag or a.brain + ("_trained" if a.train else "_random")
    print(f"=== 脉冲脑玩 FlappyBird ===")
    print(f"  brain={a.brain}  device={dev}  decide_every={a.decide_every}")

    brain = SpikingBrain.from_npz(a.brain, device=dev)
    ctrl = SpikingController(brain, readout_dim=a.head_dim,
                             readout_mode=a.readout,
                             n_wide=a.n_wide, wide_per=a.wide_per)
    cfg = EnvConfig(n_envs=1, seed=0, auto_reset=True)
    print(f"  脑: {brain.N:,} 神经元 / {len(brain.indices):,} 突触  "
          f"gain={brain.gain} tonic={brain.tonic}")
    print(f"  读出: {a.readout}"
          + (f" ({a.n_wide} 群 × {a.wide_per} 神经元, EMA α={ctrl.ema_alpha})"
             if a.readout == "wide" else f" ({ctrl.n_groups} 个下行群)"))
    n_params = sum(p.numel() for p in ctrl.parameters())
    print(f"  可学参数: {n_params:,}（对比原 connectome 方案 547,307）")

    if not a.train:
        t0 = time.time()
        sc = evaluate(ctrl, cfg, a.episodes, a.decide_every, a.max_ticks, verbose=True)
        print(f"\n  随机读出: 均分 {sc.mean():.2f}  最高 {sc.max():.0f}  "
              f"（{a.episodes} 局，{time.time()-t0:.0f}s）")
        out = {"tag": tag, "brain": a.brain, "trained": False,
               "mean": float(sc.mean()), "max": float(sc.max()),
               "n_params": n_params, "scores": sc.tolist()}
        (ROOT / "output").mkdir(exist_ok=True)
        (ROOT / "output" / f"{tag}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    # ---- 先监督预训练（行为蒸馏：用**真会玩的 MLP** 当教师）----
    # 为什么必须有这一步：ES 在同分个体之间没有梯度（所有鸟都 0 分、同一帧死），
    # 必须先让读出"会 flap"，才谈得上优化。
    # 为什么必须是 MLP 教师：手写规则教师自己 0 分（活 121 帧），蒸馏它无意义。
    #   已实测 mlp_s0 均分 368 / 最高 1657，且诊断确认 4 维脉冲特征
    #   线性可分度 0.929 —— 信息足够，蒸馏路线可行。
    print(f"\n[1/2] 行为蒸馏（教师：{a.teacher}）")
    teacher = load_teacher(a.teacher, dev)
    ctrl.reset()
    sim = FlappySim(EnvConfig(n_envs=1, seed=0, auto_reset=True))
    feats, labels = [], []
    teacher_scores = []
    for ep in range(a.pretrain_eps):
        st = sim.reset_all()
        ctrl.reset()
        for t in range(a.max_ticks):
            s = torch.as_tensor(st[0], dtype=torch.float32, device=dev)
            ctrl.advance(s, n_ticks=1)
            f = ctrl.read_features()
            want = teacher_action(teacher, st[0], dev)
            feats.append(f.cpu())
            labels.append(want)
            st, r, d, info = sim.step(np.array([want]))
            if d[0]:
                teacher_scores.append(int(info["terminal_scores"][0]))
                break
    ts = teacher_scores if teacher_scores else [0]
    F = torch.stack(feats).to(dev)
    Y = torch.tensor(labels, dtype=torch.long, device=dev)
    print(f"    收集 {len(labels)} 个样本（教师 {a.pretrain_eps} 局均分 "
          f"{np.mean(ts):.1f} 最高 {max(ts)}，flap 占比 {np.mean(labels):.2f}）")
    # ⚠️ 必须做类别加权（2026-10-02 修复）：
    #   flap 只占 7%，用普通 cross_entropy 训练，模型会直接塌缩成"永远 noop"
    #   —— 实测 loss 稳定在 0.2595，正好是 7/93 分布的熵，acc 卡在 0.928
    #   （= 多数类基线），一步也没学会拍翅，实跑 28 帧落地。
    #   给 flap 类加 n_neg/n_pos 的权重，逼模型真正去学"什么时候该拍翅"。
    n_pos = int(Y.sum()); n_neg = int(len(Y) - n_pos)
    w_pos = n_neg / max(n_pos, 1)
    cw = torch.tensor([1.0, w_pos], dtype=torch.float32, device=dev)
    print(f"    类别权重: noop=1.0 flap={w_pos:.1f}")
    opt = torch.optim.Adam(list(ctrl.parameters()), lr=3e-3)

    def report(it, logits, loss):
        pred = logits.argmax(1)
        tp = int(((pred == 1) & (Y == 1)).sum())
        fp = int(((pred == 1) & (Y == 0)).sum())
        fn = int(((pred == 0) & (Y == 1)).sum())
        tn = int(((pred == 0) & (Y == 0)).sum())
        r_pos = tp / max(tp + fn, 1)
        r_neg = tn / max(tn + fp, 1)
        print(f"    iter {it:3d}: loss={float(loss):.4f} "
              f"flap召回={r_pos:.3f} noop召回={r_neg:.3f} "
              f"平衡={0.5*(r_pos+r_neg):.3f} flap预测占比={float((pred==1).float().mean()):.3f}")

    for it in range(500):
        logits = ctrl.q_values_from_feat(F)
        loss = torch.nn.functional.cross_entropy(logits, Y, weight=cw)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0:
            with torch.no_grad():
                report(it, ctrl.q_values_from_feat(F), loss.detach())
    with torch.no_grad():
        lg = ctrl.q_values_from_feat(F)
        report(500, lg, torch.nn.functional.cross_entropy(lg, Y, weight=cw))
    sc = evaluate(ctrl, cfg, 10, a.decide_every, a.max_ticks, verbose=False)
    print(f"    蒸馏后实跑: 均分 {sc.mean():.2f} 最高 {sc.max():.0f}")

    # ---- DAgger：解决行为克隆的"误差累积" ----
    # 行为克隆的根本缺陷：训练数据在**教师轨迹**上采集，但测试时是**自己的策略**
    # 在驾驶。一步偏差 -> 进入训练时没见过的状态 -> 偏差放大 -> 迅速崩盘。
    # 实测：蒸馏后实跑均分仍为 0（存活 33~112 tick，一个管子都没过）。
    # DAgger 的标准解法：让**当前策略**自己跑，在每个访问到的状态上问教师
    # "这里该怎么动"，把 (当前状态特征, 教师动作) 加进训练集再训练。
    # 这样训练分布会逐步对齐到"策略实际会遇到的分布"。
    if a.dagger_rounds > 0:
        print(f"\n[1.5/2] DAgger 在线纠偏（{a.dagger_rounds} 轮）")
        F_all, Y_all = F, Y
        for rd in range(a.dagger_rounds):
            ctrl.reset()
            sim_d = FlappySim(EnvConfig(n_envs=1, seed=1000 + rd, auto_reset=True))
            Fd, Yd = [], []
            for ep in range(a.dagger_eps):
                st = sim_d.reset_all()
                ctrl.reset()
                for _ in range(a.max_ticks):
                    s = torch.as_tensor(st[0], dtype=torch.float32, device=dev)
                    ctrl.advance(s, n_ticks=1)
                    f = ctrl.read_features()
                    want = teacher_action(teacher, st[0], dev)   # 教师标注
                    Fd.append(f.cpu())
                    Yd.append(want)
                    act = ctrl.act(f)                            # 自己驾驶
                    st, r, d, info = sim_d.step(np.array([act]))
                    if d[0]:
                        break
            if not Fd:
                print(f"    轮 {rd}: 没采到样本，跳过")
                break
            Fn = torch.stack(Fd).to(dev)
            Yn = torch.tensor(Yd, dtype=torch.long, device=dev)
            F_all = torch.cat([F_all, Fn])
            Y_all = torch.cat([Y_all, Yn])
            np_, nn_ = int(Y_all.sum()), int(len(Y_all) - Y_all.sum())
            cw = torch.tensor([1.0, nn_ / max(np_, 1)], dtype=torch.float32, device=dev)
            opt = torch.optim.Adam(list(ctrl.parameters()), lr=1e-3)
            for it in range(200):
                logits = ctrl.q_values_from_feat(F_all)
                loss = torch.nn.functional.cross_entropy(logits, Y_all, weight=cw)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                lg = ctrl.q_values_from_feat(Fn)
                pred = lg.argmax(1)
                tp = int(((pred == 1) & (Yn == 1)).sum()); fn = int(((pred == 0) & (Yn == 1)).sum())
                fp = int(((pred == 1) & (Yn == 0)).sum()); tn = int(((pred == 0) & (Yn == 0)).sum())
                rpos = tp / max(tp + fn, 1); rneg = tn / max(tn + fp, 1)
            sc = evaluate(ctrl, cfg, 6, a.decide_every, a.max_ticks,
                          seed=2000 + rd, verbose=False)
            print(f"    轮 {rd}: 新增 {len(Yd)} 样本  flap召回={rpos:.3f} "
                  f"noop召回={rneg:.3f} 平衡={0.5*(rpos+rneg):.3f} | "
                  f"实跑均分 {sc.mean():.2f} 最高 {sc.max():.0f}", flush=True)

    # ---- 再用 ES 微调 ----
    print(f"\n[2/2] ES 微调（pop={a.pop}, gen={a.generations}, sigma={a.sigma}）")
    params = [p for p in ctrl.parameters()]
    base = [p.detach().clone() for p in params]
    best_score, best_params = -1e9, [p.clone() for p in base]
    t0 = time.time()
    for gen in range(a.generations):
        eps = [torch.randn_like(p) for p in params]
        scores = []
        for sign in (+1, -1):
            with torch.no_grad():
                for p, b, e in zip(params, base, eps):
                    p.copy_(b + sign * a.sigma * e)
            sc = evaluate(ctrl, cfg, episodes=5, decide_every=a.decide_every,
                          max_ticks=a.max_ticks, seed=1000 + gen)
            scores.append(sc.mean())
        grad = (scores[0] - scores[1]) / (2 * a.sigma)
        with torch.no_grad():
            for i, (p, b, e) in enumerate(zip(params, base, eps)):
                b.add_(a.lr * grad * e)
                p.copy_(b)
        cur = (scores[0] + scores[1]) / 2
        if cur > best_score:
            best_score = cur
            best_params = [b.clone() for b in base]
        if gen % 5 == 0 or gen == a.generations - 1:
            print(f"  gen {gen:3d}: 训练分 {cur:6.2f}  (best {best_score:.2f})  "
                  f"{time.time()-t0:5.0f}s")
    with torch.no_grad():
        for p, b in zip(params, best_params):
            p.copy_(b)

    ctrl.reset()
    sc = evaluate(ctrl, cfg, a.episodes, a.decide_every, a.max_ticks, verbose=True)
    print(f"\n  训练后: 均分 {sc.mean():.2f}  最高 {sc.max():.0f}  ({a.episodes} 局)")
    out = {"tag": tag, "brain": a.brain, "trained": True,
           "mean": float(sc.mean()), "max": float(sc.max()),
           "n_params": n_params, "scores": sc.tolist(),
           "es": {"pop": a.pop, "gens": a.generations, "sigma": a.sigma, "lr": a.lr}}
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / f"{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"params": [p.detach().cpu() for p in params]},
               ROOT / "checkpoints" / f"{tag}.pt")
    print(f"  -> output/{tag}.json   checkpoints/{tag}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
