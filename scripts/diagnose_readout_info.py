#!/usr/bin/env python
"""阶段 3 诊断：脉冲脑的读出特征**是否携带足够信息**学会玩？

为什么必须先做这个诊断
----------------------
如果 4 个下行神经元群的发放率（ESCAPE/TARGET/LOOM/OTHER）在"该 flap"和
"不该 flap"的状态上**分布完全重叠**，那么无论用什么算法训练读出，都不可能
学会——这是信息论上的天花板，不是优化问题。

本脚本：
1. 用**会玩的 MLP 教师**（checkpoints/mlp_s0.pt）生成专家轨迹
   （专家在每一步会 flap 或 not，这是"正确动作"的 ground truth）
2. 记录每一步脉冲脑的 4 维特征
3. 用**逻辑回归**（线性可分性）和**1-近邻**（非线性可分性）测：
       给定 4 维脉冲特征，能多准地预测专家的动作？
4. 作为对照，同样测用 18 维原始状态能多准

结论解读：
- 若脉冲特征准确率 >> 0.5（随机）：信息够，可以用它学会玩 -> 继续
- 若 ≈ 0.5：信息不足，需要改读出设计（更多群 / 更长时间积分）
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.setrecursionlimit(10000)

from fpv.spiking_brain import SpikingBrain  # noqa: E402
from fpv.spiking_controller import SpikingController  # noqa: E402
from fpv.vendor_flappyrl.config import AgentConfig, EnvConfig  # noqa: E402
from fpv.vendor_flappyrl.networks import RainbowNet  # noqa: E402
from fpv.vendor_flappyrl.sim import FlappySim  # noqa: E402


def load_teacher(tag: str, device) -> RainbowNet:
    cfg = AgentConfig(double=True, dueling=True, noisy=False, per=False,
                      n_step=1, distributional=False)
    net = RainbowNet(cfg).to(device)
    p = torch.load(ROOT / "checkpoints" / f"{tag}.pt", map_location=device,
                   weights_only=False)
    net.load_state_dict(p["online"])
    net.eval()
    return net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="spiking_circuit")
    ap.add_argument("--teacher", default="mlp_s0")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=600,
                    help="每局最多采样多少步（CPU 上要控住，专家能活几千帧）")
    a = ap.parse_args()
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    brain = SpikingBrain.from_npz(a.brain, device=dev)
    ctrl = SpikingController(brain)
    teacher = load_teacher(a.teacher, dev)
    cfg = EnvConfig(n_envs=1, seed=0, auto_reset=True)
    sim = FlappySim(cfg)

    print(f"脑: {brain.N:,} 神经元  |  教师: {a.teacher}")

    # ---- 生成专家轨迹（教师驱动），同时记录脉冲特征 ----
    Xf, Xs, Y, teacher_scores = [], [], [], []
    for ep in range(a.episodes):
        st = sim.reset_all()
        ctrl.reset()
        for t in range(a.max_steps):
            raw = st[0]
            s = torch.as_tensor(raw, dtype=torch.float32, device=dev)
            ctrl.advance(s, n_ticks=1)
            feat = ctrl.read_features()
            with torch.no_grad():
                q = teacher(s.view(1, -1))
                act = int(torch.argmax(q, dim=1).item())
            Xf.append(feat.cpu().numpy())
            Xs.append(raw.copy())
            Y.append(act)
            st, r, d, info = sim.step(np.array([act]))
            if d[0]:
                teacher_scores.append(int(info["terminal_scores"][0]))
                break
        if ep % 5 == 0:
            print(f"  ep{ep}: 累计 {len(Y)} 步  (教师当前局已得 {sim.scores[0]} 分)", flush=True)
    Xf = np.array(Xf); Xs = np.array(Xs); Y = np.array(Y)
    ts = teacher_scores if teacher_scores else [0]
    print(f"收集 {len(Y)} 步样本，教师 {a.episodes} 局均分 {np.mean(ts):.1f} "
          f"最高 {max(ts)}  (flap 占比 {Y.mean():.3f})")

    # ---- 评估：4 维脉冲特征 vs 18 维原始状态的可分性 ----
    n = len(Y)
    idx = np.random.default_rng(0).permutation(n)
    split = int(n * 0.7)
    tr, te = idx[:split], idx[split:]

    def test_separability(X, name):
        # 逻辑回归（在线梯度）
        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        Yt = torch.tensor(Y, dtype=torch.long, device=dev)
        mu, sd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-6
        Xn = (Xt - mu) / sd
        lin = torch.nn.Linear(X.shape[1], 2, device=dev)
        opt = torch.optim.Adam(lin.parameters(), lr=0.05)
        for it in range(400):
            loss = torch.nn.functional.cross_entropy(lin(Xn[tr]), Yt[tr])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc_te = float((lin(Xn[te]).argmax(1) == Yt[te]).float().mean())
            acc_tr = float((lin(Xn[tr]).argmax(1) == Yt[tr]).float().mean())
        # 1-NN（非线性）
        with torch.no_grad():
            d = torch.cdist(Xn[te], Xn[tr])
            nn = d.argmin(1)
            acc_knn = float((Yt[tr][nn] == Yt[te]).float().mean())
        base = max(Y.mean(), 1 - Y.mean())
        print(f"  {name:<22} 线性 {acc_te:.3f}(train {acc_tr:.3f})  1-NN {acc_knn:.3f}  "
              f"[多数类基线 {base:.3f}]")
        return acc_te, acc_knn

    print(f"\n可分性检验（预测教师动作的准确率，70/30 split）:")
    af = test_separability(Xf, "4 维脉冲特征")
    as_ = test_separability(Xs, "18 维原始状态")
    print(f"\n结论：脉冲特征 {'✅ 信息充足' if af[0] > 0.6 or af[1] > 0.65 else '⚠️ 信息不足'} "
          f"（线性 {af[0]:.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
