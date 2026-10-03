#!/usr/bin/env python
"""诊断 v4：感觉编码是不是信息瓶颈？（对比三种编码方案的 AUC）

背景
----
`spiking_controller.encode()` 目前的编码是：
    col = 1 - s[4]        (dx)
    tgt = 1 - 3*|s[6]|    (dy_gap)
    mot = 2*|s[1]|        (vy)
**只用了 18 维状态里的 3 维**，而且同一通道内**所有神经元拿到同一个标量**。
也就是说整个输入被压成 3 个数字 —— 剩下 15 维状态**根本没进脑子**。
作为对照，教师 MLP 看的是全部 18 维。这是严重的信息瓶颈。

本脚本对比三种编码，用"预测教师动作的 AUC / 平衡准确率"来打分：
  A `hand3`  : 现状，3 个手工标量
  B `proj18` : 18 维状态 -> 感觉群体的**固定随机投影**（ReLU）
               （储备池计算的标准做法：脑仍冻结，编码也无参数）
  C `both`   : A + B 叠加

随机投影为什么可能更好：
  - 保留了全部 18 维信息（Johnson-Lindenstrauss：随机投影近似保距）
  - 每个感觉神经元有**各自的**感受野（不再是同一标量），形成真正的群体编码
  - 脑（2.45M 突触）负责把群体编码变成下行指令，这正是"用结构换参数"的本意

用法
----
    python scripts/diag4_encoding.py --device cpu --ticks 1500
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


def load_teacher(tag, device):
    cfg = AgentConfig(double=True, dueling=True, noisy=False, per=False,
                      n_step=1, distributional=False)
    net = RainbowNet(cfg).to(device)
    p = torch.load(ROOT / "checkpoints" / f"{tag}.pt", map_location=device,
                   weights_only=False)
    net.load_state_dict(p["online"])
    net.eval()
    return net


def auc_of(y, score):
    y = np.asarray(y).astype(int); s = np.asarray(score, dtype=float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan"), 0.5
    order = np.argsort(s); ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    auc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    best = 0.5
    for thr in np.unique(np.quantile(s, np.linspace(0.01, 0.99, 50))):
        pred = (s > thr).astype(int)
        rpos = ((pred == 1) & (y == 1)).sum() / max(npos, 1)
        rneg = ((pred == 0) & (y == 0)).sum() / max(nneg, 1)
        best = max(best, 0.5 * (rpos + rneg))
    return float(auc), float(best)


def lin_auc(X, Y, dev, iters=400, lr=0.05):
    n = len(Y)
    idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[: int(n * 0.7)], idx[int(n * 0.7):]
    if len(np.unique(Y[tr])) < 2 or len(np.unique(Y[te])) < 2:
        return float("nan"), float("nan")
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Y, dtype=torch.long, device=dev)
    mu, sd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-6
    Xn = (Xt - mu) / sd
    lin = torch.nn.Linear(X.shape[1], 2, device=dev)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(lin(Xn[tr]), Yt[tr])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.softmax(lin(Xn[te]), dim=1)[:, 1].cpu().numpy()
    return auc_of(Y[te], p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="spiking_circuit")
    ap.add_argument("--teacher", default="mlp_s0")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--injs", default="0.4,1.0")
    ap.add_argument("--state-dim", type=int, default=18)
    a = ap.parse_args()
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    brain = SpikingBrain.from_npz(a.brain, device=dev)
    ctrl = SpikingController(brain)
    teacher = load_teacher(a.teacher, dev)
    cfg = EnvConfig(n_envs=1, seed=0, auto_reset=True)
    sim = FlappySim(cfg)

    sens_idx = torch.cat([v for v in ctrl.ch_idx.values()])
    n_sens = int(sens_idx.numel())
    print(f"感觉神经元 {n_sens} 个 | 脑 {brain.N:,} | device={dev}", flush=True)

    # 固定随机投影：每个感觉神经元一个自己的 18 维感受野
    g = np.random.default_rng(0)
    W = torch.tensor(g.normal(0, 1.0 / np.sqrt(a.state_dim), (n_sens, a.state_dim)),
                     dtype=torch.float32, device=dev)

    def make_stim(s, mode, inj):
        s = s.reshape(-1)
        stim = torch.zeros(brain.N, device=dev)
        if mode in ("hand3", "both"):
            stim = stim + ctrl.encode(s)
        if mode in ("proj18", "both"):
            v = torch.relu(W @ s) * inj
            stim = stim.index_add(0, sens_idx, v)
        return stim * (inj if mode == "hand3" else 1.0)

    print(f"\n{'编码':<8} {'inj':>5} | {'DN率':>7} | {'4维AUC':>7} {'4维bal':>7} "
          f"| {'宽AUC':>7}")
    print("-" * 60)
    g2 = np.random.default_rng(1)
    wide_idx = torch.as_tensor(
        g2.choice(brain.N, size=512, replace=False), device=dev)

    for mode in ("hand3", "proj18", "both"):
        for inj in [float(x) for x in a.injs.split(",")]:
            brain.gain = ctrl.brain.gain
            ctrl.reset()
            st = sim.reset_all()
            F4, FW, Y = [], [], []
            for _ in range(a.ticks):
                s = torch.as_tensor(st[0], dtype=torch.float32, device=dev)
                with torch.no_grad():
                    stim = make_stim(s, mode, inj)
                    brain.step(stim)
                    act = int(torch.argmax(teacher(s.view(1, -1)), dim=1).item())
                    wide = brain.S[wide_idx].float().cpu().numpy()
                F4.append(ctrl.read_features().cpu().numpy())
                FW.append(wide)
                Y.append(act)
                st, r, d, info = sim.step(np.array([act]))
                if d[0]:
                    st = sim.reset_all(); ctrl.reset()
            F4 = np.array(F4); FW = np.array(FW); Y = np.array(Y)
            a4, b4 = lin_auc(F4, Y, dev)
            aw, _ = lin_auc(FW, Y, dev)
            dn_rate = F4.mean(1).mean()
            print(f"{mode:<8} {inj:>5.2f} | {dn_rate:>7.3f} | {a4:>7.3f} {b4:>7.3f} "
                  f"| {aw:>7.3f}", flush=True)

    print("\n对照基线：")
    print("  18 维原始状态本身 -> AUC 0.971 / 平衡 0.710（教师策略的可预测上限）")
    print("  永远猜 noop       -> 平衡 0.500")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
