#!/usr/bin/env python
"""诊断 v5：读出宽度扫描 —— 是不是"读出的神经元太少"在限制信息量？

动机
----
前面几个诊断里，4 维读出（ESCAPE/TARGET/LOOM/OTHER 群）的 AUC 一直在
0.55~0.70 徘徊，而 18 维原始状态是 0.971。同时观察到一个反常现象：
**宽读出（512 个随机神经元）反而更好**（diag4 里 proj18@1.0 宽 AUC 0.743
> 4 维 0.579）。

怀疑根因是**读出群体太小、率估计全是噪声**：
    ESCAPE 群 = DNp01 + DNp04，DNp01 **全脑只有 2 个**（左右各 1）
    从 2 个神经元估计发放率，再 EMA 平滑，本质上是二值噪声，
    根本不可能支撑 FlappyBird 这种需要精确时机的决策。

本脚本把"读出宽度"作为唯一变量，画出信息量-宽度曲线：
    width ∈ {4(现状), 8, 32, 128, 512, 2048, 8192}
并顺带扫 EMA 平滑系数 α ∈ {0.05, 0.14, 0.30}

做法：只跑一次脑（慢的部分），把一段轨迹里**所有池内神经元的逐 tick 发放**
记下来，之后所有宽度/α 的组合都在这份缓存上离线评估（快）。

用法
----
    python scripts/diag5_width.py --device cpu --ticks 3000
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


def ema(X, alpha):
    """沿时间轴做 EMA。X: (T, D) -> (T, D)"""
    out = np.empty_like(X, dtype=np.float32)
    acc = X[0].astype(np.float32)
    out[0] = acc
    for t in range(1, len(X)):
        acc = acc * (1 - alpha) + X[t].astype(np.float32) * alpha
        out[t] = acc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="spiking_circuit")
    ap.add_argument("--teacher", default="mlp_s0")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--pool", type=int, default=16384,
                    help="记录多少个神经元的逐 tick 发放")
    ap.add_argument("--encoding", default="hand3", choices=["hand3", "proj18", "both"])
    ap.add_argument("--inj", type=float, default=0.4)
    ap.add_argument("--widths", default="4,8,32,128,512,2048,8192")
    ap.add_argument("--alphas", default="0.05,0.14,0.30")
    a = ap.parse_args()
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    brain = SpikingBrain.from_npz(a.brain, device=dev)
    ctrl = SpikingController(brain)
    teacher = load_teacher(a.teacher, dev)
    cfg = EnvConfig(n_envs=1, seed=0, auto_reset=True)
    sim = FlappySim(cfg)

    # ---- 池：全部下行 + 全部感觉 + 随机补充 ----
    dn_idx = torch.cat([v for v in ctrl.grp_idx.values()])
    sens_idx = torch.cat([v for v in ctrl.ch_idx.values()])
    g = np.random.default_rng(0)
    rand_idx = torch.as_tensor(
        g.choice(brain.N, size=min(a.pool, brain.N), replace=False), device=dev)
    pool = torch.unique(torch.cat([dn_idx, sens_idx, rand_idx]))
    pool_np = pool.cpu().numpy()
    print(f"池大小 {len(pool_np)}（含 {len(dn_idx)} 下行 + {len(sens_idx)} 感觉）"
          f" | 脑 {brain.N:,} | device={dev}", flush=True)

    # 池内下标映射（用于离线取"4 维群"和"全部下行"）
    pos = {int(v): i for i, v in enumerate(pool_np)}
    dn_pos = np.array([pos[int(v)] for v in dn_idx.cpu().numpy()])
    grp_pos = {k: np.array([pos[int(v)] for v in v_.cpu().numpy()])
               for k, v_ in ctrl.grp_idx.items()}

    # 随机投影编码（如果需要）
    W = None
    if a.encoding in ("proj18", "both"):
        W = torch.tensor(
            np.random.default_rng(0).normal(0, 1 / np.sqrt(18),
                                            (int(sens_idx.numel()), 18)),
            dtype=torch.float32, device=dev)

    # ---- 跑一次脑，记录逐 tick 发放 ----
    ctrl.reset()
    st = sim.reset_all()
    spikes = np.zeros((a.ticks, len(pool_np)), dtype=np.uint8)
    Y = []
    for t in range(a.ticks):
        s = torch.as_tensor(st[0], dtype=torch.float32, device=dev)
        with torch.no_grad():
            stim = ctrl.encode(s)
            if a.encoding in ("proj18", "both"):
                v = torch.relu(W @ s.reshape(-1)) * a.inj
                stim = stim.index_add(0, sens_idx, v)
            if a.encoding == "hand3":
                stim = stim * a.inj
            brain.step(stim)
            act = int(torch.argmax(teacher(s.view(1, -1)), dim=1).item())
            spikes[t] = brain.S[pool].to(torch.uint8).cpu().numpy()
        Y.append(act)
        st, r, d, info = sim.step(np.array([act]))
        if d[0]:
            st = sim.reset_all(); ctrl.reset()
    Y = np.array(Y)
    print(f"记录 {a.ticks} tick × {len(pool_np)} 神经元，flap 占比 {Y.mean():.3f}",
          flush=True)

    # ---- 离线评估：宽度 × α ----
    print(f"\n{'宽度':>8} {'α':>5} | {'AUC':>7} {'平衡acc':>7}")
    print("-" * 34)

    def feats_for(width, alpha):
        if width == 4:
            cols = [grp_pos[k] for k in sorted(grp_pos)]
            X = np.stack([spikes[:, c].mean(1) for c in cols], axis=1)
        elif width == "dn":
            X = spikes[:, dn_pos].mean(1, keepdims=True)
        else:
            sub = g.choice(len(pool_np), size=min(width, len(pool_np)), replace=False)
            X = spikes[:, sub].mean(1, keepdims=False)
            # 用多个随机组的均值做特征（更稳）
            ngrp = min(width, 64)
            groups = np.array_split(sub, ngrp)
            X = np.stack([spikes[:, gg].mean(1) for gg in groups], axis=1)
        return ema(X.astype(np.float32), alpha)

    results = []
    for width in ([int(x) for x in a.widths.split(",")] + ["dn"]):
        for alpha in [float(x) for x in a.alphas.split(",")]:
            X = feats_for(width, alpha)
            au, ba = lin_auc(X, Y, dev)
            label = str(width)
            results.append((label, alpha, au, ba))
            print(f"{label:>8} {alpha:>5.2f} | {au:>7.3f} {ba:>7.3f}", flush=True)

    best = max(results, key=lambda r: (r[2] if not np.isnan(r[2]) else -1))
    print(f"\n最佳：宽度={best[0]} α={best[1]} -> AUC {best[2]:.3f} "
          f"平衡 {best[3]:.3f}")
    print("\n对照：18 维原始状态 AUC 0.971 / 平衡 0.710；永远猜 noop 平衡 0.500")
    print("判据：若宽读出能把 AUC 推到 >0.85，说明【读出太窄】确实是瓶颈，"
          "放宽即可；若宽度加到大几千仍停在 0.7 附近，说明是脑动力学本身"
          "在毁信息，需要换方向。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
