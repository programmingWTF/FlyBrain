#!/usr/bin/env python
"""诊断 v2：脉冲特征到底能不能区分 flap / noop？

为什么需要 v2
-------------
v1（diagnose_readout_info.py）报 "4 维脉冲特征线性 0.929 ✅ 信息充足"，
但**多数类基线是 0.926** —— 0.929 只比"永远不 flap"高了 0.3 个百分点。
在 flap 占比仅 7% 的极度不平衡数据上，**准确率是有欺骗性的指标**：
一个永远输出 noop 的模型就能拿 0.926。

v1 的判据 `acc > 0.6` 是绝对阈值，没和基线比 —— 这是个设计错误，
它把一个"几乎无信息"的特征误判成"信息充足"。

v2 用**不会被类别不平衡欺骗的指标**：
  - 每类召回率（flap recall / noop recall）
  - 平衡准确率 balanced accuracy = (flap_recall + noop_recall) / 2
  - ROC AUC（阈值无关）
对照组：18 维原始状态（教师自己看的就是这个，是信息量的上界参考）

判定标准
--------
- 平衡准确率 > 0.65 且 AUC > 0.7  → 特征真有信息，蒸馏有望
- 平衡准确率 ≈ 0.5（等价随机/单类）→ 特征无信息，必须改读出设计

同时把特征缓存到 npz，方便后续反复分析而不用重跑脑（脑很慢）。
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


def metrics(y_true, score_pos, thr=0.5):
    """score_pos: 属于 flap 类的打分（越高越像 flap）。"""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score_pos, dtype=float)
    pred = (s > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    r_pos = tp / max(tp + fn, 1)          # flap 召回
    r_neg = tn / max(tn + fp, 1)          # noop 召回
    bal = 0.5 * (r_pos + r_neg)
    acc = (tp + tn) / max(len(y), 1)
    # AUC（秩和法，无需 sklearn）
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        auc = float("nan")
    else:
        order = np.argsort(s)
        ranks = np.empty(len(s), dtype=float)
        ranks[order] = np.arange(1, len(s) + 1)
        auc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return dict(acc=acc, balanced=bal, recall_flap=r_pos, recall_noop=r_neg,
                auc=auc, tp=tp, fp=fp, fn=fn, tn=tn)


def eval_lin(X, Y, dev, iters=600, lr=0.05):
    """逻辑回归，返回在测试集上的 flap 类概率。"""
    n = len(Y)
    idx = np.random.default_rng(0).permutation(n)
    split = int(n * 0.7)
    tr, te = idx[:split], idx[split:]
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
    return metrics(Y[te], p), Y[te], p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="spiking_circuit")
    ap.add_argument("--teacher", default="mlp_s0")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--cache", default="diag2_cache")
    ap.add_argument("--load-cache", action="store_true",
                    help="直接用缓存的 npz 分析，不重跑脑")
    a = ap.parse_args()
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    cache_p = ROOT / "output" / f"{a.cache}.npz"

    if a.load_cache:
        d = np.load(cache_p)
        Xf, Xs, Y = d["Xf"], d["Xs"], d["Y"]
        print(f"从缓存载入 {len(Y)} 样本（flap 占比 {Y.mean():.3f}）")
    else:
        brain = SpikingBrain.from_npz(a.brain, device=dev)
        ctrl = SpikingController(brain)
        teacher = load_teacher(a.teacher, dev)
        cfg = EnvConfig(n_envs=1, seed=0, auto_reset=True)
        sim = FlappySim(cfg)
        print(f"脑: {brain.N:,} 神经元 | 教师: {a.teacher} | device={dev}", flush=True)
        Xf, Xs, Y, tscores = [], [], [], []
        for ep in range(a.episodes):
            st = sim.reset_all()
            ctrl.reset()
            for _ in range(a.max_steps):
                s = torch.as_tensor(st[0], dtype=torch.float32, device=dev)
                ctrl.advance(s, n_ticks=1)
                Xf.append(ctrl.read_features().cpu().numpy())
                with torch.no_grad():
                    act = int(torch.argmax(teacher(s.view(1, -1)), dim=1).item())
                Xs.append(st[0].copy())
                Y.append(act)
                st, r, d, info = sim.step(np.array([act]))
                if d[0]:
                    tscores.append(int(info["terminal_scores"][0]))
                    break
            print(f"  ep{ep}: 累计 {len(Y)} 步", flush=True)
        Xf, Xs, Y = np.array(Xf), np.array(Xs), np.array(Y)
        ts = tscores if tscores else [0]
        print(f"收集 {len(Y)} 步，教师均分 {np.mean(ts):.1f} 最高 {max(ts)} "
              f"(flap 占比 {Y.mean():.3f})")
        cache_p.parent.mkdir(exist_ok=True)
        np.savez_compressed(cache_p, Xf=Xf, Xs=Xs, Y=Y)
        print(f"缓存 -> {cache_p}")

    print(f"\n多数类基线准确率 = {max(Y.mean(), 1 - Y.mean()):.4f} "
          f"（永远猜 noop 就能拿这个分）\n")
    print(f"{'特征':<16} {'acc':>7} {'平衡acc':>8} {'flap召回':>8} "
          f"{'noop召回':>8} {'AUC':>7}")
    print("-" * 62)
    for X, name in ((Xf, "4维脉冲特征"), (Xs, "18维原始状态")):
        m, _, _ = eval_lin(X, Y, dev)
        print(f"{name:<16} {m['acc']:>7.3f} {m['balanced']:>8.3f} "
              f"{m['recall_flap']:>8.3f} {m['recall_noop']:>8.3f} {m['auc']:>7.3f}")

    mf, _, _ = eval_lin(Xf, Y, dev)
    print(f"\n判定：{'✅ 有信息' if mf['balanced'] > 0.65 and mf['auc'] > 0.7 else '⚠️ 信息不足'} "
          f"（平衡acc {mf['balanced']:.3f}, AUC {mf['auc']:.3f}）")
    print("注：v1 用绝对准确率 > 0.6 判据，在 7% flap 的不平衡数据上会误判；")
    print("    本版改用平衡准确率 + AUC，不被类别不平衡欺骗。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
