#!/usr/bin/env python
"""诊断 v3：在**带载**条件下重新标定脉冲脑的工作点。

为什么必须做这个
----------------
`diag2.py` 证明 4 维下行读出**几乎没有信息**（平衡 acc 0.500 / flap 召回 0.000 /
AUC 0.617）。查特征分布发现根因不是"读出错"，而是**动力学饱和**：

    ESCAPE 群发放率 mean=0.654  max=0.718  -> 长期顶在上限
    即 65% 的下行神经元每一 tick 都在发放，发放率被钉死，
    自然编码不了任何状态差异。

而 `tune_dynamics.py` 标定的 gain=3.0/tonic=0.0 是在**无感觉驱动的静息态**下做的
（静息 0% + 区分度 0.611）。接上持续感觉注入后，循环驱动把网络推进了饱和区。
**工作点必须在带载条件下重新标定。**

本脚本扫描 (gain, tonic, inj_scale)，对每个组合同时测两件事：
  1. **饱和度**：下行群平均发放率（想要 5%~30% 的敏感中段，不要 0% 也不要 65%）
  2. **信息量**：用该工作点下的 4 维特征预测教师动作的 AUC / 平衡准确率
     —— 另外附一个 512 维随机群体读出的 AUC 作对照，用来区分
        "是动力学坏了" 还是 "只是读出太窄"

用法
----
    python scripts/diag3_operating_point.py --device cpu --ticks 1200
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


def auc_of(y, score):
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan"), 0.5
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    auc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    # 平衡准确率：取使平衡acc最大的阈值
    best = 0.5
    for thr in np.unique(np.quantile(s, np.linspace(0.01, 0.99, 50))):
        pred = (s > thr).astype(int)
        rpos = ((pred == 1) & (y == 1)).sum() / max(npos, 1)
        rneg = ((pred == 0) & (y == 0)).sum() / max(nneg, 1)
        best = max(best, 0.5 * (rpos + rneg))
    return float(auc), float(best)


def lin_auc(X, Y, dev, iters=300, lr=0.05):
    """逻辑回归 -> 测试集上 flap 概率 -> AUC / 平衡acc。"""
    n = len(Y)
    if n < 50 or len(np.unique(Y)) < 2:
        return float("nan"), float("nan")
    idx = np.random.default_rng(0).permutation(n)
    split = int(n * 0.7)
    tr, te = idx[:split], idx[split:]
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
    ap.add_argument("--ticks", type=int, default=1200)
    ap.add_argument("--wide", type=int, default=512,
                    help="宽读出对照用多少随机神经元")
    ap.add_argument("--gains", default="0.3,0.6,1.0,1.5,2.0,3.0")
    ap.add_argument("--tonics", default="0.0")
    ap.add_argument("--injs", default="0.15,0.4,1.0")
    a = ap.parse_args()
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    brain = SpikingBrain.from_npz(a.brain, device=dev)
    ctrl = SpikingController(brain)
    teacher = load_teacher(a.teacher, dev)
    cfg = EnvConfig(n_envs=1, seed=0, auto_reset=True)
    sim = FlappySim(cfg)

    # 宽读出对照：固定随机神经元子集
    g = np.random.default_rng(0)
    wide_idx = torch.as_tensor(
        g.choice(brain.N, size=min(a.wide, brain.N), replace=False), device=dev)
    dn_idx = torch.cat([v for v in ctrl.grp_idx.values()])

    gains = [float(x) for x in a.gains.split(",")]
    tonics = [float(x) for x in a.tonics.split(",")]
    injs = [float(x) for x in a.injs.split(",")]

    print(f"扫描 {len(gains)*len(tonics)*len(injs)} 个工作点，每个 {a.ticks} tick，"
          f"device={dev}", flush=True)
    print(f"{'gain':>5} {'tonic':>6} {'inj':>5} | {'全脑率':>7} {'DN率':>7} "
          f"{'DNstd':>7} | {'4维AUC':>7} {'4维bal':>7} | {'宽AUC':>7} {'宽bal':>7}")
    print("-" * 84)

    rows = []
    for gain in gains:
        for tonic in tonics:
            for inj in injs:
                brain.gain = gain
                brain.tonic = tonic
                ctrl.inj_scale = inj
                ctrl.reset()
                st = sim.reset_all()
                F4, FW, Y = [], [], []
                pop_rates = []
                for _ in range(a.ticks):
                    s = torch.as_tensor(st[0], dtype=torch.float32, device=dev)
                    ctrl.advance(s, n_ticks=1)
                    with torch.no_grad():
                        act = int(torch.argmax(teacher(s.view(1, -1)), dim=1).item())
                        wide = brain.S[wide_idx].float().mean().item()
                        dn = brain.S[dn_idx].float().mean().item()
                    F4.append(ctrl.read_features().cpu().numpy())
                    FW.append(brain.S[wide_idx].float().cpu().numpy())
                    Y.append(act)
                    pop_rates.append((wide, dn))
                    st, r, d, info = sim.step(np.array([act]))
                    if d[0]:
                        st = sim.reset_all()
                        ctrl.reset()
                F4 = np.array(F4); FW = np.array(FW); Y = np.array(Y)
                pr = np.array(pop_rates)
                a4, b4 = lin_auc(F4, Y, dev)
                aw, bw = lin_auc(FW, Y, dev)
                rows.append(dict(gain=gain, tonic=tonic, inj=inj,
                                 pop=pr[:, 0].mean(), dn=pr[:, 1].mean(),
                                 dnstd=F4.mean(1).std(), a4=a4, b4=b4, aw=aw, bw=bw))
                print(f"{gain:>5.2f} {tonic:>6.2f} {inj:>5.2f} | {pr[:,0].mean():>7.3f} "
                      f"{pr[:,1].mean():>7.3f} {F4.mean(1).std():>7.4f} | "
                      f"{a4:>7.3f} {b4:>7.3f} | {aw:>7.3f} {bw:>7.3f}", flush=True)

    ok = [r for r in rows if not np.isnan(r["a4"])]
    if ok:
        best = max(ok, key=lambda r: r["a4"])
        print(f"\n最佳工作点：gain={best['gain']} tonic={best['tonic']} "
              f"inj={best['inj']}  -> 4维AUC {best['a4']:.3f} 平衡 {best['b4']:.3f}")
        bw = max(ok, key=lambda r: (r["aw"] if not np.isnan(r["aw"]) else -1))
        print(f"宽读出的最佳：gain={bw['gain']} inj={bw['inj']} -> 宽AUC {bw['aw']:.3f}")
    print("\n解读：DN率 接近 0 = 网络死寂；接近 0.6+ = 饱和（都学不到东西）。"
          "目标是 0.05~0.30 的敏感中段。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
