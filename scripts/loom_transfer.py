#!/usr/bin/env python
"""视觉逼近逃避反射 第 1 步：**冻结脑的 LC4 -> DNp01 传递函数**。

为什么先做这个
--------------
FINDINGS.md 判死的是"能不能**学出**一个策略"。这里问的是一个更基本、
而且**不需要任何可学参数**的问题：

    冻结的连接组把上游感觉群的发放率，忠实地中继到下行运动神经元了吗？
    中继曲线是不是"跨过某个阈值才触发"的反射式非线性？

方法：把 LC4（以及 LPLC2）群的发放率**钳制**成已知值 r_in（伯努利抽样），
其余 5.9 万神经元照常按 LIF 演化，然后直接数 DNp01 的脉冲。
r_in 是我们**精确知道**的自变量，所以整条曲线是干净的传递函数测量。

判据（全部相对基线，绝对值无效 —— 见 FINDINGS.md 第四节教训）
------------------------------------------------------------
  Δ      = 刺激期发放率 - 同一次运行的基线期发放率
  z      = Δ / 基线期波动的标准差
  AUC    = 用 DNp01 发放率区分"有逼近 vs 无逼近"的判别力（0.5 = 零信息）
  特异性 = 在全部下行神经元里，DNp01 的响应排在第几

对照组（缺一个都不算数）
------------------------
  real     真实拓扑
  shuffled 目标全局洗牌：保持每个神经元出度 + 权重分布，只毁掉"谁连到谁"
  cut      删掉 LC4->DNp01 的直接边：证明响应确实走这条单跳通路
  ctrl-sens钳制 LC10（小目标追踪，与逼近无关）而不是 LC4：感觉通路特异性

用法
----
    D:/Code/FlyBrain/env/python.exe scripts/loom_transfer.py --quick
    D:/Code/FlyBrain/env/python.exe scripts/loom_transfer.py --sweep --reps 3
    D:/Code/FlyBrain/env/python.exe scripts/loom_transfer.py --controls
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv import looming                      # noqa: E402
from fpv.spiking_brain import SpikingBrain   # noqa: E402

OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

# 带载标定过的工作点（HANDOFF.md：静息态标定不可用）
GAIN, TONIC, INJ_REF = 3.0, 0.0, 0.4


def build_groups(brain):
    g = looming.resolve(brain, looming.LOOM_SENSE + looming.ESCAPE_MOTOR
                        + looming.CONTROL_SENSE)
    meta = pd.read_feather(ROOT / "data" / "fafb_783_meta.feather")
    meta["id"] = meta["fafb_783_id"].astype("int64")
    dn_mask = (meta["super_class"].eq("descending")
               | meta["cell_class"].eq("descending_neuron")).to_numpy()
    dn_ids = set(meta.loc[dn_mask, "id"].to_numpy(np.int64).tolist())
    pos_of = dict(zip([int(x) for x in brain.node_ids], range(brain.N)))
    dn_idx = np.array(sorted(pos_of[i] for i in dn_ids if i in pos_of), dtype=np.int64)
    g["ALL_DN"] = torch.as_tensor(dn_idx, device=brain.device)
    return g


def run_condition(brain, g, rate_in: float, *, clamp_groups=("LC4",),
                  ticks: int = 300, warm: int = 60, seed: int = 0,
                  mode: str = "clamp", dc: float = 0.0):
    """一次运行：先跑 warm 个 tick 基线期，再给 rate_in 的刺激期 ticks 个 tick。

    返回 (基线发放率, 刺激发放率, 刺激脉冲原始计数, DNp01 逐 tick 轨迹,
          基线脉冲原始计数)。窗口长度取 warm==ticks 时两者可直接比。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    brain.reset()
    watch = {k: g[k] for k in list(g) if k != "ALL_DN"}
    watch["ALL_DN"] = g["ALL_DN"]

    idx = torch.cat([g[k] for k in clamp_groups])
    if mode == "clamp":
        clamp_base = (idx, torch.zeros(idx.numel()))
    else:
        clamp_base = None
    stim = None
    if mode == "dc":
        stim = torch.zeros(brain.N, device=brain.device)
        stim[idx] = dc

    def clamp_at(p):
        if clamp_base is None:
            return None
        return (clamp_base[0], torch.full((idx.numel(),), float(p),
                                          device=brain.device))

    def phase(p, n):
        spk = {k: 0 for k in watch}
        traj = np.zeros(n, dtype=np.float64)
        for i in range(n):
            brain.step(stim, clamp=clamp_at(p))
            for k, ix in watch.items():
                spk[k] += int(brain.S[ix].sum())
            traj[i] = float(brain.S[g["DNp01"]].float().mean())
        rate = {k: v / (watch[k].numel() * n) for k, v in spk.items()}
        return rate, spk, traj

    base, base_raw, _ = phase(0.0, warm)
    dur, raw, tr = phase(rate_in, ticks)
    return base, dur, raw, tr, base_raw


def auc_mannwhitney(a, b):
    """P(a > b) 的秩统计量，不受类别不平衡影响（判据必须相对 0.5）。"""
    from scipy.stats import rankdata
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.size == 0 or b.size == 0:
        return 0.5
    r = rankdata(np.concatenate([a, b]))
    ra = r[:a.size].sum()
    u = ra - a.size * (a.size + 1) / 2.0
    return float(u / (a.size * b.size))


def sweep(brain, g, rates, *, reps, ticks, warm, mode, dc=0.0, label="real"):
    rows = []
    for r_in in rates:
        for rep in range(reps):
            t0 = time.time()
            base, dur, raw, traj, base_raw = run_condition(brain, g, r_in, ticks=ticks,
                                                            warm=warm, seed=1000 + rep,
                                                            mode=mode, dc=dc)
            nz = np.flatnonzero(traj)
            lat = float(nz[0] * 20.0) if nz.size else np.nan   # ms，相对刺激起始
            rows.append(dict(
                graph=label, r_in=float(r_in), rep=rep,
                dn_base=base["DNp01"], dn_dur=dur["DNp01"],
                dn_delta=dur["DNp01"] - base["DNp01"],
                dn_spk=raw["DNp01"], dn_spk_base=base_raw["DNp01"],
                latency_ms=lat,
                lc4_base=base["LC4"], lc4_dur=dur["LC4"],
                dn04_dur=dur["DNp04"], all_dn_dur=dur["ALL_DN"],
                brain_spk=sum(raw.values()),
                secs=time.time() - t0,
                traj=traj,
            ))
            print(f"  [{label}] r_in={r_in:4.2f} rep{rep}  DNp01 "
                  f"{base['DNp01']:.4f} -> {dur['DNp01']:.4f}  "
                  f"(脉冲 {raw['DNp01']}/{2*ticks})  "
                  f"LC4实测 {dur['LC4']:.3f}  潜时 {lat:.0f}ms  {time.time()-t0:.1f}s",
                  flush=True)
    return rows


def fit_logistic(sub: pd.DataFrame):
    """P_fire(r_in) = sigmoid(slope*(r_in - r50))，最小二乘拟合阈值与斜率。"""
    grp = sub.groupby("r_in")["dn_spk"]
    xs, ps, ns = [], [], []
    for r_in, s in grp:
        xs.append(float(r_in)); ps.append(float((s > 0).mean())); ns.append(len(s))
    xs = np.asarray(xs); ps = np.asarray(ps); ns = np.asarray(ns, float)
    if ps.min() > 0.05 or ps.max() < 0.95:      # 没跨过 0.5，拟合无意义
        return None
    from scipy.optimize import curve_fit
    f = lambda x, r50, k: 1.0 / (1.0 + np.exp(-k * (x - r50)))
    try:
        popt, _ = curve_fit(f, xs, ps, p0=[0.7, 10.0], sigma=1.0 / np.maximum(ns, 1),
                            maxfev=20000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return None


def summarize(df: pd.DataFrame) -> None:
    print("\n" + "=" * 76)
    print("传递函数汇总（发放率单位：每神经元每 tick；dt=20ms -> ×50 = Hz）")
    print("=" * 76)
    for lab, sub in df.groupby("graph"):
        m = sub.groupby("r_in").agg(dn=("dn_dur", "mean"), dl=("dn_delta", "mean"),
                                    lc4=("lc4_dur", "mean"),
                                    dn04=("dn04_dur", "mean"),
                                    alldn=("all_dn_dur", "mean")).reset_index()
        b = float(m["dn"].iloc[0])
        sd = float(np.nanstd(sub.loc[sub.r_in == 0.0, "dn_dur"].to_numpy()))
        sd = max(sd, 1e-9)
        m["d_hz"] = (m["dn"] - b) * 50.0          # 相对基线的增量，单位 Hz
        m["z_vs_base"] = np.where(m["d_hz"] > 0, (m["dn"] - b) / sd, 0.0)
        print(f"\n--- {lab}  (基线 DNp01={b:.4f}/tick；基线窗内 0 脉冲 -> "
              f"z 退化，主看 ΔHz 与 P_fire) ---")
        print(f"{'r_in':>6}{'DNp01率/tick':>14}{'ΔHz':>8}{'DNp04率':>10}"
              f"{'LC4实测':>10}{'全部DN率':>11}")
        for _, r in m.iterrows():
            print(f"{r.r_in:6.2f}{r.dn:14.5f}{r.d_hz:8.2f}{r.dn04:10.4f}"
                  f"{r.lc4:10.3f}{r.alldn:11.6f}")
        rho = pd.Series(m["dn"]).corr(pd.Series(m["r_in"]), method="spearman")
        lo = sub.loc[sub.r_in <= 0.05, "dn_dur"].to_numpy()
        hi = sub.loc[sub.r_in >= 0.8, "dn_dur"].to_numpy()
        print(f"  Spearman(r_in, DNp01率) = {rho:.3f}   "
              f"AUC(高逼近 vs 无逼近) = {auc_mannwhitney(hi, lo):.3f}")
        # --- P_fire：把"2 个神经元的有/无脉冲"当成反射的检出问题 ---
        # 行为学里对应"给了这个逼近刺激，逃还是不逃"，可以直接和文献的概率曲线比。
        pb = float((sub["dn_spk_base"] > 0).mean())
        print(f"  {'r_in':>6}{'P_fire':>9}{'n':>4}   "
              f"（DNp01 在等长基线窗内至少放 1 个脉冲的试次比例；基线 P_fire={pb:.2f}）")
        for r_in, grp in sub.groupby("r_in"):
            fired = float((grp["dn_spk"] > 0).mean())
            print(f"  {r_in:6.2f}{fired:9.2f}{len(grp):>4}")
        pfit = fit_logistic(sub)
        if pfit is not None:
            r50, slope = pfit
            print(f"  logistic 拟合：P_fire=0.5 处 r_in = {r50:.3f}"
                  f"（即每只 LC4 发放率 {r50*50:.1f} Hz），斜率 {slope:.2f}")


def rank_dns(brain, g, meta_ct, *, r_in=1.0, ticks=400, seed=0,
             clamp_groups=("LC4",), meta_side=None):
    """全部下行神经元的逐个刺激期/基线期发放率 -> DNp01 排第几。

    这是**特异性**检验：如果 LC4 的驱动只是"把脑子点亮"，那 DN 之间的响应
    应该没有系统差别；只有当 DNp01（文献里的逃逸指令神经元）显著排在前面，
    才能说"这条通路在连接组里确实是指向逃逸的"。
    """
    torch.manual_seed(seed); np.random.seed(seed)
    brain.reset()
    idx = g["ALL_DN"]
    n = idx.numel()
    acc = torch.zeros(n, device=brain.device)

    cidx = torch.cat([g[k] for k in clamp_groups])
    for phase_n, p in ((ticks, 0.0), (ticks, r_in)):
        acc.zero_()
        for i in range(phase_n):
            pr = torch.full((cidx.numel(),), float(p), device=brain.device)
            brain.step(clamp=(cidx, pr))
            acc += brain.S[idx].float()
        if p == 0.0:
            base = acc.detach().cpu().numpy() / phase_n
        else:
            dur = acc.detach().cpu().numpy() / phase_n
    df = pd.DataFrame({
        "asset_idx": idx.detach().cpu().numpy(),
        "base_rate": base, "stim_rate": dur, "delta": dur - base,
    })
    ids = np.asarray(brain.node_ids, dtype=np.int64)
    df["fafb_id"] = ids[df["asset_idx"].to_numpy()]
    # 必须用 map：reindex(Series) 会把结果的 index 设成传入的标签，
    # 回写时按 index 对齐 -> 整列变成 NaN（踩过一次了）。
    df["cell_type"] = df["fafb_id"].map(meta_ct).fillna("?")
    if meta_side is not None:
        df["side"] = df["fafb_id"].map(meta_side).fillna("?")
    df = df.sort_values("delta", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--asset", default="spiking_circuit",
                    help="spiking_circuit=旧资产(min-count 3 + BFS r3 截断)；"
                         "spiking_full=全脑未截断(144,837 神经元/1502 万边)")
    ap.add_argument("--quick", action="store_true", help="只测几个点，验证能跑")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--warm", type=int, default=0, help="0 = 与 --ticks 等长（可直接比脉冲数）")
    ap.add_argument("--rates", default="", help="逗号分隔的 r_in 网格，留空用默认")
    ap.add_argument("--mode", default="clamp", choices=["clamp", "dc"])
    ap.add_argument("--dc", type=float, default=0.5)
    ap.add_argument("--rank", action="store_true",
                    help="特异性检验：全部下行神经元逐个排序，看 DNp01 排第几")
    ap.add_argument("--rank-rate", type=float, default=1.0)
    a = ap.parse_args()
    if a.warm <= 0:
        a.warm = a.ticks

    brain = SpikingBrain.from_npz(a.asset, device=a.device)
    sfx = "" if a.asset == "spiking_circuit" else f"_{a.asset}"
    brain.gain, brain.tonic = GAIN, TONIC
    g = build_groups(brain)
    print(f"资产 N={brain.N:,}  E={len(brain.codes):,}")
    print("精确类型群：" + "  ".join(f"{k}={g[k].numel()}" for k in g if k != "ALL_DN")
          + f"  ALL_DN={g['ALL_DN'].numel()}")

    if a.rank:
        meta = pd.read_feather(ROOT / "data" / "fafb_783_meta.feather")
        meta["id"] = meta["fafb_783_id"].astype("int64")
        meta["cell_type"] = meta["cell_type"].fillna("")
        meta["side"] = meta["side"].fillna("?")
        ct = pd.Series(meta["cell_type"].to_numpy(), index=meta["id"].to_numpy())
        side = pd.Series(meta["side"].to_numpy(), index=meta["id"].to_numpy())
        variants = [("real", brain)]
        if a.controls:
            variants += [("cut", looming.variant(brain, cut=(g["LC4"], g["DNp01"]))),
                         ("shuffled", looming.variant(brain, shuffle_seed=7)),
                         ("ctrl_lc10a", brain)]
        for lab, br in variants:
            cg = ("LC10a",) if lab == "ctrl_lc10a" else ("LC4",)
            df = rank_dns(br, g, ct, r_in=a.rank_rate, ticks=a.ticks, clamp_groups=cg,
                          meta_side=side)
            df.to_csv(OUT / f"loom_dn_rank_{lab}{sfx}.csv", index=False)
            n = len(df)
            cols = ["rank", "cell_type", "side", "base_rate", "stim_rate", "delta"]
            print(f"\n=== {lab}：钳制 {'LC4' if cg[0]=='LC4' else 'LC10a'} 到 "
                  f"r_in={a.rank_rate:.2f} 时，{n} 个下行神经元的响应排名 Top 15 ===")
            print(df.head(15)[cols].to_string(index=False,
                  float_format=lambda x: f"{x:.5f}"))
            for nm in ("DNp01", "DNp04"):
                r = df[df["cell_type"] == nm]
                if len(r):
                    print(f"  >>> {nm}: 排名 {int(r['rank'].min())}~{int(r['rank'].max())}"
                          f" / {n}   Δ率 {r['delta'].max():.5f}/tick "
                          f"({r['delta'].max()*50:.2f} Hz)")
            n_resp = int((df["delta"] > 0.001).sum())
            print(f"  响应显著（Δ>0.001/tick）的 DN 数：{n_resp}/{n}")
        return 0

    if a.rates:
        rates = [float(x) for x in a.rates.split(",") if x != ""]
    elif a.quick:
        rates = [0.0, 0.3, 0.6, 1.0]
    else:
        rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    all_rows = []
    print("\n=== 真实拓扑：钳制 LC4 ===")
    all_rows += sweep(brain, g, rates, reps=a.reps, ticks=a.ticks, warm=a.warm,
                      mode=a.mode, dc=a.dc, label="real")

    if a.controls:
        print("\n=== 对照 1：钳制 LC4，但目标全局洗牌（毁掉真实拓扑）===")
        sh = looming.variant(brain, shuffle_seed=7)
        all_rows += sweep(sh, g, rates, reps=a.reps, ticks=a.ticks, warm=a.warm,
                          mode=a.mode, dc=a.dc, label="shuffled")
        print("\n=== 对照 2：钳制 LC4，但删掉 LC4->DNp01 直接边 ===")
        cu = looming.variant(brain, cut=(g["LC4"], g["DNp01"]))
        all_rows += sweep(cu, g, rates, reps=a.reps, ticks=a.ticks, warm=a.warm,
                          mode=a.mode, dc=a.dc, label="cut_lc4_dn01")
        print("\n=== 对照 3：钳制 LC10（非逼近感觉群），真实拓扑 ===")
        for r_in in rates:
            base, dur, raw, traj, base_raw = run_condition(
                brain, g, r_in, clamp_groups=("LC10a",),
                ticks=a.ticks, warm=a.warm, seed=1000, mode=a.mode, dc=a.dc)
            all_rows.append(dict(graph="ctrl_lc10a", r_in=r_in, rep=0,
                                 dn_base=base["DNp01"], dn_dur=dur["DNp01"],
                                 dn_delta=dur["DNp01"] - base["DNp01"],
                                 dn_spk=raw["DNp01"], dn_spk_base=base_raw["DNp01"],
                                 lc4_base=base["LC4"], lc4_dur=dur["LC4"],
                                 dn04_dur=dur["DNp04"], all_dn_dur=dur["ALL_DN"],
                                 secs=0.0, traj=traj))
            print(f"  [ctrl_lc10a] r_in={r_in:4.2f}  DNp01 "
                  f"{base['DNp01']:.4f} -> {dur['DNp01']:.4f}")

    df = pd.DataFrame(all_rows)
    df.drop(columns=["traj"]).to_csv(OUT / f"loom_transfer{sfx}.csv", index=False)
    np.savez_compressed(OUT / f"loom_transfer_traj{sfx}.npz",
                        **{f"{r['graph']}_{r['r_in']:.2f}_{r['rep']}": r["traj"]
                           for r in all_rows})
    summarize(df)
    print(f"\n✓ {OUT/f'loom_transfer{sfx}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
