#!/usr/bin/env python
"""视觉逼近逃避反射 第 2 步：**逼近刺激 -> DNp01 的触发时刻**（用视觉坐标表达）。

第 1 步（scripts/loom_transfer.py）量到的是抽象的"输入率 -> 输出率"传递函数：
DNp01 的阈值 r50 ≈ 0.58 满量程 ≈ 每只 LC4 约 29 Hz，且对照干净
（切直接边 / 打乱拓扑 / 换感觉群 LC10a 后 DNp01 全为 0 脉冲）。
这一步把自变量换回**真实刺激参数**：一个正前方逼近的暗盘，逐 tick 算
角直径 θ(t)、角扩张速度 dθ/dt(t)、剩余碰撞时间 τ(t)，据此驱动 LC4，
看 DNp01 在**哪个视觉角 / 还剩多少 τ** 上首次发放。

于是有三个能和文献对表的量：
  θ_esc      触发时的角直径        —— 行为学的"逃逸触发角"
  τ_esc      触发时的剩余碰撞时间  —— "逃逸用的时间预算"
  Δ_brain    越过 LC4 阈值 -> DNp01 首脉冲的延迟
             这个量**只属于脑**（连接组的积分+阈值），不是刺激给的

⚠️ 分解（本实验唯一的外部假设）
  刺激参数 --(LC4 调谐曲线 looming_drive)--> 发放率 --(冻结脑 纯测量)--> DNp01
  LC4 的调谐形状/灵敏度我们并不知道，所以把它当**显式扫描的参数**
  （--sens），报出结论对它的依赖，而不是假装知道。
  「静止/远离对照」由调谐曲线的构造决定（dθ/dt≤0 -> 驱动 0），检验的是刺激模型；
  真正检验**脑**的对照是 cut / shuffled / 换感觉群 LC10a。

用法
----
    D:/Code/FlyBrain/env/python.exe scripts/loom_reflex.py --controls
    D:/Code/FlyBrain/env/python.exe scripts/loom_reflex.py --source theta --sens 20,35,45
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv import looming                      # noqa: E402
from fpv.spiking_brain import SpikingBrain   # noqa: E402

OUT = ROOT / "output"
GAIN, TONIC = 3.0, 0.0          # 带载标定过的工作点（HANDOFF.md）
R50 = 0.577                     # 第 1 步量到的 DNp01 激发阈值（LC4 群发放率）


def first_spike(spikes: np.ndarray, pre: int):
    """DNp01（2 个细胞）在刺激期内首次发放的 tick；返回 (tick, 各细胞脉冲数)。"""
    post = spikes[pre:]
    fired = post.sum(axis=1) > 0
    per_cell = post.sum(axis=0)
    if not fired.any():
        return None, per_cell
    return int(np.flatnonzero(fired)[0]), per_cell


def run_grid(brain, g, speeds, radius, start, max_ticks, pre, reps, *,
             tunes: list[tuple[str, str, float, float]],
             watch=("DNp01",), label="real"):
    """tunes: [(驱动群, 调谐变量, s50, n), ...]。

    支持**双通道**驱动：文献里 GF(DNp01) 的视叶输入 = LC4 的角速度通道
    + LPLC2 的角大小通道（Ache 2019 / von Reyn 2017 / Gaitanidis 2025），
    所以这里让 LC4 走 dtheta、LPLC2 走 theta，两路各自钳制各自的群。
    """
    rows = []
    for v in speeds:
        kin = looming.loom_kinematics(radius, v, start, max_ticks)
        ticks = len(kin)
        series, cross = [], None
        for name, src, s50, nn in tunes:
            p_stim = looming.looming_drive(kin, s50=s50, n=nn, source=src)
            series.append((name, np.concatenate([np.zeros(pre), p_stim])))
            above = np.flatnonzero(p_stim >= R50)
            if above.size:
                cross = int(above[0]) if cross is None else min(cross, int(above[0]))
        widx = torch.cat([g[k] for k in watch])
        for rep in range(reps):
            inputs = [(g[name], p) for name, p in series]
            spikes = looming.run_relay(brain, inputs, widx, seed=7000 + rep)
            k, per_cell = first_spike(spikes, pre)
            row = dict(graph=label,
                       drive="+".join(t[0] for t in tunes),
                       source="+".join(t[1] for t in tunes),
                       sens="/".join(f"{t[2]:g}" for t in tunes),
                       radius_m=radius, speed_ms=v, start_m=start,
                       stim_ticks=ticks, rep=rep, fired=k is not None,
                       esc_tick=k, esc_ms=None if k is None else (k + 1) * 20.0,
                       cross_tick=cross,
                       brain_ms=None if (k is None or cross is None)
                       else (k - cross) * 20.0,
                       dn_spk_base=int(spikes[:pre].sum()),
                       dn_spk_total=int(per_cell.sum()))
            if k is not None:
                row.update(theta_deg=float(kin["theta_deg"].iloc[k]),
                           dtheta_dps=float(kin["dtheta_dps"].iloc[k]),
                           tau_s=float(kin["tau_s"].iloc[k]),
                           dist_m=float(kin["dist_m"].iloc[k]))
            rows.append(row)
    return rows


def report(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg = df.groupby(["graph", "source", "sens", "speed_ms"], sort=True).agg(
        p_fire=("fired", "mean"), n=("fired", "size"),
        theta_deg=("theta_deg", "mean"), theta_sd=("theta_deg", "std"),
        dtheta_dps=("dtheta_dps", "mean"), tau_s=("tau_s", "mean"),
        lat_ms=("esc_ms", "mean"), brain_ms=("brain_ms", "mean"),
        spk_stim=("dn_spk_total", "mean"), spk_base=("dn_spk_base", "mean"),
    ).reset_index()
    pd.set_option("display.width", 240)
    print("\n" + "=" * 110)
    print("DNp01 逃避触发点。P_fire=激发概率(≈行为学逃逸概率)；spk_base 必须为 0；"
          "brain_ms = 驱动越过 r50 到 DNp01 首脉冲的延迟（只属于连接组）")
    print("=" * 110)
    cols = ["graph", "source", "sens", "speed_ms", "p_fire", "n", "theta_deg",
            "dtheta_dps", "tau_s", "lat_ms", "brain_ms", "spk_stim", "spk_base"]
    print(agg[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}",
                              na_rep="   -"))
    print("\n标度律（P_fire>=0.5 的点，log-log 斜率；0 = 与速度无关）")
    # 必须按 (图, s50) 分组：只按 graph 分会把不同 LC4 灵敏度的点混进同一次拟合
    for (lab, sens), sub0 in agg.groupby(["graph", "sens"]):
        sub = sub0[sub0.p_fire >= 0.5]
        if len(sub) < 3:
            continue
        x = np.log(sub["speed_ms"].to_numpy(float))
        parts = []
        for nm, col in (("θ_esc", "theta_deg"), ("τ_esc", "tau_s")):
            y = np.log(sub[col].to_numpy(float))
            sl = float(np.polyfit(x, y, 1)[0])
            parts.append(f"{nm}∝v^({sl:+.2f})")
        print(f"  {lab:<24} s50={str(sens):<9}: " + "  ".join(parts)
              + f"   brain_ms 均值 {sub['brain_ms'].mean():.0f}"
              f"   θ_esc {sub['theta_deg'].min():.0f}~{sub['theta_deg'].max():.0f}°"
              f"   (n={len(sub)})")
    df.to_csv(OUT / "loom_reflex.csv", index=False)
    agg.to_csv(OUT / "loom_reflex_agg.csv", index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--radius", type=float, default=0.05, help="暗盘半径(米)")
    ap.add_argument("--start", type=float, default=1.0, help="起始距离(米)")
    ap.add_argument("--speeds", default="0.2,0.4,0.7,1.0,1.5,2.0")
    ap.add_argument("--max-ticks", type=int, default=250, help="刺激最长 tick 数")
    ap.add_argument("--pre", type=int, default=150, help="刺激前的基线 tick 数")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--sources", default="dtheta,theta")
    ap.add_argument("--sens", default="15,30,60,120",
                    help="LC4 半饱和常数列表（dtheta 单位 度/秒；theta 单位 度）")
    ap.add_argument("--n", type=float, default=3.0)
    ap.add_argument("--codrive", action="store_true",
                    help="加跑文献的双通道条件：LC4(角速度)+LPLC2(角大小)，以及只驱动 LPLC2")
    ap.add_argument("--lplc2-s50", type=float, default=30.0,
                    help="LPLC2 角大小通道的半饱和角(度)")
    ap.add_argument("--asset", default="spiking_circuit")
    ap.add_argument("--controls", action="store_true")
    a = ap.parse_args()

    brain = SpikingBrain.from_npz(a.asset, device=a.device)
    brain.gain, brain.tonic = GAIN, TONIC
    g = looming.resolve(brain, looming.LOOM_SENSE + looming.ESCAPE_MOTOR
                        + looming.CONTROL_SENSE)
    print(f"资产 {a.asset}  N={brain.N:,}  "
          + "  ".join(f"{k}={g[k].numel()}" for k in g)
          + f"\n第 1 步阈值 r50={R50}（= 每只 LC4 {R50*50:.1f} Hz）")
    speeds = [float(x) for x in a.speeds.split(",")]
    sources = a.sources.split(",")
    sens_list = [float(x) for x in a.sens.split(",")]
    common = dict(radius=a.radius, start=a.start, max_ticks=a.max_ticks,
                  pre=a.pre, reps=a.reps)

    rows = []
    for src in sources:
        # theta 码与角速度码的合理 s50 差 1~2 个数量级，各自缩一下
        mult = 1.0 if src != "theta" else 0.3
        for s in sens_list:
            rows += run_grid(brain, g, speeds, tunes=[("LC4", src, s * mult, a.n)],
                             label=f"real[{src}]", **common)
    if a.codrive:
        for s in sens_list:
            rows += run_grid(brain, g, speeds,
                             tunes=[("LC4", "dtheta", s, a.n),
                                    ("LPLC2", "theta", a.lplc2_s50, a.n)],
                             label=f"2ch(LC4+LPLC2) s50={s:g}", **common)
        rows += run_grid(brain, g, speeds,
                         tunes=[("LPLC2", "theta", a.lplc2_s50, a.n)],
                         label="only[LPLC2]", **common)
    if a.controls:
        s = sens_list[1]
        t1 = [("LC4", "dtheta", s, a.n)]
        print(f"\n--- 对照（LC4 角速度通道, s50={s}）：切边 / 打乱拓扑 / 换感觉群 ---")
        cu = looming.variant(brain, cut=(g["LC4"], g["DNp01"]))
        rows += run_grid(cu, g, speeds, tunes=t1, label="cut", **common)
        sh = looming.variant(brain, shuffle_seed=7)
        rows += run_grid(sh, g, speeds, tunes=t1, label="shuffled", **common)
        rows += run_grid(brain, g, speeds, tunes=[("LC10a", "dtheta", s, a.n)],
                         label="ctrl_lc10a", **common)

    df = report(rows)
    (OUT / "loom_reflex_summary.json").write_text(json.dumps({
        "params": vars(a), "operating_point": {"gain": GAIN, "tonic": TONIC},
        "r50_from_transfer": R50, "trials": int(len(df)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ {OUT/'loom_reflex.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
