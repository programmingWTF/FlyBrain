"""视觉逼近逃避反射：神经元群解析 + 冻结脑的通路工具。

和 spiking_controller.py 的关系
------------------------------
spiking_controller 是给 FlappyBird 用的（多通道感觉编码 + 可学读出），
那条路已经判定是死路（见 FINDINGS.md）。本模块服务一个新问题：

    **冻结的连接组本身，能不能把 LC4 的逼近信号中继到下行神经元 DNp01，
     并且表现出"跨过阈值才触发"的反射非线性？**

这里**没有任何可学参数** —— 直接观察神经元的放电。所以 FlappyBird 那套
"读出学不会"的问题在这里不存在，我们测的是脑本身的传递特性。

⚠️ 类型学修正（重要）
--------------------
`extract_key_types.py` 用的是 `cell_type.startswith(prefix)`，于是：
    "LC4" 前缀匹配 207 个 —— 里面混了 LC40(37) / LC45(24) / LC46(14) /
                            LC41(12) / LC43(12) / LC44(4)
    精确 `cell_type == "LC4"` 只有 **104 个**（全部胆碱能，左 54 / 右 50）
本模块一律用**精确**类型名，不用前缀。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"

# 精确 cell_type 名（不是前缀）
LOOM_SENSE = ["LC4", "LPLC2", "LPLC1"]
ESCAPE_MOTOR = ["DNp01", "DNp04"]
# 对照组：与逼近无关的感觉群 / 与下行无关的中间神经元
# （注意：精确类型名是 LC10a/LC10b/...，没有裸的 "LC10"）
CONTROL_SENSE = ["LC10a"]


def resolve(brain, names: list[str]) -> dict[str, torch.Tensor]:
    """精确 cell_type 名 -> 冻结资产里的子图索引（张量）。"""
    meta = pd.read_feather(DATA / "fafb_783_meta.feather")
    meta["id"] = meta["fafb_783_id"].astype("int64")
    meta["cell_type"] = meta["cell_type"].fillna("")
    pos = pd.Series(np.arange(brain.N), index=np.asarray(brain.node_ids, dtype=np.int64))
    out: dict[str, torch.Tensor] = {}
    for n in names:
        ids = meta.loc[meta["cell_type"] == n, "id"].to_numpy(np.int64)
        hit = pos.reindex(pd.Index(ids)).to_numpy()
        ok = hit[~pd.isna(hit)].astype(np.int64)
        if len(ok) == 0:
            raise KeyError(f"{n} 在冻结资产里一个都没有（原始 {len(ids)} 个）")
        out[n] = torch.as_tensor(np.sort(ok), dtype=torch.long, device=brain.device)
    return out


def edge_table(brain, pre_idx: torch.Tensor, post_idx: torch.Tensor) -> pd.DataFrame:
    """两组之间的直接边：(pre, post, w)；w 是解码后的带符号权重。"""
    lut = brain.lut.detach().cpu().numpy()
    indptr = brain.indptr.detach().cpu().numpy()
    indices = brain.indices.detach().cpu().numpy()
    codes = brain.codes.detach().cpu().numpy()
    post_np = post_idx.detach().cpu().numpy()
    rows = []
    for s in pre_idx.detach().cpu().numpy():
        lo, hi = int(indptr[s]), int(indptr[s + 1])
        if hi <= lo:
            continue
        tgt = indices[lo:hi]
        sel = np.isin(tgt, post_np)
        if not sel.any():
            continue
        w = lut[codes[lo:hi][sel]]
        for t, ww in zip(tgt[sel], w):
            rows.append((int(s), int(t), float(ww)))
    return pd.DataFrame(rows, columns=["pre", "post", "w"])


def incoming_weights(brain, targets: torch.Tensor) -> dict[int, np.ndarray]:
    """每个目标神经元收到的带符号权重数组（需要转置 CSR，一次性建好）。"""
    lut = brain.lut.detach().cpu().numpy()
    indptr = brain.indptr.detach().cpu().numpy()
    indices = brain.indices.detach().cpu().numpy()
    codes = brain.codes.detach().cpu().numpy()
    n = brain.N
    src = np.repeat(np.arange(n), np.diff(indptr))
    import scipy.sparse as sp
    A = sp.csr_matrix((lut[codes], (src, indices)), shape=(n, n))
    AT = A.T.tocsr()
    out = {}
    for t in targets.detach().cpu().numpy():
        t = int(t)
        out[t] = AT.data[AT.indptr[t]:AT.indptr[t + 1]]
    return out


def resolve_side(brain, names: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    """精确类型名 -> {"left"/"right"/"unknown": 子图索引}，用于方向性检验。

    DNp01 的轴突越过中线，所以"同侧感觉 -> 对侧下行"是逃避方向的关键预测；
    要测它就必须能把刺激只打给一侧的 LC4。
    """
    meta = pd.read_feather(DATA / "fafb_783_meta.feather")
    meta["id"] = meta["fafb_783_id"].astype("int64")
    meta["cell_type"] = meta["cell_type"].fillna("")
    meta["side"] = meta["side"].fillna("unknown")
    pos = pd.Series(np.arange(brain.N), index=np.asarray(brain.node_ids, dtype=np.int64))
    out: dict[str, dict[str, torch.Tensor]] = {}
    for n in names:
        sub = meta[meta["cell_type"] == n]
        d = {}
        for sd, grp in sub.groupby("side"):
            hit = pos.reindex(pd.Index(grp["id"].to_numpy(np.int64))).to_numpy()
            ok = hit[~pd.isna(hit)].astype(np.int64)
            if len(ok):
                d[str(sd)] = torch.as_tensor(np.sort(ok), dtype=torch.long,
                                             device=brain.device)
        out[n] = d
    return out


def variant(brain, *, shuffle_seed: int | None = None,
            cut: tuple[torch.Tensor, torch.Tensor] | None = None,
            block_inhibition: bool = False):
    """造对照图。返回一个新的 SpikingBrain（复用 indptr，只改 indices/codes/lut）。

    shuffle_seed   把所有边的目标全局洗牌：精确保持每个神经元的**出度**、
                   边数、权重分布；破坏"谁连到谁"（真实拓扑）。
    cut=(pre,post) 删除 pre->post 的直接边（把权重置 0），用于证明
                   "响应确实走这条单跳通路"。
    block_inhibition 把所有负权改成 0（去掉前馈抑制）。
    """
    import copy
    b = copy.copy(brain)                    # 浅拷贝：N/meta/indptr 等直接复用
    b.device = brain.device
    indices = brain.indices.detach().cpu().numpy().copy()
    codes = brain.codes.detach().cpu().numpy().copy()
    lut = brain.lut.detach().cpu().numpy().copy()
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(indices)                # 出度不变（每行条数不变），目标打乱
    if block_inhibition:
        lut = np.maximum(lut, 0.0)
    if cut is not None:
        pre_idx, post_idx = cut
        indptr = brain.indptr.detach().cpu().numpy()
        post_np = post_idx.detach().cpu().numpy()
        zeroed = 0
        for s in pre_idx.detach().cpu().numpy():
            lo, hi = int(indptr[s]), int(indptr[s + 1])
            if hi <= lo:
                continue
            sel = np.isin(indices[lo:hi], post_np)
            if sel.any():
                k = np.flatnonzero(sel) + lo
                codes[k] = 0                # code 0 -> lut 权重 0
                zeroed += int(sel.sum())
        print(f"  [variant] cut 删除 {zeroed} 条直接边")
    b.indices = torch.as_tensor(indices, dtype=torch.long, device=brain.device)
    b.codes = torch.as_tensor(codes, dtype=torch.long, device=brain.device)
    b.lut = torch.as_tensor(lut, dtype=torch.float32, device=brain.device)
    # 状态张量必须是各图独立的，否则两个变体交替跑会串状态
    b.G = torch.zeros(b.N, dtype=torch.float32, device=brain.device)
    b.S = torch.zeros(b.N, dtype=torch.bool, device=brain.device)
    b._cur = torch.zeros(b.N, dtype=torch.float32, device=brain.device)
    return b


# ------------------------------------------------------- 逼近刺激的运动学
# 模型时钟：dt=20ms/tick -> 单个神经元的发放率上限是 50 Hz。
# 所以下面所有"发放率"都用 **脉冲/tick**（0..1）表示，乘 50 才是 Hz。
DT_S = 0.02
TICK_HZ = 50.0


def loom_kinematics(radius_m: float, speed_ms: float, start_dist_m: float,
                    max_ticks: int = 300, *, dt_s: float = DT_S) -> pd.DataFrame:
    """正前方逼近球体的角动力学，**跑到撞击为止**（物理上刺激就消失了）。

    θ  = 2·arctan(R / D)          瞬时角直径（度）
    dθ/dt = 2·R·v / (D² + R²)      角扩张速度（度/秒），逼近时为正
    τ  = D / v                    剩余碰撞时间（秒）

    为什么让刺激在撞击处结束：这给出了一个**有物理意义的读出**——
    DNp01 越过阈值后还剩多少 τ 可以真正逃出去。"太快所以来不及逃"
    是真实的生物学结果，而不是仿真瑕疵；反过来若人为把刺激拖长
    （固定时长 / 固定终角），就会造出"快速逼近反而不触发"的假象。
    """
    n_hit = int(np.ceil(start_dist_m / max(speed_ms, 1e-9) / dt_s))
    ticks = max(2, min(max_ticks, n_hit))
    t = np.arange(ticks) * dt_s
    d = np.maximum(start_dist_m - speed_ms * t, 1e-6)
    theta = np.degrees(2.0 * np.arctan(radius_m / d))
    dtheta = np.degrees(2.0 * radius_m * speed_ms / (d ** 2 + radius_m ** 2))
    return pd.DataFrame({"t_s": t, "dist_m": d, "theta_deg": theta,
                         "dtheta_dps": dtheta, "tau_s": d / max(speed_ms, 1e-9)})


def looming_drive(kin: pd.DataFrame, *, s50: float = 30.0, n: float = 3.0,
                  source: str = "dtheta") -> np.ndarray:
    """角动力学 -> LC4 群体的目标发放率（脉冲/tick）。

    ⚠️ 这是本实验**唯一的外部假设**：从"刺激参数"到"LC4 发放率"的调谐曲线。
    冻结脑那一段（LC4 -> DNp01）是纯测量，没有任何假设，所以结论对这里的
    选择不敏感 —— 用 --s50/--n 扫一遍即可验证（见 scripts/loom_reflex.py）。

    Naka–Rushton 形式：p = s^n / (s^n + s50^n)，s = **角扩张速度**(度/秒，只取正向)
    source="theta" 则用角直径本身做自变量。

    ⚠️ 负向（远离/收缩）在这里被夹成 0，所以"远离对照"是**由构造决定的**，
    它检验的是刺激模型而不是脑。真正检验脑的对照是 cut / shuffled / 换感觉群。
    """
    if source == "dtheta":
        s = kin["dtheta_dps"].to_numpy(dtype=np.float64)
    elif source == "theta":
        s = kin["theta_deg"].to_numpy(dtype=np.float64)
    elif source == "invtau":
        # 1/τ = v/D：文献里"剩余碰撞时间编码"的代理量（度/秒量纲无关，只用相对值）
        s = 1.0 / np.maximum(kin["tau_s"].to_numpy(dtype=np.float64), 1e-6)
    else:
        raise ValueError(f"未知 source: {source}")
    s = np.clip(np.nan_to_num(s, nan=0.0), 0.0, None)
    p = s ** n / (s ** n + s50 ** n)
    return np.nan_to_num(p, nan=0.0)


def run_relay(brain, inputs: list[tuple[torch.Tensor, np.ndarray]],
              watch_idx: torch.Tensor, *, seed: int = 0) -> np.ndarray:
    """按逐 tick 的目标发放率钳制输入群，返回 (T, len(watch)) 的 0/1 脉冲矩阵。

    inputs: [(神经元索引张量, 长度 T 的发放率序列), ...] —— 可以同时驱动
            LC4 和 LPLC2 两条逼近通路。
    seed: 背景噪声的种子（每个试次换一个，用来算"激发概率"）。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    cidx = torch.cat([i for i, _ in inputs])
    T = min(len(p) for _, p in inputs)
    series = [np.asarray(p, dtype=np.float64) for _, p in inputs]
    sizes = [i.numel() for i, _ in inputs]
    brain.reset()
    out = np.zeros((T, watch_idx.numel()), dtype=np.int8)
    for i in range(T):
        p = np.concatenate([np.full(s, series[k][i])
                            for k, s in enumerate(sizes)])
        pt = torch.as_tensor(p, dtype=torch.float32, device=brain.device)
        brain.step(clamp=(cidx, pt))
        out[i] = brain.S[watch_idx].detach().cpu().numpy().astype(np.int8)
    return out
