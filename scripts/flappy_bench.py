#!/usr/bin/env python
"""Flappy 无头评测台：把 demo 里的游戏物理 + 冻结脑反射原样搬到命令行。

为什么需要它
------------
`demo/app.js` 里的游戏物理跑在浏览器里，改一次感觉投射要手动点、靠肉眼记分，
既慢又不可复现，也没法做对照。SCORE_PROMPT.md 明确要求每个加分改动都要同时报
**分数变化 + eff/need 变化 + 关掉该改动的退化**。所以先把评测台搭出来：
物理、拍翅判据、脑的推进全部与前端同源（constants 见下），只把"感觉投射"抽成
可插拔的模块 —— 这样任何改动都能量化，而且能跑零模型对照。

与 demo/app.js 的同源性（改前端时这些要一起改）
-----------------------------------------------
    GRAV=1180  FLAP_V=-340  PIPE_W=62  GAP=168  PX_PER_M=240  birdX=120
    G.W=520  G.H=620  GROUND=92  间距 300  首管 x=birdX+0.75*PX_PER_M-PIPE_W
    spawnPipe: gapTop = 70 + rand*(G.H-GAP-150) = 70..400   (upper∈[70,540])
    脑: dt=20ms 一 tick，拍翅判据 = 最近 5 tick 的 DNp01 脉冲 >= need_spikes
    FLAP_COOLDOWN=0.14 s（与前端一致；每 tick 都放行的话反射会被严重高估）

⚠️ 时基差异（必须记住，否则分数和页面不一样）
前端是 60fps，`ticks = floor(acc/0.02)`，也就是**每帧最多推 1 个 20ms tick**，
而游戏物理按真实 dt 走 —— 所以页面上的脑实际上只跑到约 0.83× 实时，物理却也
按同一个 dt 缩放。本评测台一律 **1 tick = 1 个物理步 = 20ms**，物理与脑严格同步，
这是"同一颗脑"最保守也最可复现的耦合方式。

用法
----
    PY=D:/Code/FlyBrain/env/python.exe
    # 单模式
    $PY scripts/flappy_bench.py --asset spiking_full --mode pipe_only --games 20
    # 基线（当前页面的做法）对照
    $PY scripts/flappy_bench.py --asset spiking_full --mode baseline --games 20
    # 一次扫一批模式（推荐排障/找方向时用）
    $PY scripts/flappy_bench.py --asset spiking_full --games 12 --modes baseline,ground_only,wall_gap
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
try:                                       # Windows 控制台默认 GBK
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from fpv import looming                     # noqa: E402
from fpv.spiking_brain import SpikingBrain  # noqa: E402

# ------------------------------------------------------------------ 游戏常数（= app.js）
G_W, G_H = 520, 620
BIRD_X = 120
BIRD_R = 11
GROUND = 92
GAP = 168
PIPE_W = 62
PX_PER_M = 240.0
SPACING = 300
GRAV = 1180.0
FLAP_V = -340.0
WARM_S = 1.2                     # 开局悬空展示时长（= 前端 readyT）
TICK_S = 0.02
FLAP_COOLDOWN = 0.14
SPIKE_WINDOW = 5                 # dn01_recent 的窗口（tick）
GROUND_Y = G_H - GROUND
NATURAL_GAP_TOP = (70.0, float(G_H - GAP - 150))   # 70..400（前端原样）

INF = float("inf")


# ================================================================== 世界
class World:
    """只包含游戏物理 + 几何，不知道脑的存在。"""

    def __init__(self, rng: np.random.Generator, *, gap_top, solvable: bool = True):
        self.rng = rng
        self.gap_top = gap_top
        self.solvable = solvable
        self.reset()

    def reset(self) -> None:
        self.y = 300.0
        self.vy = 0.0
        self.t = 0.0
        self.pipes: list[dict] = []
        self.score = 0
        self.dead = False
        self.cause = ""
        self.spawned = 0
        self.y_at_gap: list[float] = []          # 每根被穿过的管子：通过瞬间的 y
        self.t_first_flap = None
        self.last_flap_t = -INF
        self._spawn(self._first_gap_top())
        self.pipes[0]["x"] = BIRD_X + 0.75 * PX_PER_M - PIPE_W

    # ---- 生成
    def _first_gap_top(self) -> float:
        # 前端硬编码 top=250（中心 y=334 → 开局就在缺口里）
        return 250.0

    def _spawn(self, top: float) -> None:
        self.pipes.append(dict(x=G_W + 30.0, top=top, passed=False))
        self.spawned += 1

    def _next_gap_top(self) -> float:
        lo, hi = self.gap_top
        if not self.solvable:
            return float(self.rng.uniform(lo, hi))
        # 可解性约束：两根管子之间鸟能爬升的高度上界。
        # 一次拍翅给 vy=-340，之后以 1180 减速，vy=0 耗时 0.288 s、上升 49 px；
        # 管距 300px / 240px·s⁻¹ = 1.25 s 内可以连续拍，所以"必须爬升"的极限
        # 远大于屏幕高度；真正的约束是**开局**那一根：鸟必须能在管子到达前从
        # y=300 爬到缺口。这里只排除"杯口比上一根高得离谱"的地方，保持分布
        # 与前端一致（不偷偷把游戏改简单，只保证物理可达）。
        return float(self.rng.uniform(lo, hi))

    # ---- 几何查询
    def nearest_pipe(self):
        best = None
        for p in self.pipes:
            if p["x"] + PIPE_W > BIRD_X - 6 and (best is None or p["x"] < best["x"]):
                best = p
        return best

    def front_distance_px(self) -> float:
        """前方**最近那面墙**的水平距离（px）。管子没到就先按整屏算。"""
        p = self.nearest_pipe()
        if p is None:
            return float(G_W)
        return max(p["x"] + PIPE_W - BIRD_X, 0.0)

    def wall_present(self) -> bool:
        """前方是否真有一面会撞上的墙（管子）。"""
        return self.front_distance_px() < G_W - 1

    def gap_center(self) -> float:
        p = self.nearest_pipe()
        return (p["top"] + GAP / 2.0) if p else (G_H / 2.0)

    # ---- 物理推进
    def step(self, flap: bool) -> None:
        dt = TICK_S
        if self.dead:
            return
        self.t += dt
        if self.t < WARM_S:                       # 开局悬空，不落体
            self.y = 300.0 + math.sin(self.t * 3.3) * 10.0
            self.vy = 0.0
            return
        vpx = PX_PER_M                                # 管速 = 1 m/s
        if flap:
            self.vy = FLAP_V
            self.last_flap_t = self.t
            if self.t_first_flap is None:
                self.t_first_flap = self.t
        self.vy += GRAV * dt
        self.y += self.vy * dt
        for p in self.pipes:
            p["x"] -= vpx * dt
            if not p["passed"] and p["x"] + PIPE_W < BIRD_X:
                p["passed"] = True
                self.score += 1
                self.y_at_gap.append(self.y)
        self.pipes = [p for p in self.pipes if p["x"] > -PIPE_W - 10]
        last = self.pipes[-1] if self.pipes else None
        if last is None or last["x"] < G_W - SPACING:
            self._spawn(self._next_gap_top())
        if self.y > G_H - 14:
            return self.die("撞到地面")
        if self.y < 6:
            return self.die("撞到天花板")
        for p in self.pipes:
            if BIRD_X + BIRD_R > p["x"] and BIRD_X - BIRD_R < p["x"] + PIPE_W:
                if self.y - BIRD_R < p["top"] or self.y + BIRD_R > p["top"] + GAP:
                    return self.die("撞上管子")

    def die(self, cause: str) -> None:
        self.dead = True
        self.cause = cause


# ================================================================== 感觉投射
class Projection:
    """感觉投射 = 把**世界几何**变成每个感觉群的逐细胞发放率。

    这里没有任何可学参数、没有任何"该不该拍翅"的规则 —— 只有
    "威胁在视野的哪个位置 / 扩张多快 → 哪些细胞被驱动到多少"。
    """

    name = "unnamed"

    def __init__(self, harness, spec: dict):
        self.h = harness
        self.spec = spec
        self.s50 = float(spec.get("s50", 15.0))
        self.s50size = float(spec.get("s50size", 30.0))
        self.n = float(spec.get("n", 3.0))
        self.gain = float(spec.get("gain", 1.0))
        self.max_ticks = int(spec.get("max_ticks", 900))
        self.baseline = float(spec.get("baseline", 0.25))
        self.ground_aware = bool(spec.get("ground_aware", 1))
        self.u = harness.u                 # {群名: (n,) 视野坐标}
        self.idx = harness.gidx            # {群名: (n,) 资产索引}
        self.w = harness.wname             # {群名: (n,) -> DNp01 权重}

    def size_amp(self, theta_deg: float) -> float:
        x = max(theta_deg, 0.0) ** self.n
        return x / (x + self.s50size ** self.n) if self.s50size > 0 else 0.0

    def rate_amp(self, dtheta_dps: float) -> float:
        x = max(dtheta_dps, 0.0) ** self.n
        return x / (x + self.s50 ** self.n) if self.s50 > 0 else 0.0

    def drive(self, w: World) -> list[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    # ---- 工具：把一次"威胁"写成 (群, 模式, 参数) 的驱动图样
    def _window(self, group: str, mode: str, center: float, width: float,
                amp: float) -> tuple[np.ndarray, np.ndarray]:
        u = self.u[group]
        if mode == "flat":
            shape = np.ones_like(u)
        elif mode == "gauss":
            shape = np.exp(-((u - center) / max(width / 2.355, 1e-6)) ** 2 / 2.0)
        elif mode == "ramp_up":                     # u 越小越强（腹侧）
            shape = np.clip(1.0 - u / max(width, 1e-6), 0.0, 1.0)
        elif mode == "ramp_dn":                     # u 越大越强（背侧）
            shape = np.clip((u - (1.0 - width)) / max(width, 1e-6), 0.0, 1.0)
        elif mode == "half_lo":
            shape = (u < center).astype(np.float64)
        elif mode == "half_hi":
            shape = (u >= center).astype(np.float64)
        else:
            raise ValueError(mode)
        return self.idx[group], shape * amp


class BaselineProjection(Projection):
    """当前页面的做法：只把"最近碰撞"当作正前方的暗盘，单通道 LC4 角速度码。

    与 app.js 的 threat()+stepBrain() 完全一致：半径取 0.05（管子）或 0.55（地面/天花板），
    用 dθ/dt 过 Naka-Rushton(s50=15)，整群 LC4 钳同一个值。
    """
    name = "baseline"

    def drive(self, w: World):
        vpx = PX_PER_M
        cands = []
        d = w.front_distance_px()
        if w.wall_present():
            cands.append((d / vpx, 0.05, vpx, d))
        if w.y > WARM_S and w.vy > 0:
            cands.append((max(GROUND_Y - w.y, 0) / max(w.vy, 1e-6), 0.55, w.vy,
                          max(GROUND_Y - w.y, 0)))
        if w.y > WARM_S and w.vy < 0:
            cands.append((max(w.y, 0) / max(-w.vy, 1e-6), 0.55, -w.vy, max(w.y, 0)))
        if not cands:
            return []
        ttc, rad, spd, dpx = min(cands, key=lambda c: c[0])
        dist = max(dpx / PX_PER_M, 0.02)
        speed = max(spd / PX_PER_M, 0.02)
        dtheta = math.degrees(2 * rad * speed / (dist ** 2 + rad ** 2))
        p = self.gain * self.rate_amp(dtheta)
        g = "LC4"
        idx, pv = self._window(g, "flat", 0.5, 1.0, p)
        self.last = dict(p=p, dtheta=dtheta, dist=dist, rad=rad)
        return [(idx, pv)]


class GroundLockProjection(Projection):
    """只让**地面**驱动腹侧 LPLC2（安全反射），完全不管管子。

    这不是用来拿分的，是用来分离两件事：腹侧 LPLC2 到底能不能把鸟从
    "下坠"里救出来（= 高度维持），以及它和"导航到缺口"必须分开算账。
    """
    name = "ground_lock"

    def drive(self, w: World):
        if w.t < WARM_S or w.vy <= 0:
            return []
        h_px = max(GROUND_Y - w.y, 1.0)
        dist = h_px / PX_PER_M
        speed = w.vy / PX_PER_M
        tau = dist / max(speed, 1e-6)
        theta = math.degrees(2 * math.atan2(0.55, max(dist, 1e-3)))
        amp = self.gain * self.size_amp(theta)
        idx, pv = self._window("LPLC2", self.spec.get("window", "half_lo"),
                               0.5, self.spec.get("width", 0.5), amp)
        self.last = dict(p=amp, theta=theta, tau=tau)
        return [(idx, pv)]


class GroundLockVariant(GroundLockProjection):
    """ground_lock 的消融变体 —— 用来证明分数确实来自"腹侧 LPLC2 这一路"。

    `group` 决定驱动哪个感觉群：
        LPLC2（默认）  腹侧半视野、到 DNp01 的权重占全群 72%
        LPLC2_full     整群（腹+背）—— 检验"只用腹侧"是不是真的更好
        LC4            54 个 LC4 —— 检验"腹侧 LPLC2 相对 LC4 的优势"（ESCAPE §3.6 说
                       LC4 单通道延迟 ~260ms，应该更差）
        LC10a          与逼近无关的感觉群 —— 对照组，应该完全不触发（ESCAPE §3.2）
    """
    name = "ground_lock_x"

    def drive(self, w: World):
        grp = str(self.spec.get("group", "LPLC2"))
        win = ("flat" if grp in ("LPLC2_full", "LC4", "LC10a")
               else self.spec.get("window", "ramp_up"))
        if w.t < WARM_S or w.vy <= 0:
            return []
        h_px = max(GROUND_Y - w.y, 1.0)
        dist = h_px / PX_PER_M
        theta = math.degrees(2 * math.atan2(0.55, max(dist, 1e-3)))
        amp = self.gain * self.size_amp(theta)
        idx, pv = self._window(grp, win, 0.5, self.spec.get("width", 0.5), amp)
        self.last = dict(p=amp, theta=theta)
        return [(idx, pv)]


class WallGapProjection(Projection):
    """把**整面管子墙**当成逼近物，用"洞口在视野里的高度"决定哪个视野段被驱动。

    这是 SCORE_PROMPT.md 第 1 条建议的落地：
      · 逼近量 = 墙的逼近强度（角大小 θ 过 LPLC2 的 Naka-Rushton，s50size）
      · 位置   = 洞口相对鸟的**视觉高度**，映射到 LPLC2 的视野坐标 u∈[0,1]
                  （低 u = 腹侧 = "缺口在头顶上方" → 应该爬升；高 u = 背侧 → 下潜）
      · 腹侧那半视野对 DNp01 的突触权重是背侧的 2.6 倍
        （scripts/loom_ventral_probe.py：0.0752 vs 0.0291），所以"往上爬"这一支
        天生比"往下潜"更容易触发 —— 这正好是 Flappy 里最需要的那一支。

    没有方向控制器：方向完全由**威胁落在视野的哪一段**决定。
    """
    name = "wall_gap"

    def drive(self, w: World):
        if w.t < WARM_S or not w.wall_present():
            return []
        d = w.front_distance_px() / PX_PER_M
        rad = 0.55                                   # 管子迎面压来的等效半宽（demo 同值）
        theta = math.degrees(2 * math.atan2(rad, max(d, 1e-3)))
        amp = self.gain * self.size_amp(theta)
        center = w.gap_center()
        u_gap = float(np.clip(0.5 + (center - w.y) / (0.5 * G_H), 0.0, 1.0))
        w_half = float(self.spec.get("width", 0.6))
        base = self.baseline
        out = []
        if u_gap <= 0.5:                             # 缺口在头顶 → 腹侧被驱动 → 爬升
            sc = base + (1.0 - base) * (1.0 - 2.0 * u_gap) ** self.spec.get("pow", 1.0)
            idx, pv = self._window("LPLC2", "ramp_up", 0.5, w_half, amp * sc)
            out.append((idx, pv))
        else:                                        # 缺口在脚下 → 背侧被驱动 → 下潜
            sc = base + (1.0 - base) * (2.0 * u_gap - 1.0) ** self.spec.get("pow", 1.0)
            idx, pv = self._window("LPLC2", "ramp_dn", 0.5, w_half, amp * sc)
            out.append((idx, pv))
        self.last = dict(p=amp, theta=theta, u_gap=u_gap, dist=d)
        return out


class GroundEquilibriumProjection(Projection):
    """腹侧 LPLC2 读**地面在视野里的角尺寸**，用它当逃逸触发；缺口高度只调"要多怕"。

    为什么是这个结构（这条是整个任务的钥匙）
    ----------------------------------------
    1. DNp01 有硬阈值（need = 0.0604/tick），而**局部小物体够不到它**：
       Flappy 里管子的角宽只有 13~20°，最多覆盖 LPLC2 群体的一小块，
       Σw·p 永远到不了 1（本文件的 pipe_only 模式实测 0 触发）。
       唯一能覆盖大部分视野的逼近物是**地面**（0.55 的等效半径在低空能撑满视野），
       所以反射必须由地面驱动 —— 这正是 ESCAPE.md §3.8 补记里那条结论。
    2. 于是"维持高度"来自"地面角尺寸越大 → 腹侧驱动越强 → 拍翅"，
       这构成一个**负反馈**：拍翅 → 爬升 → 地面角变小 → 停止拍翅。
       平衡高度由 s50size 决定（默认 30° → 约几百 px）。
    3. 那怎么去**另一根管子**的缺口？不需要任何方向控制器：把同一个地面信号
       乘上"鸟相对缺口的垂直误差"，让它只在地面**和**缺口都对的时候才够强。
       缺口在下方 → 抑制地面那一支 → 鸟自己往下掉；缺口在上方 → 放行 →
       地面那支立刻把鸟往上推。两个纯几何量相乘，没有任何可学参数。

    注意门控用的是闭环误差本身（center-y），不是"如果继续自由落体会掉到哪"。
    自由的预测门控试过，会把鸟钉在缺口上、无法在两根管子之间移动（见
    output/flappy_bench*.json 的历史记录）。
    """
    name = "ground_eq"

    def _ground_drive(self, w: World, u_center: float, width: float):
        """地面（鸟在缺口**之上**时）或天花板（鸟在缺口之下时）的角尺寸驱动。"""
        if w.vy > 0:
            h_px = max(GROUND_Y - w.y, 1.0)
            rad = 0.55
            mode = "ramp_up"
        else:
            h_px = max(w.y, 1.0)
            rad = 0.55
            mode = "ramp_dn"
        dist = max(h_px / PX_PER_M, 0.02)
        theta = math.degrees(2 * math.atan2(rad, dist))
        amp = self.gain * self.size_amp(theta)
        return self._window("LPLC2", mode, 0.5, width, amp), theta

    def drive(self, w: World):
        if w.t < WARM_S:
            return []
        base = self.baseline
        dev = (w.y - w.gap_center()) / (0.5 * G_H)         # >0：鸟在缺口下方
        lo, hi = self.spec.get("gate_lo", -0.08), self.spec.get("gate_hi", 0.25)
        # 纯几何门控：鸟比缺口低得越多，地面那一支被放行得越多；
        # 鸟高于缺口时地面支被压到底（base），于是自由下落去找缺口。
        x = (dev - lo) / max(hi - lo, 1e-6)
        gate = base + (1.0 - base) * float(np.clip(x, 0.0, 1.0))
        out = []
        if w.vy > 0:                                       # 下落 → 地面逼近
            (idx, pv), theta = self._ground_drive(w, 0.5, self.spec.get("width", 0.5))
            out.append((idx, pv * gate))
            self.last = dict(p=float(pv.max()) * gate, theta=theta, gate=gate, dev=dev)
        elif w.vy < 0 and dev > -0.05:                     # 上爬且还没超过缺口 → 天花板支
            (idx, pv), theta = self._ground_drive(w, 0.5, self.spec.get("width", 0.5))
            out.append((idx, pv))
            self.last = dict(p=float(pv.max()), theta=theta, gate=1.0, dev=dev)
        else:
            self.last = dict(p=0.0, theta=0.0, gate=gate, dev=dev)
        return out


class GroundSetpointProjection(GroundEquilibriumProjection):
    """地面 looming 当触发，**缺口高度只按比例改触发的强度**（线性整定）。

    为什么只能是这个结构（把前面所有失败的教训浓缩成三句）
    ------------------------------------------------------
    1. **管子的角尺寸太小，够不到 DNp01。** `loom_trigger_curve.csv` 实测：
       要让 DNp01 稳定发放，LPLC2 整群被驱动的等效刺激要 ~34° 以上
       （Σw·p ≥ 1.0，need=0.0604）。而 Flappy 里管子的角宽只有 13~20°，
       所以 pipe_only / wall_gap 实测都是 **0 触发**，不是参数没调好。
       唯一能覆盖这么大视野的逼近物是**地面**（低空时下视野被它填满）。
    2. **触发不是一个"比值过线"就完事，要连续高驱动才能攒够膜电位。**
       膜电位 R = Σ leak^k·(eff·gain)：eff/need 只要 ≤1 就永远到不了阈值，
       实测要 eff/need≈1.1 且**连续**好几个 tick 才放一个脉冲。所以
       ground_eq/ground_eq2 里那种"把驱动乘一个 0~1 的门控"会直接把驱动
       压到阈值以下 → 一个脉冲都没有（实测 0 分、拍翅 0 次）。
       正确做法是让触发**自己**决定平衡高度（地面角尺寸的负反馈），
       而缺口只调这个平衡点的高低。
    3. 于是：`驱动 = p_ground(落速/高度) × gate(缺口相对高度)`，
       gate 是 [0,1] 的**线性**量。gate=1 → 鸟在被驱动的那套高度上悬停；
       gate→0 → 触发被压掉，鸟自由下坠去更低的位置。缺口越高，需要的
       平衡高度越高，gate 把地面支放行得越多。

    这也是"苍蝇看到了什么"而不是"苍蝇学到了什么"：gate 只用到缺口在
    视野里的**高低**这一个几何量，没有任何控制器、没有时序记忆、没有可学参数。
    """
    name = "ground_sp"

    def drive(self, w: World):
        if w.t < WARM_S:
            return []
        dev = (w.y - w.gap_center()) / (0.5 * G_H)     # >0：鸟在缺口下方
        s = self.spec.get("elev_scale", 1.0)
        gate = float(np.clip(0.5 + s * dev, 0.0, 1.0))
        # 安全网：地面真的要到脸上了就无条件拍（真实反射里"再不逃就死"，不是时序策略）
        h_px = max(GROUND_Y - w.y, 1.0)
        if w.vy > 0 and (h_px / PX_PER_M) / max(w.vy / PX_PER_M, 1e-6) < 0.22:
            gate = 1.0
        out = []
        if w.vy > 0:                                   # 下落 → 地面逼近
            (idx, pv), theta = self._ground_drive(w, 0.5, self.spec.get("width", 0.5))
            out.append((idx, pv * gate))
            self.last = dict(p=float(pv.max()) * gate, theta=theta, gate=gate, dev=dev)
        elif w.vy < 0 and dev > -0.02:                 # 上爬且没越过缺口 → 天花板支
            (idx, pv), theta = self._ground_drive(w, 0.5, self.spec.get("width", 0.5))
            out.append((idx, pv))
            self.last = dict(p=float(pv.max()), theta=theta, gate=1.0, dev=dev)
        else:
            self.last = dict(p=0.0, theta=0.0, gate=gate, dev=dev)
        return out



class DoubleChannelProjection(GroundSetpointProjection):
    """腹侧（地面）+ 背侧（天花板）两条视野通道都接，**只用几何做方向选择**。

    这是"腹侧权重是背侧 2.6 倍"这条解剖事实的直接检验：
      · 缺口在鸟下方 → 地面在逼近 → 腹侧 LPLC2（权重 0.0752）→ 爬升
      · 缺口在鸟上方 → 天花板在逼近 → 背侧 LPLC2（权重 0.0291）→ 下潜
    如果背侧那半单独够不到 need（0.0291 < 0.0604，静态上就不可能），
    那么"往下潜"这支永远推不动 DNp01 —— 鸟只能往上爬，飞不过低缺口。
    这个模式就是把这件事测出来，而不是嘴上说。
    """
    name = "ground_dual"

    def drive(self, w: World):
        if w.t < WARM_S:
            return []
        dev = (w.y - w.gap_center()) / (0.5 * G_H)
        out = []
        if dev < 0:
            # 鸟在缺口**之上**：威胁来自下方 → 腹侧 LPLC2 → 爬升
            (idx, pv), theta = self._ground_drive(w, 0.5, self.spec.get("width", 0.5))
            out.append((idx, pv * float(self.spec.get("vent_gain", 1.0))))
            self.last = dict(p=float(pv.max()), theta=theta, chan="ventral", dev=dev)
        else:
            # 鸟在缺口**之下**：威胁来自上方 → 背侧 LPLC2 → 下潜
            h_px = max(w.y, 1.0)
            dist = max(h_px / PX_PER_M, 0.02)
            theta = math.degrees(2 * math.atan2(0.55, dist))
            amp = self.gain * self.size_amp(theta)
            idx, pv = self._window("LPLC2", "ramp_dn", 0.5,
                                   float(self.spec.get("width", 0.5)), amp)
            out.append((idx, pv * float(self.spec.get("dors_gain", 1.0))))
            self.last = dict(p=float(pv.max()), theta=theta, chan="dorsal", dev=dev)
        return out


class PipeEdgeVentralProjection(GroundEquilibriumProjection):
    """腹侧通道**同时**接地面和"管子近端边缘"的 looming —— 缺口高度写入触发强度。

    动机（前面几轮失败留下的唯一缺口）
    ----------------------------------
    ground_lock 真的能飞，但它是一个**固定高度的极限环**：平衡点由"地面角尺寸
    达到腹侧 LPLC2 阈值"决定，实测过管时 y≈267±20。缺口中心在 y∈[154,624] 上随机，
    所以只有缺口恰好跨过这条带时才过得去。要提分就必须让**平衡点跟着缺口走**。

    为什么不能简单地"把驱动乘一个门控"：DNp01 的驱动上限只有 need 的 1.25 倍
    （腹侧半群 0.0752 vs need 0.0604），任何能把驱动压到 1.0 以下的门控
    都会直接把它打回"一个脉冲都不放"（这正是 ground_eq/ground_eq2/ground_sp
    实测 0 分的原因）。所以门控只能关掉**附加**的那一路，不能动地面这一路。

    这里用的附加路是**管子近端边缘**：缺口在鸟上方时，下管的上沿就在鸟的下方，
    距离 0.2~0.4 m 时它自己就撑出 ~60°+ 的角，是货真价实的大 looming 刺激；
    而它到 DNp01 的投射走的是**同一半腹侧视野**（下视野 → 腹侧 LPLC2）。
    两个来源在腹侧群体上线性叠加 → 触发阈值等效降低 → 平衡点被拉到缺口中心。
    没有控制器、没有时序记忆：仍然只是"威胁落在视野哪一段、扩张多快"。
    """
    name = "pipe_edge"

    def drive(self, w: World):
        if w.t < WARM_S:
            return []
        # ---- 主路：地面/天花板（安全反射），决定大致的悬停高度
        out = GroundSetpointProjection.drive(self, w)
        # ---- 附加路：管子近端边缘，只在鸟**低于缺口**（需要往上爬）时给
        pipe_w = float(self.spec.get("pipe_w", 1.0))
        if pipe_w <= 0:
            return out
        p = w.nearest_pipe()
        if p is None:
            return out
        near = p["top"] + GAP if w.y > p["top"] + GAP / 2 else p["top"]
        d_fwd = max(w.front_distance_px(), 12.0) / PX_PER_M      # 水平距离（真实透视）
        d_vert = abs(w.y - near) / PX_PER_M
        dist = math.hypot(d_fwd, d_vert)
        if d_vert <= 0 or dist <= 0:
            return out
        theta = math.degrees(math.atan2(0.31, dist))             # 管壁半厚 0.31 m
        amp = self.gain * self.size_amp(theta) * pipe_w
        idx, pv = self._window("LPLC2", "ramp_up", 0.5,
                               float(self.spec.get("width", 0.5)), amp)
        out.append((idx, pv))                                    # 与主路逐细胞相加
        self.last = dict(self.last, pipe_theta=theta, pipe_amp=amp, near=near)
        return out


class GroundEqualGateProjection(GroundEquilibriumProjection):
    """（已否决）对称门控版：缺口在下时把地面支压到 0。保留作为负结果证据。

    实测 **0 分、拍翅 0 次**：把驱动乘一个能到 0 的门控 = 把 eff 压到阈值以下 =
    DNp01 一个脉冲都不放。这直接否掉了"用一个 [0,1] 门控去做比例控制"这条路，
    也是 ground_sp 存在的理由（触发强度只能在线性整定里动，不能把它关掉）。
    """
    name = "ground_eq2"

    def drive(self, w: World):
        if w.t < WARM_S:
            return []
        dev = (w.y - w.gap_center()) / (0.5 * G_H)
        span = self.spec.get("span", 0.25)
        gate = float(np.clip(0.5 + dev / max(span, 1e-6), 0.0, 1.0))
        out = []
        if w.vy > 0:
            (idx, pv), theta = self._ground_drive(w, 0.5, self.spec.get("width", 0.5))
            out.append((idx, pv * gate))
            self.last = dict(p=float(pv.max()) * gate, theta=theta, gate=gate, dev=dev)
        elif w.vy < 0 and dev > -0.05:
            (idx, pv), theta = self._ground_drive(w, 0.5, self.spec.get("width", 0.5))
            out.append((idx, pv))
            self.last = dict(p=float(pv.max()), theta=theta, gate=1.0, dev=dev)
        else:
            self.last = dict(p=0.0, theta=0.0, gate=gate, dev=dev)
        return out


class WallGapGroundProjection(WallGapProjection):
    """墙（导航）+ 地面/天花板（安全）双通道，全部走 LPLC2 的视野段。

    地面是一个大面积逼近物：下坠时它扩张 → 腹侧 LPLC2 → 拍翅 → 停止下坠。
    这一支不需要"知道"缺口在哪，是纯反射；管子那一支负责把鸟送到缺口高度。
    """
    name = "wall_gap_ground"

    def drive(self, w: World):
        out = WallGapProjection.drive(self, w)
        if w.t < WARM_S or not self.ground_aware:
            return out
        h_px = GROUND_Y - w.y
        if w.vy > 0 and h_px > 0:                    # 正在下落 → 地面逼近
            dist = max(h_px / PX_PER_M, 0.02)
            speed = max(w.vy / PX_PER_M, 0.02)
            theta = math.degrees(2 * math.atan2(0.55, dist))
            amp = self.gain * self.size_amp(theta)
            idx, pv = self._window("LPLC2", "ramp_up", 0.5,
                                   float(self.spec.get("ground_width", 0.5)), amp)
            out.append((idx, pv))
        elif w.vy < 0 and w.y > 0:                   # 正在上爬 → 天花板逼近
            dist = max(w.y / PX_PER_M, 0.02)
            speed = max(-w.vy / PX_PER_M, 0.02)
            theta = math.degrees(2 * math.atan2(0.55, dist))
            amp = self.gain * self.size_amp(theta)
            idx, pv = self._window("LPLC2", "ramp_dn", 0.5,
                                   float(self.spec.get("ground_width", 0.5)), amp)
            out.append((idx, pv))
        return out


class PipeOnlyProjection(Projection):
    """只把**管子实体**（上下两段管体）当威胁，按它在视野里的高度定位。

    这是"只盯管子"的对照组 —— ESCAPE.md 说它全是 0 分。放进来是为了确认
    评测台能复现这个已知的负结果，而不是为了拿分。
    """
    name = "pipe_only"

    def drive(self, w: World):
        if w.t < WARM_S or not w.wall_present():
            return []
        vpx = PX_PER_M
        d = w.front_distance_px() / PX_PER_M
        theta = math.degrees(2 * math.atan2(0.05, max(d, 1e-3)))
        dtheta = math.degrees(2 * 0.05 * (vpx / PX_PER_M) / (d ** 2 + 0.05 ** 2))
        amp = self.gain * self.rate_amp(dtheta)
        p = w.nearest_pipe()
        out = []
        if p is not None:
            c = p["top"] + GAP / 2.0
            u_gap = float(np.clip(0.5 + (c - w.y) / (0.5 * G_H), 0.0, 1.0))
            sc = self.baseline + (1.0 - self.baseline) * abs(2 * u_gap - 1) ** 1.5
            center = 1.0 - u_gap                     # 缺口在哪，威胁（管体）就在对面
            idx, pv = self._window("LC4", "gauss", float(np.clip(center, 0.05, 0.95)),
                                   float(self.spec.get("width", 0.6)), amp * sc)
            out.append((idx, pv))
        self.last = dict(p=amp, theta=theta, d=d)
        return out


PROJECTIONS = {p.name: p for p in (
    BaselineProjection, GroundLockProjection, WallGapProjection,
    WallGapGroundProjection, PipeOnlyProjection, GroundEquilibriumProjection,
    GroundEqualGateProjection, GroundSetpointProjection, DoubleChannelProjection,
    PipeEdgeVentralProjection, GroundLockVariant)}
# --modes 的键：值是 (投射类名, 拍翅判据)
MODES = {
    "baseline": ("baseline", "brain"),
    "ground_lock": ("ground_lock", "brain"),
    # ground_lock 的消融：只需换 --group
    "abl_lc4": ("ground_lock_x", "brain"),
    "abl_lplc2_full": ("ground_lock_x", "brain"),
    "abl_lc10a": ("ground_lock_x", "brain"),
    "ground_eq": ("ground_eq", "brain"),
    "ground_eq2": ("ground_eq2", "brain"),
    "ground_sp": ("ground_sp", "brain"),
    "ground_dual": ("ground_dual", "brain"),
    "pipe_edge": ("pipe_edge", "brain"),
    "wall_gap": ("wall_gap", "brain"),
    "wall_gap_ground": ("wall_gap_ground", "brain"),
    "pipe_only": ("pipe_only", "brain"),
    "passive": ("baseline", "passive"),          # 只喂脑、从不拍翅：零模型
    "oracle": ("baseline", "oracle"),            # 外部理想控制器：世界几何的上限刻度
}
ALIAS = {"ground": "ground_lock", "ground_only": "ground_lock",
         "wall": "wall_gap", "wall+ground": "wall_gap_ground",
         "pipe": "pipe_only", "eq": "ground_eq", "eq2": "ground_eq2",
         "sp": "ground_sp", "setpoint": "ground_sp"}


# ================================================================== 评测台
class Harness:
    def __init__(self, asset: str, *, device: str = "cpu", gain: float = 3.0,
                 tonic: float = 0.0, need_spikes: int = 1, cooldown: float = FLAP_COOLDOWN):
        t0 = time.time()
        self.base = SpikingBrain.from_npz(asset, device=device)
        self.base.gain, self.base.tonic = gain, tonic
        self.brain = self.base
        self.asset = asset
        self.need_spikes = need_spikes
        self.cooldown = cooldown
        self._init_groups()
        self.load_time = time.time() - t0

    def set_graph(self, name: str) -> str:
        """干预脑的接线（都是 ESCAPE.md 里已验证过的对照）。

        real     原图
        cut      切掉 LC4+LPLC2 -> DNp01 的**全部直接边**（证明响应走这条单跳通路）
        shuffled 全部边的目标全局洗牌（保持出度/权重分布，毁掉拓扑）
        no_inh   把所有负权清零（去掉前馈抑制 → 阈值应当下降）
        """
        if name in (None, "real"):
            self.brain = self.base
            return "real"
        b = SpikingBrain.from_npz(self.asset, device=str(self.base.device))
        b.gain, b.tonic = self.base.gain, self.base.tonic
        if name == "cut":
            gg = looming.resolve(self.base, ["LC4", "LPLC2", "DNp01"])
            pre = torch.cat([gg["LC4"], gg["LPLC2"]])
            self.brain = looming.variant(b, cut=(pre, gg["DNp01"]))
        elif name == "shuffled":
            self.brain = looming.variant(b, shuffle_seed=7)
        elif name == "no_inhibition":
            self.brain = looming.variant(b, block_inhibition=True)
        else:
            raise ValueError(name)
        return name

    def _init_groups(self) -> None:
        g = looming.resolve(self.base, ["LC4", "LPLC2", "LC10a", "DNp01", "DNp04"])
        self.g = g
        self.need = (1 - self.base.leak) * self.base.threshold / self.base.gain
        # 视野轴 + 逐细胞权重（与 demo/server.py 的 _retino_groups/_weights_to_dn01 同法）
        coords = self._coords()
        self.u, self.gidx, self.wname = {}, {}, {}
        self.axis_info, self.idx_of_first = {}, {}
        for nm in ("LC4", "LPLC2", "LC10a"):
            idx = g[nm].detach().cpu().numpy()
            self.u[nm], self.axis_info[nm] = self._field_u(coords, idx)
            self.gidx[nm] = idx
            self.idx_of_first[int(idx[0])] = nm
            self.wname[nm] = self._w_to_dn01(idx)
        # 整群 LPLC2 的窗口是"平"的，但 u 仍然需要（flat 模式只用它做形状）
        self.gidx["LPLC2_full"] = self.gidx["LPLC2"]
        self.u["LPLC2_full"] = self.u["LPLC2"]
        self.wname["LPLC2_full"] = self.wname["LPLC2"]
        self.idx_of_first[int(self.gidx["LPLC2"][0])] = "LPLC2"

    # ---- 坐标 / 视野轴（半球轴先剔除）
    def _coords(self) -> np.ndarray:
        mf = json.loads((REPO / "data" / "brain" / "manifest.json").read_text(encoding="utf-8"))
        xs = mf["bbox"]
        ctr = np.array([(xs[0] + xs[3]) / 2, (xs[1] + xs[4]) / 2, (xs[2] + xs[5]) / 2])
        half = np.array([(xs[3] - xs[0]) / 2, (xs[4] - xs[1]) / 2, (xs[5] - xs[2]) / 2])
        row_of = {str(n["id"]): i for i, n in enumerate(mf["neurons"])}
        secs = mf["coarse"]["sections"]
        out = np.zeros((self.brain.N, 3), dtype=np.float32)
        for k, nid in enumerate(np.asarray(self.brain.node_ids, np.int64)):
            j = row_of.get(str(int(nid)))
            if j is not None and j < len(secs) and secs[j].get("bbox"):
                b = secs[j]["bbox"]
                out[k] = (np.array([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2,
                                    (b[2] + b[5]) / 2]) - ctr) / half
        return out

    @staticmethod
    def _field_u(coords, idx):
        pts = coords[idx]
        hemi = int(np.argmax([pts[:, a].std() for a in range(3)]))
        cand = [a for a in range(3) if a != hemi]
        # 侧别：用半球轴符号近似（这里只用来在**同侧内部**归一化 u）
        sgn = np.sign(pts[:, hemi])
        axis = int(max(cand, key=lambda a: np.mean([pts[sgn == s, a].std()
                                                    for s in (-1, 1)
                                                    if (sgn == s).sum() > 3])))
        u = np.zeros(len(idx), dtype=np.float64)
        for s in (-1, 1):
            m = sgn == s
            if m.sum() < 2:
                continue
            v = pts[m, axis]
            u[m] = (v - v.min()) / max(v.max() - v.min(), 1e-9)
        return u, {"hemi": "xyz"[hemi], "field": "xyz"[axis]}

    def _w_to_dn01(self, group: np.ndarray) -> np.ndarray:
        lut = self.brain.lut.detach().cpu().numpy()
        indptr = self.brain.indptr.detach().cpu().numpy()
        indices = self.brain.indices.detach().cpu().numpy()
        codes = self.brain.codes.detach().cpu().numpy()
        dn = self.g["DNp01"].detach().cpu().numpy()
        w = np.zeros(len(group), dtype=np.float64)
        for k, s in enumerate(group):
            lo, hi = int(indptr[s]), int(indptr[s + 1])
            if hi <= lo:
                continue
            sel = np.isin(indices[lo:hi], dn)
            if sel.any():
                w[k] = float(lut[codes[lo:hi][sel]].sum())
        return w / max(len(dn), 1)

    # ---- 一条命
    def dn01_idx(self) -> torch.Tensor:
        key = "_dn01"
        if not hasattr(self, key):
            setattr(self, key, looming.resolve(self.base, ["DNp01"])["DNp01"])
        return getattr(self, key)

    def play(self, proj_name: str, spec: dict, seed: int, *, max_ticks: int,
             solvable: bool = True, policy: str = "brain", trace: bool = False) -> dict:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        gap_top = (NATURAL_GAP_TOP if solvable else NATURAL_GAP_TOP)
        world = World(rng, gap_top=gap_top, solvable=solvable)
        brain = self.brain
        brain.reset()
        proj = PROJECTIONS[proj_name](self, spec) if proj_name in PROJECTIONS else None
        dn01 = self.dn01_idx()
        recent: list[int] = []
        eff_hist: list[float] = []
        p_hist: list[float] = []
        flaps = 0
        t_last_flap = -INF
        tr: list[dict] = []
        for tick in range(max_ticks):
            # ---- 感觉 → 脑
            eff = 0.0
            if proj is not None:
                drives = proj.drive(world)
                if drives:
                    # 同一个群可能被多个威胁同时驱动（墙 + 地面），
                    # 必须先**逐细胞求和**再钳制，不能把同一个群 concat 两遍
                    # （那样 clamp 只会用到第一段，而且 eff 会重复计数）。
                    merged: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                    for ix, p in drives:
                        key = int(ix[0])
                        if key in merged:
                            merged[key] = (ix, merged[key][1] + p)
                        else:
                            merged[key] = (ix, p.copy())
                    cidx = torch.as_tensor(
                        np.concatenate([ix for ix, _ in merged.values()]),
                        dtype=torch.long, device=brain.device)
                    pv = torch.as_tensor(
                        np.concatenate([np.clip(p, 0.0, 1.0)
                                        for _, p in merged.values()]),
                        dtype=torch.float32, device=brain.device)
                    # Σw·p：权重必须和索引**一起**取子集（形状不对当场报错，
                    # 别让一个恒为 0 的仪表悄悄骗过判据）
                    for ix, p in merged.values():
                        nm = self.idx_of_first.get(int(ix[0]))
                        if nm is not None:
                            eff += float((self.wname[nm] * np.clip(p, 0, 1)).sum())
                    brain.step(clamp=(cidx, pv))
                else:
                    brain.step()
            else:
                brain.step()
            eff_hist.append(eff)
            p_hist.append(eff / self.need)
            spk = int(brain.S[dn01].sum())
            recent.append(spk)
            if len(recent) > SPIKE_WINDOW:
                recent.pop(0)
            # ---- 读出 → 拍翅（纯反射，无学习）
            if policy == "brain":
                want = sum(recent) >= self.need_spikes
            elif policy == "passive":
                want = False
            elif policy == "rule":
                c = world.gap_center()
                want = world.y > c + 6
            elif policy == "oracle":
                want = self._oracle(world)
            else:
                raise ValueError(policy)
            if want and (world.t - t_last_flap) >= self.cooldown and world.t >= WARM_S:
                flaps += 1
                t_last_flap = world.t
            else:
                want = False                      # 冷却期内不算拍翅（计数要诚实）
            world.step(want)
            if trace:
                tr.append(dict(tick=tick, t=round(world.t, 3), y=round(world.y, 1),
                               vy=round(world.vy, 1), gap=round(world.gap_center(), 1),
                               dev=round(world.y - world.gap_center(), 1),
                               eff=round(eff, 5), ratio=round(eff / self.need, 2),
                               spk=spk, recent=sum(recent), flap=bool(want),
                               score=world.score,
                               d_front=round(world.front_distance_px(), 1)))
            if world.dead:
                break
        return dict(mode=proj_name, policy=policy, seed=seed, score=world.score,
                    ticks=tick + 1, cause=world.cause, flaps=flaps,
                    eff_mean=float(np.mean(eff_hist)) if eff_hist else 0.0,
                    ratio_mean=float(np.mean(p_hist)) if p_hist else 0.0,
                    ratio_max=float(np.max(p_hist)) if p_hist else 0.0,
                    frac_over_1=float(np.mean([r >= 1.0 for r in p_hist]))
                    if p_hist else 0.0,
                    y_at_gap=[round(v, 1) for v in world.y_at_gap],
                    t_first_flap=world.t_first_flap, trace=tr)

    @staticmethod
    def _oracle(w: World) -> bool:
        """外部规则基线：知道确切状态的理想 Flappy 控制器。

        它**不是**策略候选（那样就是"教它玩"），只是一个上限刻度：
        世界几何到底能不能拿高分。分数上限低就说明该先修世界而不是改脑。
        """
        if w.y > w.gap_center() + 4:
            return True
        if w.vy > 260 and w.y > w.gap_center() - GAP * 0.28:
            return True
        return False


# ================================================================== 主
def summarize(rows: list[dict]) -> dict:
    sc = np.array([r["score"] for r in rows], dtype=float)
    ck = np.array([r["ticks"] for r in rows], dtype=float)
    fl = np.array([r["flaps"] for r in rows], dtype=float)
    causes: dict[str, int] = {}
    for r in rows:
        causes[r["cause"] or "存活到上限"] = causes.get(r["cause"] or "存活到上限", 0) + 1
    return dict(n=len(rows), score_mean=float(sc.mean()), score_median=float(np.median(sc)),
                score_max=float(sc.max()), score_std=float(sc.std()),
                pass_rate=float((sc >= 1).mean()), ticks_mean=float(ck.mean()),
                flaps_mean=float(fl.mean()),
                eff_mean=float(np.mean([r["eff_mean"] for r in rows])),
                ratio_mean=float(np.mean([r["ratio_mean"] for r in rows])),
                frac_over_1=float(np.mean([r["frac_over_1"] for r in rows])),
                causes=causes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spiking_full")
    ap.add_argument("--modes", default="baseline",
                    help="逗号分隔；键见 ALIAS（baseline/ground_only/wall_gap/"
                         "wall_gap_ground/pipe_only/passive/oracle）")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--max-ticks", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gain", type=float, default=1.0, help="投射增益（不改脑动力学）")
    ap.add_argument("--s50", type=float, default=15.0, help="LC4 角速度灵敏度（度/秒）")
    ap.add_argument("--s50size", type=float, default=30.0, help="LPLC2 角大小灵敏度（度）")
    ap.add_argument("--width", type=float, default=0.5, help="视野窗宽")
    ap.add_argument("--baseline", type=float, default=0.25, help="方向门控的底（0=纯几何门控）")
    ap.add_argument("--elev-scale", type=float, default=1.0,
                    help="缺口高度对地面触发的线性整定斜率（ground_sp 用）")
    ap.add_argument("--pipe-w", type=float, default=1.0,
                    help="管子近端边缘的 looming 权重（pipe_edge 用；0 = 关掉该附加路）")
    ap.add_argument("--cooldown", type=float, default=FLAP_COOLDOWN,
                    help="两次拍翅最小间隔（秒，前端为 0.14；用来分离'游戏限制'和'脑限制'）")
    ap.add_argument("--group", default=None,
                    help="ground_lock_x 消融用：LPLC2 / LPLC2_full / LC4 / LC10a")
    ap.add_argument("--graph", default="real",
                    choices=["real", "cut", "shuffled", "no_inhibition"],
                    help="脑侧干预（对照）")
    ap.add_argument("--need-spikes", type=int, default=1)
    ap.add_argument("--trace-mode", default=None, help="打印该模式的逐 tick 轨迹")
    ap.add_argument("--trace-n", type=int, default=80)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    h = Harness(a.asset, gain=3.0, tonic=0.0, need_spikes=a.need_spikes,
                cooldown=a.cooldown)
    print(f"资产 {a.asset}: N={h.brain.N:,}  E={len(h.brain.codes):,}  "
          f"gain={h.brain.gain} tonic={h.brain.tonic}  载入 {h.load_time:.1f}s")
    print(f"拍翅冷却 {h.cooldown}s   游戏常数 = demo/app.js（GRAV={GRAV} "
          f"FLAP_V={FLAP_V} GAP={GAP} 间距={SPACING}）")
    print(f"need = (1-leak)*thr/gain = {h.need:.5f}")
    for nm in ("LC4", "LPLC2"):
        print(f"  {nm:<6} n={len(h.gidx[nm]):<4} 视野轴={h.axis_info[nm]}"
              f"  Σw(DNp01)={h.wname[nm].sum():.4f}"
              f"  腹侧(u<.5)权重 {h.wname[nm][h.u[nm] < 0.5].sum():.4f}")

    modes = [ALIAS.get(m.strip(), m.strip()) for m in a.modes.split(",") if m.strip()]
    spec = dict(s50=a.s50, s50size=a.s50size, gain=a.gain, width=a.width,
                baseline=a.baseline, ground_aware=1, elev_scale=a.elev_scale,
                pipe_w=a.pipe_w)
    summary: dict[str, dict] = {}
    all_rows: list[dict] = []
    t0 = time.time()
    for mode in modes:
        if mode not in MODES:
            raise SystemExit(f"未知模式 {mode!r}；可选 {sorted(MODES)}")
        proj_name, policy = MODES[mode]
        if a.graph != "real":
            h.set_graph(a.graph)
            print(f"\n[干预] 脑的接线 = {a.graph}")
        else:
            h.brain = h.base
        spec["group"] = a.group or {"abl_lc4": "LC4", "abl_lplc2_full": "LPLC2_full",
                                    "abl_lc10a": "LC10a"}.get(mode, "LPLC2")
        rows = []
        for k in range(a.games):
            r = h.play(proj_name, spec, seed=a.seed * 1000 + k, max_ticks=a.max_ticks,
                       policy=policy, trace=bool(a.trace_mode == mode))
            r["mode"] = mode
            rows.append(r)
            all_rows.append(r)
        summary[mode] = summarize(rows)
        s = summary[mode]
        print(f"\n[{mode}] n={s['n']} 均分 {s['score_mean']:.2f} ± {s['score_std']:.2f}  "
              f"中位 {s['score_median']:.0f} 最高 {s['score_max']:.0f}  "
              f"过管率 {s['pass_rate']:.0%}")
        print(f"        存活 {s['ticks_mean']:.0f} tick  拍翅 {s['flaps_mean']:.1f} 次  "
              f"eff={s['eff_mean']:.4f}  eff/need 均值 {s['ratio_mean']:.2f}  "
              f"超阈值时间占比 {s['frac_over_1']:.0%}")
        print(f"        死因 {s['causes']}")
        if mode in ("baseline",) and s["flaps_mean"] < 0.5:
            print("        ⚠️ 几乎不拍翅 —— 先看 eff/need 有没有到 1，别急着调游戏")
        if a.trace_mode == mode:
            row = next((r for r in rows if r["trace"]), None)
            if row:
                print(f"        轨迹（第 1 局）：每 tick 一行，共 {len(row['trace'])} tick")
                for e in row["trace"][:a.trace_n]:
                    print("          " + " ".join(f"{k}={e[k]}" for k in
                          ("tick", "t", "y", "vy", "gap", "d_front", "ratio", "spk",
                           "recent", "flap")))
    print(f"\n总耗时 {time.time() - t0:.1f}s")

    out = pathlib.Path(a.out) if a.out else (ROOT / "output" / "flappy_bench.json")
    payload = dict(asset=a.asset, spec=spec, max_ticks=a.max_ticks, games=a.games,
                   seed=a.seed, summary=summary, rows=all_rows)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
