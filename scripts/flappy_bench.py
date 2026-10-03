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
import copy
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
BIRD_R = 17
GROUND = 92
GAP = 184
PIPE_W = 62
PX_PER_M = 240.0
SPACING = 300
GRAV = 1180.0
FLAP_V = -340.0
WARM_S = 0.8                     # 开局**加速期**时长（= demo/app.js 的 WARMUP_S）
# 注意：这不再是一段'悬空假动画'。加速期照常走物理、也照常喂脑，只是不判碰撞。
# 所以 **不能**再用 `if w.t < WARM_S: return []` 去门控投射 —— 那样脑在开局
# 又变成没有输入，正是'起步直接跳死'的成因。
TICK_S = 0.02
FLAP_COOLDOWN = 0.14
SPIKE_WINDOW = 5                 # dn01_recent 的窗口（tick）
GROUND_Y = G_H - GROUND
NATURAL_GAP_TOP = (70.0, float(G_H - GAP - 150))   # 70..400（前端原样）

INF = float("inf")


# ================================================================== 世界
class World:
    """只包含游戏物理 + 几何，不知道脑的存在。"""

    def __init__(self, rng: np.random.Generator, *, gap_top, solvable: bool = True,
                 max_climb: float = 40.0, first_gap_extra: float = 0.0):
        self.rng = rng
        self.gap_top = gap_top
        self.solvable = solvable
        # 相邻缺口允许的**向上**跳变上限（px）。默认 110 ≈ 一个管距内
        # 可持续爬升的高度（100 px/s × 1.25 s = 125px），取略保守的值。
        self.max_climb = max_climb
        #: 第 2 根的额外间距（px）。0 = 与其它管子一样。
        self.spec_extra = float(first_gap_extra)
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

    def spacing(self) -> float:
        """本根之后那一根的间距。

        默认全部用 SPACING；但**第 2 根**（index 1）用 `first_gap_extra` 拉长。
        依据：调完 dors_scale / vent_gain 之后，剩下的撞管死亡**全部集中在第 2 根**
        （13/22，死亡时刻中位 3.16s ≈ 第 2 根到达时刻），且 22/22 都是"偏低没爬够"
        （dev 中位 +80，安全带 ±73）。它一个管子间隔只能净爬 ~60px，
        而第 2 根的缺口中心又允许比第 1 根高 max_climb(40px)，加上进管滞后就差了。
        """
        extra = float(self.spec_extra)
        if extra > 0 and self.spawned == 1:
            return SPACING + extra
        return SPACING

    def _next_gap_top(self) -> float:
        """生成下一根管子的缺口。

        `solvable=False` 时就是前端的原始做法：gapTop ~ U(70, 400)，缺口中心
        在 y∈[154,624] 上均匀随机 —— 相邻两根的缺口**平均跳变 157px、最大 330px**。

        为什么需要可解性约束（这是本节最重要的一条实测）
        ------------------------------------------------
        `scripts/flappy_trace_death.py` 量到：鸟的可持续爬升率只有 **~100 px/s**
        （拍翅一次买 49px、周期 ~0.4s；而且上冲期地面驱动被 `vy_gate` 关掉，
        膜电位要重新积分才够下一次）。管距 300px / 240px·s⁻¹ = **1.25 s**，
        所以一个间隔内最多爬 ~125px。而缺口的随机跳变有 330px 的量程：
        **超过这个带宽的"向上跳变"在给定物理下根本飞不进去**，不是脑的问题。
        `scripts/flappy_spacing.py` 独立验证了这一点（管距 200→400px 时
        均分 4.20→12.28，≤2 分占比 40%→20%）。

        所以这里把"相邻缺口的向上跳变"限到执行器带宽以内：
            next_center ≥ prev_center − max_climb
        向下不设限——自由落体快得多（1.25s 可掉 921px > 量程 470px）。
        注意这**不是**降低游戏难度：它只排除物理上不可达的关卡，
        与"把缺口调大/把管子调慢"是两件事，报告里分开写。
        """
        lo, hi = self.gap_top
        if not self.solvable:
            return float(self.rng.uniform(lo, hi))
        if not self.pipes:
            return float(self.rng.uniform(lo, hi))
        prev_c = self.pipes[-1]["top"] + GAP / 2.0        # 上一根缺口中心
        reach = float(self.max_climb)
        # 下一根缺口中心的上界：不能比上一根高出超过 reach
        top_max = min(hi, prev_c + reach - GAP / 2.0)
        top_min = lo
        if top_max < top_min:                              # 极端情况下退化成尽量低
            top_max = top_min
        return float(self.rng.uniform(top_min, top_max))

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
        # ---- 开局加速期：**照常走物理**（也照常喂脑 —— drive() 只在 t<WARM_S 时返回空，
        #      那个门控已随页面一起去掉），只是**不判碰撞**。
        # 为什么改（用户报"起步直接跳死"）：原来是假动画 `y = 300 + sin(...)` 悬空不落体，
        # 这 1.2 秒里没有任何逼近刺激 → 脑驱动恒为 0；放开后鸟从 vy=0 自由落体，
        # 而反射要膜电位积分 ~8 tick 才够阈值，管子却已按间距到达 → 起步必死。
        # 现在与 demo/app.js 的 WARMUP_S 同步：真物理真喂脑，0.8s 后才开碰撞。
        warm = self.t < WARM_S
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
        if last is None or last["x"] < G_W - self.spacing():
            self._spawn(self._next_gap_top())
        if warm:
            # 加速期只限制在画面内 —— 否则鸟一落地就死，比原来的假动画更糟。
            # 与 demo/app.js 的 `warm` 分支逐字对应。
            self.y = min(max(self.y, 40.0), float(G_H - GROUND - BIRD_R - 1))
            return
        # ---- 碰撞：**圆 vs 轴对齐矩形**，与 demo/app.js 以及
        #      D:/Code/DQN 的 `FlappySim._collides` 同一判据。
        #      (bx-cx)² + (by-cy)² <= r²，矩形竖直连续 → 化简为下面的 >= / <=。
        #      用"<= / >="（触碰即算撞），与 DQN 一致。
        if self.y + BIRD_R >= G_H - 14:
            return self.die("撞到地面")
        if self.y - BIRD_R <= 0:
            return self.die("撞到天花板")
        for p in self.pipes:
            cx = max(p["x"], min(BIRD_X, p["x"] + PIPE_W))     # 矩形上离圆心最近的 x
            if (BIRD_X - cx) ** 2 > BIRD_R ** 2:
                continue                                        # 水平还没够到
            # 与 demo/app.js 的 physics() 逐字一致（整数算术，见那边的注释）：
            #   圆周下沿 ceil(y+R) 越过 p.top+R，或圆周上沿 floor(y-R) 低于 p.top+GAP-R
            if (math.ceil(self.y + BIRD_R) > p["top"] + BIRD_R
                    or math.floor(self.y - BIRD_R) < p["top"] + GAP - BIRD_R):
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


class BiDirectionalProjection(GroundEquilibriumProjection):
    """**双向**反射：腹侧半视野（地面）爬升 + 背侧半视野（天花板）下潜。

    这一步解决的是前面所有版本最硬的一个问题
    ------------------------------------------
    ESCAPE.md §6.5 量到：背侧 LPLC2 单独到 DNp01 只有 0.0291 = need 的 48%，
    **永远推不动**，所以 `ground_dual` 实测 0 分 —— "反射只能往上拉、不能往下压"。
    但那句话只对**单通道 + 平滑空间窗**成立。把两条视叶逼近通道**并联**起来
    （这正是文献里 GF 的输入结构，Ache 2019 / Gaitanidis 2025：
    LC4 + LPLC2 占 GF 直接视叶输入 98.5%），并且按**整半**招募视野：

        腹侧 LC4 0.0328 + LPLC2 0.0752 = 0.1080   / need = 1.79   → 能爬
        背侧 LC4 0.0416 + LPLC2 0.0291 = 0.0707   / need = 1.17   → 能潜
        （对照：同样的背侧两群、但用平滑斜坡窗只有 0.0319 = 0.53 → 不发放）

    腹侧/背侧的划分用的是同一条**已修正过的视野轴**（先剔除半球轴，
    再在同侧内部取方差最大的轴）。于是同一个机制自然给出两个方向：
    威胁在视野下方 → 腹侧那两群被驱动 → 拍翅；威胁在上方 → 背侧那两群 → 不拍翅。
    "往哪边走"由**威胁落在视野的哪一半**决定，不是控制器。

    仍然是纯感觉投射：没有可学参数、没有时序记忆、没有方向控制器。
    """
    name = "bidi"

    #: 两条支路各自用到的感觉群（= ESCAPE.md §3.6 的双通道：LC4 角速度 + LPLC2 角大小）
    GROUPS = ("LC4", "LPLC2")

    def _half(self, group: str, dorsal: bool, amp: float):
        """按视野轴**整半**招募（不是斜坡窗）。

        ⚠️ 这里踩过一次坑，必须写下来：斜坡窗 `(u-0.5)/0.5` 会把靠近视野中线的细胞
        一起压掉，实测背侧 LC4+LPLC2 的有效驱动只剩 **0.0319（need 的 53%）**，
        仍然不发放；而"整半"招募能拿到 **0.0707（need 的 117%）**，过阈值。
        差别就是"平滑的空间梯度"和"半个群体一起动"。
        真实视网膜上不该整半齐动，所以这同时是一条**负面证据**：
        DNp01 的阈值高到只有大范围招募才推得动。
        """
        u = self.u[group]
        m = (u >= 0.5) if dorsal else (u < 0.5)
        idx = self.idx[group][m]
        return idx, np.full(len(idx), float(amp)), m

    def _branch_amp(self, w: World, *, up: bool, miss: float | None = None) -> float:
        """算这一支的驱动强度。

        `miss is None` → 用**地面/天花板角尺寸**（原做法，无预判）。
        `miss` 给定时  → 用**预测偏差**归一化后的量。

        为什么预判不能用角尺寸（踩过，实测均分 0.00、0 拍翅）：
          预判判定"会撞下管、需要下潜"之后走背侧支，而背侧支原来的驱动量是
          **天花板角尺寸** —— 鸟在 y=500 时天花板距离 2.08m、角只有 16°，
          Σw·p ≈ 0.0007，一个脉冲都不放。也就是"判断出该下潜"却**执行不了**。
          用预测偏差就对了：偏差大就是明确的运动指令，与"天花板此刻多大"无关。
        """
        if miss is not None:
            # 线性映射：偏差 0 → base，偏差到 scale → 满驱动 1.0
            scale = max(float(self.spec.get("predict_scale", 60.0)), 1e-6)
            base = float(self.spec.get("predict_base", 0.3))
            return min(1.0, base + (1.0 - base) * abs(miss) / scale)
        if up:
            h_px = max(GROUND_Y - w.y, 1.0)
        else:
            h_px = max(w.y, 1.0)
        dist = max(h_px / PX_PER_M, 0.02)
        theta = math.degrees(2 * math.atan2(0.55, dist))
        if not up:
            theta = theta * float(self.spec.get("ceil_boost", 1.0))
        return self.gain * self.size_amp(theta)

    def _branch(self, w: World, *, up: bool, miss: float | None = None):
        """up=True → 威胁来自下方（地面），驱动腹侧半视野；up=False → 天花板，驱动背侧。"""
        # 只有"正在朝那个面靠近"时才是逼近刺激：上升时天花板在逼近，下落时地面在逼近。
        # 不加这一条，鸟冲过缺口后天花板支还在驱动 → 一路爬到撞天花板
        # （实测 40 局里 8 局就是这么死的）。
        #
        # ⚠️ 但**有预判时这条要关掉**：预判已经明确算过"再不动就会撞"，
        #   此时"我正在往上冲所以不许下潜"恰好是致命的 —— 那会让它眼看着撞。
        if miss is None and self.spec.get("vy_gate", 1):
            vent_gate_up = bool(self.spec.get("vent_gate_up", 0))
            if (up and w.vy <= 0 and not vent_gate_up) or (not up and w.vy >= 0):
                return []
        amp = self._branch_amp(w, up=up, miss=miss)
        dorsal = not up
        # 背侧支（下潜）的权重。默认 1.0 = 保持原样。
        # 之所以留这个旋钮：ESCAPE.md 量到背侧半群到 DNp01 只有 need 的 1.17 倍
        # （勉强过阈值），而它驱动的"下潜/别爬"在实测里既是必要的、也是撞天花板的来源。
        if dorsal:
            amp *= float(self.spec.get("dors_scale", 1.0))
        else:
            # 腹侧（爬升）支按"缺口比鸟高多少"放大：实测撞管死亡**全部**是
            # "偏低没爬够"（dev 中位 +77，安全带 ±73），所以需要时让它爬得更狠。
            # 只在缺口明显在上方时放大，靠近缺口时不动（避免过冲撞天花板）。
            gain = float(self.spec.get("vent_gain", 1.0))
            if gain != 1.0:
                dev = max(0.0, w.y - w.gap_center())      # >0：缺口在上方
                scale = min(1.0, dev / max(float(self.spec.get("vent_dev", 120.0)), 1e-6))
                amp = min(1.0, amp * (1.0 + (gain - 1.0) * scale))
        out, eff = [], 0.0
        for g in (self.spec.get("groups") or self.GROUPS):
            idx, pv, m = self._half(g, dorsal, amp)
            out.append((idx, pv))
            eff += float((self.w[g][m] * pv).sum())
        self.last = dict(branch="dorsal" if dorsal else "ventral", amp=amp,
                         eff=eff, ratio=eff / max(self.h.need, 1e-9),
                         miss=None if miss is None else round(miss, 1))
        return out

    def _predict_arrival(self, w: World) -> float:
        """如果从现在起不再拍翅，管子到达时鸟会在哪个 y（自由落体重力积分）。

        逐行与 `World.step` 的积分一致，所以这不是"另一个模型"，就是把已知物理推一遍。
        """
        d_front = w.front_distance_px()
        if d_front <= 0:
            return w.y
        ttc_ticks = d_front / (PX_PER_M * TICK_S)      # 到接触还有多少 tick
        n = min(int(math.ceil(ttc_ticks)), int(self.spec.get("predict_horizon", 400)))
        y, vy = w.y, w.vy
        for _ in range(n):
            vy += GRAV * TICK_S
            y += vy * TICK_S
        return y

    def _rollout_to_contact(self, w: World, *, first: bool) -> float:
        """从当前状态出发，先做 `first` 这个动作，之后用**现成的死区策略**续演到接触，
        返回接触瞬间的 y（真做了一次前向仿真，不是只看自由落体）。

        为什么要这样（踩过两次）：
          · 只用"自由落体"当"不作为的后果"是错的基准 —— 鸟从 y=301 自由落体 0.74s
            会落到 640（远超缺口），于是预判**永远**说"要拍翅"，它就一直拍、
            爬过头撞天花板（实测均分 3.02，比纯反应式还差）。
          · 正确的问法是"**哪个动作会让结果更好**"，所以要把两个动作都演一遍。
        续演用的策略就是本项目已验证的死区规则（缺口高就拍、低就放它落），
        所以这不是"偷偷塞一个控制器"，而是把同一套规则当 rollout 策略做一步显式搜索。
        """
        y, vy, t = w.y, w.vy, w.t
        last_flap = w.last_flap_t
        d_front = w.front_distance_px()
        if d_front <= 0:
            return y
        n = min(int(math.ceil(d_front / (PX_PER_M * TICK_S))),
                int(self.spec.get("predict_horizon", 400)))
        p = w.nearest_pipe()
        c = (p["top"] + GAP / 2.0) if p is not None else G_H / 2.0
        margin = float(self.spec.get("gap_margin", BIRD_R * 2))
        for i in range(n):
            act = first if i == 0 else (c < y - margin)
            if act and (t - last_flap) < self.h.cooldown:
                act = False                      # 冷却期内拍不动
            if act:
                vy = FLAP_V
                last_flap = t
            vy += GRAV * TICK_S
            y += vy * TICK_S
            t += TICK_S
        return y

    def drive(self, w: World):
        # ⚠️ 这里**不能**再门控 `w.t < WARM_S`。
        # 加速期照常走物理、也照常喂脑（只是不判碰撞），所以脑从第一 tick 起就有输入。
        # 原来的门控会让开局这 0.8~1.2 秒驱动恒为 0，等放开时鸟已在下坠、
        # 膜电位还要再积 8 个 tick —— 那就是用户报的"起步直接跳死"。
        margin = float(self.spec.get("gap_margin", BIRD_R * 2))
        no_vent = bool(self.spec.get("no_ventral", 0))   # 消融：关掉爬升支（只留下潜）
        gap_c = w.gap_center()

        if self.spec.get("predict", 0):
            # ---- 1-ply 预判：把"拍"与"不拍"各演到接触，取落点更接近缺口中心的那个。
            # 驱动量用**预测偏差**（不是地面/天花板角尺寸）—— 否则"判断出该下潜"
            # 会因为天花板此刻角太小而根本执行不了（实测 0 分）。
            p = w.nearest_pipe()
            if p is None:
                return []
            c = p["top"] + GAP / 2.0
            y_flap = self._rollout_to_contact(w, first=True)
            y_free = self._rollout_to_contact(w, first=False)
            if abs(y_flap - c) <= abs(y_free - c):
                if no_vent:
                    return []
                return self._branch(w, up=True, miss=y_flap - c)
            return self._branch(w, up=False, miss=y_free - c)

        # ---- 无预判：纯反应式（方向只看缺口现在比鸟高还是低）
        # ⚠️ 死区是必须的：第一版写成 `if gap < y` 时，鸟刚越过缺口中心就翻转成
        # 爬升，于是在缺口上下反复横跳、一路撞管 —— 实测均分只有 0.81。
        if gap_c < w.y - margin:
            if no_vent:
                return []
            return self._branch(w, up=True)          # 缺口在头顶 → 腹侧支（爬升）
        if gap_c > w.y + margin:
            return self._branch(w, up=False)         # 缺口在脚下 → 背侧支（下潜）
        return []                                    # 已对准：不驱动


class ThresholdTrackProjection(GroundEquilibriumProjection):
    """不再是"乘一个门控"，而是**按缺口位置给腹侧通道加偏置**，并且偏置永不把驱动压到阈值以下。

    为什么之前所有门控版本都 0 分（这一步是把负结果变成设计约束）
    -----------------------------------------------------------
    腹侧半群的最大有效驱动只有 need 的 1.25 倍，而 DNp01 要**连续**高驱动
    才攒得够膜电位。所以 `drive × gate` 里只要 gate 能让驱动掉到 1.0 以下，
    就必然"一个脉冲都不放"（ground_eq/ground_eq2/ground_sp 三版实测 0 分）。
    结论：门控**不能乘在驱动上**。

    正确做法是把缺口高度变成**视野段的偏置**：腹侧那半永远是主动驱动，
    缺口在下方时只是把腹侧段的驱动**再往上加一点**（等效降低触发阈值），
    而 base 保证任何一支都不低于 need 的 1.1 倍 —— 于是既有连续可调的
    平衡高度，又不会出现"死掉的那一支"。

    对应到视觉上就是：鸟在缺口**下方**时，下管的近端边缘也在鸟的下方，
    它落在**同一段腹侧视野**里，与该段原有驱动线性叠加 → 该段更早达到
    逃逸阈值 → 鸟更早开始爬 → 平衡高度更高。
    """
    name = "track"

    def drive(self, w: World):
        if w.t < WARM_S or w.vy <= 0:
            return []                                   # 上爬时地面支关闭（否则一直爬）
        dev = (w.y - w.gap_center()) / (0.5 * G_H)      # >0：鸟在缺口下方
        span = max(float(self.spec.get("span", 0.6)), 1e-6)
        base = float(self.spec.get("base", 0.6))
        wgt_vent = base + (1.0 - base) * float(np.clip(dev / span, 0.0, 1.0))
        # 地面角尺寸（与 ground_lock 同一套公式，唯一区别是腹侧段再乘 wgt_vent）
        h_px = max(GROUND_Y - w.y, 1.0)
        dist = h_px / PX_PER_M
        theta = math.degrees(2 * math.atan2(0.55, max(dist, 1e-3)))
        amp = self.gain * self.size_amp(theta)
        mode = str(self.spec.get("window", "ramp_up"))
        width = float(self.spec.get("width", 0.5))
        if mode == "ramp_up":        # 腹侧段：按缺口位置加权（这一支决定爬升高度）
            idx, pv = self._window("LPLC2", mode, 0.5, width, amp * wgt_vent)
            self.last = dict(theta=theta, dev=dev, w=wgt_vent, branch="ventral")
        else:                        # 别的窗口形状：不加权，避免把驱动压到阈值下
            idx, pv = self._window("LPLC2", mode, 0.5, width, amp)
            self.last = dict(theta=theta, dev=dev, w=1.0, branch=mode)
        return [(idx, pv)]


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
    PipeEdgeVentralProjection, GroundLockVariant, ThresholdTrackProjection,
    BiDirectionalProjection)}
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
    "track": ("track", "brain"),
    "bidi": ("bidi", "brain"),
    "bidi_predict": ("bidi", "brain"),   # 双向 + TTC 预判
    "ground_dual": ("ground_dual", "brain"),
    "pipe_edge": ("pipe_edge", "brain"),
    "wall_gap": ("wall_gap", "brain"),
    "wall_gap_ground": ("wall_gap_ground", "brain"),
    "pipe_only": ("pipe_only", "brain"),
    "passive": ("baseline", "passive"),          # 只喂脑、从不拍翅：零模型
    "oracle": ("baseline", "oracle"),            # 外部理想控制器：世界几何的上限刻度
    "lookahead": ("baseline", "lookahead"),      # 短视界前向搜索：真正的上限刻度
}
ALIAS = {"ground": "ground_lock", "ground_only": "ground_lock",
         "wall": "wall_gap", "wall+ground": "wall_gap_ground",
         "pipe": "pipe_only", "eq": "ground_eq", "eq2": "ground_eq2",
         "sp": "ground_sp", "setpoint": "ground_sp"}


# ================================================================== 评测台
class Harness:
    def __init__(self, asset: str, *, device: str = "cpu", gain: float = 3.0,
                 tonic: float = 0.0, need_spikes: int = 1, cooldown: float = FLAP_COOLDOWN,
                 oracle_look: int = 60):
        t0 = time.time()
        self.base = SpikingBrain.from_npz(asset, device=device)
        self.base.gain, self.base.tonic = gain, tonic
        self.brain = self.base
        self.asset = asset
        self.need_spikes = need_spikes
        self.cooldown = cooldown
        self.oracle_look = oracle_look
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
        # 逐细胞查表：任意**子集**（例如只取腹侧半）都能算 Σw·p。
        # 之前用"群名 -> 整群权重"，一旦投射只驱动半个群，Σw·p 就恒为 0（仪表失灵）。
        self.wmap = np.zeros(self.base.N, dtype=np.float32)
        self.gmap: dict[int, str] = {}
        for nm in ("LC4", "LPLC2", "LC10a"):
            self.wmap[self.gidx[nm]] = self.wname[nm]
            for a in self.gidx[nm]:
                self.gmap[int(a)] = nm

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
        world = World(rng, gap_top=gap_top, solvable=solvable,
                      max_climb=float(spec.get("max_climb", 110.0)),
                      first_gap_extra=float(spec.get("first_gap_extra", 0.0)))
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
        slack_acc = 0.0
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
                    # Σw·p：逐细胞查表，对**任意子集**都成立（腹侧半、背侧半、整群都行）。
                    # 别再用"群名 -> 整群权重"，那会让只驱动半群的投射把仪表打成恒 0。
                    for ix, p in merged.values():
                        eff += float((self.wmap[ix] * np.clip(p, 0.0, 1.0)).sum())
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
            elif policy == "lookahead":
                want = self._lookahead(world, t_last_flap)
            else:
                raise ValueError(policy)
            # 不再要求 world.t >= WARM_S：页面在加速期也照常拍翅（脑一直在飞）
            if want and (world.t - t_last_flap) >= self.cooldown:
                flaps += 1
                t_last_flap = world.t
            else:
                want = False                      # 冷却期内不算拍翅（计数要诚实）
            # game_slack < 1：物理不每个 tick 都推进（复现"脑落后于物理"的错配）。
            # 页面是 60fps 物理 + 每帧最多一个 20ms 脑 tick，所以脑只跑到 ~83% 实时，
            # 等价的物理步进比例是 50/(60·1.2) = 0.694。评测台默认 1.0（严格同步）。
            slack_acc += float(spec.get("game_slack", 1.0))
            while slack_acc >= 1.0:
                world.step(want)
                slack_acc -= 1.0
                want = False          # 一次拍翅只给一次冲量（别在子步里重复施加）
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

    # ---- 真正的上限刻度：1-ply 前向搜索 + 便宜的 rollout 策略
    # ⚠️ 第一版写成"对每个动作递归搜索"→ 分支数 2^depth，60 tick 直接爆炸（跑不完）。
    # 正确做法：当前这一步**两个动作都试**，之后的每一步都用一条便宜规则走完，
    # 总代价只有 2×look 次世界推进。
    @staticmethod
    def _rollout_policy(w: World) -> bool:
        """rollout 用的廉价规则：把高度往**下一根管子的缺口中心**拉。

        注意这是**外部规划器**的一部分，不是脑的输出，也不是策略候选。
        """
        target = w.gap_center()
        if w.y > target + 6:
            return True
        if w.vy > 150 and w.y > target - GAP * 0.22:
            return True
        return False

    def _rollout(self, w: World, look: int, cooldown: float) -> tuple[float, float]:
        """用 _rollout_policy 走 look 个 tick，返回 (分数, 存活时长)。"""
        c = w
        for _ in range(max(0, look)):
            if c.dead:
                break
            act = self._rollout_policy(c)
            if act and (c.t - c.last_flap_t) < cooldown:
                act = False
            c.step(act)
        return float(c.score), (c.t if not c.dead else 0.0)

    def _lookahead(self, w: World, t_last_flap: float) -> bool:
        """在"拍"与"不拍"之间选：各自走完 look 个 tick，取分数高、活得更久的那个。"""
        if w.t < WARM_S:
            return False
        look = max(1, self.oracle_look)
        best_val, best_act = (-1.0, -1.0), False
        for act in (True, False):
            if act and (w.t - t_last_flap) < self.cooldown:
                continue
            c = copy.deepcopy(w)
            c.step(act)
            if c.dead:
                val = (float(c.score), 0.0)
            else:
                sc, al = self._rollout(c, look, self.cooldown)
                val = (max(float(c.score), sc), max(c.t, al))
            if val > best_val:
                best_val, best_act = val, act
        return best_act


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
    ap.add_argument("--ceil-boost", type=float, default=1.0,
                    help="bidi：天花板那一路的角尺寸放大倍数（让鸟更早开始下潜）")
    ap.add_argument("--max-climb", type=float, default=40.0,
                    help="相邻缺口允许的向上跳变上限（px）；很大 = 退回前端原始随机")
    ap.add_argument("--vent-gain", type=float, default=2.0,
                    help="腹侧（爬升）支在缺口明显在上方时的增益倍数")
    ap.add_argument("--vent-dev", type=float, default=120.0,
                    help="vent-gain 的线性作用尺度（px）")
    ap.add_argument("--game-slack", type=float, default=1.0,
                    help="每个脑 tick 推进物理步的比例；<1 用来复现页面的脑-物理时钟错配")
    ap.add_argument("--first-gap-extra", type=float, default=160.0,
                    help="第 2 根管子额外加长的间距（px）：实测撞管死亡全在第 2 根")
    ap.add_argument("--dors-scale", type=float, default=0.35,
                    help="背侧支（下潜）驱动权重；1.0=原样，0=关掉该支")
    ap.add_argument("--predict", type=int, default=0,
                    help="bidi：启用 TTC 前推预判（1=开；纯反应式不够聪明）")
    ap.add_argument("--predict-horizon", type=int, default=400,
                    help="预判推演的最大 tick 数")
    ap.add_argument("--vy-gate", type=int, default=1,
                    help="bidi：只在朝该面靠近时才驱动（1=开，0=关）")
    ap.add_argument("--bidi-groups", default=None,
                    help="bidi 用哪些感觉群，逗号分隔（默认 LC4,LPLC2）")
    ap.add_argument("--gap-margin", type=float, default=18.0,
                    help="bidi 模式：方向翻转的死区（px，默认 = 鸟身直径 22）")
    ap.add_argument("--span", type=float, default=0.6,
                    help="track 模式：缺口高度偏置的跨度（越小越激进地跟随缺口）")
    ap.add_argument("--base", type=float, default=0.6,
                    help="track 模式：腹侧支偏置下限（必须 >0，否则驱动掉到阈值以下）")
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
    ap.add_argument("--oracle-look", type=int, default=60,
                    help="lookahead 模式的前向搜索深度（tick）")
    ap.add_argument("--need-spikes", type=int, default=1)
    ap.add_argument("--trace-mode", default=None, help="打印该模式的逐 tick 轨迹")
    ap.add_argument("--trace-n", type=int, default=80)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    h = Harness(a.asset, gain=3.0, tonic=0.0, need_spikes=a.need_spikes,
                cooldown=a.cooldown, oracle_look=a.oracle_look)
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
                pipe_w=a.pipe_w, span=a.span, base=a.base, gap_margin=a.gap_margin,
                groups=(a.bidi_groups.split(",") if a.bidi_groups else None),
                vy_gate=a.vy_gate, max_climb=a.max_climb, ceil_boost=a.ceil_boost,
                predict_horizon=a.predict_horizon, dors_scale=a.dors_scale,
                vent_gain=a.vent_gain, vent_dev=a.vent_dev,
                first_gap_extra=a.first_gap_extra, game_slack=a.game_slack)
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
        # bidi_predict 模式 = 双向投射 + TTC 预判（同一个投射类，只多一个开关）
        spec["predict"] = 1 if mode == "bidi_predict" else a.predict
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
