#!/usr/bin/env python
"""逃跑反射可交互演示 —— 后端。

设计原则：**不重写仿真**。这里只是把已经验证过的
`src/fpv/spiking_brain.py` + `src/fpv/looming.py` 挂到一个 HTTP 接口上，
所以网页上看到的每一个数字，都和 scripts/loom_*.py 量出来的是同一套代码产出的。

浏览器负责游戏物理（60fps），每帧请求本服务推进若干个 20ms tick 并取回：
  - DNp01 在最近 100ms 内的脉冲数  -> 是否拍翅（纯反射，无任何学习）
  - LC4 群实测发放率、逼近角 θ、角扩张速度 dθ/dt、剩余碰撞时间 τ
  - 用于 3D 点亮的发放神经元索引

可以实时切换的干预（都是 scripts 里已验证过的对照）：
  real / cut（删掉 LC4->DNp01 的 102 条直接边）/ shuffled（目标全局洗牌）
  / no_inhibition（负权清零）；以及 LC4 灵敏度 s50。

用法
----
    D:/Code/FlyBrain/env/python.exe demo/server.py            # 然后浏览器打开 http://127.0.0.1:8620
    D:/Code/FlyBrain/env/python.exe demo/server.py --asset spiking_full --port 8620
"""
from __future__ import annotations

# ⚠️ 必须在 import torch 之前锁线程数。这个仿真的热点是**成千上万个极小的
# 张量操作**（一次 index_add、一次 346 元素的 gather），torch 默认按逻辑核数
# 开满线程池，每个小操作都要唤醒并自旋等待 —— 实测 16 线程时 CPU 时间
# 2.76 ms/tick(约 3.6 个核)却比单线程的 0.57 ms/tick(1.0 个核)**更慢**。
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import pathlib
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                                  # 仓库根
sys.path.insert(0, str(ROOT / "src"))

# ---- 可选资产：全脑 manifest 与前端 vendor ----
#
# ⚠️ 历史背景：这个 demo 原来住在 `<大仓库>/flyflappy/demo/`，而全脑资产在
#    `<大仓库>/data/brain/` 与 `<大仓库>/viewer/vendor/`，当时 `REPO = ROOT.parent`
#    正好是大仓库根。后来项目被拆成**只含 Flappy 的独立仓库**（`flyflappy/`
#    提为仓库根），`ROOT.parent` 就指到仓库外面去了。
# 现在两边都能跑：优先用**本仓库自带**的，找不到再退回老的大仓库布局。
#
# ⚠️ 三条路径**必须一起改**，它们是配套的：
#      ① 服务器读 manifest（localize_coords / load_manifest）
#      ② 3D 点云端点 /api/coords.bin（依赖 ① 产出的 self.coords）
#      ③ 前端 three.js 的 importmap（demo/index.html）+ 本文件的 /vendor/ 路由
#    漏改任何一条的后果：**服务器照样起得来、Flappy 照样能玩**，但右侧脑图
#    要么空白、要么 404 —— 不会报错，所以很容易被当成"面板就这样"。


def _find_optional(*cands: pathlib.Path) -> pathlib.Path | None:
    """按顺序返回第一个存在的**文件**；都不在就返回 None（不抛异常）。"""
    for c in cands:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


def _find_dir(*cands: pathlib.Path) -> pathlib.Path | None:
    """按顺序返回第一个存在的**目录**；都不在就返回 None。"""
    for c in cands:
        try:
            if c.is_dir():
                return c
        except OSError:
            continue
    return None


#: 全脑 manifest：骨架包围盒中心 → 解剖坐标。**没有它游戏照常，只是脑图点云为空。**
def find_manifest() -> pathlib.Path | None:
    env = os.environ.get("FLAPPY_BRAIN_MANIFEST")
    return _find_optional(
        *([pathlib.Path(env)] if env else []),
        ROOT / "data" / "brain" / "manifest.json",         # 独立 Flappy 仓库（自带）
        ROOT.parent / "data" / "brain" / "manifest.json",  # 老的大仓库布局
    )


#: three.js 所在目录。demo/vendor 优先（自包含），再退回 viewer/vendor。
def find_vendor() -> pathlib.Path | None:
    return _find_dir(
        HERE / "vendor",                       # 本仓库自带 ← 首选
        ROOT / "viewer" / "vendor",
        ROOT.parent / "viewer" / "vendor",
    )

from fpv import looming                              # noqa: E402
from fpv.spiking_brain import SpikingBrain           # noqa: E402

#: 两次拍翅的最小间隔（秒）—— 与 demo/app.js 的 FLAP_COOLDOWN 一致。
#: 服务端自持回路由 GameWorld 自己管这个门；评测台那边由 Harness 统一管。
FLAP_COOLDOWN = 0.14
#: 服务端仿真速率（Hz）。**固定 50**（= 1/20ms，评测台的"真实时间"）。
#: 这是整个重构的目的：脑与物理在同一个进程里按这个节拍 1:1 锁步，
#: HTTP 只用来取状态，所以游戏速度与网络、与浏览器定时器都无关。
SIM_HZ = 50.0

GAIN, TONIC = 3.0, 0.0            # 带载标定过的工作点
# ---- 游戏几何（必须与 demo/app.js 的常数一致；双向投射要用）----
G_H = 620.0                       # 画布高
GROUND = 92.0                     # 地面高度
GROUND_Y = G_H - GROUND           # 地面线 y = 528
PX_PER_M = 240.0                  # 世界尺度（= app.js 的 PX_PER_M）
# ---- 双向逼近反射的投射参数（与 scripts/flappy_bench.py 的默认值一致）----
#: dors_scale / vent_gain 是本轮调出来的两个旋钮，性质上**是外部介入**，
#: 不是连接组里量出来的 —— 报告里必须这么写（见 ESCAPE.md §6.7 的"分账"）。
#:   dors_scale=0.35：背侧（下潜）支按 0.35 缩。实测越强越差（1.0→撞天花板 24/80），
#:                    0.35 时撞天花板归零、均分 31→41。
#:   vent_gain=2.0 ：缺口明显在上方时把腹侧（爬升）驱动放大。实测撞管死亡**全部**
#:                    是"偏低没爬够"，放大后均分 40→62、存活满 69/80。
BIDI = dict(s50size=30.0, n=3.0, gain=1.0, gap_margin=18.0, vy_gate=1,
            ceil_boost=1.0, groups=["LC4", "LPLC2"],
            dors_scale=0.35, vent_gain=2.0, vent_dev=120.0)
#: 相邻缺口的向上跳变上限（px）—— 与 bench 的 max_climb 一致：
#: 这是"关卡按执行器带宽生成"那一条（ESCAPE.md 6.3）
MAX_CLIMB = 40.0
#: 第 2 根管子的额外间距（px）。实测调完上面两个旋钮后，剩下的撞管死亡
#: **全部集中在第 2 根**（13/22，死亡时刻 3.16s ≈ 第 2 根到达），且都是"偏低没爬够"。
#: 给它更多时间即可：≤2 分局占比 12% → 6%。这同样是**改游戏**，单独报账。
FIRST_GAP_EXTRA = 160.0
WINDOW = 5                        # 100ms = 5 个 20ms tick
VIZ_SAMPLE = 12000                # 全脑"点亮"用的随机抽样规模
# 会被【点名】点亮（白/彩点）的关键神经元类型。
# 必须包含 Flappy 实际驱动的两个感觉群：LC4 和 LPLC2。
# 之前漏了 LPLC2 —— 而它到 DNp01 的权重比 LC4 还大（Sigma_w 0.1043 vs 0.0744），
# 也就是脑图里最主要的驱动细胞其实没被点亮（用户问过这个）。
LIT_TYPES = ["LC4", "LPLC2", "DNp01", "DNp04", "LC10a", "DNp02", "DNp11"]


# ------------------------------------------------------------------ 坐标
def load_manifest() -> dict | None:
    """读全脑 manifest；**不在就返回 None**（不抛异常）。

    没有 manifest 时，视觉面板会退化成"所有点堆在原点"，但 Flappy 游戏、
    脑仿真、分数**完全不受影响** —— 因为 manifest 只提供解剖坐标。
    """
    p = find_manifest()
    if p is None:
        print("  [提示] 没找到 data/brain/manifest.json —— 3D 脑图点云将为空，"
              "其余功能正常。要恢复脑图：把该文件放到 data/brain/ 下，"
              "或设环境变量 FLAPPY_BRAIN_MANIFEST 指向它。", flush=True)
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                      # noqa: BLE001
        print(f"  [警告] manifest 读取失败（{p}）：{e} —— 3D 脑图将为空", flush=True)
        return None


def coords_npz() -> pathlib.Path | None:
    """预烤的坐标文件 `data/coords.npz`（由 `scripts/bake_coords.py` 生成）。

    它把 manifest 里我们真正需要的三件事压成 ~1.6 MB：
      · 每个神经元的归一化解剖坐标（右侧 3D 脑图按**真实解剖位置**摆点）
      · 关键神经元的坐标（点亮那 556 个）
      · 关键群的资产索引（LC4 / LPLC2 / DNp01 ...）
    有它就**完全不需要 51.5 MB 的 manifest**，启动也更快 —— 这是仓库自包含的关键。
    """
    env = os.environ.get("FLAPPY_COORDS_NPZ")
    return _find_optional(
        *([pathlib.Path(env)] if env else []),
        ROOT / "data" / "coords.npz",
        ROOT.parent / "data" / "coords.npz",
    )


def precomputed() -> dict | None:
    """读 `data/coords.npz`（一次性缓存）。没有任何预烤文件时返回 None。"""
    global _PRECOMPUTED
    if _PRECOMPUTED is not _UNSET:
        return _PRECOMPUTED                       # type: ignore[return-value]
    p = coords_npz()
    if p is None:
        _PRECOMPUTED = None
        return None
    try:
        with np.load(p) as d:
            _PRECOMPUTED = {k: np.asarray(d[k]) for k in d.files}
    except Exception as e:                        # noqa: BLE001
        print(f"  [警告] 预烤坐标读取失败（{p}）：{e} —— 退回 manifest", flush=True)
        _PRECOMPUTED = None
    return _PRECOMPUTED


_UNSET = object()
_PRECOMPUTED: object = _UNSET


def load_coords(fafb_ids: set[str]) -> dict[str, list[float]]:
    """fafb id -> 归一化到 [-1,1] 的解剖坐标（用 manifest 里该神经元粗几何的包围盒中心）。

    优先用预烤的 `data/coords.npz`；没有才退回读 manifest。
    """
    pre = precomputed()
    if pre is not None and "lit" in pre and "lit_node_id" in pre:
        ids = pre["lit_node_id"].astype(np.int64)
        xyz = pre["lit"].astype(np.float32)
        return {str(int(i)): [float(v) for v in xyz[k]]
                for k, i in enumerate(ids) if str(int(i)) in fafb_ids}
    mf = load_manifest()
    if mf is None:
        return {}
    xs = mf["bbox"]
    ctr = np.array([(xs[0] + xs[3]) / 2, (xs[1] + xs[4]) / 2, (xs[2] + xs[5]) / 2])
    half = np.array([(xs[3] - xs[0]) / 2, (xs[4] - xs[1]) / 2, (xs[5] - xs[2]) / 2])
    out: dict[str, list[float]] = {}
    secs = mf["coarse"]["sections"]
    for i, n in enumerate(mf["neurons"]):
        nid = str(n["id"])
        if nid not in fafb_ids or i >= len(secs):
            continue
        b = secs[i].get("bbox")
        if not b:
            continue
        c = np.array([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2])
        out[nid] = ((c - ctr) / half).tolist()
    return out


# ------------------------------------------------------------------ 会话
def build_all_coords(brain) -> np.ndarray:
    """(N,3) float32：每个资产神经元的归一化解剖坐标（没有几何的填 0）。

    这是"能碾压 pinme 那个 demo"的关键：它的 16 万个点是按黄金角螺旋**摆**出来的，
    我们这里每个点都是 FlyWire 骨架的真实包围盒中心。
    """
    pre = precomputed()
    if pre is not None and "all" in pre:
        arr = pre["all"].astype(np.float32)
        if arr.shape == (brain.N, 3):
            return arr
        print(f"  [警告] 预烤坐标形状 {arr.shape} 与资产 {brain.N} 不符 —— 退回 manifest",
              flush=True)
    mf = load_manifest()
    arr = np.zeros((brain.N, 3), dtype=np.float32)
    if mf is None:
        return arr                                # 没有解剖坐标 → 全零（_retino_groups 会自行跳过）
    xs = mf["bbox"]
    ctr = np.array([(xs[0] + xs[3]) / 2, (xs[1] + xs[4]) / 2, (xs[2] + xs[5]) / 2])
    half = np.array([(xs[3] - xs[0]) / 2, (xs[4] - xs[1]) / 2, (xs[5] - xs[2]) / 2])
    pos_of = {str(int(i)): k for k, i in enumerate(np.asarray(brain.node_ids, dtype=np.int64))}
    secs = mf["coarse"]["sections"]
    for i, n in enumerate(mf["neurons"]):
        j = pos_of.get(str(n["id"]))
        if j is None or i >= len(secs):
            continue
        b = secs[i].get("bbox")
        if not b:
            continue
        arr[j] = (np.array([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2,
                            (b[2] + b[5]) / 2]) - ctr) / half
    return arr


class GameWorld:
    """Flappy 的游戏物理与关卡生成 —— **服务端权威**版本。

    为什么要在服务端再写一遍（而不是继续用前端的 `demo/app.js`）
    ------------------------------------------------------------
    前端那版跑不出恒定速度，根因是结构性的：**游戏时钟必须挂在某个外部节奏上**，
    而浏览器里可选的只有两个，实测都不行：

        setTimeout(4)          → 29.8 次/s（被浏览器压到 ~33ms）
        requestAnimationFrame  → 28.2 次/s

    更糟的是 HTTP 往返中位 60ms、p90 86ms、最快 4ms（差十几倍），
    于是"这一步物理该配多少时间"时多时少 → 表现就是"时快时慢"。
    试过的补救（固定节拍+队列预取、大批量预取、预取与消费拆分）都不稳，
    因为**只要决策要靠一次跨进程往返拿到，时钟就一定是抖的**。

    放到服务端之后：脑与物理在**同一个进程、同一个锁**里按 1:1 跑固定 50 Hz，
    没有任何网络或定时器节拍进入游戏时钟。HTTP 只用来**取状态**，
    取晚了只会让画面晚一点，不会改变游戏速度。

    物理与碰撞判据与 `demo/app.js`、`scripts/flappy_bench.py` **逐字一致**
    （`demo/verify_server_collision.js` 逐像素守着「服务端判定 vs 前端画法」；
    `scripts/verify_server_physics.py` 守着这里与评测台逐 tick 等价）。
    """

    #: 与 app.js / flappy_bench 一致的常数（改这里必须同时改那两处）
    G_W, G_H = 520, 620
    BIRD_X = 120
    BIRD_R = 17
    GROUND = 92
    GAP = 184
    PIPE_W = 62
    PX_PER_M = 240.0
    SPACING = 300.0
    GRAV = 1180.0
    FLAP_V = -340.0
    TICK_S = 0.02
    WARM_S = 0.8

    def __init__(self, *, max_climb: float = 40.0, first_gap_extra: float = 160.0,
                 seed: int | None = None):
        self.max_climb = float(max_climb)
        self.first_gap_extra = float(first_gap_extra)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.y = 300.0
        self.vy = 0.0
        self.t = 0.0
        self.pipes: list[dict] = []
        self.score = 0
        self.best = getattr(self, "best", 0)
        self.dead = False
        self.cause = ""
        self.spawned = 0
        self.cooldown = 0.0
        self.flap_t = 0.0                      # 最近一次拍翅的游戏时刻（给前端画翅膀）
        # 开局那根：与 app.js 一致 —— top=250，位置 birdX + 0.75m - pipeW
        self.pipes.append(dict(x=self.BIRD_X + 0.75 * self.PX_PER_M - self.PIPE_W,
                               top=250.0, passed=False))
        self.spawned += 1

    # ---------------------------------------------------------------- 关卡生成
    def spacing(self) -> float:
        """本根之后那根的间距。第 2 根用 first_gap_extra 拉长（与 bench 一致）。"""
        if self.first_gap_extra > 0 and self.spawned == 1:
            return self.SPACING + self.first_gap_extra
        return self.SPACING

    def _spawn(self, top: float) -> None:
        self.pipes.append(dict(x=self.G_W + 30.0, top=float(top), passed=False))
        self.spawned += 1

    def _next_gap_top(self) -> float:
        """相邻缺口的**向上**跳变限在 max_climb 以内（向下不设限）。

        这不是降低难度，而是排除物理上不可达的关卡：鸟的可持续爬升率约 100 px/s，
        管距 300px/240px·s⁻¹ = 1.25 s，一个间隔最多爬约 125px；而 `70 + rand*330`
        的缺口跳变有 330px 量程 —— 超出的关卡怎么飞都进不去。见 ESCAPE.md §6。
        """
        lo = 70.0
        hi = self.G_H - self.GAP - 150.0                     # = 286
        if not self.pipes:
            return float(self.rng.uniform(lo, hi))
        prev_c = self.pipes[-1]["top"] + self.GAP / 2.0
        top_max = min(hi, prev_c + self.max_climb - self.GAP / 2.0)
        if top_max < lo:
            top_max = lo
        return float(self.rng.uniform(lo, top_max))

    # ---------------------------------------------------------------- 几何查询
    def nearest_pipe(self):
        best = None
        for p in self.pipes:
            if p["x"] + self.PIPE_W > self.BIRD_X - 6 and (best is None or p["x"] < best["x"]):
                best = p
        return best

    def gap_center(self) -> float:
        p = self.nearest_pipe()
        return (p["top"] + self.GAP / 2.0) if p else (self.G_H / 2.0)

    # ---------------------------------------------------------------- 物理
    def step(self, flap: bool) -> None:
        """推进恰好一个 tick。`flap` 是**已经过冷却门**的最终决策。

        ⚠️ 这个函数必须与 `scripts/flappy_bench.py` 的 `World.step` **逐字等价**：
        它只做物理，**不管冷却**。原因是评测台那边冷却由 harness 统一管，
        `World.step(flap)` 收到什么就无条件执行什么。
        我最初把冷却塞进这里，结果 `scripts/verify_server_physics.py` 立刻抓到：
        评测台 `vy=-316.4`（= -340 + 1180×0.02），服务端 `-292.8`（拍翅被吞）。
        所以冷却门移到调用方（`Session._game_tick`），两边形状就一致了。
        """
        dt = self.TICK_S
        if self.dead:
            return
        self.t += dt
        if flap:
            self.vy = self.FLAP_V
            self.flap_t = self.t
        self.vy += self.GRAV * dt
        self.y += self.vy * dt
        warm = self.t < self.WARM_S
        for p in self.pipes:
            p["x"] -= self.PX_PER_M * dt
            if not p["passed"] and p["x"] + self.PIPE_W < self.BIRD_X:
                p["passed"] = True
                self.score += 1
                self.best = max(self.best, self.score)
        self.pipes = [p for p in self.pipes if p["x"] > -self.PIPE_W - 10]
        last = self.pipes[-1] if self.pipes else None
        if last is None or last["x"] < self.G_W - self.spacing():
            self._spawn(self._next_gap_top())
        if warm:
            # 加速期只限制在画面内，不判碰撞（与 app.js 的 warm 分支逐字对应）
            self.y = min(max(self.y, 40.0), float(self.G_H - self.GROUND - self.BIRD_R - 1))
            return
        # ---- 碰撞：像素级判据，与 app.js / flappy_bench / DQN 一致
        if self.y + self.BIRD_R >= self.G_H - 14:
            return self.die("撞到地面")
        if self.y - self.BIRD_R <= 0:
            return self.die("撞到天花板")
        for p in self.pipes:
            cx = max(p["x"], min(self.BIRD_X, p["x"] + self.PIPE_W))
            if (self.BIRD_X - cx) ** 2 > self.BIRD_R ** 2:
                continue
            if (self.y - self.BIRD_R <= p["top"] + self.BIRD_R
                    or self.y + self.BIRD_R >= p["top"] + self.GAP - self.BIRD_R - 1):
                return self.die("撞上管子")

    def die(self, cause: str) -> None:
        self.dead = True
        self.cause = cause

    def snapshot(self, **extra) -> dict:
        """给前端的状态快照。刻意做小（前端每帧都要取）。"""
        d = {
            "t": round(self.t, 4),
            "y": round(self.y, 2),
            "vy": round(self.vy, 2),
            "score": self.score,
            "best": self.best,
            "dead": self.dead,
            "cause": self.cause,
            "flap_t": round(self.flap_t, 4),
            # 只报可见的管子（画面外的不必发）
            "pipes": [[round(p["x"], 2), round(p["top"], 2)]
                      for p in self.pipes if p["x"] > -self.PIPE_W - 4],
        }
        d.update(extra)
        return d


class Session:
    """一个持锁的活脑会话。"""

    def __init__(self, asset: str, device: str):
        self.lock = threading.Lock()
        self.base = SpikingBrain.from_npz(asset, device=device)
        self.base.gain, self.base.tonic = GAIN, TONIC
        self.asset = asset
        self.graph = "real"
        self.brain = self.base
        # 关键群：优先用预烤文件里存好的（那样连 meta 的 cell_type 都不需要）；
        # 没有就按老路走 meta 解析。
        _pre = precomputed()
        if _pre and "key_LC4" in _pre:
            self.g = {k[4:]: torch.as_tensor(_pre[k], dtype=torch.long,
                                             device=self.base.device)
                      for k in _pre if k.startswith("key_")}
        else:
            self.g = looming.resolve(self.base, looming.LOOM_SENSE
                                     + looming.ESCAPE_MOTOR + looming.CONTROL_SENSE
                                     + ["DNp02", "DNp11"])
        # ---- 关键群自检：**必须在 _build_lit() 之前**。
        # 两种都要查，缺一不可：
        #   ① 群**不存在**（空数组）—— 例如不含 LC4 的子图
        #   ② 群存在但**索引越界** —— 这个更阴：`data/coords.npz` 里的 `key_*`
        #      数组是为**全脑 144,837 个神经元**预烤的。换成更小的资产
        #      （`spiking_circuit` 只有 59,548 个）时这些索引会整体越界，
        #      而 `len() != 0` 的检查**完全看不出来**。
        #      于是报出一个跟根因毫无关系的错，把人往"坐标文件坏了"的方向带：
        #          IndexError: index 61485 is out of bounds for axis 0 with size 59548
        N = self.base.N
        _missing = [nm for nm in ("LC4", "LPLC2", "DNp01") if len(self.g.get(nm, [])) == 0]
        _oob = [nm for nm, v in self.g.items()
                if len(v) and (int(v.min()) < 0 or int(v.max()) >= N)]
        if _missing or _oob:
            why = []
            if _missing:
                why.append("缺少关键群：" + "、".join(_missing))
            if _oob:
                why.append("预烤坐标里的索引越界（" + "、".join(_oob[:4]) + " …）："
                           f"该资产只有 {N:,} 个神经元，"
                           "而 data/coords.npz 是为全脑 144,837 个预烤的")
            raise SystemExit(
                f"\n❌ 资产 `{asset}` 不能用：\n   " + "\n   ".join(why) +
                "\n\n   Flappy 的感觉输入全部打在 LC4/LPLC2 上，输出只读 DNp01。\n"
                "   缺了它们分数会**恒为 0**，而且不会报错 —— 很容易被当成"
                "「反射不行」。\n"
                "   请用全脑资产：--asset spiking_full\n")
        rng = np.random.default_rng(0)
        self.viz_idx = torch.as_tensor(
            rng.choice(self.base.N, size=min(VIZ_SAMPLE, self.base.N), replace=False),
            dtype=torch.long, device=self.base.device)
        self.lit = self._build_lit()
        self.coords = build_all_coords(self.base)
        self.g2 = dict(self.g)
        self.g2.update(self._retino_groups())
        self.w_dn01 = self._weights_to_dn01()
        self._wmap_cache: dict = {}
        self.last_plan: dict = {}
        self.reset()
        # ---- 服务端锁步仿真（见 GameWorld 的说明）
        self.game = GameWorld(max_climb=MAX_CLIMB, first_gap_extra=FIRST_GAP_EXTRA)
        self.game_lock = threading.Lock()
        self.game_stats = dict(ticks=0, flaps=0, rate=0.0, dead=0, restarts=0)
        self._stop = False
        self._thread = threading.Thread(target=self._game_loop, daemon=True,
                                        name="flappy-sim")
        self._thread.start()

    # ------------------------------------------------------------ 服务端锁步仿真
    def _game_tick(self) -> None:
        """推进**恰好一个 tick**：按当前世界算驱动 → 走一个脑 tick → 用 DNp01
        的发放决定拍不拍 → 走一步物理。

        这就是与 `scripts/flappy_bench.py` **同构**的控制回路，也是整个重构的要点：
        决策与物理步严格 1:1 配对（先算驱动、再问脑、再用这个决策走物理），
        而且全程在**同一个进程、同一个锁**里 —— 没有任何网络往返进入时钟。
        """
        w = self.game
        gap_c = w.gap_center()
        body = dict(BIDI)                       # 与前端同一套默认参数（从 /api/info 读同一份）
        body.update(y=w.y, vy=w.vy, gap=gap_c, ground_y=GROUND_Y)
        idx, pv, _eff = self._bidi_plan(body)
        clamp = (torch.cat(idx), torch.cat(pv)) if idx else None
        self.brain.step(clamp=clamp)
        dn = int(self.brain.S[self.g["DNp01"]].sum())
        self.dn_tick.append(dn)
        flap = sum(self.dn_tick) >= 1
        # 手动接管（空格/点击）：人在场时优先于人，否则玩了半天没反应
        if getattr(self, "_manual_flap", False):
            self._manual_flap = False
            flap = True
        # ---- 冷却门：与 demo/app.js 的 `if (flap && G.cooldown <= 0)` 同构 ——
        #      冷却期内这一拍**什么都不做**（不是把 vy 改成别的值）。
        #      它必须在**这里**而不是 GameWorld 里：GameWorld 要与评测台的
        #      `World.step(flap)` 逐字等价（那个是无条件的），见 GameWorld.step 的注释。
        w.cooldown = max(0.0, w.cooldown - 0.02)
        if flap and w.cooldown > 0:
            flap = False
        elif flap:
            w.cooldown = FLAP_COOLDOWN
        if flap:
            self.game_stats["flaps"] += 1
        w.step(flap)
        self.game_stats["ticks"] += 1

    def _game_loop(self) -> None:
        """固定 50 Hz 的仿真线程。

        用**墙钟差值**决定这一轮走几个 tick（而不是"每次醒过来走一步"），
        所以即使这个线程被系统调度得忽快忽慢，游戏速度依然恒定 ——
        这是与前端那版最大的区别：前端没有任何一个可信的墙钟节拍可用
        （setTimeout 被压到 ~30 次/s、rAF 在后台会被完全暂停）。
        """
        step_s = 1.0 / SIM_HZ
        last = time.perf_counter()
        carry = 0.0
        rate_t0, rate_n = last, 0
        while not self._stop:
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0.5:
                dt = 0.5                            # 挂起/暂停后不要一次补太多
            carry += dt
            n = 0
            with self.game_lock:
                while carry >= step_s and n < 12:   # 上限防螺旋
                    if self.game.dead:
                        # 撞了：停一小会儿再重开，让玩家看见"撞了"
                        break
                    self._game_tick()
                    carry -= step_s
                    n += 1
                    rate_n += 1
                if self.game.dead:
                    if not hasattr(self, "_dead_since"):
                        self._dead_since = now
                    elif now - self._dead_since > 1.6:
                        self.game.reset()
                        self.reset()                # 脑也归零，下一局从零开始
                        self.game_stats["restarts"] += 1
                        del self._dead_since
                        carry = 0.0
            if now - rate_t0 >= 1.0:
                self.game_stats["rate"] = round(rate_n / (now - rate_t0), 1)
                self.game_stats["dead"] = 1 if self.game.dead else 0
                rate_t0, rate_n = now, 0
            # 睡到下一个节拍（留一点余量，避免忙等）
            slack = step_s - (time.perf_counter() - now)
            time.sleep(slack if slack > 0.0005 else 0.0005)

    def game_snapshot(self) -> dict:
        """前端每帧取的状态快照。刻意做小。"""
        with self.game_lock:
            st = dict(self.game_stats)
            snap = self.game.snapshot(**st)
            snap["sim_hz"] = SIM_HZ
            snap["need"] = 1
            # 最近一次脑读出的几个数（前端画曲线用）
            snap["dn01_recent"] = int(sum(self.dn_tick)) if self.dn_tick else 0
            snap["plan"] = dict(self.last_plan)
            # ---- 3D 脑图要点亮的细胞。
            # ⚠️ 这几个量原来由**前端自己的脑循环**从 /api/step 的响应里拿。
            #    改成服务端权威之后那条路没了，脑图会**静默变空**（不报错，
            #    很容易被当成"面板就这样"）。所以在这里补回来。
            S = self.brain.S
            snap["viz_spike"] = [int(i) for i in self.viz_idx[S[self.viz_idx].bool()].tolist()]
            lit = self.lit_assets[S[self.lit_assets].bool()]
            snap["lit_spike"] = [int(i) for i in lit.tolist()]
            # need 是**膜电位阈值**（不是 need_spikes 那个计数）—— 前端用它算
            # "到达阈值的比例"来画驱动条，两处口径必须一致
            snap["need"] = round((1 - self.base.leak) * self.base.threshold
                                 / self.base.gain, 5)
            # eff：本 tick 从 DNp01 读出的驱动量（与 /api/step 同口径）
            snap["eff"] = round(
                (1 - self.base.leak) * float(self.base.threshold)
                * float(self.brain.S[self.g["DNp01"]].float().sum())
                / max(self.base.gain, 1e-9), 5)
            return snap

    def game_reset(self) -> None:
        with self.game_lock:
            self.game.reset()
            self.reset()
            self.game_stats["restarts"] += 1
            if hasattr(self, "_dead_since"):
                del self._dead_since

    def _build_lit(self):
        """给前端用的『可点亮神经元』清单：资产索引 + 类型 + 侧别 + 解剖坐标。"""
        ids = np.asarray(self.base.node_ids, dtype=np.int64)
        want: dict[str, dict] = {}
        for nm in LIT_TYPES:
            for i in self.g[nm].detach().cpu().numpy():
                want[str(int(ids[int(i)]))] = {"asset": int(i), "type": nm}
        coords = load_coords(set(want))
        out = []
        for nid, d in want.items():
            if nid in coords:
                out.append(dict(id=nid, type=d["type"], asset=d["asset"],
                                xyz=[round(v, 4) for v in coords[nid]]))
        # 预张量化，供 step() 每次直接取这批细胞的发放状态（前端按此顺序点亮）
        self.lit_assets = torch.as_tensor([d["asset"] for d in out], dtype=torch.long,
                                          device=self.base.device)
        return out

    def _retino_groups(self):
        """按**解剖**把关键群再切分，供"苍蝇真实视野"模式用。

        LC4 是柱状神经元，其树突在叶上按视野位置拼贴（retinotopy）。
        所以 104 个 LC4 的包围盒中心，**方差最大的那根解剖轴**就是它的视野轴；
        按该轴的中位数把 LC4 分成"视野上半 / 下半"两半 —— 这是数据自己选出来的轴，
        不是我硬指定 x/y/z。
        """
        dev = self.base.device
        gs: dict[str, torch.Tensor] = {}
        self.retino_axis: dict[str, dict] = {}
        self.retino_u: dict[str, torch.Tensor] = {}
        self.retino_u_idx: dict[str, torch.Tensor] = {}
        c = self.coords
        for nm in ("LC4", "LPLC2", "LC10a"):
            ix = self.g[nm].detach().cpu().numpy()
            pts = c[ix]
            ok = np.abs(pts).sum(1) > 1e-6
            if ok.sum() < 8:
                continue
            # ⚠️ 不能取"全体方差最大的轴"：那是 x 轴，而 x 完美区分左右半球
            # （分侧 AUC=0/1），拿它当视野轴等于把"左脑/右脑"说成"视野上/下"。
            # 正确做法：先认半球轴（方差最大那根）并剔除它，再在**同侧内部**
            # 取方差最大的轴作为视野轴，u 在每个半球内部各自归一化。
            hemi = int(np.argmax(pts[ok].std(0)))
            sgn = np.sign(pts[:, hemi])
            cand = [a for a in range(3) if a != hemi]
            axis = max(cand, key=lambda a: np.mean([
                pts[sgn == s, a].std() for s in (-1, 1) if (sgn == s).sum() > 3]))
            u = np.zeros(len(ix), dtype=np.float32)
            for s in (-1, 1):
                m = sgn == s
                if m.sum() < 2:
                    continue
                v = pts[m, axis]
                u[m] = (v - v.min()) / max(v.max() - v.min(), 1e-9)
            spread = pts[ok].std(axis=0)
            self.retino_axis[nm] = {"hemi_axis": "xyz"[hemi], "field_axis": "xyz"[axis],
                                    "std_within_side": [round(float(v), 3) for v in spread]}
            v = pts[:, axis]
            med = np.median(v[ok])
            gs[f"{nm}_hi"] = torch.as_tensor(ix[u >= 0.5], dtype=torch.long, device=dev)
            gs[f"{nm}_lo"] = torch.as_tensor(ix[u < 0.5], dtype=torch.long, device=dev)
            # 归一化到 0..1 的"视野位置"坐标：威胁落在哪个位置，就只驱动
            # 偏好位置与之重叠的那批 LC4（这才是真实视网膜会发生的事）
            self.retino_u[nm] = torch.as_tensor(u, dtype=torch.float32, device=dev)
            self.retino_u_idx[nm] = self.g[nm]
        sd = looming.resolve_side(self.base, ["DNp01", "DNp02", "DNp04", "DNp11"])
        self.lr = []                                # 需要逐侧计数的读出群
        for nm, d in sd.items():
            for side, t in d.items():
                gs[f"{nm}_{side}"] = t
                if side in ("left", "right"):
                    self.lr.append((f"{nm}_{side}", t))
        # 堆成一个矩阵，逐 tick 只做一个操作（而不是每群一个 int() 同步）
        self.lr_sizes = [int(t.numel()) for _, t in self.lr]
        self.lr_stack = (torch.cat([t for _, t in self.lr]) if self.lr
                         else torch.zeros(0, dtype=torch.long, device=self.base.device))
        return gs

    def _weights_to_dn01(self) -> dict:
        """每个可驱动群的**逐细胞**权重：该细胞 -> DNp01(两只取均值) 的突触权重和。
        决定逃逸指令是否发放的是这个加权和，不是细胞个数占比。"""
        lut = self.base.lut.detach().cpu().numpy()
        indptr = self.base.indptr.detach().cpu().numpy()
        indices = self.base.indices.detach().cpu().numpy()
        codes = self.base.codes.detach().cpu().numpy()
        dn = self.g["DNp01"].detach().cpu().numpy()
        out = {}
        for name, t in self.g2.items():
            grp = t.detach().cpu().numpy()
            w = np.zeros(len(grp), dtype=np.float32)
            for k, s_ in enumerate(grp):
                lo, hi = int(indptr[s_]), int(indptr[s_ + 1])
                if hi <= lo:
                    continue
                sel = np.isin(indices[lo:hi], dn)
                if sel.any():
                    w[k] = float(lut[codes[lo:hi][sel]].sum())
            out[name] = torch.as_tensor(w / max(len(dn), 1), device=self.base.device)
        return out

    # ---- 干预开关
    def set_graph(self, name: str):
        with self.lock:
            if name == "real":
                self.brain = self.base
            elif name == "cut":
                self.brain = looming.variant(self.base, cut=(self.g["LC4"], self.g["DNp01"]))
            elif name == "shuffled":
                self.brain = looming.variant(self.base, shuffle_seed=7)
            elif name == "no_inhibition":
                self.brain = looming.variant(self.base, block_inhibition=True)
            else:
                raise ValueError(name)
            self.graph = name
            self.reset()

    def reset(self):
        self.brain.reset()
        self.dn_tick = deque(maxlen=WINDOW)
        self.dn04_tick = deque(maxlen=WINDOW)
        self.lr_tick = deque(maxlen=WINDOW)
        self.ticks_total = 0

    # ---- 感觉投射（与 scripts/flappy_bench.py 同一套公式，不许各写一遍）
    def _wmap(self, gname: str) -> torch.Tensor:
        """该群逐细胞到 DNp01 的权重（每只细胞口径，与 loom_ventral_probe 一致）。"""
        if gname not in self._wmap_cache:
            self._wmap_cache[gname] = self.w_dn01[gname]
        return self._wmap_cache[gname]

    def _bidi_plan(self, b: dict) -> tuple[list, list, float]:
        """**双向逼近反射**：视野哪一半 + 是否正在靠近 → 驱动腹侧或背侧半群。

        与 `scripts/flappy_bench.py` 的 `BiDirectionalProjection` 逐行对应：
          · 威胁量 = 2·atan(0.55 / 距离) 的角尺寸，过 LPLC2 的 Naka-Rushton(s50size)
          · 方向：缺口比鸟高一个死区 → 腹侧（u<0.5）；比鸟低一个死区 → 背侧（u>=0.5）
          · vy_gate：地面只在下落时逼近、天花板只在上升时逼近（纯物理）
          · **整半招募**（不是平滑窗）—— 实测平滑斜坡窗只有 need 的 53%，不发放
        这几条都是 ESCAPE.md §7 量出来的，不是这里另调的参数。
        """
        y = float(b["y"])
        vy = float(b["vy"])
        gap = float(b["gap"])
        s50size = float(b.get("s50size", 30.0))
        n = float(b.get("n", 3.0))
        margin = float(b.get("gap_margin", 18.0))
        vy_gate = int(b.get("vy_gate", 1))
        boost = float(b.get("ceil_boost", 1.0))
        dors_scale = float(b.get("dors_scale", 1.0))
        vent_gain = float(b.get("vent_gain", 1.0))
        vent_dev = float(b.get("vent_dev", 120.0))
        groups = b.get("groups") or ["LC4", "LPLC2"]

        # ---- 方向（纯几何）
        if gap < y - margin:
            up = True
        elif gap > y + margin:
            up = False
        else:
            self.last_plan = dict(branch="aligned", theta=0.0, amp=0.0, eff=0.0)
            return [], [], 0.0                      # 已对准：不驱动
        # ---- vy_gate：只有正在朝那一面靠近才是逼近刺激
        if vy_gate and ((up and vy <= 0) or ((not up) and vy >= 0)):
            self.last_plan = dict(branch="gated", theta=0.0, amp=0.0, eff=0.0)
            return [], [], 0.0
        # ---- 角尺寸（地面用离地高度，天花板用离顶高度）
        h_px = (float(b["ground_y"]) - y) if up else y
        h_px = max(h_px, 1.0)
        dist = max(h_px / PX_PER_M, 0.02)
        theta = math.degrees(2 * math.atan2(0.55, dist))
        if not up:
            theta *= boost
        x = max(theta, 0.0) ** n
        amp = float(b.get("gain", 1.0)) * (x / (x + s50size ** n))
        # ---- 两条支路各自的权重（这两个旋钮是**外部介入**，不是连接组事实）
        if not up:
            # 背侧（下潜）支：实测越强越差（它会把鸟推去撞天花板）
            amp *= dors_scale
        elif vent_gain != 1.0:
            # 腹侧（爬升）支：缺口明显在上方时放大 —— 撞管死亡 100% 是"偏低没爬够"
            dev = max(0.0, y - gap)
            sc = min(1.0, dev / max(vent_dev, 1e-6))
            amp = min(1.0, amp * (1.0 + (vent_gain - 1.0) * sc))
        # ---- 整半招募
        idx, pv, eff = [], [], 0.0
        for gname in groups:
            u = self.retino_u[gname]
            m = (u >= 0.5) if not up else (u < 0.5)
            t = self.retino_u_idx[gname][m]
            p = torch.full((int(m.sum()),), amp, device=self.brain.device)
            idx.append(t)
            pv.append(p)
            eff += float((self._wmap(gname)[m.cpu().numpy()] * amp).sum())
        self.last_plan = dict(branch="ventral" if up else "dorsal",
                              theta=round(theta, 1), amp=round(amp, 4),
                              eff=round(eff, 5))
        return idx, pv, eff

    def _plan_drives(self, drives: list) -> tuple[list, list, float]:
        """把请求里的驱动描述翻译成 (逐细胞索引, 逐细胞发放率, Σw·p)。

        支持三种写法：
          ["LC4", 0.8]                       整群同一个发放率（老写法）
          {"group","amp","center","width",...} 视野高斯窗 / hole（老写法）
          {"group","half":"ventral|dorsal","amp"}  整半招募（新：双向反射用）
          {"type":"bidi", ...}               双向逼近反射（新）
        """
        idx: list = []
        pv: list = []
        eff = 0.0
        for spec in drives:
            if isinstance(spec, dict) and spec.get("type") == "bidi":
                i2, p2, e2 = self._bidi_plan(spec)
                idx += i2
                pv += p2
                eff += e2
            elif isinstance(spec, dict) and "half" in spec:
                gname = str(spec["group"])
                u = self.retino_u[gname]
                m = (u >= 0.5) if str(spec["half"]).startswith("d") else (u < 0.5)
                t = self.retino_u_idx[gname][m]
                amp = float(spec.get("amp", 1.0))
                idx.append(t)
                pv.append(torch.full((int(m.sum()),), amp, device=self.brain.device))
                eff += float((self._wmap(gname)[m.cpu().numpy()] * amp).sum())
            elif isinstance(spec, dict):
                gname = str(spec["group"])
                t = self.g2.get(gname, self.retino_u_idx[gname])
                u = self.retino_u[gname]
                amp, c, w = (float(spec.get(k, d)) for k, d in
                             (("amp", 1.0), ("center", 0.5), ("width", 0.25)))
                g = torch.exp(-((u - c) / max(w / 2.355, 1e-6)) ** 2 / 2.0)
                shape = (1.0 - g) if spec.get("hole") else g
                idx.append(t)
                pv.append(shape * amp)
                eff += float((self.w_dn01[gname] * shape * amp).sum())
            else:
                name, p = spec
                t = self.g2[str(name)]
                idx.append(t)
                pv.append(torch.full((t.numel(),), float(p), device=self.brain.device))
                eff += float(self.w_dn01[str(name)].sum()) * float(p)
        return idx, pv, eff

    # ---- 主循环
    def step(self, req: dict) -> dict:
        """推进脑若干 tick。两种驱动写法：

        旧（简化逼近）: drive=["LC4"]，p 由 dist/speed/radius 单一算出
        新（真实视野）: drives=[["LC4_hi", .83], ["LC10a", .2], ...]
            每个群各自一个发放率 —— 用来把**径向扩张**喂给逼近通道、
            把**横向平移**喂给非逼近通道。文献里 LC4/LPLC2 对平移不响应，
            这个分解正是"苍蝇真实看到管子时会不会躲"要检验的东西。

        时间窗改成**按 tick** 记（100ms = 5 tick）。之前按"请求"记，
        而每个请求含的 tick 数会变，窗口长度不稳定。
        """
        ticks = max(1, int(req.get("ticks", 1)))
        n = float(req.get("n", 3.0))
        drives = req.get("drives")
        extra: dict = {}
        if drives is None:
            radius = float(req.get("radius", 0.05))
            speed = max(float(req.get("speed", 1.0)), 0.0)
            dist = max(float(req.get("dist", 1.0)), 1e-3)
            s50 = float(req.get("s50", 30.0))
            source = req.get("source", "dtheta")
            tau_s = dist / max(speed, 1e-6)
            theta = float(np.degrees(2 * np.arctan(radius / dist)))
            dtheta = float(np.degrees(2 * radius * speed / (dist ** 2 + radius ** 2)))
            var = {"dtheta": dtheta, "theta": theta, "invtau": 1.0 / tau_s}[source]
            p = float(np.clip(var ** n / (var ** n + s50 ** n), 0.0, 1.0)) if s50 > 0 else 0.0
            drives = [[g, p] for g in req.get("drive", ["LC4"])]
            extra = dict(theta_deg=round(theta, 2), dtheta_dps=round(dtheta, 1),
                         tau_s=round(tau_s, 3), drive=round(p, 4))

        with self.lock:
            idx, pv, eff = self._plan_drives(drives)
            # 空驱动是合法状态（双向反射里"已对准"和"vy_gate 关掉"都会返回空）：
            # 此时不钳制任何细胞，脑照常按自己的动力学跑。别去 cat 空列表。
            clamp = None
            if idx:
                clamp = (torch.cat(idx), torch.cat(pv))
            dn01, dn04 = self.g["DNp01"], self.g["DNp04"]
            # 累加全部留在张量里做，循环内**一次都不回 Python**
            acc_dn = torch.zeros((), device=self.brain.device)
            acc_dn04 = torch.zeros((), device=self.brain.device)
            acc_lr = torch.zeros(len(self.lr), device=self.brain.device)
            acc_lit = torch.zeros(self.lit_assets.numel(), device=self.brain.device)
            for _ in range(ticks):
                self.brain.step(clamp=clamp)
                s_ = self.brain.S
                acc_dn += s_[dn01].sum()
                acc_dn04 += s_[dn04].sum()
                acc_lr += s_[self.lr_stack].float().view(len(self.lr_sizes), -1).sum(1)
                acc_lit += s_[self.lit_assets].float()
                self.ticks_total += 1
            dn_now = int(acc_dn)
            self.dn_tick.append(dn_now)
            self.dn04_tick.append(int(acc_dn04))
            lit_hits = {i for i, v in enumerate(acc_lit.tolist()) if v > 0}
            lc4_rate = (float(self.brain.S[clamp[0]].float().mean()) if clamp is not None
                        else 0.0)
            lr_row = dict(zip([n for n, _ in self.lr], [int(v) for v in acc_lr.tolist()]))
            self.lr_tick.append(lr_row)
            # 回传**资产索引**（前端才能查到解剖坐标）
            vz = self.brain.S[self.viz_idx]
            recent = sum(self.dn_tick)
            out = dict(
                drive_max=round(float(self.last_plan.get("amp", 0.0)), 4)
                if self.last_plan else
                float(max((float(d[1] if isinstance(d, (list, tuple))
                                 else d.get("amp", 0.0)) for d in drives), default=0.0)),
                eff=round(eff, 5), need=round((1 - self.base.leak)
                                               * self.base.threshold
                                               / self.base.gain, 5),
                plan=dict(self.last_plan),      # 双向投射：哪一支 / 角尺寸 / Σw·p
                lc4_rate=round(lc4_rate, 4),
                dn01_recent=recent, dn04_recent=sum(self.dn04_tick),
                flap=bool(recent >= int(req.get("need_spikes", 1))),
                lr={k: sum(d.get(k, 0) for d in self.lr_tick)
                    for k in self.lr_tick[-1]} if self.lr_tick else {},
                graph=self.graph, ticks=self.ticks_total,
                viz_spike=self.viz_idx[vz].tolist()[:1500],
                lit_spike=sorted(lit_hits),
                **extra,
            )
        return out

    def info(self) -> dict:
        ids = np.asarray(self.base.node_ids, dtype=np.int64)
        return dict(
            asset=self.asset, graph=self.graph, n_neurons=int(self.base.N),
            n_synapses=int(len(self.base.codes)),
            dt_ms=self.base.dt * 1000, gain=self.base.gain, tonic=self.base.tonic,
            groups={k: int(v.numel()) for k, v in self.g.items()},
            retino={k: int(v.numel()) for k, v in self.g2.items()
                    if k not in self.g},
            retino_axis=getattr(self, "retino_axis", None),
            lit=self.lit, r50=0.577,
            # 前端要用这些常数复现引擎的关卡生成与双向投射（别在前端另写一套数）
            bidi=dict(BIDI), max_climb=MAX_CLIMB, first_gap_extra=FIRST_GAP_EXTRA,
            geom=dict(g_h=G_H, ground=GROUND, ground_y=GROUND_Y, px_per_m=PX_PER_M),
        )


# ------------------------------------------------------------------ HTTP
class H(BaseHTTPRequestHandler):
    sess: Session = None
    demo_dir: pathlib.Path = HERE
    # 默认 HTTP/1.0 = 每个请求新建一条 TCP 连接；本地 demo 每帧都发请求，
    # 实测把脑拖到 24% 实时速度。开 1.1 + keep-alive 才跑得动。
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _static(self, path: str):
        if path in ("/", ""):
            path = "/index.html"
        if path.startswith("/vendor/"):
            name = path[len("/vendor/"):]
            vend = find_vendor()
            f = (vend / name) if vend else (self.demo_dir / "vendor" / name)
        else:
            f = self.demo_dir / path.lstrip("/")
        try:
            f = f.resolve()
            assert f.is_file() and (str(f).startswith(str(self.demo_dir))
                                    or str(f).startswith(str(REPO / "viewer" / "vendor")))
        except Exception:
            self.send_error(404, "not found")
            return
        ctype = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".json": "application/json",
                 ".png": "image/png"}.get(f.suffix.lower(), "application/octet-stream")
        data = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # 本地开发用：一律不缓存，否则改了 app.js 浏览器还在跑旧版本
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/info"):
            return self._json(self.sess.info())
        # ---- 服务端权威游戏状态：前端每帧取这个来渲染（见 GameWorld 的说明）
        if self.path.startswith("/api/game/state"):
            return self._json(self.sess.game_snapshot())
        if self.path.startswith("/api/coords.bin"):
            data = self.sess.coords.tobytes(order="C")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)
            return None
        return self._static(self.path.split("?")[0])

    def do_POST(self):
        ln = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(ln) or b"{}")
        except Exception:
            return self._json({"error": "bad json"}, 400)
        try:
            if self.path.startswith("/api/step"):
                return self._json(self.sess.step(req))
            # 重开一局（前端"再来一局"按钮）
            if self.path.startswith("/api/game/reset"):
                self.sess.game_reset()
                return self._json({"ok": True})
            # 手动拍翅：**必须由服务端执行**。前端自己改 vy 会被下一次状态同步覆盖 ——
            # 这是"服务端权威"下最容易踩的坑。
            if self.path.startswith("/api/game/flap"):
                with self.sess.game_lock:
                    self.sess._manual_flap = True
                return self._json({"ok": True})
            if self.path.startswith("/api/config"):
                if "graph" in req:
                    self.sess.set_graph(req["graph"])
                return self._json({"ok": True, "graph": self.sess.graph})
            if self.path.startswith("/api/reset"):
                self.sess.reset()
                return self._json({"ok": True})
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        return self._json({"error": "no route"}, 404)


def main() -> int:
    ap = argparse.ArgumentParser()
    # ⚠️ 默认值是 **spiking_full**，不是 spiking_circuit。
    #    这个默认值曾经是个**静默失败陷阱**：spiking_circuit 子图不含 LC4/LPLC2，
    #    而 Flappy 的所有感觉输入都打在这两个群上 —— 用它的后果是
    #    **分数恒为 0、脑一次都不拍翅，而且不报任何错**。
    #    现在只有 Flappy 一个模式了，没有理由再默认到那个子图。
    ap.add_argument("--asset", default="spiking_full",
                    help="脉冲脑资产。**必须含 LC4/LPLC2**，否则分数恒 0 且不报错")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--port", type=int, default=8620)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    print(f"载入冻结脉冲脑 {a.asset} …")
    sess = Session(a.asset, a.device)
    print(f"  N={sess.base.N:,}  E={len(sess.base.codes):,}  "
          f"用时 {time.time()-t0:.1f}s")
    print("  " + "  ".join(f"{k}={v}" for k, v in sess.info()["groups"].items()))
    print(f"  可点亮神经元（有解剖坐标的）：{len(sess.lit)}")

    # 关键群自检已经在 Session.__init__ 里做了（那里能给出更准的错误位置）。
    g = sess.info()["groups"]
    print(f"  ✅ 关键群齐全：LC4={g.get('LC4')} LPLC2={g.get('LPLC2')} "
          f"DNp01={g.get('DNp01')}")

    H.sess = sess

    # ⚠️ 两个必须的服务器设置（都是实测踩出来的）：
    #
    # 1) `request_queue_size`：socketserver 默认只有 **5**。客户端（网页）以
    #    ~60 次/秒发 /api/step，而每次脑仿真要 ~10ms（全脑 1502 万突触），
    #    于是监听队列经常是满的 —— 满的时候连接请求被丢弃/挂起。
    #    实测：并发 8 个请求，7 个 ~80ms、**1 个卡 4.7 秒**；浏览器里表现为
    #    fetch 3.7 秒才 settle，inFlight 长期为 true，脑一个 tick 都推不动，
    #    页面分数停在 0~1（而评测台 111）。这是页面"看起来不聪明"的真正原因。
    #
    # 2) `daemon_threads`：否则 Ctrl+C 退出时会等残留连接线程。
    class _Srv(ThreadingHTTPServer):
        request_queue_size = 256
        daemon_threads = True
        allow_reuse_address = True

    srv = _Srv((a.host, a.port), H)
    url = f"http://{a.host}:{a.port}/"
    print(f"\n✓ 打开 {url}\n  （Ctrl+C 退出）\n")
    if not a.no_browser:
        # 用标准库 webbrowser —— 跨平台。
        # 原来写的是 `subprocess.Popen(["cmd", "/c", "start", "", url])`，那是
        # **Windows 专有**的：在 Linux 上会抛 FileNotFoundError（被下面的 except 吞掉），
        # 结果是"浏览器静默不打开"，很难察觉。服务器上仍然建议加 --no-browser。
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
