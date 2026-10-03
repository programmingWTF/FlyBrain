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
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                                  # flyflappy/
REPO = ROOT.parent                                  # FlyBrain/
sys.path.insert(0, str(ROOT / "src"))

from fpv import looming                              # noqa: E402
from fpv.spiking_brain import SpikingBrain           # noqa: E402

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
LIT_TYPES = ["LC4", "DNp01", "DNp04", "LC10a", "DNp02", "DNp11"]


# ------------------------------------------------------------------ 坐标
def load_coords(fafb_ids: set[str]) -> dict[str, list[float]]:
    """fafb id -> 归一化到 [-1,1] 的解剖坐标（用 manifest 里该神经元粗几何的包围盒中心）。"""
    mf = json.loads((REPO / "data" / "brain" / "manifest.json").read_text(encoding="utf-8"))
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
    mf = json.loads((REPO / "data" / "brain" / "manifest.json").read_text(encoding="utf-8"))
    xs = mf["bbox"]
    ctr = np.array([(xs[0] + xs[3]) / 2, (xs[1] + xs[4]) / 2, (xs[2] + xs[5]) / 2])
    half = np.array([(xs[3] - xs[0]) / 2, (xs[4] - xs[1]) / 2, (xs[5] - xs[2]) / 2])
    pos_of = {str(int(i)): k for k, i in enumerate(np.asarray(brain.node_ids, dtype=np.int64))}
    arr = np.zeros((brain.N, 3), dtype=np.float32)
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


class Session:
    """一个持锁的活脑会话。"""

    def __init__(self, asset: str, device: str):
        self.lock = threading.Lock()
        self.base = SpikingBrain.from_npz(asset, device=device)
        self.base.gain, self.base.tonic = GAIN, TONIC
        self.asset = asset
        self.graph = "real"
        self.brain = self.base
        self.g = looming.resolve(self.base, looming.LOOM_SENSE
                                 + looming.ESCAPE_MOTOR + looming.CONTROL_SENSE
                                 + ["DNp02", "DNp11"])
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
            f = REPO / "viewer" / "vendor" / path[len("/vendor/"):]
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
    ap.add_argument("--asset", default="spiking_circuit")
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
        try:
            subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
