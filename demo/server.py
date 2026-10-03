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

import argparse
import json
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
            idx, pv = [], []
            eff = 0.0
            for spec in drives:
                if isinstance(spec, dict):
                    # 视野定位驱动。两种空间形状：
                    #   高斯窗（默认）：威胁只覆盖视野的一小块
                    #   hole：整面压过来的墙都驱动，只有"能穿过去的洞"那块不驱动
                    #       —— 第一人称走廊版用这个，因为墙才是覆盖大部分视野的逼近物
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
            cidx = torch.cat(idx)
            pvec = torch.cat(pv)
            dn01, dn04 = self.g["DNp01"], self.g["DNp04"]
            lit_hits: set[int] = set()
            for _ in range(ticks):
                self.brain.step(clamp=(cidx, pvec))
                self.dn_tick.append(int(self.brain.S[dn01].sum()))
                self.dn04_tick.append(int(self.brain.S[dn04].sum()))
                self.lr_tick.append({nm: int(self.brain.S[t].sum()) for nm, t in self.lr})
                # 本次请求覆盖的所有 tick 里，只要放过脉冲就标记（前端按帧点亮）
                lit_hits.update(self.brain.S[self.lit_assets].nonzero(
                    as_tuple=False).flatten().tolist())
                self.ticks_total += 1
            lc4_rate = float(self.brain.S[cidx].float().mean())
            # 回传**资产索引**（前端才能查到解剖坐标）
            vz = self.brain.S[self.viz_idx]
            recent = sum(self.dn_tick)
            out = dict(
                drive_max=float(max((float(d[1] if isinstance(d, (list, tuple))
                                     else d.get("amp", 0.0)) for d in drives),
                                    default=0.0)),
                eff=round(eff, 5), need=round((1 - self.base.leak)
                                               * self.base.threshold
                                               / self.base.gain, 5),
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
    srv = ThreadingHTTPServer((a.host, a.port), H)
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
