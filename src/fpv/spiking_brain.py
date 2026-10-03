"""阶段 2：真·脉冲（LIF）全脑仿真内核。

对照 pinme.dev 逆向出的动力学（见 _recon/REPORT.md A4）
--------------------------------------------------------
    e  = exp(-dt/tau)                     # 0.81873 每步泄漏
    r  = e*G[i] + gain*C[i] + tonic + P[i]
    if (rand < noise_hz*dt): r += noise_amp
    if r >= 1.0:  spike, S[i]=1, G[i]=0   # 阈值 1.0，硬重置（不是减法重置）
    else:         G[i]=r

其中：
    G      膜电位（leaky 状态）
    C      本步收到的突触电流（= 加权入边之和，符号已含在权重里）
    P      外部注入电流（感觉编码 / 或实验用的人工刺激）
    阈值 1.0，硬重置到 0，无显式不应期

和之前 ReLU 版的关键区别（这是负结果的根因之一）
-----------------------------------------------
1. **脉冲**：神经元只在越过阈值时发放一个瞬时事件，而不是连续的 ReLU 激活值。
2. **真泄漏**：膜电位按 exp(-dt/tau) 衰减，有真实的时间常数。
3. **符号**：GABA 能突触是**抑制性**的（负权重），能把膜电位往下拉——
   这是 ReLU + 全正权重永远做不到的（ReLU 把所有负值截断成 0，抑制完全失效）。

实现要点
--------
- CSR 用 scatter_add 在 GPU 上做：每个 tick 把活跃神经元的出边电流加到目标。
  这比稠密 (N,N) matmul 快得多（N=6 万时稠密矩阵是 36 亿槽位）。
- 突变电流用 `code` 解码：lut[code] = sign(code) * exp(ln_min*(1-mag/127))
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"


def encode_lut(ln_min: float, log_range: tuple[float, float] | None = None) -> torch.Tensor:
    """把 0..255 的量化 code 解码成真实权重。

    code < 128 : 兴奋，mag = code      -> +w
    code >= 128: 抑制，mag = code-128  -> -w
    code == 0  : 权重 0（未知符号 / 零强度）

    权重的解码规则必须与 build_spiking_asset.py 的量化规则严格互逆：
        w = exp(lo + (mag-1)/126 * (hi-lo))      当 log_range=(lo,hi) 给出
        w = exp(ln_min*(1 - mag/127))            否则（pinme.dev 兼容模式）

    ⚠️ 早期版本对全部 code 一律用 ln_min 公式，而我们的 norm 值远小于 1，
    导致解码出的权重几乎全是 exp(-11.69)≈8e-6 -> 突触电流累积不起来 ->
    DNp01 恒不发放（tune_io_coupling 全是 0）。这是那个 bug 的修复。
    """
    code = np.arange(256, dtype=np.int64)
    mag = np.where(code >= 128, code - 128, code).astype(np.float64)
    sign = np.where(code >= 128, -1.0, 1.0)
    if log_range is not None:
        lo, hi = float(log_range[0]), float(log_range[1])
        magc = np.clip(mag, 1, 127)
        w = np.exp(lo + (magc - 1.0) / 126.0 * (hi - lo))
    else:
        w = np.exp(ln_min * (1.0 - mag / 127.0))
    w = sign * w
    w[code == 0] = 0.0
    return torch.tensor(w, dtype=torch.float32)


class SpikingBrain:
    """冻结的 LIF 脉冲连接组。唯一的"学习"发生在外部的读出层。

    用法
    ----
        brain = SpikingBrain.from_npz("spiking_circuit", device="cuda")
        stim = torch.zeros(brain.N, device=...)
        stim[brain.key_idx("LC4")] = 0.8         # 视觉 loom 输入
        spikes = brain.step(stim)                 # (N,) bool
        rate = brain.population_rate(brain.key_idx("DNp01"), ticks=5)
    """

    def __init__(self, indptr, indices, codes, node_ids, meta, device="cpu"):
        self.device = torch.device(device)
        self.meta = meta
        self.N = int(meta["n_neurons"])
        self.ln_min = float(meta.get("ln_min", -11.694705))

        # CSR（转成 torch 张量常驻显存）
        self.indptr = torch.as_tensor(indptr, dtype=torch.long, device=self.device)
        self.indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        self.codes = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        self.node_ids = np.asarray(node_ids, dtype=np.int64)

        # code -> 权重查表（解码规则必须与构建端一致）
        log_range = meta.get("weight_log_range")
        self.lut = encode_lut(self.ln_min, tuple(log_range) if log_range else None).to(self.device)

        # 动力学参数（对齐 pinme.dev meta.bin 的 params）
        d = meta.get("dynamics", {})
        self.dt = float(d.get("dt", 0.02))
        self.tau = float(d.get("tau", 0.1))
        self.gain = float(d.get("gain", 3.0))
        self.tonic = float(d.get("tonic", 0.14))
        self.noise_hz = float(d.get("noise_hz", 1.2))
        self.noise_amp = float(d.get("noise_amp", 0.22))
        self.threshold = float(d.get("threshold", 1.0))
        self.leak = float(np.exp(-self.dt / self.tau))     # 0.81873

        # 状态
        self.G = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        self.S = torch.zeros(self.N, dtype=torch.bool, device=self.device)

        # 本步的突触电流缓冲（复用，避免每 tick 分配）
        self._cur = torch.zeros(self.N, dtype=torch.float32, device=self.device)

        # 关键神经元索引缓存
        self._key = meta.get("key_neurons", {})
        self._key_flat = {}
        for group in ("sensory", "motor"):
            for name, lst in self._key.get(group, {}).items():
                self._key_flat[name] = torch.as_tensor(lst, dtype=torch.long,
                                                       device=self.device)

    # ---------------------------------------------------------------- 构造
    @classmethod
    def from_npz(cls, name: str = "spiking_circuit", device: str = "cpu") -> "SpikingBrain":
        npz = DATA / f"{name}.npz"
        js = DATA / f"{name}.json"
        if not npz.exists():
            raise FileNotFoundError(f"找不到 {npz}；先跑 scripts/build_spiking_asset.py")
        d = np.load(npz)
        meta = json.loads(js.read_text(encoding="utf-8")) if js.exists() else {}
        return cls(d["indptr"], d["indices"], d["codes"], d["node_ids"], meta, device=device)

    def key_idx(self, name: str) -> torch.Tensor:
        """取某个命名神经元群的子图索引（LC4 / LPLC2 / DNp01 ...）。"""
        if name not in self._key_flat:
            raise KeyError(f"没有名为 {name!r} 的关键神经元群；"
                           f"可选：{sorted(self._key_flat)}")
        return self._key_flat[name]

    def key_names(self) -> list[str]:
        return sorted(self._key_flat)

    # ---------------------------------------------------------------- 传播
    def _synaptic_current(self) -> torch.Tensor:
        """把上一步发放的神经元的出边电流散射到目标（CSR 结构）。"""
        self._cur.zero_()
        src = torch.nonzero(self.S, as_tuple=False).flatten()
        if src.numel() == 0:
            return self._cur
        # 每个活跃神经元 src 的目标区间 [indptr[s], indptr[s+1])
        starts = self.indptr[src]
        ends = self.indptr[src + 1]
        counts = ends - starts
        total = int(counts.sum())
        if total == 0:
            return self._cur
        # 展开成逐个突触（向量化，无 python 循环）
        rep = torch.repeat_interleave(src, counts)
        # 目标索引：对每个 src，从 start 数到 end
        offset = torch.arange(total, device=self.device) - torch.repeat_interleave(
            torch.cumsum(counts, 0) - counts, counts)
        tgt_pos = torch.repeat_interleave(starts, counts) + offset
        tgt = self.indices[tgt_pos]
        w = self.lut[self.codes[tgt_pos]]
        self._cur.index_add_(0, tgt, w)
        return self._cur

    def step(self, stim: torch.Tensor | None = None, add_noise: bool = True,
             clamp: tuple | None = None) -> torch.Tensor:
        """推进一个 tick（20ms）。返回本步发放的布尔掩码 (N,)。

        stim: (N,) 外部注入电流（感觉编码 / 人工刺激），None 表示无。
        clamp: (idx, p) —— 把 idx 这批神经元的发放**强制**成伯努利(p) 抽样，
               无视它们自己的膜电位。用来精确设定上游输入群的发放率，
               从而测下游神经元对「已知输入率」的传递函数（逃避反射实验用）。
               被钳制的神经元仍会正常把脉冲传给下游，且发放后照常复位。

        注意：这里用 `with torch.no_grad():` 而不是 `@torch.no_grad()` 装饰器。
        在深调用栈（play_spiking 的 rollout 链）下，`@torch.no_grad()` 装饰器会
        因 `_contextlib` 反复构造而触发 RecursionError，显式上下文没有这个问题。
        """
        with torch.no_grad():
            C = self._synaptic_current()
            r = self.leak * self.G + self.gain * C + self.tonic
            if stim is not None:
                r = r + stim
            if add_noise and self.noise_amp > 0:
                # 泊松噪声：每 tick 以 noise_hz*dt 概率给某些神经元加 noise_amp
                p = self.noise_hz * self.dt
                n_noisy = int(np.random.binomial(self.N, p))
                if n_noisy:
                    idx = torch.randint(0, self.N, (n_noisy,), device=self.device)
                    r.index_add_(0, idx, torch.full((n_noisy,), self.noise_amp,
                                                    device=self.device))
            spike = r >= self.threshold
            if clamp is not None:
                cidx, cp = clamp
                prob = torch.as_tensor(cp, dtype=r.dtype, device=self.device)
                forced = torch.rand(cidx.numel(), device=self.device,
                                    dtype=r.dtype) < prob
                spike = spike.clone()
                spike[cidx] = forced
            self.G = torch.where(spike, torch.zeros_like(r), r)
            self.S = spike
            return spike

    def run(self, stim_fn=None, ticks: int = 1) -> list[torch.Tensor]:
        """连续跑 ticks 步；stim_fn(t) -> (N,) 可选。返回每步的发放掩码列表。"""
        out = []
        for t in range(ticks):
            out.append(self.step(stim_fn(t) if stim_fn else None))
        return out

    # ---------------------------------------------------------------- 读出
    def population_rate(self, idx: torch.Tensor, ticks: int = 5) -> float:
        """给定种群在最近 ticks 步里的平均发放率（0..1），用于运动解码。"""
        cnt = 0
        for _ in range(ticks):
            cnt += int(self.S[idx].sum())
        return cnt / max(1, idx.numel() * ticks)

    def reset(self) -> None:
        self.G.zero_()
        self.S.zero_()
        self._cur.zero_()
