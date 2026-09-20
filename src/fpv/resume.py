"""检查点：保存/恢复**完整训练状态**，让长跑能被中断后续上。

为什么放在自己的包里而不是改 vendored agent：
  - vendored agent 是原项目的逐行副本，改得越少越好；
  - agent 已经把 online_net / target_net / optimizer / buffer / eps / beta
    这些字段都暴露成属性了，外部照样能存取。

存了什么（full 模式）
    nets          online/target 的 state_dict
    opt           AdamW 的 state（含动量，少了它会明显影响续训质量）
    buffer        回放缓冲的环形存储（只存已使用的那部分）+ PER 的 SumTree
    rng          python random / numpy / torch 的随机状态
    progress     step、eps、beta、近期分数、近期 loss、历史评测
    meta         cfg 等，用于校验「续跑的配置是否一致」

注意：Resume 后回放缓冲是完整恢复的，所以**不需要重新预热**。
"""
from __future__ import annotations

import pathlib
import random
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

VERSION = 2

# 稠密矩阵里其实只有 E 个非零（4000²=16M 槽位 vs 356k 条真实突触）。
# 直接存稠密会让一个检查点变成 150 MB（online 37 + target 37 + 优化器动量 75 MB）。
# 所以对「带 edge_mask 的网络」把这些矩阵**按掩码稀疏化**，读回时再重建稠密。
SPARSE_KEYS = ("A",)


def _sparsify_state(sd: dict, mask: torch.Tensor) -> dict:
    """把 state_dict 里的稠密 A 替换成掩码下的非零值（+形状），其余键原样保留。"""
    out = {}
    for k, v in sd.items():
        if k in SPARSE_KEYS and torch.is_tensor(v) and v.dim() == 2 and tuple(v.shape) == tuple(mask.shape):
            idx = mask.nonzero(as_tuple=False)
            out[k] = {"__sparse__": True, "shape": tuple(v.shape),
                      "idx": idx.cpu(), "vals": v[idx[:, 0], idx[:, 1]].cpu()}
        else:
            out[k] = v
    return out


def _densify_state(sd: dict, like: dict, mask: torch.Tensor, device) -> dict:
    """把 _sparsify_state 的结果还原成可以被 load_state_dict 接受的稠密张量。"""
    out = {}
    for k, v in sd.items():
        if isinstance(v, dict) and v.get("__sparse__"):
            ref = like[k]
            dense = torch.zeros(v["shape"], dtype=ref.dtype, device=device)
            idx = v["idx"].to(device)
            dense[idx[:, 0], idx[:, 1]] = v["vals"].to(device=device, dtype=ref.dtype)
            out[k] = dense
        else:
            out[k] = v
    return out


def _net_state(net) -> dict:
    sd = net.state_dict()
    mask = getattr(net, "edge_mask", None)
    return _sparsify_state(sd, mask) if mask is not None else sd


def _load_net_state(net, sd: dict) -> None:
    mask = getattr(net, "edge_mask", None)
    if mask is not None:
        sd = _densify_state(sd, net.state_dict(), mask, net.A.device if hasattr(net, "A") else "cpu")
    net.load_state_dict(sd)


def _opt_state(agent) -> dict:
    """优化器状态：对 A 的动量也只存非零。"""
    st = agent.optimizer.state_dict()
    net = agent.online_net
    mask = getattr(net, "edge_mask", None)
    if mask is None:
        return st
    idx = mask.nonzero(as_tuple=False)
    new_state = {}
    for pid, s in st["state"].items():
        ns = {}
        for k, v in s.items():
            if torch.is_tensor(v) and v.dim() == 2 and tuple(v.shape) == tuple(mask.shape):
                ns[k] = {"__sparse__": True, "shape": tuple(v.shape),
                         "idx": idx.cpu(), "vals": v[idx[:, 0], idx[:, 1]].cpu()}
            else:
                ns[k] = v
        new_state[pid] = ns
    return {"state": new_state, "param_groups": st["param_groups"]}


def _load_opt_state(agent, st: dict) -> None:
    net = agent.online_net
    mask = getattr(net, "edge_mask", None)
    if mask is None:
        agent.optimizer.load_state_dict(st)
        return
    dense_params = {p for pg in agent.optimizer.param_groups for p in pg["params"]}
    # 找到 A 对应的参数张量，用来做 dtype/device 参考
    a_param = None
    for name, p in net.named_parameters():
        if name in SPARSE_KEYS:
            a_param = p
            break
    new_state = {}
    for pid, s in st["state"].items():
        ns = {}
        for k, v in s.items():
            if isinstance(v, dict) and v.get("__sparse__"):
                ref = a_param
                dense = torch.zeros(v["shape"], dtype=ref.dtype, device=ref.device)
                idx = v["idx"].to(ref.device)
                dense[idx[:, 0], idx[:, 1]] = v["vals"].to(device=ref.device, dtype=ref.dtype)
                ns[k] = dense
            else:
                ns[k] = v
        new_state[pid] = ns
    agent.optimizer.load_state_dict({"state": new_state, "param_groups": st["param_groups"]})


def _buffer_state(buf) -> dict[str, Any] | None:
    """把回放缓冲序列化成可 torch.save 的 dict（只存已写入的部分）。"""
    st = getattr(buf, "store", None)
    if st is None:
        return None
    size = int(st.size)
    # 环形缓冲：已使用区间是 [0, size)（还没绕圈时）或整圈
    if size < st.capacity:
        sl = slice(0, size)
    else:
        sl = slice(0, st.capacity)
    out = {
        "kind": type(buf).__name__,
        "capacity": int(st.capacity),
        "state_dim": int(st.state_dim),
        "pos": int(st.pos),
        "size": size,
        "s": st.s[sl].copy(),
        "a": st.a[sl].copy(),
        "r": st.r[sl].copy(),
        "s2": st.s2[sl].copy(),
        "d": st.d[sl].copy(),
    }
    if hasattr(buf, "tree"):
        out["tree"] = buf.tree.tree.copy()
        out["tree_ptr"] = int(buf.tree.data_ptr)
        out["max_priority"] = float(getattr(buf, "max_priority", 1.0))
        out["alpha"] = float(getattr(buf, "alpha", 0.6))
    return out


def _restore_buffer(buf, st: dict[str, Any] | None) -> None:
    if not st:
        return
    ring = buf.store
    n = int(st["size"])
    ring.s[:n] = st["s"][:n]
    ring.a[:n] = st["a"][:n]
    ring.r[:n] = st["r"][:n]
    ring.s2[:n] = st["s2"][:n]
    ring.d[:n] = st["d"][:n]
    ring.pos = int(st["pos"])
    ring.size = n
    if "tree" in st and hasattr(buf, "tree"):
        buf.tree.tree[:] = st["tree"]
        buf.tree.data_ptr = int(st.get("tree_ptr", 0))
        buf.max_priority = float(st.get("max_priority", 1.0))


def save(agent, path: str | pathlib.Path, *, loop: int, cfg, extra: dict | None = None,
         skip_buffer: bool = False) -> pathlib.Path:
    """保存完整训练状态。

    参数名是 ``loop``（**循环数**，不是 env-step）：控制流用它，避免和
    ``--steps``（env-step）混用单位导致续训跑飞。旧参数名 ``step`` 仍兼容。
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": VERSION,
        "loop": int(loop),
        "step": int(loop),          # 兼容旧读取方
        "online": _net_state(agent.online_net),
        "target": _net_state(agent.target_net),
        "opt": _opt_state(agent),
        "eps": float(getattr(agent, "eps", 0.0)),
        "beta": float(getattr(agent, "beta", 0.0)),
        "cfg": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(cfg),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "extra": extra or {},
    }
    if not skip_buffer:
        payload["buffer"] = _buffer_state(agent.buffer)
    torch.save(payload, p)
    return p


def load(agent, path: str | pathlib.Path, device=None) -> dict[str, Any]:
    """恢复完整训练状态。返回 payload（含 step / eps / extra）供训练循环续上。"""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到检查点 {p}")
    payload = torch.load(p, map_location=device or agent.device, weights_only=False)
    if payload.get("version") != VERSION:
        raise ValueError(f"检查点版本不匹配：{payload.get('version')} != {VERSION}")

    agent.online_net.load_state_dict(_densify_state(
        payload["online"], agent.online_net.state_dict(),
        agent.online_net.edge_mask, getattr(agent.online_net, "A").device
    ) if hasattr(agent.online_net, "edge_mask") else payload["online"])
    agent.target_net.load_state_dict(_densify_state(
        payload["target"], agent.target_net.state_dict(),
        agent.target_net.edge_mask, getattr(agent.target_net, "A").device
    ) if hasattr(agent.target_net, "edge_mask") else payload["target"])
    agent.target_net.eval()
    try:
        _load_opt_state(agent, payload["opt"])
    except Exception as e:  # 优化器结构变了就跳过，但要说清楚
        print(f"  ! 优化器状态恢复失败（{type(e).__name__}: {e}），将用新优化器继续")

    if "eps" in payload:
        agent.eps = float(payload["eps"])
    if "beta" in payload:
        agent.beta = float(payload["beta"])

    if payload.get("buffer"):
        _restore_buffer(agent.buffer, payload["buffer"])
        print(f"  回放缓冲已恢复 {len(agent.buffer):,} 条（容量 {agent.buffer.store.capacity:,}）")
    else:
        print("  检查点不含回放缓冲 -> 续训需要重新预热")

    rng = payload.get("rng") or {}
    # 逐项恢复，互不牵连：CUDA 的 RNG 在跨进程恢复时容易出问题，
    # 不能因为它失败就把 python/numpy/torch 的 CPU 随机状态也一起跳过。
    restored = []
    for name, setter in (
        ("python", lambda s: random.setstate(s)),
        ("numpy", lambda s: np.random.set_state(s)),
        # torch 的 RNG 状态是 ByteTensor，且 set_rng_state 明确要求**CPU** 张量
        # （map_location 到 cuda 时会变成 cuda 张量，那样会报 "RNG state must be a torch.ByteTensor"）
        ("torch", lambda s: torch.set_rng_state(s.detach().to("cpu", torch.uint8))),
    ):
        if rng.get(name) is None:
            continue
        try:
            setter(rng[name])
            restored.append(name)
        except Exception as e:
            print(f"  ! RNG[{name}] 恢复失败：{type(e).__name__}: {e}")
    if rng.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
            restored.append("cuda")
        except Exception as e:
            print(f"  ! RNG[cuda] 恢复失败（不影响 CPU 随机性）：{type(e).__name__}: {e}")
    if restored:
        print(f"  RNG 已恢复：{', '.join(restored)}")

    # 续训后网络权重里非边位置应当是 0（被 mask 过），这里顺手校验
    if hasattr(agent.online_net, "edge_mask"):
        with torch.no_grad():
            off = agent.online_net.A[~agent.online_net.edge_mask]
            mx = float(off.abs().max()) if off.numel() else 0.0
        if mx != 0.0:
            print(f"  ! 警告：恢复后非边位置不全为 0（max {mx:.2e}），已重新 mask")
            agent.online_net.mask_edges_()
            agent.target_net.mask_edges_()

    return payload
