"""子图数据加载：把 scripts/build_subgraph.py 产出的 npz 变成 GPU 就绪的图对象。

刻意只用 **numpy + torch**，不依赖 pandas/scipy —— 这样
`D:\\Code\\DQN\\env`（CUDA torch，但没有 pandas）也能直接跑训练。

对外暴露 `SubgraphData`：
    src, dst      (E,) int64   边端点（dst 收到 src 的消息）
    weight        (E,) float32 边权初始值（norm / count / binary）
    n_nodes       int
    sensory_idx   (S,) int64   输入注入的神经元
    motor_idx     (M,) int64   读出用的神经元
    meta          dict         规模、可达率等（用于记录实验）
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # flyflappy/
DATA = ROOT / "data"


@dataclass
class SubgraphData:
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    n_nodes: int
    sensory_idx: np.ndarray
    motor_idx: np.ndarray
    meta: dict = field(default_factory=dict)
    name: str = ""

    @property
    def n_edges(self) -> int:
        return int(len(self.src))

    @property
    def n_sensory(self) -> int:
        return int(len(self.sensory_idx))

    @property
    def n_motor(self) -> int:
        return int(len(self.motor_idx))

    @property
    def density(self) -> float:
        return self.n_edges / max(1, self.n_nodes * self.n_nodes)

    def param_count(self, hidden: int = 256, state_dim: int = 18, heads: int = 0) -> dict:
        """结构相关的**有效**可学参数（用于公平性对照）。

        注意：ConnectomeNet 在实现上把 A 存成稠密 (N,N) 参数（为了用 cuBLAS 提速），
        但非边位置由 mask 强制恒为 0，所以有效自由度就是真实突触条数 E，
        不是 N²。报告时必须用这个数，否则 4000²=16M 会严重高估。
        """
        return {
            "edges": self.n_edges,
            "w_in": self.n_sensory * (state_dim + 1),
            "readout": self.n_motor * hidden + hidden,
            "heads": heads,
            "total": self.n_edges + self.n_sensory * (state_dim + 1)
                     + self.n_motor * hidden + hidden + heads,
            "dense_matrix_slots": self.n_nodes * self.n_nodes,
        }

    def __repr__(self) -> str:
        return (f"<SubgraphData {self.name}: {self.n_nodes} 节点 / {self.n_edges:,} 边 "
                f"({self.density*100:.1f}%) / {self.n_sensory} 感觉 / {self.n_motor} 读出>")


def _weights(kind: str, norm: np.ndarray, count: np.ndarray) -> np.ndarray:
    if kind == "norm":
        return norm.astype(np.float32)
    if kind == "count":
        # 原始突触数是重尾的（中位 2 / p99 42 / max 4105），线性会爆炸，先 log1p 压一下
        return np.log1p(count).astype(np.float32)
    if kind == "binary":
        return np.ones_like(count, dtype=np.float32)
    raise ValueError(f"未知 weight_mode: {kind!r}")


def load_subgraph(name: str = "sg_collision_s300_n4000", weight_mode: str = "norm") -> SubgraphData:
    npz_path = DATA / f"{name}.npz"
    json_path = DATA / f"{name}.json"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"找不到 {npz_path}；先跑 scripts/build_subgraph.py --out {name}"
        )
    d = np.load(npz_path)
    meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}

    return SubgraphData(
        src=d["src"].astype(np.int64),
        dst=d["dst"].astype(np.int64),
        weight=_weights(weight_mode, d["norm"], d["count"]),
        n_nodes=int(d["node_ids"].shape[0]),
        sensory_idx=np.nonzero(d["is_sensory"])[0].astype(np.int64),
        motor_idx=np.nonzero(d["is_motor"])[0].astype(np.int64),
        meta=meta,
        name=name,
    )
