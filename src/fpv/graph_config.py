"""我们的配置扩展：在 vendored 的 AgentConfig 上追加「图网络」相关字段。

**刻意不改 vendored/config.py 里的 AgentConfig 默认值**，而是用子类 +
一个构造器，这样：
  - 原项目的 baseline/rainbow 预设原样可用（net_arch 默认 'mlp'）；
  - 我们自己的实验配置集中在这一个文件里，一眼能看全。
"""

from __future__ import annotations

from dataclasses import dataclass

from .vendor_flappyrl.config import AgentConfig, EnvConfig, TrainConfig


@dataclass
class GraphConfig(AgentConfig):
    """在 AgentConfig 基础上追加图网络字段。"""

    # 'mlp' = 原项目基线；'connectome' = 真实接线图；'rewired' = 度分布保持的随机图
    net_arch: str = "mlp"

    # 图传播步数 K（W_g 跨步复用，所以 K 不影响参数量）
    graph_steps: int = 3
    # 边权整体缩放（稳定训练用）
    graph_gain: float = 1.0
    # 边权初始化方式：'norm'（数据自带归一化）/ 'count'（原始突触数）/ 'binary'（全 1）
    weight_mode: str = "norm"
    # 是否按神经递质把 GABA/谷氨酸能边取负号
    signed_weights: bool = False
    # 读出时的池化：'full'（全连接读出）/ 'mean'（对下行神经元取均值）
    readout: str = "full"


def graph_config(**kw) -> GraphConfig:
    """一个稳健的起点：Double + Dueling，先别上 Noisy/PER/C51，把结构差异隔离出来。"""
    base = dict(
        # --- 与 baseline 对齐的部分 ---
        double=True,
        dueling=True,
        noisy=False,
        per=False,
        n_step=1,
        distributional=False,
        lr=1e-4,
        batch_size=128,
        buffer_size=100_000,
        learning_starts=2_000,
        target_update_freq=1_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=50_000,
        per_alpha=0.0,
    )
    base.update(kw)
    return GraphConfig(**base)


def env_config(**kw) -> EnvConfig:
    base = dict(n_envs=8, seed=0, auto_reset=True)
    base.update(kw)
    return EnvConfig(**base)


__all__ = ["GraphConfig", "graph_config", "env_config", "EnvConfig", "TrainConfig", "AgentConfig"]
