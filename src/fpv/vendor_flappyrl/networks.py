"""网络定义 —— 本项目自己的版本（**不是** D:\Code\DQN 里的那一份，那边保持原样）。

包含三个东西：
  1. ``NoisyLinear`` / ``RainbowNet`` —— 与 D:\\Code\\DQN\\flappyrl\\networks.py **逐行等价**的副本，
     作为对照组基线，保证「MLP 基线」和原项目跑出来的是同一个东西。
  2. ``ConnectomeNet`` —— 实验组：同样接口，但把 MLP 躯干换成「沿 FlyWire 真实接线做稀疏消息传递」。
  3. ``build_net(cfg)`` —— 按 ``cfg.net_arch`` 分发，供 vendored agent.py 调用。

输出契约（必须和 RainbowNet 完全一致，agent.py 依赖它）：
    forward(x)      -> (B, A, atoms)  若 distributional，否则 (B, A)
    distribution(x) -> (B, A, atoms)  原子上的 softmax
    expected_q(x)   -> (B, A)
    reset_noise()                        （Noisy 层存在时才有实际作用）
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AgentConfig


# ============================================================================
# 1) 基线：与原项目逐行等价的 MLP / Dueling / Noisy / C51 网络
# ============================================================================
class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer (Fortunato et al., 2018)."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


def _linear(cfg: AgentConfig, in_f: int, out_f: int) -> nn.Module:
    if cfg.noisy:
        return NoisyLinear(in_f, out_f, std_init=cfg.noisy_std)
    return nn.Linear(in_f, out_f)


class RainbowNet(nn.Module):
    """Dueling + (optional) Noisy + (optional) Distributional Q-network."""

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.num_atoms = cfg.num_atoms
        self.dist = cfg.distributional
        self.dueling = cfg.dueling

        if self.dist:
            self.register_buffer("atoms", torch.linspace(cfg.v_min, cfg.v_max, cfg.num_atoms))
            self.delta = (cfg.v_max - cfg.v_min) / (cfg.num_atoms - 1)

        self.fc1 = _linear(cfg, cfg.state_dim, cfg.hidden)
        self.ln1 = nn.LayerNorm(cfg.hidden)
        self.fc2 = _linear(cfg, cfg.hidden, cfg.hidden)
        self.ln2 = nn.LayerNorm(cfg.hidden)
        self.act = nn.ReLU(inplace=True)

        if self.dueling:
            if self.dist:
                self.val_fc = _linear(cfg, cfg.hidden, cfg.num_atoms)
                self.adv_fc = _linear(cfg, cfg.hidden, cfg.action_dim * cfg.num_atoms)
            else:
                self.val_fc = _linear(cfg, cfg.hidden, 1)
                self.adv_fc = _linear(cfg, cfg.hidden, cfg.action_dim)
        else:
            if self.dist:
                self.out_fc = _linear(cfg, cfg.hidden, cfg.action_dim * cfg.num_atoms)
            else:
                self.out_fc = _linear(cfg, cfg.hidden, cfg.action_dim)

        self._init_orthogonal()

    def _init_orthogonal(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                nn.init.orthogonal_(m.weight_mu, gain=math.sqrt(2))
                nn.init.zeros_(m.bias_mu)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.ln1(self.fc1(x)))
        h = self.act(self.ln2(self.fc2(h)))
        return self._heads(h)

    def _heads(self, h: torch.Tensor) -> torch.Tensor:
        """从特征 h 出 Q / 分布。ConnectomeNet 复用这一段，保证两个网络只有躯干不同。"""
        if self.dueling:
            if self.dist:
                v = self.val_fc(h).view(-1, 1, self.num_atoms)
                a = self.adv_fc(h).view(-1, self.action_dim, self.num_atoms)
                q = v + a - a.mean(dim=1, keepdim=True)
            else:
                v = self.val_fc(h)
                a = self.adv_fc(h)
                q = v + a - a.mean(dim=1, keepdim=True)
        else:
            if self.dist:
                q = self.out_fc(h).view(-1, self.action_dim, self.num_atoms)
            else:
                q = self.out_fc(h)
        return q

    def distribution(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def expected_q(self, x: torch.Tensor) -> torch.Tensor:
        if self.dist:
            p = self.distribution(x)
            return (p * self.atoms.view(1, 1, -1)).sum(dim=-1)
        return self.forward(x)


# ============================================================================
# 2) 实验组：连接组稀疏图躯干
# ============================================================================
class ConnectomeNet(RainbowNet):
    """把 MLP 躯干换成「沿 FlyWire 真实接线做 K 步稀疏消息传递」。

    数据流：
        x (18)  --W_in-->  感觉神经元(300)      每个状态维度投到一部分感觉神经元
                                    |
                    K 步：h <- ReLU( Ŵ h )      Ŵ = W_g ⊙ A，A 是固定真实接线
                                    |
        读出(1301 下行神经元) --R--> (hidden)  --Dueling/C51 头--> Q

    与 MLP 的关键差异：
      * 连接是**稀疏且固定**的（只有生物学上真实存在的突触）；
      * ``W_g`` 每条真实突触一个标量，**K 步复用同一套**（= 用固定参数换深度）；
      * 输入投到哪个神经元、从哪读出来是**可学**的，图本身只是骨架。
    """

    def __init__(
        self,
        cfg: AgentConfig,
        src: np.ndarray,
        dst: np.ndarray,
        weight: np.ndarray,
        n_nodes: int,
        sensory_idx: np.ndarray,
        motor_idx: np.ndarray,
    ):
        # 不走 RainbowNet.__init__ 的 MLP 躯干，但保留它的头与 C51 缓冲
        nn.Module.__init__(self)
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.num_atoms = cfg.num_atoms
        self.dist = cfg.distributional
        self.dueling = cfg.dueling
        if self.dist:
            self.register_buffer("atoms", torch.linspace(cfg.v_min, cfg.v_max, cfg.num_atoms))
            self.delta = (cfg.v_max - cfg.v_min) / (cfg.num_atoms - 1)

        self.n_nodes = int(n_nodes)
        self.steps = int(getattr(cfg, "graph_steps", 3))
        self.gain = float(getattr(cfg, "graph_gain", 1.0))
        self.sensory = torch.as_tensor(np.asarray(sensory_idx, dtype=np.int64))
        self.motor = torch.as_tensor(np.asarray(motor_idx, dtype=np.int64))
        self.register_buffer("sensory_buf", self.sensory, persistent=False)
        self.register_buffer("motor_buf", self.motor, persistent=False)

        # 边端点（固定结构）。方向：A[dst, src] = 该突触权重
        #   -> 传播 h_new = relu(h @ A)，正好把 src 的活动加权送到 dst。
        e_src = torch.as_tensor(np.asarray(src, dtype=np.int64))
        e_dst = torch.as_tensor(np.asarray(dst, dtype=np.int64))
        self.register_buffer("edge_src", e_src, persistent=False)
        self.register_buffer("edge_dst", e_dst, persistent=False)

        base = np.asarray(weight, dtype=np.float32).copy()

        # 边权表示：**稠密 (N,N) 参数 + 结构掩码**
        # 实测（scripts/bench_propagation.py, RTX 5060 Ti, K=3 batch=128，前向+反向）：
        #   h[:, src]*w + index_add  52.3 ms   ← 最直白，带宽杀手
        #   torch.sparse.mm(Coo)     10.9 ms   ← 稀疏对 4000² 这种小矩阵开销过高
        #   h @ A_dense               4.2 ms   ← 胜出（密度仅 2.2%，但 cuBLAS 极快）
        # 非边位置**恒为 0**：初始化即 0，且每次 optimizer.step() 之后由 mask_edges_()
        # 强行清零 -> 有效可学参数严格等于真实突触条数 E。
        A_init = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
        A_init[e_dst, e_src] = torch.from_numpy(base)
        self.A = nn.Parameter(A_init)

        mask = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
        mask[e_dst, e_src] = True
        self.register_buffer("edge_mask", mask, persistent=False)

        # 输入注入：18 维 -> 感觉神经元（全连接，可学）
        self.w_in = nn.Linear(cfg.state_dim, len(sensory_idx))
        # 读出：下行神经元 -> hidden
        self.readout = nn.Linear(len(motor_idx), cfg.hidden)
        self.ln_out = nn.LayerNorm(cfg.hidden)
        self.act = nn.ReLU(inplace=True)

        # 与基线同构的头（复用 _heads）
        if self.dueling:
            if self.dist:
                self.val_fc = _linear(cfg, cfg.hidden, cfg.num_atoms)
                self.adv_fc = _linear(cfg, cfg.hidden, cfg.action_dim * cfg.num_atoms)
            else:
                self.val_fc = _linear(cfg, cfg.hidden, 1)
                self.adv_fc = _linear(cfg, cfg.hidden, cfg.action_dim)
        else:
            if self.dist:
                self.out_fc = _linear(cfg, cfg.hidden, cfg.action_dim * cfg.num_atoms)
            else:
                self.out_fc = _linear(cfg, cfg.hidden, cfg.action_dim)

        self._init_orthogonal()

    def _propagate(self, inj: torch.Tensor) -> torch.Tensor:
        """inj: (B, S) 注入到感觉神经元的电流。返回 (B, n_nodes) 的传播结果。"""
        B = inj.shape[0]
        h = torch.zeros(B, self.n_nodes, device=inj.device, dtype=inj.dtype)
        h = h.index_add(1, self.sensory_buf, inj)

        for _ in range(self.steps):
            h = F.relu(h @ self.A)                       # 稠密 matmul：实测最快
            h = h.index_add(1, self.sensory_buf, inj)    # 每步重新注入输入
        return h

    @torch.no_grad()
    def mask_edges_(self) -> None:
        """把非边位置的权重强行清零。

        因为 A 是稠密参数，Adam 只会给「梯度非零」的位置更新动量；非边位置的梯度
        恒为 0，所以理论上不会被改动。但为了杜绝任何浮点/优化器边角情况，
        每步显式清零一次，保证非边处严格为 0 -> 有效参数 = 真实突触数。
        """
        self.A.mul_(self.edge_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inj = self.w_in(x)                       # (B, S)
        h = self._propagate(inj)                 # (B, n_nodes)
        m = h[:, self.motor_buf]                 # (B, M) 从下行神经元读出
        feat = self.act(self.ln_out(self.readout(m)))
        return self._heads(feat)


# ============================================================================
# 3) 分发
# ============================================================================
def build_net(cfg: AgentConfig, graph=None) -> nn.Module:
    """按 cfg.net_arch 造网络。graph 为 None 时（MLP 组）忽略。"""
    arch = getattr(cfg, "net_arch", "mlp")
    if arch == "mlp":
        return RainbowNet(cfg)
    if arch in ("connectome", "rewired"):
        # 'rewired' 用的是同一套网络，只是传进来的图是度分布保持的随机图 ——
        # 这样两组的网络代码、参数量、训练流程完全相同，唯一变量是拓扑。
        if graph is None:
            raise ValueError(f"net_arch={arch!r} 需要传入 graph（SubgraphData）")
        return ConnectomeNet(
            cfg,
            src=graph.src,
            dst=graph.dst,
            weight=graph.weight,
            n_nodes=graph.n_nodes,
            sensory_idx=graph.sensory_idx,
            motor_idx=graph.motor_idx,
        )
    raise ValueError(f"未知 net_arch: {arch!r}")
