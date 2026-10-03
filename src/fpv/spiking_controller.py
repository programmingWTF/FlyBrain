"""阶段 2.4 / 3：把「冻结的脉冲连接组」接到 FlappyBird 上。

架构（对齐 pinme.dev 的验证过的做法）
------------------------------------
    Flappy 18 维状态
        |
        v  感觉编码（可学或固定）—— 注入到文献已知的感觉神经元群
    LC4 / LPLC2（碰撞）  LC10（目标）  T4/T5（运动）
        |
        v  冻结的 LIF 脉冲连接组（59,548 神经元 / 245 万突触）
        |
        v  运动解码 —— 读出命名下行神经元群的群体发放率
    ESCAPE / TARGET / LOOM 组的发放率
        |
        v  可学读出（唯一训练的部分，~几百参数）
    flap / noop

关键设计（这正是它区别于失败版本的地方）
----------------------------------------
1. **连接组冻结**：内部 245 万突触永不更新。图是"先验骨架"，不是可学参数。
2. **只有小读出可学**：参数量从 547k 降到几百，这才是真正的"用生物结构换参数效率"。
3. **脉冲 + 符号**：GABA 抑制真实存在，能把膜电位往下拉。
4. **工作点**：gain=3.0 / tonic=0.0（tune_dynamics.py 标定：静息 0% + 区分度 0.61）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .spiking_brain import SpikingBrain

# 感觉通道 -> 注入的神经元群（文献已知的视觉通路）
SENSORY_CHANNELS = {
    "collision": ["LC4", "LPLC2"],        # loom / 逼近
    "target": ["LC10"],                    # 小目标追踪
    "motion": ["T4", "T5"],               # ON/OFF 运动
}

# 运动读出：命名下行神经元群（pinme.dev 的分组思路）
READOUT_GROUPS = {
    "ESCAPE": ["DNp01", "DNp04"],
    "TARGET": ["DNae002", "DNae001", "DNg111", "DNge109", "DNb01"],
    "LOOM": ["DNa07", "DNp06"],
    "OTHER": ["DNp07", "DNp09", "DNp11", "DNa02", "DNg13", "DNge103", "DNp54"],
}

# 动力学报工作点
#
# ⚠️ 修订记录（2026-10-02）：
#   旧值 gain=3.0/tonic=0.0/inj=1.0 来自 tune_dynamics.py，但那是**静息态**
#   （无感觉驱动）下标定的。接上持续感觉注入后，实测：
#     - ESCAPE 群（DNp01/DNp04，只有 2 个神经元）发放率长期 0.65，顶到上限
#     - 4 维特征预测教师动作：flap 召回 0.000，平衡 acc 0.500，AUC 0.617
#       -> 即"完全学不到什么时候该拍翅"
#   diag3_operating_point.py 在**带载**条件下重新扫描 (gain, tonic, inj_scale)：
#     gain=3.0 tonic=0.0 inj=0.4  ->  4维 AUC 0.704 / 平衡 0.729  （最佳）
#     （对照：18 维原始状态本身也只有 平衡 0.710，说明已接近教师策略的可预测上限）
BEST_GAIN = 3.0
BEST_TONIC = 0.0
BEST_INJ = 0.4


class SpikingController(nn.Module):
    """冻结脉冲脑 + 可学小读出。"""

    def __init__(self, brain: SpikingBrain, readout_dim: int = 64,
                 lr: float = 1e-2, state_dim: int = 18,
                 readout_mode: str = "groups", n_wide: int = 512,
                 wide_per: int = 64, seed: int = 0):
        """readout_mode:
            "groups" —— 4 个文献下行群（现状，但信息量不足，见 diag5）
            "wide"   —— n_wide 个随机群体，每群 wide_per 个神经元
                        （diag5 实测 AUC 0.78 vs 4 维 0.55，且平衡 acc 0.745
                          高于 18 维原始状态自己的 0.710）
        """
        super().__init__()
        self.brain = brain
        self.readout_mode = readout_mode
        # 工作点（带载标定，见上方 BEST_* 注释）
        self.brain.gain = BEST_GAIN
        self.brain.tonic = BEST_TONIC

        dev = brain.device
        # 感觉编码权重：18 维状态 -> 每个感觉通道一个标量（可学）
        self.n_ch = len(SENSORY_CHANNELS)
        self.ch_w = nn.Parameter(torch.zeros(self.n_ch, state_dim, device=dev))
        with torch.no_grad():
            self.ch_w.fill_(0.3)

        # 感觉通道 -> 神经元索引
        self.ch_idx = {}
        for ch, names in SENSORY_CHANNELS.items():
            ids = []
            for n in names:
                try:
                    ids.append(brain.key_idx(n))
                except KeyError:
                    pass
            if ids:
                self.ch_idx[ch] = torch.cat(ids)

        # 读出群 -> 神经元索引
        self.grp_idx = {}
        for g, names in READOUT_GROUPS.items():
            ids = []
            for n in names:
                try:
                    ids.append(brain.key_idx(n))
                except KeyError:
                    pass
            if ids:
                self.grp_idx[g] = torch.cat(ids)

        if readout_mode == "wide":
            # n_wide 个随机群体，每群 wide_per 个神经元
            rng = np.random.default_rng(seed)
            total = n_wide * wide_per
            pick = rng.choice(brain.N, size=min(total, brain.N), replace=False)
            self.wide_flat = torch.as_tensor(pick, device=dev).reshape(n_wide, -1)
        else:
            self.wide_flat = None
        self.n_groups = n_wide if readout_mode == "wide" else len(self.grp_idx)

        # 可学读出：群发放率 -> 2 个动作 Q 值
        # readout_dim=0 时退化成**纯线性**读出（n_groups -> 2）。
        # 这正是 diag5 测 AUC 时用的模型（逻辑回归），宽读出下 AUC 0.781；
        # 而且参数只有 ~1k，进化策略（ES）才搜得动。
        self.linear_head = readout_dim == 0
        if self.linear_head:
            self.readout = None
            self.head = nn.Linear(self.n_groups, 2, device=dev)
            self.cmp_head = None
        else:
            # ⚠️ 关键修复（2026-10-02）：原来是 ReLU(readout(feat)) -> head，
            #    即 q = W2·relu(W1·f)。这是个**单调**映射：对 2 类决策，
            #    它只能学出 "f 的线性组合 > 阈值"，无法表达"该 flap 当且仅当
            #    某一个群领先另一个群"这类**相对比较**。
            #    修复：头改成双线性/成对比较（把成对差喂给线性头），
            #    参数量仍只有几百量级，却足以表达"群 A 强于群 B -> flap"。
            self.readout = nn.Linear(self.n_groups, readout_dim, device=dev)
            self.head = nn.Linear(readout_dim, 2, device=dev)
            # 成对比较头：只在读出很窄时才建（宽读出下 npair 是 O(n^2)，
            # 512 维会变成 13 万个参数，失去"小读出"的意义）
            if self.n_groups <= 16:
                npair = self.n_groups * (self.n_groups - 1) // 2
                self.cmp_head = nn.Linear(npair + self.n_groups, 2, device=dev)
                with torch.no_grad():
                    self.cmp_head.weight.zero_()
                    self.cmp_head.bias.zero_()
            else:
                self.cmp_head = None
        # ⚠️ 不能叫 self.act —— 会和下面的动作选择方法 act() 撞名，
        #    导致 self.act(...) 无限递归（RecursionError 伪装成 torch 的锅）。
        self.act_fn = nn.ReLU()
        self.optim = torch.optim.Adam(list(self.parameters()), lr=lr)

        # EMA 平滑（对齐 pinme.dev 的 L = L*0.86 + f*0.14）
        # diag5 扫描：宽读出下 α=0.05 明显更好（AUC 0.781 vs α=0.14 的 0.692），
        # 窄读出则差别不大。故宽读出默认 0.05。
        self.ema = torch.zeros(self.n_groups, device=dev)
        self.ema_alpha = 0.05 if readout_mode == "wide" else 0.14

        # 感觉注入总强度缩放（动力学标定用）--------------------------------
        # ⚠️ 背景：tune_dynamics.py 是**在没有感觉驱动**的静息态下标定的
        # （gain=3/tonic=0 -> 静息 0%）。但一旦接上持续的感觉注入，
        # 循环驱动会把网络推到**饱和区**：实测 ESCAPE 群发放率长期 0.65，
        # 顶到上限 0.718，发放率被钉死 -> 携带不了信息 -> flap 召回 0。
        # 所以工作点必须在**带载**条件下重新标定，inj_scale 就是这个旋钮。
        self.inj_scale = BEST_INJ

        # 冷却（对齐 pinme.dev 的去抖）
        self.cooldown = 0

    # ---------------------------------------------------------------- 感觉编码
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """state: (18,) 或 (18,) 归一化状态 -> (N,) 注入电流。

        对齐 pinme.dev 的语义：
          collision 通道 <- 与管道距离/逼近程度（状态里的 dx、dy_gap）
          target 通道    <- 与缺口中心的偏移
          motion 通道    <- 垂直速度
        """
        s = state.detach().clone().float().reshape(-1)
        dx, dy_gap = s[4], s[6]
        vy = s[1]
        # 三个通道的标量（都 clamp 到 [0,1] 附近）
        # ⚠️ 不要写 `1.0 - t`（python float 在左）：PyTorch 2.11 + Py3.13 下
        #    对某些 tensor 会触发 `__instancecheck__` 无限递归。改成 t.neg().add_(1)
        col = dx.neg().add(1.0).clamp(0.0, 1.0)
        tgt = dy_gap.abs().mul(3.0).neg().add(1.0).clamp(0.0, 1.0)
        mot = vy.abs().mul(2.0).clamp(0.0, 1.0)
        raw = torch.stack([col, tgt, mot])                       # (3,)
        ch_val = raw.mul(2.0).add(0.1).clamp(0.0, 3.0)           # 注入强度标度
        if self.inj_scale != 1.0:
            ch_val = ch_val.mul(self.inj_scale)
        stim = torch.zeros(self.brain.N, device=self.brain.device)
        for i, ch in enumerate(self.ch_idx):
            stim.index_add_(0, self.ch_idx[ch],
                            torch.full((self.ch_idx[ch].numel(),), float(ch_val[i]),
                                       device=self.brain.device))
        return stim

    # ---------------------------------------------------------------- 前向
    def advance(self, state: torch.Tensor, n_ticks: int = 1) -> None:
        """推进脉冲脑 n_ticks 步，但**不读出**。

        关键：脑必须**持续运行**（每游戏帧 tick），膜电位才能跨帧累积。
        之前每个决策点只跑 2 tick，信号根本传不到下行神经元，
        读出特征恒为 0（feat=[0,0,0,0]），ES 完全没有梯度可用。
        """
        with torch.no_grad():
            stim = self.encode(state)
            for _ in range(n_ticks):
                self.brain.step(stim)

    def read_features(self) -> torch.Tensor:
        """读出当前各下行神经元群的发放率（带 EMA 平滑）。"""
        with torch.no_grad():
            if self.readout_mode == "wide":
                r = self.brain.S[self.wide_flat].float().mean(1)   # (n_wide,)
            else:
                rates = []
                for g, ix in self.grp_idx.items():
                    rates.append(float(self.brain.S[ix].float().mean()))
                r = torch.tensor(rates, device=self.brain.device)
            self.ema = self.ema * (1 - self.ema_alpha) + r * self.ema_alpha
            return self.ema

    def neural_response(self, state: torch.Tensor, n_ticks: int = 3) -> torch.Tensor:
        """跑 n_ticks 步脉冲并读出（便捷入口，主要用于诊断）。"""
        self.advance(state, n_ticks)
        return self.read_features()

    def q_values_from_feat(self, feat: torch.Tensor) -> torch.Tensor:
        """n_groups 维群发放率 -> 2 个动作 Q 值。支持 (n_groups,) 或 (B, n_groups)。

        两路并联相加：
          (a) MLP 路：head(relu(readout(feat)))
          (b) 成对比较路：cmp_head([feat, 所有成对差])
        成对比较路让读出能表达"ESCAPE 群比 OTHER 群更活跃 -> 拍翅"这类
        相对关系，参数代价极小（21 个数）。
        """
        f = feat if feat.dim() == 2 else feat.reshape(1, -1)   # (B, G)
        if self.linear_head:
            return self.head(f)
        q_mlp = self.head(self.act_fn(self.readout(f)))
        # 成对差（向量化，支持 batch）
        if self.cmp_head is not None and self.n_groups > 1:
            iu = torch.triu_indices(self.n_groups, self.n_groups, offset=1,
                                    device=f.device)
            diff = f[:, iu[0]] - f[:, iu[1]]                  # (B, npair)
            q_cmp = self.cmp_head(torch.cat([f, diff], dim=1))
            return q_mlp + q_cmp
        return q_mlp

    def q_values(
        self,
        state: torch.Tensor | None = None,
        n_ticks: int = 3,
        feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if feat is None:
            feat = self.neural_response(state, n_ticks)
            self._last_feat = feat
        return self.q_values_from_feat(feat)

    def act(self, feat: torch.Tensor | None = None, *, tie_break: bool = True) -> int:
        """按 Q 值选动作。feat 为 None 时用最近一次 read_features 的结果。

        tie_break=True 时，若两个动作 Q 完全相等则用一个小随机打破平局，
        避免网络在"什么都学不到"时退化成永远不动。
        """
        with torch.no_grad():
            if feat is None:
                feat = getattr(self, "_last_feat", None)
            if feat is None:
                return 0
            q = self.q_values_from_feat(feat).reshape(-1)
            if tie_break and abs(float(q[0] - q[1])) < 1e-9:
                return int(np.random.randint(0, 2))
            return int(torch.argmax(q).item())

    def reset(self):
        self.brain.reset()
        self.ema.zero_()
        self.cooldown = 0
