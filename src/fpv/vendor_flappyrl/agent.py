"""RainbowDQN agent.

Combines the network, replay buffer and all algorithmic components behind a
small interface used by ``train.py``:

    act(states, training=True) -> actions
    store_transition(s, a, r, s2, d)   # n-step aggregation happens here
    learn()                            # one gradient step
    anneal(global_step, total)         # epsilon + PER beta
    sync_target()
    save(path) / load(path)
    evaluate(env_cfg, episodes)        # greedy play, returns scores
"""

from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AgentConfig, EnvConfig
from .networks import build_net
from .buffer import ReplayBuffer, PrioritizedReplayBuffer
from .sim import FlappySim


class RainbowDQN:
    def __init__(self, cfg: AgentConfig, env_cfg: EnvConfig, device: torch.device,
                 graph=None):
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.device = device
        self.n_envs = env_cfg.n_envs
        self.graph = graph

        # 唯一的本地改动：网络改由 build_net 分发（cfg.net_arch）
        # net_arch='mlp' -> 与原项目逐行等价的 RainbowNet
        # net_arch='connectome' -> 连接组稀疏图躯干（需要 graph）
        self.online_net = build_net(cfg, graph).to(device)
        self.target_net = build_net(cfg, graph).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.online_net.parameters(), lr=cfg.lr, eps=cfg.adam_eps,
            weight_decay=1e-5,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        # Replay buffer
        if cfg.per:
            self.buffer = PrioritizedReplayBuffer(
                cfg.buffer_size, cfg.state_dim, alpha=cfg.per_alpha
            )
        else:
            self.buffer = ReplayBuffer(cfg.buffer_size, cfg.state_dim)

        # N-step buffers (one per env)
        self.n_step = max(1, cfg.n_step)
        self.nstep_buf: List[deque] = [deque() for _ in range(self.n_envs)]

        # Exploration / annealing state
        self.eps = cfg.eps_start
        self.beta = cfg.per_beta_start
        self.gamma = cfg.gamma

        # C51 support cached on device
        self.v_min = cfg.v_min
        self.v_max = cfg.v_max
        self.num_atoms = cfg.num_atoms
        self.delta = (cfg.v_max - cfg.v_min) / (cfg.num_atoms - 1)
        self.atoms = torch.linspace(cfg.v_min, cfg.v_max, cfg.num_atoms, device=device)

    # ----------------------------------------------------------- action
    def act(self, states: np.ndarray, training: bool = True) -> np.ndarray:
        states_t = torch.as_tensor(
            np.asarray(states, dtype=np.float32), device=self.device
        )
        if not training:
            self.online_net.eval()
            with torch.no_grad():
                q = self.online_net.expected_q(states_t)
            return q.argmax(1).cpu().numpy().astype(np.int64)

        if self.cfg.noisy:
            self.online_net.train()
            self.online_net.reset_noise()
            with torch.no_grad():
                q = self.online_net.expected_q(states_t)
            return q.argmax(1).cpu().numpy().astype(np.int64)

        # epsilon-greedy (noisy disabled)
        self.online_net.eval()
        with torch.no_grad():
            q = self.online_net.expected_q(states_t)
        a = q.argmax(1).cpu().numpy().astype(np.int64)
        mask = np.random.rand(self.n_envs) < self.eps
        if mask.any():
            a[mask] = np.random.randint(self.cfg.action_dim, size=int(mask.sum()))
        return a

    # ----------------------------------------------------------- storage
    def store_transition(self, s, a, r, s2, d):
        s = np.asarray(s, dtype=np.float32)
        a = np.asarray(a).astype(np.int64)
        r = np.asarray(r, dtype=np.float32)
        s2 = np.asarray(s2, dtype=np.float32)
        d = np.asarray(d, dtype=bool)
        for i in range(self.n_envs):
            buf = self.nstep_buf[i]
            trans = (s[i], int(a[i]), float(r[i]), s2[i], bool(d[i]))
            buf.append(trans)
            if trans[4]:  # terminal -> flush the whole (partial) window
                if len(buf) > 0:
                    self._form_and_push(buf, len(buf))
                buf.clear()
                continue
            if len(buf) >= self.n_step:
                self._form_and_push(buf, self.n_step)
                buf.popleft()

    def _form_and_push(self, buf: deque, m: int):
        s0 = buf[0][0]
        a0 = buf[0][1]
        R = 0.0
        for k in range(m):
            R += (self.gamma ** k) * buf[k][2]
        s2 = buf[m - 1][3]
        d = any(buf[k][4] for k in range(m))
        self.buffer.push(s0, a0, R, s2, d)

    # ----------------------------------------------------------- learning
    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.cfg.learning_starts:
            return None
        batch = self.buffer.sample(self.cfg.batch_size, self.beta)
        s = torch.as_tensor(batch.s, device=self.device)
        a = torch.as_tensor(batch.a, device=self.device)
        r = torch.as_tensor(batch.r, device=self.device)
        s2 = torch.as_tensor(batch.s2, device=self.device)
        d = torch.as_tensor(batch.d, device=self.device)
        w = torch.as_tensor(batch.weights, device=self.device)
        gamma_n = self.gamma ** self.n_step

        self.online_net.train()
        if self.cfg.noisy:
            self.online_net.reset_noise()
        self.target_net.eval()

        if self.cfg.distributional:
            loss_per, prio = self._learn_dist(s, a, r, s2, d, w, gamma_n)
        else:
            loss_per, prio = self._learn_q(s, a, r, s2, d, w, gamma_n)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss_per.mean()).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.cfg.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 本地改动②：图网络每步把「非边位置」的权重清零，保证有效可学参数
        # 严格等于真实突触条数（A 是稠密参数，但结构由 connectome 固定）。
        # MLP 基线没有 mask_edges_ 这个方法，所以判断一下。
        for net in (self.online_net, self.target_net):
            hook = getattr(net, "mask_edges_", None)
            if callable(hook):
                hook()

        if self.cfg.per:
            self.buffer.update_priorities(batch.idx, prio)
        return float(loss_per.mean().item())

    def _learn_q(self, s, a, r, s2, d, w, gamma_n):
        q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.cfg.double:
                self.online_net.eval()
                a_next = self.online_net(s2).argmax(1)
                self.online_net.train()
                # NOTE: do NOT reset_noise() here — it would mutate the noisy
                # buffers in-place and corrupt the graph of the prediction
                # forward (logits_o) we still need to backprop through.
                next_q = self.target_net(s2).gather(1, a_next.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_net(s2).max(1).values
            target = r + gamma_n * (1.0 - d) * next_q
        loss_per = F.smooth_l1_loss(q, target, reduction="none")
        loss = (w * loss_per).mean()
        prio = (loss_per + 1e-6).detach().cpu().numpy()
        return loss, prio

    def _learn_dist(self, s, a, r, s2, d, w, gamma_n):
        logits_o = self.online_net(s)                       # (B, A, atoms)
        a_idx = a.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_atoms)
        logits_a = logits_o.gather(1, a_idx).squeeze(1)     # (B, atoms)
        log_p = F.log_softmax(logits_a, dim=-1)

        with torch.no_grad():
            # double: select action with online net (greedy, deterministic)
            self.online_net.eval()
            a_next = self.online_net.expected_q(s2).argmax(1)
            self.online_net.train()
            # NOTE: do NOT reset_noise() here (see _learn_q for why).
            p_next = self.target_net.distribution(s2)        # (B, A, atoms)
            p_next = p_next.gather(1, a_next.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_atoms)).squeeze(1)

            tz = r.unsqueeze(1) + gamma_n * self.atoms.unsqueeze(0) * (1.0 - d).unsqueeze(1)
            tz = tz.clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / self.delta
            l = b.floor().long().clamp(0, self.num_atoms - 1)
            u = b.ceil().long().clamp(0, self.num_atoms - 1)
            eq = (l == u)
            w_l = torch.where(eq, p_next, p_next * (u.float() - b))
            w_u = torch.where(eq, torch.zeros_like(p_next), p_next * (b - l.float()))
            target_dist = torch.zeros_like(p_next)
            offset = torch.arange(r.shape[0], device=self.device) * self.num_atoms
            target_dist.view(-1).index_add_(
                0, (offset.unsqueeze(1) + l).view(-1), w_l.view(-1)
            )
            target_dist.view(-1).index_add_(
                0, (offset.unsqueeze(1) + u).view(-1), w_u.view(-1)
            )

        loss_per = -(target_dist * log_p).sum(-1)           # (B,)
        loss = (w * loss_per).mean()
        prio = (loss_per + 1e-6).detach().cpu().numpy()
        return loss, prio

    # ----------------------------------------------------------- annealing
    def anneal(self, global_step: int, total: int):
        if not self.cfg.noisy:
            frac = min(1.0, global_step / max(1, self.cfg.eps_decay_steps))
            self.eps = max(self.cfg.eps_end,
                           self.cfg.eps_start - (self.cfg.eps_start - self.cfg.eps_end) * frac)
        frac = min(1.0, global_step / max(1, total))
        self.beta = self.cfg.per_beta_start + \
            (self.cfg.per_beta_end - self.cfg.per_beta_start) * frac

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ----------------------------------------------------------- persistence
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "online": self.online_net.state_dict(),
                "target": self.target_net.state_dict(),
                "meta": {
                    "agent": self.cfg.__dict__,
                    "state_dim": self.cfg.state_dim,
                    "action_dim": self.cfg.action_dim,
                    "hidden": self.cfg.hidden,
                    "distributional": self.cfg.distributional,
                    "num_atoms": self.cfg.num_atoms,
                    "v_min": self.cfg.v_min,
                    "v_max": self.cfg.v_max,
                },
            },
            path,
        )

    def load(self, path: str, device: Optional[torch.device] = None):
        device = device or self.device
        ckpt = torch.load(path, map_location=device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt.get("target", ckpt["online"]))
        self.target_net.eval()
        return ckpt.get("meta", {})

    # ----------------------------------------------------------- evaluation
    def evaluate(self, env_cfg: EnvConfig, episodes: int = 20,
                  max_steps: int = 5_000_000, max_steps_per_episode: int = 50_000):
        from .config import EnvConfig
        eval_cfg = EnvConfig(
            width=env_cfg.width, height=env_cfg.height, seed=env_cfg.seed + 999,
            pipe_gap=env_cfg.pipe_gap, pipe_speed=env_cfg.pipe_speed,
            pipe_spawn_dist=env_cfg.pipe_spawn_dist, bird_radius=env_cfg.bird_radius,
            n_envs=1, shared_world=False, auto_reset=True,
        )
        sim = FlappySim(eval_cfg)
        scores = []
        ep = 0
        steps = 0
        while ep < episodes and steps < max_steps:
            s = sim.reset_all() if ep == 0 else sim.states()
            done = False
            ep_steps = 0
            while not done and steps < max_steps and ep_steps < max_steps_per_episode:
                a = self.act(s, training=False)
                s, r, dones, info = sim.step(a)
                steps += 1
                ep_steps += 1
                if dones[0]:
                    scores.append(int(info["terminal_scores"][0]))
                    ep += 1
                    done = True
        return scores
