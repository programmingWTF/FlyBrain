"""Configuration objects for the Flappy Bird RL project.

Everything that controls the environment, the agent architecture and the
training loop lives in dataclasses here so it can be serialized, diffed and
reproduced exactly. Two ready-made presets are provided:

- baseline_config() : a solid Double + Dueling DQN (no fancy tricks) — used to
  validate the pipeline and as a comparison point.
- rainbow_config()  : full Rainbow (Double + Dueling + Noisy + PER + N-step +
  Distributional). This is the recommended "best" configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class EnvConfig:
    width: int = 288
    height: int = 512
    fps: int = 60
    seed: int = 0
    # Physics / terrain
    pipe_gap: int = 120
    pipe_speed: float = 2.5
    pipe_spawn_dist: int = 180
    bird_radius: int = 12
    # Vectorization
    n_envs: int = 8
    # When True all birds share ONE pipe field (used for human-vs-AI).
    shared_world: bool = False
    # Whether finished envs auto-reset (training=True, versus=False).
    auto_reset: bool = True


@dataclass
class AgentConfig:
    # --- network architecture ---
    state_dim: int = 18
    action_dim: int = 2
    hidden: int = 256

    # --- algorithmic components (Rainbow-style toggles) ---
    double: bool = True          # Double DQN target
    dueling: bool = True         # Dueling network
    noisy: bool = True           # NoisyNet exploration
    noisy_std: float = 0.5       # initial std of noisy layers
    per: bool = True             # Prioritized Experience Replay
    n_step: int = 3              # multi-step returns (1 = one-step TD)
    distributional: bool = True  # C51 distributional Q-learning
    num_atoms: int = 51          # atoms for C51
    # C51 support must span the *actual* Q range. With alive=+0.01, shaping up
    # to +0.05/step and death=-1, Q-values live in roughly [-1, 6], so a tight
    # support gives the atoms far better resolution than the naive [-10, 10].
    v_min: float = -2.0          # C51 value support lower bound
    v_max: float = 8.0           # C51 value support upper bound

    # --- optimization ---
    lr: float = 6.25e-5
    adam_eps: float = 1.5e-4
    gamma: float = 0.99
    batch_size: int = 256
    buffer_size: int = 200_000
    learning_starts: int = 2_000     # env-steps before first gradient update
    grad_steps_per_env_step: int = 1 # gradient updates per collected env-step
    target_update_freq: int = 1_000  # env-steps between target-net syncs
    max_grad_norm: float = 10.0

    # --- epsilon-greedy (only used when noisy=False) ---
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000

    # --- PER ---
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0

    # --- checkpointing ---
    checkpoint_dir: str = "checkpoints"


@dataclass
class TrainConfig:
    total_timesteps: int = 300_000   # env-steps
    save_every: int = 50_000        # env-steps between checkpoints
    log_every: int = 2_000          # env-steps between console logs
    eval_every: int = 25_000        # env-steps between greedy evaluations
    eval_episodes: int = 20
    render: bool = False            # render during training (slow!)
    render_every: int = 1
    tag: str = "rainbow"            # run name (used for log/ckpt subdirs)
    device: str = "auto"            # "auto" -> cuda if available else cpu
    amp: bool = True                # mixed precision (CUDA only)


def baseline_config() -> AgentConfig:
    """A plain, robust Double + Dueling DQN for validation / comparison."""
    return AgentConfig(
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
        per_beta_start=0.4,
        per_beta_end=1.0,
    )


def rainbow_config() -> AgentConfig:
    """Full Rainbow — the recommended strong configuration."""
    return AgentConfig(
        double=True,
        dueling=True,
        noisy=True,
        noisy_std=0.5,
        per=True,
        n_step=3,
        distributional=True,
        num_atoms=51,
        v_min=-2.0,
        v_max=8.0,
        lr=6.25e-5,
        adam_eps=1.5e-4,
        gamma=0.99,
        batch_size=256,
        buffer_size=200_000,
        learning_starts=2_000,
        grad_steps_per_env_step=1,
        target_update_freq=1_000,
        max_grad_norm=10.0,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=50_000,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
    )


def to_dict(cfg) -> dict:
    return asdict(cfg)


def from_dict(cls, d: dict):
    known = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in d.items() if k in known})
