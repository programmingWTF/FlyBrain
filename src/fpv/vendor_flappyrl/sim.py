"""Fast, vectorized, headless Flappy Bird simulator (pure numpy).

Design notes
------------
* The simulation is fully decoupled from rendering. Training never touches
  pygame, so it runs headless and fast.
* Supports ``n_envs`` independent environments (default) or a single shared
  world (``shared_world=True``) where every bird faces the *same* pipes — this
  is what the human-vs-AI mode needs.
* The observation is the same 18-dimensional engineered feature vector used by
  the original single-file implementation, so older checkpoints remain
  structurally compatible.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EnvConfig


class FlappySim:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.W = cfg.width
        self.H = cfg.height
        self.n = cfg.n_envs
        self.shared = cfg.shared_world
        self.auto_reset = cfg.auto_reset

        # Physics
        self.bird_radius = cfg.bird_radius
        self.gravity = 0.5
        self.flap_impulse = 8.5
        self.max_vel = 10.0

        # Pipes
        self.pipe_width = 52
        self.pipe_gap = cfg.pipe_gap
        self.pipe_speed = cfg.pipe_speed
        self.pipe_spawn_dist = cfg.pipe_spawn_dist
        self.ground_y = int(self.H * 0.87)
        self.sky_y = 0

        # Bird geometry
        self.bird_x = int(self.W * 0.2)

        # Per-env RNG (divergent pipe sequences when not shared)
        if self.shared:
            self.rngs = [random.Random(cfg.seed)]
        else:
            self.rngs = [random.Random(cfg.seed + i * 100003) for i in range(self.n)]

        # Bird state arrays
        self.bird_y = np.zeros(self.n, dtype=np.float32)
        self.bird_vel = np.zeros(self.n, dtype=np.float32)
        self.scores = np.zeros(self.n, dtype=np.int32)
        self.done = np.zeros(self.n, dtype=bool)        # persistent dead flag (versus)
        self.terminal_scores = np.zeros(self.n, dtype=np.int32)  # score at death
        self.ticks = np.zeros(self.n, dtype=np.int32)

        # Pipe storage:
        #   shared world  -> self.pipes is a single list[dict]
        #   independent   -> self.pipes is list[ list[dict] ] (one per env)
        if self.shared:
            self.pipes: Any = []
        else:
            self.pipes: Any = [[] for _ in range(self.n)]

        self.reset_all()

    # ---------------------------------------------------------------- helpers
    def _spawn_pipe(self, rng: random.Random, pipes: list, x: float,
                    gap_y: Optional[float] = None) -> None:
        margin = 60
        if gap_y is None:
            gap_y = rng.randint(
                margin + self.pipe_gap // 2,
                self.ground_y - margin - self.pipe_gap // 2,
            )
        # `passed` is tracked PER BIRD (a list of length n) so that in a shared
        # world every bird can score the same pipe independently.
        pipes.append({"x": float(x), "gap_y": float(gap_y),
                      "passed": [False] * self.n})

    def _spawn_initial(self, rng: random.Random, pipes: list) -> None:
        pipes.clear()
        start_x = self.W + 80
        for i in range(3):
            self._spawn_pipe(rng, pipes, start_x + i * self.pipe_spawn_dist)

    # ---------------------------------------------------------------- reset
    def reset_all(self) -> np.ndarray:
        # Shared world: rebuild the pipe field so a restart begins from a clean
        # scene. (Without this the old pipes stay put and a restarted bird can
        # spawn inside one.)
        if self.shared:
            self._spawn_initial(self.rngs[0], self.pipes)
            self.terminal_scores[:] = 0
        for i in range(self.n):
            self._reset_env(i)
        return self.states()

    def _reset_env(self, i: int) -> None:
        self.bird_y[i] = int(self.H * 0.5)
        self.bird_vel[i] = 0.0
        self.scores[i] = 0
        self.done[i] = False
        self.ticks[i] = 0
        if self.shared:
            if not self.pipes:
                self._spawn_initial(self.rngs[0], self.pipes)
        else:
            self._spawn_initial(self.rngs[i], self.pipes[i])

    # ---------------------------------------------------------------- collision
    def _collides(self, i: int, bx: int, by: int, pipes: list) -> bool:
        r = self.bird_radius
        if by - r <= self.sky_y:
            return True
        if by + r >= self.ground_y:
            return True
        for p in pipes:
            top_h = int(p["gap_y"] - self.pipe_gap / 2)
            bottom_y = int(p["gap_y"] + self.pipe_gap / 2)
            rects = [
                (int(p["x"]), 0, self.pipe_width, top_h),
                (int(p["x"]), bottom_y, self.pipe_width, self.ground_y - bottom_y),
            ]
            for rx, ry, rw, rh in rects:
                cx = max(rx, min(bx, rx + rw))
                cy = max(ry, min(by, ry + rh))
                if (bx - cx) ** 2 + (by - cy) ** 2 <= r ** 2:
                    return True
        return False

    # ---------------------------------------------------------------- state
    def _next_pipes(self, i: int, pipes: list, k: int = 2) -> list:
        bx = self.bird_x
        r = self.bird_radius
        cands = [p for p in pipes if p["x"] + self.pipe_width >= bx - r]
        cands.sort(key=lambda p: p["x"])
        if len(cands) < k:
            pad = cands[-1] if cands else (pipes[-1] if pipes else None)
            while len(cands) < k and pad is not None:
                cands.append(pad)
        return cands[:k]

    def _state_for(self, i: int, pipes: list) -> np.ndarray:
        by = self.bird_y[i] / self.H
        vy = self.bird_vel[i] / self.max_vel
        dist_ground = (self.ground_y - self.bird_y[i]) / self.H
        dist_sky = (self.bird_y[i] - self.sky_y) / self.H

        bird_x = self.bird_x
        nxt = self._next_pipes(i, pipes, k=2)

        def feats(p):
            dx_px = p["x"] - bird_x
            dx = dx_px / self.W
            gy = p["gap_y"] / self.H
            top = (p["gap_y"] - self.pipe_gap / 2) / self.H
            bottom = (p["gap_y"] + self.pipe_gap / 2) / self.H
            dy_gap = (p["gap_y"] - self.bird_y[i]) / self.H
            dy_top = ((p["gap_y"] - self.pipe_gap / 2) - self.bird_y[i]) / self.H
            dy_bottom = ((p["gap_y"] + self.pipe_gap / 2) - self.bird_y[i]) / self.H
            denom = self.W / max(1e-6, self.pipe_speed)
            t_norm = max(0.0, dx_px) / denom
            return (dx, gy, dy_gap, top, bottom, dy_top, dy_bottom, t_norm)

        z = (0.0,) * 8
        p0 = feats(nxt[0]) if len(nxt) > 0 and nxt[0] is not None else z
        if len(nxt) > 1 and nxt[1] is not None:
            p1 = feats(nxt[1])
            p1 = (p1[0], p1[1], p1[2], p1[3], p1[4], p1[7])  # 6 feats used
        else:
            p1 = (0.0,) * 6

        return np.array(
            [
                by, vy, dist_ground, dist_sky,
                p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7],
                p1[0], p1[1], p1[2], p1[3], p1[4], p1[5],
            ],
            dtype=np.float32,
        )

    def states(self) -> np.ndarray:
        out = np.empty((self.n, 18), dtype=np.float32)
        for i in range(self.n):
            pipes = self.pipes if self.shared else self.pipes[i]
            out[i] = self._state_for(i, pipes)
        return out

    # ---------------------------------------------------------------- step
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Advance the simulation.

        actions : int array (n,) — 0 = noop, 1 = flap.
        Returns (states, rewards, dones, infos) where ``dones`` is the
        *terminal* flag for this step (True on the step a bird dies). In
        auto-reset (training) mode the env is immediately reset afterwards,
        but the terminal flag and ``terminal_scores`` are preserved for the
        caller.
        """
        actions = np.asarray(actions).astype(np.int64).reshape(-1)
        rewards = np.zeros(self.n, dtype=np.float32)
        just_done = np.zeros(self.n, dtype=bool)
        info: Dict[str, object] = {}

        # Shared world: advance the single pipe field ONCE per step,
        # unconditionally (as long as at least one bird is alive). Previously
        # the move was gated on bird 0, so when the first bird died the pipes
        # froze and the survivor faced a static scene forever.
        if self.shared and not self.done.all():
            self._move_pipes(self.pipes, self.rngs[0])

        for i in range(self.n):
            if (not self.auto_reset) and self.done[i]:
                # Versus mode: dead bird is frozen, world keeps going.
                rewards[i] = 0.0
                continue

            # Action
            if actions[i] == 1:
                self.bird_vel[i] = -self.flap_impulse

            # Physics
            self.bird_vel[i] = max(
                -self.max_vel,
                min(self.max_vel, self.bird_vel[i] + self.gravity),
            )
            self.bird_y[i] = int(self.bird_y[i] + self.bird_vel[i])

            pipes = self.pipes if self.shared else self.pipes[i]
            rng = self.rngs[0] if self.shared else self.rngs[i]

            # Per-env pipes move every env; shared pipes already moved above.
            if not self.shared:
                self._move_pipes(pipes, rng)

            # Scoring — each bird tracks its OWN "passed" flag so both birds
            # can score the same pipe in a shared scene (fair comparison).
            for p in pipes:
                if (not p["passed"][i]) and (p["x"] + self.pipe_width < self.bird_x - self.bird_radius):
                    p["passed"][i] = True
                    self.scores[i] += 1

            # Collision / reward
            done = self._collides(i, self.bird_x, int(self.bird_y[i]), pipes)
            reward = 0.01  # alive reward
            if done:
                reward = -1.0
            else:
                nxt = self._next_pipes(i, pipes, k=1)
                if nxt:
                    dy_gap = (nxt[0]["gap_y"] - self.bird_y[i]) / self.H
                    reward += 0.05 * (1.0 - min(1.0, abs(dy_gap) * 2.0))

            self.ticks[i] += 1
            rewards[i] = reward

            if done:
                just_done[i] = True
                self.terminal_scores[i] = self.scores[i]
                if self.auto_reset:
                    self._reset_env(i)   # ready for next collection step
                else:
                    self.done[i] = True  # frozen in versus mode

        info["scores"] = self.scores.copy()
        info["terminal_scores"] = self.terminal_scores.copy()
        info["done"] = just_done.copy()
        if self.shared:
            env_done = bool(self.done.all())
        else:
            env_done = bool(just_done.any())
        info["env_done"] = env_done
        return self.states(), rewards, just_done.copy(), info

    def _move_pipes(self, pipes: list, rng: random.Random) -> None:
        for p in pipes:
            p["x"] -= self.pipe_speed
        if pipes and pipes[0]["x"] + self.pipe_width < 0:
            pipes.pop(0)
            next_x = pipes[-1]["x"] + self.pipe_spawn_dist
            self._spawn_pipe(rng, pipes, next_x)


# Type alias for the Any annotations above (kept local to avoid import churn)
from typing import Any  # noqa: E402
