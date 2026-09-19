"""Replay buffers: uniform and prioritized (SumTree).

Both expose the same interface so the agent can switch via config:

    push(s, a, r, s2, d)            # store one (already n-step aggregated) transition
    sample(batch, beta) -> batch    # Batch namedtuple
    update_priorities(idx, prio)    # PER only

N-step aggregation is performed by the agent *before* ``push`` so the buffer
itself stays simple and episode-agnostic.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Optional

import numpy as np

Batch = namedtuple("Batch", ["s", "a", "r", "s2", "d", "weights", "idx"])


class _RingStorage:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.bool_)
        self.pos = 0
        self.size = 0

    def store(self, s, a, r, s2, d):
        i = self.pos
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = d
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def get(self, idx):
        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.s2[idx],
            self.d[idx],
        )


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.store = _RingStorage(capacity, state_dim)

    def __len__(self):
        return self.store.size

    def push(self, s, a, r, s2, d):
        self.store.store(s, a, r, s2, d)

    def sample(self, batch_size: int, beta: float = 1.0):
        n = self.store.size
        idx = np.random.choice(n, batch_size, replace=False)
        s, a, r, s2, d = self.store.get(idx)
        return Batch(
            s=np.asarray(s),
            a=np.asarray(a, dtype=np.int64),
            r=np.asarray(r, dtype=np.float32),
            s2=np.asarray(s2),
            d=np.asarray(d, dtype=np.float32),
            weights=np.ones(batch_size, dtype=np.float32),
            idx=idx,
        )

    def update_priorities(self, idx, priorities):
        pass  # no-op for uniform buffer


class SumTree:
    """Binary sum tree for prioritized sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_ptr = 0

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, idx: int, value: float):
        value = max(value, 1e-6)
        tree_idx = idx + self.capacity - 1
        change = value - self.tree[tree_idx]
        self.tree[tree_idx] = value
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get(self, s: float) -> int:
        """Retrieve leaf index for a cumulative-sum value ``s`` in [0, total)."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)

    def reset(self):
        self.tree[:] = 0.0
        self.data_ptr = 0


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, alpha: float = 0.6):
        # The SumTree heap navigation (2*idx+1) is only correct for a
        # power-of-2 number of leaves, so round capacity up to the next pow2.
        cap = 1
        while cap < capacity:
            cap *= 2
        self.capacity = cap
        self.store = _RingStorage(cap, state_dim)
        self.alpha = alpha
        self.tree = SumTree(cap)
        self.max_priority = 1.0

    def __len__(self):
        return self.store.size

    def push(self, s, a, r, s2, d):
        idx = self.store.store(s, a, r, s2, d)
        self.tree.update(idx, self.max_priority ** self.alpha)
        return idx

    def sample(self, batch_size: int, beta: float = 0.4):
        n = self.store.size
        total = self.tree.total()
        seg = total / batch_size
        idxs = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        for i in range(batch_size):
            a, b = seg * i, seg * (i + 1)
            s = np.random.uniform(a, b)
            leaf = self.tree.get(s)
            idxs[i] = leaf
            priorities[i] = self.tree.tree[leaf + self.tree.capacity - 1]
        s, a, r, s2, d = self.store.get(idxs)
        probs = priorities / total
        weights = (n * probs) ** (-beta)
        weights = weights / weights.max()
        return Batch(
            s=np.asarray(s),
            a=np.asarray(a, dtype=np.int64),
            r=np.asarray(r, dtype=np.float32),
            s2=np.asarray(s2),
            d=np.asarray(d, dtype=np.float32),
            weights=np.asarray(weights, dtype=np.float32),
            idx=idxs,
        )

    def update_priorities(self, idx, priorities):
        priorities = np.asarray(priorities, dtype=np.float64)
        self.max_priority = max(self.max_priority, float(priorities.max()))
        for i, p in zip(np.asarray(idx), priorities):
            self.tree.update(int(i), p ** self.alpha)
