"""对照组②：度分布保持的随机图。

为什么一定要做这个对照
----------------------
只拿真实连接组和 MLP 比是**不可解释的**——两者同时差了两件事：
  (a) 稠密 vs 稀疏；
  (b) 拓扑是「真实的」还是「随机的」。
赢了也说不清是谁的功劳。

做法：固定每条边的**源神经元和权重**，只把**目标神经元**在全部节点里洗牌。
这样严格保持：
  * 每个神经元的总入边数分布（in-degree）不变
  * 每个神经元的出边数分布（out-degree）不变
  * 边权重的分布不变
  * 边数、感觉/读出神经元的身份不变
而被破坏的只有「谁连到谁」——也就是生物学拓扑本身。

于是：
  ① MLP  ->  ② rewired 的差距 = 「稀疏结构」的贡献
  ② rewired ->  ③ connectome 的差距 = 「真实接线」在稀疏之上的额外贡献
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .subgraph_data import SubgraphData


def rewire_degree_preserving(g: SubgraphData, seed: int = 0) -> SubgraphData:
    """返回一份目标神经元被整体洗牌的副本（源与权重保持不变）。"""
    rng = np.random.default_rng(seed)
    dst = g.dst.copy()
    rng.shuffle(dst)
    return replace(g, dst=dst, name=f"{g.name}__rewired_s{seed}")


def rewire_swap(g: SubgraphData, n_swaps: int | None = None, seed: int = 0) -> SubgraphData:
    """更严格的对照：用「边交换」保持每个节点的精确出入度序列。

    上面那个 shuffle 版本保住了**度分布**，但没有严格保住**每个节点各自的度**。
    边交换（选两条边 (a,b)、(c,d) 换成 (a,d)、(c,b)）能精确保持每点度数，
    是网络科学里的标准零模型（配置模型 / Maslov-Sneppen）。
    这里默认做与边数同量级的交换次数。
    """
    rng = np.random.default_rng(seed)
    dst = g.dst.copy()
    E = len(dst)
    if n_swaps is None:
        n_swaps = E  # 每个边平均被碰一次，足够充分混合
    src = g.src
    done = 0
    attempts = 0
    max_attempts = n_swaps * 20
    while done < n_swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, E, size=2)
        if i == j:
            continue
        a, b = src[i], dst[i]
        c, d = src[j], dst[j]
        # 换成 (a,d) 和 (c,b)；自环或重复边则跳过
        if a == d or c == b:
            continue
        dst[i], dst[j] = d, b
        done += 1
    return replace(g, dst=dst, name=f"{g.name}__swapswired_s{seed}")
