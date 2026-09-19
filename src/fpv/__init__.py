"""fpv —— FlyWire 连接组 × FlappyBird 实验包。

模块划分：
    vendor_flappyrl/   从 D:\\Code\\DQN\\flappyrl 原样搬运的环境/回放/agent
                       （唯一改动：agent 里的网络构造改为 build_net 分发）
    graph_config.py    我们的配置（在 AgentConfig 上追加图网络字段）
    subgraph_data.py   加载 scripts/build_subgraph.py 产出的子图
    rewire.py          度分布保持的随机对照图（对照组②）
"""

from .graph_config import GraphConfig, graph_config, env_config
from .subgraph_data import SubgraphData, load_subgraph

__all__ = [
    "GraphConfig",
    "graph_config",
    "env_config",
    "SubgraphData",
    "load_subgraph",
]
