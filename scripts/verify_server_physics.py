"""校验：服务端的 `GameWorld`（demo/server.py）与评测台的 `World`（scripts/flappy_bench.py）
在**同一串拍翅命令 + 同一套管道布局**下必须逐 tick 等价。

⚠️ 职责边界：这个脚本**只验物理**。两边现在用的关卡生成算法已经不同了
（服务端 = 有界随机游走；评测台 = 旧的"从下方均匀抽样"，那个写法有缺口冻死的退化），
所以这里把两侧的"下一根缺口"都替换成**同一串预生成的序列** ——
管道布局因此逐根相同，任何分歧就只可能来自物理实现。
`max_climb` / `gap_hi` 之类的生成参数对这里没有影响。

为什么要这个脚本
----------------
"服务端权威"的前提是：那个 Python 物理和我们已经用 100.8 分验证过的评测台物理
**是同一套**。两边各写一份实现，很容易悄悄漂移（改了一个忘一个），
而症状是"分数变差了但不知道从哪来"。所以这里用固定的 rng 种子 + 固定的拍翅序列
逐 tick 对账：位置、速度、分数、死亡时刻、死因、管道布局，任何一项不等就报出来。

用法
----
    python scripts/verify_server_physics.py

    # 想跑更多随机局面：
    python scripts/verify_server_physics.py --games 60 --ticks 3000
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "demo"))

# flappy_bench 会 import 一堆 torch 相关的东西，但 World 本身不依赖脑
import flappy_bench as fb                      # noqa: E402
from server import GameWorld                   # noqa: E402

#: 逐 tick 比较的浮点容差。两边都是纯 Python float 的同一串运算，
#: 理论上应为 0；留一点余量避免平台差异误报。
TOL = 1e-9

#: 与 demo/server.py 的 Session 构造 GameWorld 时用的**同一套**关卡参数。
#: 这两处必须一致，否则对账会把"参数不同"误报成"物理不同"。
SPEC_MAX_CLIMB = 40.0
SPEC_FIRST_GAP = 160.0


def compare_one(seed: int, ticks: int, flap_seq: np.ndarray, *,
                max_climb: float, first_gap_extra: float) -> list[str]:
    """一局对账，返回不一致项（空 = 通过）。"""
    bad: list[str] = []

    # 两边用**同一个种子**，保证关卡随机序列一致
    bench = fb.World(np.random.default_rng(seed), gap_top=fb.NATURAL_GAP_TOP,
                     solvable=True, max_climb=max_climb,
                     first_gap_extra=first_gap_extra)
    srv = GameWorld(max_climb=max_climb, first_gap_extra=first_gap_extra, seed=seed)

    bench.reset()
    srv.reset()

    # ⚠️ **把两侧的关卡来源都换成同一串预生成的缺口序列**，这样这个脚本就
    #    严格只验物理。为什么需要这么做：服务端与评测台现在用的生成算法
    #    已经不同了（服务端 = 有界随机游走；评测台 = 旧的"从下方均匀抽样"，
    #    那个写法有"缺口冻死"的退化）。如果两边各自生成，比较出来的差异
    #    是生成算法的差异，不是物理的差异 —— 我一开始就是这么被绊住的。
    #    `spacing()` 两边是同一套（同一个 SPACING + first_gap_extra），
    #    所以只要缺口序列相同，生成出来的管道布局就逐根相同。
    seq = np.random.default_rng(seed ^ 0x5EED).uniform(70.0, 260.0, 4096)
    box_b = [0]
    box_s = [0]

    def _next_b():
        v = float(seq[min(box_b[0], len(seq) - 1)])
        box_b[0] += 1
        return v

    def _next_s():
        v = float(seq[min(box_s[0], len(seq) - 1)])
        box_s[0] += 1
        return v

    bench._next_gap_top = _next_b
    srv._next_gap_top = _next_s

    # ⚠️ 冷却门必须在**喂进去之前**统一施加。
    #    为什么不能各自在内部管：评测台的 `World.step(flap)` 是**无条件**执行拍翅的
    #    （冷却由它的 harness 管），而服务端最初把冷却塞在 GameWorld 里 →
    #    对账立刻抓到 vy 差 23.6（评测台 -316.4、服务端 -292.8，拍翅被吞）。
    #    现在服务端也把冷却移到 Session._game_tick，GameWorld 变成纯物理，
    #    于是这里用同一个门预筛一遍，两边收到的序列就完全一样。
    cooldown = 0.0
    gated = []
    for f in flap_seq:
        cooldown = max(0.0, cooldown - 0.02)
        if f and cooldown > 0:
            gated.append(False)
        else:
            if f:
                cooldown = 0.14
            gated.append(bool(f))
    flap_seq = np.asarray(gated)

    for i in range(ticks):
        f = bool(flap_seq[i])
        bench.step(f)
        # ⚠️ 关卡生成**两侧已经不一样了**，而且这是有意的：
        #    服务端 2026-10 换成了"以上一根缺口中心为中心的有界随机游走"，
        #    评测台仍是从下方均匀抽样（旧实现，会有缺口冻死的退化）。
        #    这个脚本的职责是**只验物理**，所以每 tick 把服务端的管道布局
        #    强制对齐到评测台 —— 这样两边物理的输入完全相同，
        #    任何分歧就一定是物理实现的问题，而不是关卡生成的问题。
        #    （关卡生成的差异由 tune_levels.py 与 verify_server_vs_bench.py 各自负责。）
        srv.step(f)

        for name, a, b in (("y", bench.y, srv.y),
                           ("vy", bench.vy, srv.vy),
                           ("t", bench.t, srv.t),
                           ("score", bench.score, srv.score)):
            if isinstance(a, float):
                if abs(a - b) > TOL:
                    bad.append(f"seed={seed} tick={i} {name}: 评测台 {a!r} != 服务端 {b!r}")
            elif a != b:
                bad.append(f"seed={seed} tick={i} {name}: 评测台 {a!r} != 服务端 {b!r}")
        if bench.dead != srv.dead:
            bad.append(f"seed={seed} tick={i} dead: {bench.dead} != {srv.dead}"
                       f"（评测台 cause={bench.cause!r} 服务端 cause={srv.cause!r}）")
        if bench.dead and bench.cause != srv.cause:
            bad.append(f"seed={seed} tick={i} cause: {bench.cause!r} != {srv.cause!r}")
        # 管道布局（位置与缺口）
        bp = [(round(p["x"], 9), round(p["top"], 9)) for p in bench.pipes]
        sp = [(round(p["x"], 9), round(p["top"], 9)) for p in srv.pipes]
        if bp != sp:
            bad.append(f"seed={seed} tick={i} 管道: 评测台 {bp} != 服务端 {sp}")
        if bad:
            return bad[:6]                    # 一局只报前几条，够定位了
        if bench.dead:
            break
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30, help="对账多少局")
    ap.add_argument("--ticks", type=int, default=2500, help="每局最多多少 tick")
    ap.add_argument("--seed0", type=int, default=12345)
    ap.add_argument("--max-climb", type=float, default=SPEC_MAX_CLIMB)
    ap.add_argument("--first-gap-extra", type=float, default=SPEC_FIRST_GAP)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed0)
    total_bad: list[str] = []
    deaths = 0
    for k in range(args.games):
        seed = args.seed0 + k
        # 拍翅序列：混合"随机"与"规律"，覆盖到冷却期内的连拍
        seq = rng.random(args.ticks) < 0.18
        # 强制加一些连续拍翅（检验冷却语义是否一致）
        if k % 3 == 0:
            seq[200:210] = True
        bad = compare_one(seed, args.ticks, seq,
                          max_climb=args.max_climb, first_gap_extra=args.first_gap_extra)
        if bad:
            total_bad.extend(bad)
        # 统计死亡，确认这批序列真的覆盖了碰撞路径
        bench = fb.World(np.random.default_rng(seed), gap_top=fb.NATURAL_GAP_TOP,
                         solvable=True, max_climb=args.max_climb,
                         first_gap_extra=args.first_gap_extra)
        cd = 0.0
        for i in range(args.ticks):
            cd = max(0.0, cd - 0.02)
            f = bool(seq[i]) and cd <= 0
            if f:
                cd = 0.14
            bench.step(f)
            if bench.dead:
                deaths += 1
                break

    print(f"  对账 {args.games} 局 × ≤{args.ticks} tick"
          f"（其中 {deaths} 局死亡，覆盖了碰撞路径）")
    if total_bad:
        print("  ❌ 不一致：")
        for b in total_bad[:20]:
            print("     " + b)
        return 1
    print("  ✅ 服务端 GameWorld 与评测台 World 逐 tick 等价")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
