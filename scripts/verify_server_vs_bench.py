"""服务端 vs 评测台：**同一关卡、同一参数、逐 tick 端到端对账**。

为什么需要这个脚本
------------------
`verify_server_physics.py` 只对物理（喂同一串拍翅命令），不碰脑；
`verify_server_gameplay.py` 只看服务端自己的分数。
而我在页面上观察到服务端单局均分偏低，`README` 又写着评测台 100.8 ——
必须回答"服务端跑的控制回路是不是和评测台同一条"。

这个脚本用**同一个关卡种子**把两边跑在一起，逐 tick 对比：
    每个 tick 的 y / vy / gap / DNp01 发放 / 累计 recent / 是否拍翅
最后再比总分。任何一项不等就报出来（并给出前几个 tick 的并排轨迹）。

用法
----
    python scripts/verify_server_vs_bench.py --games 4 --max-ticks 3000
    python scripts/verify_server_vs_bench.py --games 1 --trace-n 40 --first-gap-extra 0
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "demo"))

import flappy_bench as fb                      # noqa: E402
from server import BIDI, GROUND_Y, FIRST_GAP_EXTRA, MAX_CLIMB, Session  # noqa: E402


def run_bench(asset: str, seed: int, max_ticks: int, fge: float, seed0: int = 0):
    """跑评测台一局，逐 tick 记录**同样的字段**。"""
    h = fb.Harness(asset, gain=3.0, tonic=0.0, need_spikes=1)
    h.set_graph("real")
    # 脑的噪声种子**必须显式对齐**：原来噪声吃全局 RNG，两个进程消耗量不同
    # → 同一个种子也飞得完全不同（实测膜电位最大差 0.674）。修复后这里才有意义。
    h.brain.set_seed(seed0)
    spec = dict(s50=15.0, s50size=30.0, gain=1.0, width=0.5, baseline=0.25,
                ground_aware=1, elev_scale=1.0, pipe_w=1.0, span=0.6, base=0.6,
                gap_margin=18.0, groups=None, vy_gate=1, max_climb=MAX_CLIMB,
                ceil_boost=1.0, predict_horizon=400, dors_scale=0.35,
                vent_gain=2.0, vent_dev=120.0, first_gap_extra=fge,
                group="LPLC2", predict=0, game_slack=1.0)
    row = h.play("bidi", spec=spec, seed=seed, max_ticks=max_ticks,
                 solvable=True, policy="brain", trace=True)
    tr = row.get("trace") or []
    # 评测台的 trace 从 tick=1 开始（先 step 后记）。为便于并排，整体前移一位。
    out = []
    for e in tr:
        out.append(dict(tick=e["tick"] - 1, y=e["y"], vy=e["vy"], gap=e["gap"],
                        spk=e["spk"], recent=e["recent"]))
    return dict(score=row.get("score", 0), flaps=row.get("flaps", 0),
                ticks=len(tr), trace=out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spiking_full")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--max-ticks", type=int, default=3000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--first-gap-extra", type=float, default=FIRST_GAP_EXTRA)
    ap.add_argument("--trace-n", type=int, default=12, help="不一致时并排打印多少 tick")
    ap.add_argument("--tol", type=float, default=0.15, help="y/vy 的容差（浮点/顺序差异）")
    ap.add_argument("--brain-seed", type=int, default=7,
                    help="脑噪声种子；两边必须相同（原来吃全局 RNG，对不齐）")
    a = ap.parse_args()

    print(f"载入 {a.asset} 的服务端会话…")
    sess = Session(a.asset, "cpu", start_loop=False)
    ok = True

    for g in range(a.games):
        seed = a.seed0 + g
        srv_tr: list[dict] = []
        with sess.game_lock:
            from server import GameWorld
            sess.game = GameWorld(max_climb=MAX_CLIMB,
                                  first_gap_extra=a.first_gap_extra, seed=seed)
            sess.brain.set_seed(a.brain_seed)
            sess.reset()
            sess.game_stats.update(ticks=0, flaps=0)
            n = 0
            while n < a.max_ticks and not sess.game.dead:
                sess._game_tick()
                w = sess.game
                srv_tr.append(dict(tick=n, y=round(w.y, 4), vy=round(w.vy, 4),
                                   gap=round(w.gap_center(), 4),
                                   spk=int(sess.dn_tick[-1]) if sess.dn_tick else 0,
                                   recent=int(sum(sess.dn_tick))))
                n += 1
            srv = dict(score=w.score, flaps=sess.game_stats["flaps"], ticks=n)
        ben = run_bench(a.asset, seed, a.max_ticks, a.first_gap_extra,
                        seed0=a.brain_seed)

        print(f"\n  seed={seed}")
        print(f"    服务端  score={srv['score']:<4} ticks={srv['ticks']:<5} "
              f"flaps={srv['flaps']}")
        print(f"    评测台  score={ben['score']:<4} ticks={ben['ticks']:<5} "
              f"flaps={ben['flaps']}")

        # ---- 逐 tick 对比
        bad = []
        for i in range(min(len(srv_tr), len(ben['trace']))):
            s_, b_ = srv_tr[i], ben['trace'][i]
            for k in ("y", "vy"):
                if abs(s_[k] - b_[k]) > a.tol:
                    bad.append((i, k, s_[k], b_[k]))
            for k in ("spk", "recent"):
                if s_[k] != b_[k]:
                    bad.append((i, k, s_[k], b_[k]))
            if bad:
                break
        if len(srv_tr) != len(ben['trace']):
            bad.append(("长度", "ticks", len(srv_tr), len(ben['trace'])))
        if srv['score'] != ben['score']:
            bad.append(("总分", "score", srv['score'], ben['score']))

        if not bad:
            print(f"    ✅ 逐 tick 一致（{min(len(srv_tr), len(ben['trace']))} tick）"
                  f"，总分也相同")
        else:
            ok = False
            print(f"    ❌ 不一致：{bad[:4]}")
            i0 = bad[0][0] if isinstance(bad[0][0], int) else 0
            lo = max(0, i0 - 3)
            print(f"    {'tick':>5} {'服务端 y/vy/spk/rec':>34}   {'评测台 y/vy/spk/rec':>34}")
            for i in range(lo, min(lo + a.trace_n, len(ben['trace']))):
                if i >= len(srv_tr):
                    break
                s_, b_ = srv_tr[i], ben['trace'][i]
                mark = "  <<<" if i == i0 else ""
                print(f"    {i:>5} {s_['y']:>12} {s_['vy']:>12} {s_['spk']:>4} {s_['recent']:>5}"
                      f"   {b_['y']:>12} {b_['vy']:>12} {b_['spk']:>4} {b_['recent']:>5}{mark}")

    sess.close()
    print("\n" + ("✅ 服务端控制回路与评测台一致" if ok else "❌ 两边不一致，见上面的对比"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
