"""游戏性对账：**服务端真正的 `Session._game_tick`** 能拿多少分？

为什么要这个脚本
----------------
`scripts/verify_server_physics.py` 只证明了**物理**逐 tick 等价（喂同一串拍翅命令、
比位置/速度/碰撞）。它**不碰脑**，所以它证明不了"服务端跑同一个控制回路也能拿分"。

而我在页面上观察到过一件可疑的事：服务端连续跑 22 分钟、单局均分只有 ~9
（最高 149），而评测台报的是**100.8**。差一个数量级，必须查清是
① 参数不一致 ② 控制回路形状不对 ③ 还是只是关卡随机性。

这个脚本直接调用服务端的真代码（`Session._game_tick`），跑 N 局，
报均分/最高分/每局拍翅数，用来和评测台对比。

用法
----
    python scripts/verify_server_gameplay.py --games 8 --max-ticks 8000
    python scripts/verify_server_gameplay.py --games 4 --tag "对比第一根间距"
"""
from __future__ import annotations

import argparse
import pathlib
import statistics as st
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "demo"))

from server import FIRST_GAP_EXTRA, MAX_CLIMB, Session, GameWorld   # noqa: E402


def one_game(sess: Session, max_ticks: int, *, first_gap_extra: float,
             seed: int | None = None, verbose: bool = False,
             trace: int = 0) -> dict:
    """跑一局，返回分数/拍翅数/死因。**用服务端真正的 tick 逻辑**。

    `trace>0` 时记录前 N 个 tick 的轨迹，用来和评测台的 `--trace-mode bidi`
    逐 tick 对比 —— 这是定位"两边控制回路形状是否相同"的唯一可靠办法。
    """
    tr: list[dict] = []
    with sess.game_lock:
        sess.game = GameWorld(max_climb=MAX_CLIMB,
                              first_gap_extra=first_gap_extra, seed=seed)
        sess.reset()
        sess.game_stats.update(ticks=0, flaps=0)
        start = sess.game_stats["flaps"]
        n = 0
        while n < max_ticks and not sess.game.dead:
            sess._game_tick()
            if trace and n < trace:
                w = sess.game
                tr.append(dict(
                    tick=n, t=round(w.t, 3), y=round(w.y, 1), vy=round(w.vy, 1),
                    gap=round(w.gap_center(), 1), dev=round(w.y - w.gap_center(), 1),
                    amp=round(float(sess.last_plan.get("amp", 0.0)), 5),
                    branch=sess.last_plan.get("branch", ""),
                    spk=int(sess.dn_tick[-1]) if sess.dn_tick else 0,
                    recent=int(sum(sess.dn_tick)),
                    cd=round(w.cooldown, 3), score=w.score))
            n += 1
        sc = sess.game.score
        cause = sess.game.cause
        flaps = sess.game_stats["flaps"] - start
    if verbose:
        print(f"    ticks={n:5d}  score={sc:4d}  flaps={flaps:4d}  cause={cause!r}")
    return dict(score=sc, ticks=n, flaps=flaps, cause=cause, seed=seed, trace=tr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spiking_full")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--max-ticks", type=int, default=8000, help="每局上限（8000=160s）")
    ap.add_argument("--first-gap-extra", type=float, default=None,
                    help="覆盖第 2 根管子的额外间距（默认用 server.py 的值）")
    ap.add_argument("--seed", type=int, default=None,
                    help="关卡种子；给了就**可复现**（用来和评测台跑同一关）")
    ap.add_argument("--tag", default="", help="这次跑法的标注，方便对比")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--trace", type=int, default=0,
                    help="打印第 1 局前 N 个 tick 的轨迹（和评测台 --trace-mode 对比）")
    a = ap.parse_args()

    fge = FIRST_GAP_EXTRA if a.first_gap_extra is None else a.first_gap_extra
    tag = a.tag or "默认"
    print(f"载入 {a.asset}（{tag}）…")
    t0 = time.time()
    sess = Session(a.asset, a.device, start_loop=False)   # 不要后台线程抢状态
    print(f"  用时 {time.time()-t0:.1f}s   N={sess.base.N:,}")
    print(f"  参数：max_climb={MAX_CLIMB}  first_gap_extra={fge}  "
          f"BIDI={sess.last_plan!r}"[:120])

    res = [one_game(sess, a.max_ticks, first_gap_extra=fge,
                    seed=(a.seed + k) if a.seed is not None else None,
                    verbose=a.verbose,
                    trace=(a.trace if (a.trace and k == 0) else 0))
           for k in range(a.games)]
    if a.trace and res and res[0]["trace"]:
        print(f"\n  轨迹（第 1 局）：每 tick 一行")
        for e in res[0]["trace"]:
            print(f"    tick={e['tick']:<4} t={e['t']:<6} y={e['y']:<7} vy={e['vy']:<8} "
                  f"gap={e['gap']:<7} dev={e['dev']:<7} amp={e['amp']:<8} "
                  f"branch={e['branch']:<8} spk={e['spk']} recent={e['recent']:<2} "
                  f"cd={e['cd']:<5} score={e['score']}")
    scores = [r["score"] for r in res]
    flaps = [r["flaps"] for r in res]
    causes = {}
    for r in res:
        causes[r["cause"] or "(没死)"] = causes.get(r["cause"] or "(没死)", 0) + 1

    print(f"\n  {tag}：跑 {a.games} 局，每局上限 {a.max_ticks} tick")
    print(f"    均分   {st.mean(scores):.2f}")
    print(f"    中位   {st.median(scores):.1f}")
    print(f"    最高   {max(scores)}")
    print(f"    最低   {min(scores)}")
    print(f"    拍翅   均 {st.mean(flaps):.0f} / 局")
    print(f"    死因   {causes}")
    print(f"\n  评测台（real，40 局）是 **100.8**。"
          f"如果这里差很多，先比参数再比控制回路形状。")
    sess.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
