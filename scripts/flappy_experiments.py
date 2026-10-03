#!/usr/bin/env python
"""把 Flappy 的对照实验一次跑完，并汇总成一张表 + 结论 json。

单独一个脚本是因为要跑的东西不是"一个模式"，而是**一组对照**：
    基线（页面做法） / 零模型（喂脑但不拍翅） / 腹侧地面反射 / 管子边缘附加
    / 外部理想控制器（世界几何的上限刻度） / 左-右对照
再叠两件必须分开算账的事：拍翅冷却（游戏侧）和 s50size（投射侧）。

用法
    D:/Code/FlyBrain/env/python.exe scripts/flappy_experiments.py --games 40
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
BENCH = ROOT / "scripts" / "flappy_bench.py"
OUT = ROOT / "output"


def run(tag: str, modes: str, *, games: int, max_ticks: int, extra: list[str],
        asset: str = "spiking_full") -> dict:
    f = OUT / f"fx_{tag}.json"
    cmd = [PY, str(BENCH), "--asset", asset, "--modes", modes, "--games", str(games),
           "--max-ticks", str(max_ticks), "--out", str(f)] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        raise SystemExit(f"{tag} 失败")
    d = json.loads(f.read_text(encoding="utf-8"))
    return d


def row(d: dict, mode: str, tag: str, note: str = "") -> dict:
    s = d["summary"][mode]
    rows = [r for r in d["rows"] if r["mode"] == mode]
    ys = [v for r in rows for v in r["y_at_gap"]]
    return dict(exp=tag, mode=mode, games=s["n"], score_mean=round(s["score_mean"], 2),
                score_std=round(s["score_std"], 2), score_max=s["score_max"],
                pass_rate=round(s["pass_rate"], 3),
                ticks_mean=round(s["ticks_mean"]),
                flaps_mean=round(s["flaps_mean"], 2),
                ratio_mean=round(s["ratio_mean"], 3),
                over1=round(s["frac_over_1"], 3),
                y_at_gap_mean=round(st.mean(ys), 1) if ys else None,
                causes=s["causes"], note=note)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--max-ticks", type=int, default=1500)
    ap.add_argument("--asset", default="spiking_full")
    a = ap.parse_args()
    table: list[dict] = []

    # ---- 实验 1：主对照（同一颗脑、同一套物理，只换感觉投射）
    d = run("main", "passive,baseline,ground_lock,bidi,lookahead",
            games=a.games, max_ticks=a.max_ticks,
            extra=["--gap-margin", "22", "--vy-gate", "1", "--oracle-look", "80"],
            asset=a.asset)
    notes = {
        "passive": "零模型：同一套驱动喂给脑，但从不拍翅",
        "baseline": "当前页面做法：最近碰撞 + LC4 单通道角速度",
        "ground_lock": "腹侧 LPLC2 读地面角尺寸（第一轮最好）",
        "bidi": "腹侧/背侧双通道 + 接近速度门控（本轮最好）",
        "lookahead": "外部规划器：世界几何的上限刻度，非策略候选",
    }
    for m in d["summary"]:
        table.append(row(d, m, "main", notes.get(m, "")))

    # ---- 实验 1b：bidi 的成分消融（每一条都决定分数归零还是保住）
    for tag, extra, note in (
        ("bidi_no_vygate", ["--vy-gate", "0"], "关掉接近速度门控"),
        ("bidi_no_lc4", ["--bidi-groups", "LPLC2"], "只用 LPLC2（去掉 LC4）"),
        ("bidi_m8", ["--gap-margin", "8"], "死区 8 px"),
        ("bidi_m15", ["--gap-margin", "15"], "死区 15 px"),
        ("bidi_m35", ["--gap-margin", "35"], "死区 35 px"),
    ):
        d1 = run(tag, "bidi", games=a.games, max_ticks=a.max_ticks,
                 extra=["--gap-margin", "22", "--vy-gate", "1"] + extra, asset=a.asset)
        table.append(row(d1, "bidi", tag, note=note))

    # ---- 实验 1c：bidi 的脑侧干预（对照）
    for g in ("cut", "shuffled"):
        d1 = run(f"bidi_graph_{g}", "bidi", games=a.games, max_ticks=a.max_ticks,
                 extra=["--gap-margin", "22", "--vy-gate", "1", "--graph", g],
                 asset=a.asset)
        table.append(row(d1, "bidi", f"graph={g}", note="脑侧干预"))

    # ---- 实验 2：投射侧的灵敏度（s50size 是唯一的外部假设旋钮）
    for s50 in (20, 30, 45):
        d2 = run(f"s50size{s50}", "ground_lock", games=a.games,
                 max_ticks=a.max_ticks, extra=["--s50size", str(s50)], asset=a.asset)
        table.append(row(d2, "ground_lock", f"s50size={s50}"))

    # ---- 实验 3：游戏侧（拍翅冷却）——"改游戏"和"改脑的输入"必须分开算账
    for cd in (0.10, 0.14, 0.20):
        d3 = run(f"cd{cd}", "ground_lock", games=a.games,
                 max_ticks=a.max_ticks, extra=["--cooldown", str(cd)], asset=a.asset)
        table.append(row(d3, "ground_lock", f"cooldown={cd}s",
                         note="游戏侧改动，不与投射改动混算"))

    # ---- 实验 4：消融 —— 分数到底来自哪一路感觉投射
    for tag, mode, grp in (("abl_lc4", "abl_lc4", "LC4"),
                           ("abl_lplc2_full", "abl_lplc2_full", "LPLC2_full"),
                           ("abl_lc10a", "abl_lc10a", "LC10a")):
        d4 = run(tag, mode, games=a.games, max_ticks=a.max_ticks, extra=[],
                 asset=a.asset)
        table.append(row(d4, mode, tag, note=f"ground_lock 换成 {grp} 投射"))

    # ---- 实验 5：脑侧干预（对照）
    for g in ("cut", "shuffled", "no_inhibition"):
        d5 = run(f"graph_{g}", "ground_lock", games=a.games, max_ticks=a.max_ticks,
                 extra=["--graph", g], asset=a.asset)
        table.append(row(d5, "ground_lock", f"graph={g}",
                         note="脑侧干预：切断/打乱/去抑制"))

    hdr = ("exp", "mode", "games", "score_mean", "score_std", "score_max", "pass_rate",
           "ticks_mean", "flaps_mean", "ratio_mean", "over1", "y_at_gap_mean")
    print("\n" + "=" * 132)
    print("  ".join(f"{h:<14}" for h in hdr))
    for r in table:
        print("  ".join(f"{str(r[h]):<14}" for h in hdr))
    print("=" * 132)
    for r in table:
        print(f"  {r['exp']:<14}{r['mode']:<14}{r['causes']}  {r['note']}")

    (OUT / "flappy_experiments.json").write_text(
        json.dumps(dict(games=a.games, max_ticks=a.max_ticks, asset=a.asset,
                        table=table), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT / 'flappy_experiments.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
