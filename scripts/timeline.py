#!/usr/bin/env python
"""时间线估计：读各 run 的 log.csv，算出实际速率、剩余时间、以及整体关键路径。

直接用 python 读、避免 PowerShell 的编码/浮点显示问题。

用法：python scripts/timeline.py
"""
from __future__ import annotations

import csv
import json
import re
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# 目标步数：优先读 runs/PLAN.json（overnight.py 落的计划），否则从正在运行的
# 进程命令行读 --steps，最后兜底。写死数字会在换配置后显示全错。
def target_of(tag: str, plan: dict, proc: dict) -> int:
    if tag in proc and proc[tag]:
        return proc[tag]
    for g, jobs in (plan.get("groups") or {}).items():
        for j in jobs:
            if j.get("tag") == tag and j.get("steps"):
                return int(j["steps"])
    return 1_000_000


def running_steps() -> dict[str, int]:
    """tag -> --steps，取自正在运行的 train.py 命令行。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        ).stdout
    except Exception:
        return {}
    res: dict[str, int] = {}
    for line in out.splitlines():
        if "train.py" not in line:
            continue
        mt = re.search(r"--tag\s+(\S+)", line)
        ms = re.search(r"--steps\s+(\d+)", line)
        if mt:
            res[mt.group(1)] = int(ms.group(1)) if ms else 0
    return res


def running_tags() -> set[str]:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        ).stdout
    except Exception:
        return set()
    tags = set()
    for line in out.splitlines():
        if "train.py" in line and "--tag" in line:
            tail = line.split("--tag", 1)[1].strip()
            if tail:
                tags.add(tail.split()[0])
    return tags


def main() -> int:
    live = running_tags()
    proc = running_steps()
    try:
        plan = json.loads((ROOT / "runs" / "PLAN.json").read_text(encoding="utf-8"))
    except Exception:
        plan = {}
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        log = d / "log.csv"
        summ = d / "summary.json"
        tag = d.name
        tgt = target_of(tag, plan, proc)
        if summ.exists():
            rows.append((tag, tgt, tgt, 0.0, "完成", tag in live))
            continue
        if not log.exists():
            continue
        with log.open(encoding="utf-8") as f:
            r = list(csv.reader(f))
        if len(r) < 3:
            rows.append((tag, tgt, 0, 0.0, "刚启动", tag in live))
            continue
        hdr, first, last = r[0], r[1], r[-1]
        step = int(float(last[0]))
        dt = float(last[4]) - float(first[4])
        ds = step - int(float(first[0]))
        rate = ds / dt if dt > 0 else 0.0
        remain_s = (tgt - step) / rate if rate > 0 else float("inf")
        rows.append((tag, tgt, step, rate, remain_s, tag in live))

    print(f"{'run':<24}{'进度':>18}{'steps/s':>9}{'剩余':>12}  状态")
    print("-" * 76)
    total_remain = 0.0
    active_remain = []
    for tag, tgt, step, rate, rem, is_live in sorted(rows, key=lambda x: str(x[0])):
        if rem == "完成":
            print(f"{tag:<24}{'DONE':>18}{'-':>9}{'-':>12}  ✅")
            continue
        if rem == "刚启动":
            print(f"{tag:<24}{'0/'+str(tgt):>18}{'-':>9}{'?':>12}  {'运行中' if is_live else '已停'}")
            continue
        mins = rem / 60
        bar = f"{step:,}/{tgt:,}"
        print(f"{tag:<24}{bar:>18}{rate:>9.1f}{mins:>10.0f}min  {'运行中' if is_live else '⏸ 已停'}")
        if is_live:
            active_remain.append((tag, rem))

    print("-" * 76)
    if active_remain:
        slowest = max(active_remain, key=lambda x: x[1])
        print(f"运行中最慢的（= 关键路径）：{slowest[0]}  约 {slowest[1]/60:.0f} min "
              f"({slowest[1]/3600:.1f} h)")
        print(f"全部运行中的作业中最长的剩余时间：{slowest[1]/3600:.2f} 小时")
    print(f"当前运行中训练进程数：{len(live)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
