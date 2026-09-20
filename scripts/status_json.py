#!/usr/bin/env python
"""训练监控面板：把 runs/ 的实时状态汇集起来，供 dashboard 服务器读取。

只读、不干扰训练。输出一个 dict：
    {
      "now": 时间戳,
      "runs": [ {tag, arch, device, step, target, rate, eta_min, status,
                 recent_score, loss, eps, wall_min, graph, weight_mode}, ... ],
      "gpu": {util, mem_used, mem_total},
      "cpu": {percent, cores},
      "ram": {used_gb, total_gb},
      "summary": {running, done, total_remaining_h, critical_path}
    }
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def _ps(cmd: str) -> str:
    try:
        return subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45,
        ).stdout
    except Exception:
        return ""


def live_processes() -> dict[str, dict]:
    """tag -> {device, pid}，只看正在跑的 train.py。"""
    out = _ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              "Select-Object -ExpandProperty CommandLine")
    res: dict[str, dict] = {}
    for line in out.splitlines():
        if "train.py" not in line or "--tag" not in line:
            continue
        m = re.search(r"--tag\s+(\S+)", line)
        if not m:
            continue
        tag = m.group(1)
        dv = re.search(r"--device\s+(\S+)", line)
        res[tag] = {"device": dv.group(1) if dv else "auto"}
    return res


def gpu_info() -> dict:
    out = _ps("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
              "--format=csv,noheader,nounits")
    try:
        u, mu, mt = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return {"util": float(u), "mem_used": float(mu), "mem_total": float(mt)}
    except Exception:
        return {"util": None, "mem_used": None, "mem_total": None}


def cpu_ram() -> dict:
    """CPU / 内存：优先用 psutil（直接系统调用，不会有解析问题）。

    早期版本绕 PowerShell 读 WMI，结果又慢又脆（PS5 与 PS7 输出格式不同、
    内存单位还要靠猜），实测 RAM 显示成 0.02 GB。改掉。
    """
    out = {"percent": None, "cores": os.cpu_count() or 0,
           "ram_used_gb": None, "ram_total_gb": None}
    try:
        import psutil
        out["percent"] = float(psutil.cpu_percent(interval=0.25))
        vm = psutil.virtual_memory()
        out["ram_used_gb"] = (vm.total - vm.available) / 1024**3
        out["ram_total_gb"] = vm.total / 1024**3
        out["cores"] = psutil.cpu_count(logical=True) or out["cores"]
        return out
    except Exception:
        pass
    # psutil 不可用时的兜底：Windows 上查 GlobalMemoryStatusEx 太重，就不显示内存了
    try:
        out["percent"] = float(os.getloadavg()[0]) if hasattr(os, "getloadavg") else None
    except Exception:
        pass
    return out


def target_of(tag: str, proc_steps: dict[str, int] | None = None) -> int:
    """该 run 的目标步数（env-step）。

    优先级：正在运行的进程命令行 > 该 run 的 log.csv 里出现过的最大目标 > 兜底值。
    之前写死了 60k/300k，改成 1M 后显示全错，所以改成以实测为准。
    """
    if proc_steps and tag in proc_steps and proc_steps[tag]:
        return proc_steps[tag]
    # 从 summary.json 读（完成的 run）
    sj = RUNS / tag / "summary.json"
    if sj.exists():
        try:
            return int(json.loads(sj.read_text(encoding="utf-8")).get("steps") or 0) or 300_000
        except Exception:
            pass
    return 300_000


def proc_steps() -> dict[str, int]:
    """tag -> --steps，从正在运行的 train.py 命令行解析。"""
    out = _ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              "Select-Object -ExpandProperty CommandLine")
    res: dict[str, int] = {}
    for line in out.splitlines():
        if "train.py" not in line or "--tag" not in line:
            continue
        mt = re.search(r"--tag\s+(\S+)", line)
        ms = re.search(r"--steps\s+(\d+)", line)
        if mt:
            res[mt.group(1)] = int(ms.group(1)) if ms else 0
    return res


def arch_of(tag: str) -> str:
    # 注意顺序：'rewired' 里不含 'connectome'，但 tag 形如 rep_sg_xxx_connectome 时
    # 也要先判 rewired（历史上曾把 rewired 误判为 connectome）
    for a in ("rewired", "connectome", "mlp"):
        if a in tag:
            return a
    return "?"


def collect() -> dict:
    live = live_processes()
    steps_by_tag = proc_steps()
    runs = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        tag = d.name
        log = d / "log.csv"
        summ = d / "summary.json"
        tgt = target_of(tag, steps_by_tag)
        rec = {
            "tag": tag, "arch": arch_of(tag), "target": tgt,
            "device": live.get(tag, {}).get("device", "-"),
            "running": tag in live,
            "step": 0, "rate": 0.0, "eta_min": None, "status": "未开始",
            "recent_score": None, "loss": None, "eps": None, "wall_min": None,
            "final_mean": None, "final_max": None,
        }
        if summ.exists():
            try:
                sj = json.loads(summ.read_text(encoding="utf-8"))
                rec["final_mean"] = sj["final_eval"]["mean"]
                rec["final_max"] = sj["final_eval"]["max"]
                rec["status"] = "✅ 完成"
                rec["step"] = tgt
                runs.append(rec)
                continue
            except Exception:
                pass
        if not log.exists():
            runs.append(rec)
            continue
        try:
            with log.open(encoding="utf-8") as f:
                rows = list(csv.reader(f))
        except Exception:
            runs.append(rec)
            continue
        if len(rows) < 3:
            rec["status"] = "刚启动"
            rec["running"] = tag in live
            runs.append(rec)
            continue
        first, last = rows[1], rows[-1]
        step = int(float(last[0]))
        dt = float(last[4]) - float(first[4])
        ds = step - int(float(first[0]))
        rate = ds / dt if dt > 0 else 0.0
        rec.update({
            "step": step,
            "rate": rate,
            "recent_score": float(last[3]) if last[3] else None,
            "loss": float(last[1]) if last[1] else None,
            "eps": float(last[2]) if last[2] else None,
            "wall_min": float(last[4]) / 60,
            "status": "运行中" if tag in live else "⏸ 已停",
        })
        rec["eta_min"] = (tgt - step) / rate / 60 if rate > 0 else None
        runs.append(rec)

    act = [r for r in runs if r["running"]]
    etas = [r["eta_min"] for r in act if r["eta_min"]]
    crit = max(act, key=lambda r: r["eta_min"] or 0) if act else None
    return {
        "now": time.time(),
        "runs": runs,
        "gpu": gpu_info(),
        "sys": cpu_ram(),
        "summary": {
            "running": len(act),
            "done": sum(1 for r in runs if r["status"].startswith("✅")),
            "total": len(runs),
            "max_eta_min": max(etas) if etas else None,
            "critical_path": crit["tag"] if crit else None,
        },
    }


if __name__ == "__main__":
    print(json.dumps(collect(), ensure_ascii=False, indent=2))
