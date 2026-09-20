#!/usr/bin/env python
"""串行队列：等一批 run 完成后，自动接着跑下一批。

为什么需要：CPU 侧（n3000 图）比预想快得多，能在主实验结束前腾出算力。
与其让核闲着，不如排一个「独立复现」批次 —— 在**另一张子图**上把三方对照重做一遍，
这比在同一张图上多跑几个种子有价值得多（检验结论对子图选择的依赖性）。

用法：
    python scripts/queue_runs.py --wait ab_n3000_norm ab_n3000_binary ab_n3000_rewired \
        --then "connectome:s300_n3000:norm" "rewired:s300_n3000:norm" "mlp::"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PY = sys.executable


def done(tag: str) -> bool:
    return (RUNS / tag / "summary.json").exists()


def launch(arch: str, graph: str, wm: str, steps: int, seed: int, threads: int,
           device: str = "cpu") -> subprocess.Popen:
    tag = f"{'rep' if graph else 'main'}_{arch}"
    if graph:
        gname = graph.split(":")[-1] if ":" in graph else graph
        tag = f"rep_{gname}_{arch}"
    cmd = [PY, str(ROOT / "scripts" / "train.py"), "--arch", arch, "--steps", str(steps),
           "--seed", str(seed), "--no-eval", "--tag", tag, "--device", device]
    if graph:
        cmd += ["--graph", graph.split(":")[0], "--weight-mode", wm]
    env = {"PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": str(threads),
           "MKL_NUM_THREADS": str(threads), "PATH": __import__("os").environ.get("PATH", "")}
    print(f"[queue] 启动 {tag} (device={device}): {' '.join(cmd[1:])}", flush=True)
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)


def running_train_tags() -> set[str]:
    """当前正在跑的 train.py 的 --tag 集合（用来判断有没有槽位）。"""
    tags = set()
    try:
        import subprocess as sp
        out = sp.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:
        return tags
    for line in out.splitlines():
        if "train.py" in line and "--tag" in line:
            parts = line.split("--tag")
            if len(parts) > 1:
                tags.add(parts[1].strip().split()[0])
    return tags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", nargs="*", default=[], help="等这些 tag 的 summary.json 出现")
    ap.add_argument("--then", nargs="*", default=[], help="然后各起一个 run，格式 arch:graph:weight_mode")
    ap.add_argument("--max-concurrent", type=int, default=0,
                    help=">0 时改为「等运行中的 train 数 <= 该值」再启动，而不是等指定 tag")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()

    if a.max_concurrent > 0:
        print(f"[queue] 等待「运行中的 train 数 <= {a.max_concurrent}」以腾出槽位", flush=True)
        t0 = time.time()
        while True:
            cur = running_train_tags()
            if len(cur) <= a.max_concurrent:
                print(f"[queue] 当前 {len(cur)} 个: {sorted(cur)}，可以启动", flush=True)
                break
            if int(time.time() - t0) % 600 < a.poll:
                print(f"[queue] {int(time.time()-t0)}s  运行中 {len(cur)} 个: {sorted(cur)}", flush=True)
            time.sleep(a.poll)
    elif a.wait:
        print(f"[queue] 等待 {len(a.wait)} 个 run 完成: {', '.join(a.wait)}", flush=True)
        t0 = time.time()
        while not all(done(t) for t in a.wait):
            time.sleep(a.poll)
            st = ", ".join(f"{t}={'ok' if done(t) else '…'}" for t in a.wait)
            print(f"[queue] {int(time.time()-t0)}s  {st}", flush=True)
        print(f"[queue] 全部完成，用时 {int(time.time()-t0)}s", flush=True)

    if not a.then:
        return 0

    procs = []
    for spec in a.then:
        parts = (spec.split(":") + ["", ""])[:3]
        arch, graph, wm = parts[0], parts[1], parts[2] or "norm"
        # 设备选择：MLP 在 CPU 上更快（小矩阵，GPU 启动开销占主导）；
        # 图网络在 GPU 上快约 3 倍（3000² 稠密乘）。所以按架构分流。
        dev = "cpu" if arch == "mlp" else "cuda"
        procs.append(launch(arch, graph, wm, a.steps, a.seed, a.threads, dev))
    print(f"[queue] 已启动 {len(procs)} 个复现 run，等待结束...", flush=True)
    codes = [p.wait() for p in procs]
    print(f"[queue] 复现批次结束，退出码 {codes}", flush=True)
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
