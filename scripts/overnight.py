#!/usr/bin/env python
"""夜间运行器：把所有实验跑完，**崩溃/中断能自动从存档续上**。

解决的核心问题
--------------
晚上挂机最怕的是「跑了 5 小时，进程挂了，进度全丢」。
这个脚本把 train.py 的 `--save-every` + `--resume auto` 包成一个自恢复循环：
    while step < target:
        启动 train.py --resume auto
        若它异常退出 -> 等几秒，用存档续跑（丢的只是最后一个存档间隔）
        若它正常结束 -> 进入下一个 run

用法
----
    python scripts/overnight.py                    # 跑默认实验清单
    python scripts/overnight.py --steps 300000 --save-every 20000
    python scripts/overnight.py --dry-run          # 只打印计划

实验清单（改这里就能换任务）
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PY = os.environ.get("FPV_PY", r"D:\Code\DQN\env\python.exe")

# (tag, arch, graph, weight_mode, device, steps, seed)
#   steps 是 **env-step**（与 train.py 默认的 env-step 语义一致）
#   seed  必须给！早期版本忘了传 --seed，导致 s0/s1 其实是同一个种子、完全重复
JOBS: list[tuple[str, str, str, str, str, int, int]] = []


def add(tag, arch, graph="", wm="norm", device="cuda", steps=300_000, seed=0):
    JOBS.append((tag, arch, graph, wm, device, steps, seed))


# ---- A 计划：主对照 1M env-step，三方各 3 个种子 ----
# 为什么统一到 1M：原项目 baseline 要 97.5 万 env-step 才到均分 125；
# 在 30 万步时三组都挤在 0~30 分，区分度不够（第一轮 g1 已验证）。
# 为什么每方 3 个种子：g1 的种子方差极大（mlp 64 vs 4148），
# ②→③ 的差异只有 12 分，单种子无法区分信号与噪声。
N = 1_000_000
for sd in (0, 1, 2):
    add(f"mlp_s{sd}", "mlp", device="cpu", steps=N, seed=sd)
    add(f"rewired_s{sd}", "rewired", "sg_collision_s300_n4000", device="cuda", steps=N, seed=sd)
    add(f"connectome_s{sd}", "connectome", "sg_collision_s300_n4000", device="cuda", steps=N, seed=sd)

# ---- 独立复现：换一张子图（n3000），检验结论对子图选择是否敏感 ----
for sd in (10, 11):
    add(f"rep_n3000_mlp_s{sd}", "mlp", device="cpu", steps=N, seed=sd)
    add(f"rep_n3000_rewired_s{sd}", "rewired", "sg_collision_s200_n3000",
        device="cuda", steps=N, seed=sd)
    add(f"rep_n3000_connectome_s{sd}", "connectome", "sg_collision_s200_n3000",
        device="cuda", steps=N, seed=sd)

# ---- 权重消融：同一张图，只换「边权怎么初始化」----
# 问：突触强度信息有没有用？（norm 用数据自带归一化 / count 用 log1p 原始突触数 / binary 只剩拓扑）
for wm in ("norm", "count", "binary"):
    add(f"ab_n3000_{wm}", "connectome", "sg_collision_s200_n3000", wm,
        device="cuda", steps=N, seed=20)


def is_done(tag: str) -> bool:
    p = RUNS / tag / "summary.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return int(d.get("steps", 0)) > 0
    except Exception:
        return False


def cur_step(tag: str) -> int:
    """从日志最后一行读当前步数（用来看要不要续跑）。"""
    f = RUNS / tag / "log.csv"
    if not f.exists():
        return 0
    try:
        lines = f.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 2:
            return 0
        return int(float(lines[-1].split(",")[0]))
    except Exception:
        return 0


def run_group(jobs, save_every, max_retries) -> dict[str, bool]:
    """并发跑一组（每组一个进程），每个 job 带自恢复。

    为什么分组并发而不是全串行/全并发：
      - 全串行：GPU 会有一段时间被单个作业占不满，总时长最长；
      - 全并发（13 个）：GPU 算力被均分，每个作业都慢，且更容易触发 OOM/抖动；
      - 分组并发（GPU 一组 4-5 个 + CPU 一组 1-2 个）：GPU 吃满、CPU 吃满，
        总时长≈「总 GPU 计算量 / GPU 吞吐」，且每个作业仍有存档保护。
    """
    procs = {}
    results: dict[str, bool] = {}
    for (tag, arch, graph, wm, device, steps, seed) in jobs:
        if is_done(tag):
            print(f"[overnight] {tag}: 已完成，跳过", flush=True)
            results[tag] = True
            continue
        ckpt = ROOT / "checkpoints" / f"{tag}.pt"
        cmd = [PY, str(ROOT / "scripts" / "train.py"),
               "--arch", arch, "--tag", tag, "--steps", str(steps),
               "--device", device, "--save-every", str(save_every),
               "--seed", str(seed),
               # 开启中途评测：没有学习曲线就无法回答「连接组是不是学得更慢/更晚起飞」
               "--eval-every", "50000", "--eval-episodes", "10"]
        if graph:
            cmd += ["--graph", graph, "--weight-mode", wm]
        if ckpt.exists():
            cmd += ["--resume", "auto"]
            print(f"[overnight] {tag}: 续跑（已有存档，当前 {cur_step(tag):,} 步）", flush=True)
        else:
            print(f"[overnight] {tag}: 开跑 {arch} @ {device}  目标 {steps:,} 步", flush=True)
        procs[tag] = subprocess.Popen(
            cmd, cwd=str(ROOT),
            env={**os.environ, "PYTHONIOENCODING": "utf-8",
                 "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})

    # 轮询等待；某个挂了就重启它（带 resume）
    retries: dict[str, int] = {t: 0 for t in procs}
    spec = {j[0]: j for j in jobs}
    while procs:
        time.sleep(30)
        for tag in list(procs.keys()):
            p = procs[tag]
            if p.poll() is None:
                continue
            rc = p.returncode
            if rc == 0 and is_done(tag):
                print(f"[overnight] {tag}: ✅ 完成", flush=True)
                results[tag] = True
                del procs[tag]
                continue
            retries[tag] += 1
            if retries[tag] > max_retries:
                print(f"[overnight] {tag}: ✗ 退出码 {rc}，重试超限，放弃", flush=True)
                results[tag] = False
                del procs[tag]
                continue
            print(f"[overnight] {tag}: ✗ 退出码 {rc} @ {cur_step(tag):,} 步 "
                  f"-> 第 {retries[tag]} 次续跑", flush=True)
            _, arch, graph, wm, device, steps, seed = spec[tag]
            cmd = [PY, str(ROOT / "scripts" / "train.py"),
                   "--arch", arch, "--tag", tag, "--steps", str(steps),
                   "--device", device, "--save-every", str(save_every),
                   "--seed", str(seed),
                   "--eval-every", "50000", "--eval-episodes", "10",
                   "--resume", "auto"]
            if graph:
                cmd += ["--graph", graph, "--weight-mode", wm]
            procs[tag] = subprocess.Popen(
                cmd, cwd=str(ROOT),
                env={**os.environ, "PYTHONIOENCODING": "utf-8",
                     "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0, help="覆盖所有 job 的目标步数")
    ap.add_argument("--save-every", type=int, default=20_000)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=[], help="只跑这些 tag")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--group", default=None,
                    help="只跑某一组：g1(主对照) / g2(复现+消融)")
    a = ap.parse_args()

    jobs = [j for j in JOBS if not a.only or j[0] in a.only]
    if a.steps:
        # 覆盖所有 job 的步数（注意元组有 7 个字段：tag/arch/graph/wm/device/steps/seed）
        jobs = [(t, ar, g, w, d, a.steps, sd) for (t, ar, g, w, d, _, sd) in jobs]

    # 分组：先做主对照（科学上最重要），再跑复现与消融
    # 主对照 = 三个架构各 3 个种子的 *_s0/_s1/_s2
    MAIN = {f"{a_}_s{s_}" for a_ in ("mlp", "rewired", "connectome") for s_ in (0, 1, 2)}
    g1 = [j for j in jobs if j[0] in MAIN]
    g2 = [j for j in jobs if j not in g1]
    groups = {"g1": g1, "g2": g2} if not a.group else {a.group: (g1 if a.group == "g1" else g2)}

    print(f"[overnight] 清单 {len(jobs)} 个 job，分 {len(groups)} 组")
    for gname, gj in groups.items():
        print(f"  {gname}: {len(gj)} 个")
        for t, ar, gr, w, d, s, sd in gj:
            mark = "✅完成" if is_done(t) else f"{cur_step(t):,}/{s:,}"
            print(f"      {mark:>14}  {t:<24} {ar:<11} {d:<5} seed{sd}  {gr or '-'}")
    if a.dry_run:
        return 0

    # 落一份计划文件：回来看进度时能一眼核对「这次到底按什么配置跑的」
    try:
        import json as _json
        (ROOT / "runs").mkdir(parents=True, exist_ok=True)
        (ROOT / "runs" / "PLAN.json").write_text(_json.dumps({
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps_per_job": a.steps or "per-job",
            "save_every": a.save_every,
            "max_retries": a.max_retries,
            "groups": {g: [{"tag": t, "arch": ar, "graph": gr, "weight_mode": w,
                            "device": d, "steps": s, "seed": sd}
                           for (t, ar, gr, w, d, s, sd) in gj]
                       for g, gj in groups.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[overnight] 写 PLAN.json 失败（不影响训练）：{e}")

    all_res: dict[str, bool] = {}
    for gname, gj in groups.items():
        if not gj:
            continue
        todo = [j for j in gj if not is_done(j[0])]
        if not todo:
            print(f"[overnight] {gname}: 全部已完成", flush=True)
            continue
        # 先把 CPU 作业起掉（它们和 GPU 作业互不抢资源），再起 GPU 作业
        cpu_jobs = [j for j in todo if j[4] == "cpu"]
        gpu_jobs = [j for j in todo if j[4] != "cpu"]
        print(f"\n[overnight] === {gname}：{len(gpu_jobs)} 个 GPU + {len(cpu_jobs)} 个 CPU 并发 ===",
              flush=True)
        res = run_group(cpu_jobs + gpu_jobs, a.save_every, a.max_retries)
        all_res.update(res)

    failed = [t for t, okk in all_res.items() if not okk]
    print(f"\n[overnight] 收尾：成功 {sum(all_res.values())}/{len(all_res)}"
          + (f"，失败：{failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
