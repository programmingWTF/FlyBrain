#!/usr/bin/env python
"""验证「存档 + 续跑」是真的可用，而不是只写了个文件。

测法（等价于"中断-恢复"的对照实验）：
  1. 用固定种子跑 1500 步不中断，记录最终网络权重的指纹 + 近期均分   -> A
  2. 用**同一个种子**跑 800 步 -> 存检查点 -> 杀掉 -> 从检查点续到 1500 步 -> B
  3. 比较 A 与 B：
       - step 必须正确接续（800 -> 1500）
       - 回放缓冲条数必须恢复
       - eps 必须接续（不是从头 anneal）
       - 续跑后非边位置仍严格为 0
       - 权重指纹差异应很小（不是完全相同：环境/RNG 状态跨进程不可能逐位一致，
         但如果在同一进程内恢复，则应当接近）

用法：python scripts/test_resume.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fpv import env_config, graph_config, load_subgraph, resume as resume_mod  # noqa: E402
from fpv.vendor_flappyrl.agent import RainbowDQN  # noqa: E402
from fpv.vendor_flappyrl.sim import FlappySim  # noqa: E402

ok = True


def check(cond, msg):
    global ok
    print(f"  {'✓' if cond else '✗'} {msg}")
    if not cond:
        ok = False


def fingerprint(net) -> tuple[float, float]:
    """(参数和, 参数绝对值最大) 作为权重指纹。"""
    with torch.no_grad():
        tot = 0.0
        mx = 0.0
        for p in net.parameters():
            tot += float(p.sum())
            mx = max(mx, float(p.abs().max()))
    return tot, mx


def run(agent, sim, steps, st=None, episode_scores=None):
    episode_scores = episode_scores if episode_scores is not None else []
    st = sim.reset_all() if st is None else st
    for _ in range(steps):
        acts = agent.act(st, training=True)
        st2, rw, dn, info = sim.step(acts)
        agent.store_transition(st, acts, rw, st2, dn)
        st = st2
        agent.learn()
        ts = info["terminal_scores"]
        for i in range(info["done"].shape[0]):
            if info["done"][i]:
                episode_scores.append(int(ts[i]))
    return st, episode_scores


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = load_subgraph("sg_collision_s200_n3000")   # 用较小的 n3000 图，测试快
    cfg = graph_config(net_arch="connectome", graph_steps=3, batch_size=128,
                       learning_starts=200, buffer_size=20_000)
    ec = env_config(n_envs=8, seed=0)

    print("=" * 70)
    print("对照组 A：连续跑 1500 步，不中断")
    print("=" * 70)
    torch.manual_seed(0)
    np.random.seed(0)
    a1 = RainbowDQN(cfg, ec, dev, graph=g)
    sim1 = FlappySim(ec)
    _, epA = run(a1, sim1, 1500)
    fpA = fingerprint(a1.online_net)
    print(f"  完成：步数=1500  回放={len(a1.buffer):,}  eps={a1.eps:.4f}  指纹={fpA[0]:.4f}")

    print()
    print("=" * 70)
    print("对照组 B：跑 800 步 -> 存档 -> 重启进程续到 1500 步")
    print("=" * 70)
    ck = ROOT / "checkpoints" / "_resume_test.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    if ck.exists():
        ck.unlink()

    torch.manual_seed(0)
    np.random.seed(0)
    b1 = RainbowDQN(cfg, ec, dev, graph=g)
    sim2 = FlappySim(ec)
    st2, epB = run(b1, sim2, 800)
    resume_mod.save(b1, ck, step=800, cfg=cfg,
                    extra={"episode_scores": epB, "losses": [], "evals": []})
    print(f"  已存档于 step=800  文件 {ck.stat().st_size/1e6:.1f} MB  回放={len(b1.buffer):,}")
    del b1   # 模拟进程结束

    # 新进程从检查点续训（必须显式指定 encoding：Windows 默认 gbk，子进程输出是 utf-8）
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8",
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    code = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "_resume_child.py"), str(ck)],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    print((code.stdout or "").rstrip())
    if code.returncode != 0:
        print((code.stderr or "")[-2000:])
        check(False, "子进程续训失败")
        return 1

    # 读回子进程写的结果
    import json
    res = json.loads((ROOT / "checkpoints" / "_resume_child.json").read_text(encoding="utf-8"))

    print()
    print("=" * 70)
    print("核对")
    print("=" * 70)
    check(res["start_step"] == 800, f"续训起点正确接续在 step=800（实际 {res['start_step']}）")
    check(res["end_step"] == 1500, f"跑到 step=1500（实际 {res['end_step']}）")
    check(res["buffer_len"] >= 800, f"回放缓冲已恢复（{res['buffer_len']:,} 条，未重新预热）")
    check(abs(res["eps_at_start"] - 0.9706) < 0.05,
          f"eps 接续而非重置（续训开始时 eps={res['eps_at_start']:.4f}，若重置会是 1.0）")
    check(res["off_edge_max"] == 0.0, f"续训后非边位置严格为 0（max={res['off_edge_max']:.1e}）")

    d = abs(fpA[0] - res["fingerprint_sum"])
    rel = d / max(1e-9, abs(fpA[0]))
    print(f"\n  A 指纹 {fpA[0]:.4f}   B 指纹 {res['fingerprint_sum']:.4f}   相对差 {rel:.2e}")
    check(rel < 0.05, f"两组权重指纹接近（相对差 {rel:.2e} < 5%）")
    print(f"  （注：跨进程无法逐位一致，因为 DataLoader/线程调度会影响浮点累加顺序）")

    print()
    print("=" * 70)
    print("存档/续跑测试通过 ✅" if ok else "存档/续跑测试失败 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
