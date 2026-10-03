#!/usr/bin/env python
"""阶段 1.1 / 1.2：从 FlyWire v783 元数据里提取「文献已知的」关键神经元类型。

为什么要做这一步
----------------
之前的子图用「突触强度 top-N」选感觉神经元，**没有类型学依据**——这被证明是
负结果的主要原因之一（见 PLAN.md 1.2-A/E）。

pinme.dev 那个能跑的 demo，接线方式是：
    LC4 / LPLC2 / LC10a（文献已知的碰撞检测 / 运动 / 目标追踪神经元，双侧）
      -> 冻结的全脑脉冲网络 ->
    DNp01 等具名下行神经元（文献已知的 "飞 / 不飞" 指令通道）

本脚本做同样的事，但数据源是我们本地的 FlyWire v783 全脑（144,837 神经元）。

同时提取 **神经递质预测**（acetylcholine=兴奋 / GABA=抑制 / glutamate 视情况），
这是 pinme.dev 有、而我们之前的实现完全缺失的信息（全正权重）。

输出
----
data/key_types.json   各关键类型的神经元 id 列表 + 计数 + 左右侧分布
data/nt_index.npz     全部神经元的神经递质索引（供符号权重使用）
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
META = DATA / "fafb_783_meta.feather"

# ---- 文献已知的关键神经元类型（前缀匹配 cell_type）----
# 参考：pinme.dev 的 GROUP 映射 + FlyWire 注释表
SENSORY_GROUPS = {
    # 碰撞 / 逼近检测（loom）：投射到巨型纤维，生物学上对应"要不要躲"
    "LC4": ["LC4"],
    "LPLC2": ["LPLC2"],
    "LPLC1": ["LPLC1"],
    # 运动方向检测：视叶输出主力（T4/T5 是 ON/OFF 运动通路）
    "T4": ["T4a", "T4b", "T4c", "T4d"],
    "T5": ["T5a", "T5b", "T5c", "T5d"],
    # 目标追踪 / 小目标检测
    "LC10": ["LC10a", "LC10b", "LC10c", "LC10d"],
    # 广域运动
    "LPC": ["LPC1", "LPC2"],
}

# 下行神经元（DN）：文献已知的"大脑 -> VNC"指令通道
MOTOR_GROUPS = {
    # 巨型纤维逃逸（GF）：最经典的一条 —— 视觉 loom -> DNp01 -> 起飞
    "DNp01": ["DNp01"],
    # 其他与飞行/逃逸/转向相关的命名下行神经元
    "DNp04": ["DNp04"],
    "DNp06": ["DNp06"],
    "DNp07": ["DNp07"],
    "DNp09": ["DNp09"],
    "DNp11": ["DNp11"],
    "DNa02": ["DNa02"],
    "DNa07": ["DNa07"],
    "DNae001": ["DNae001"],
    "DNae002": ["DNae002"],
    "DNg111": ["DNg111"],
    "DNge109": ["DNge109"],
    "DNb01": ["DNb01"],
    "DNg13": ["DNg13"],
    "DNge103": ["DNge103"],
    "DNp54": ["DNp54"],
}

# 神经递质 -> 符号（+1 兴奋 / -1 抑制 / 0 未知）
# 依据：昆虫中枢神经系统里 ACh 主要是兴奋性、GABA 主要是抑制性、
# glutamate 在果蝇里既可是兴奋也可是抑制（这里按文献主流归为兴奋，单列出来供消融）
NT_SIGN = {
    "acetylcholine": 1,
    "glutamate": 1,       # 果蝇中多为快速兴奋性（部分为抑制，见消融）
    "gaba": -1,
    "dopamine": 0,        # 调质，不直接进快速传递
    "serotonin": 0,
    "octopamine": 0,
    "histamine": -1,      # 光感受器用组胺，是抑制性的
    "unknown": 0,
}


def match(ct: pd.Series, prefixes: list[str]) -> np.ndarray:
    hit = np.zeros(len(ct), dtype=bool)
    for p in prefixes:
        hit |= ct.str.startswith(p, na=False).to_numpy()
    return hit


def main() -> int:
    m = pd.read_feather(META)
    m["cell_type"] = m["cell_type"].fillna("")
    m["side"] = m["side"].fillna("unknown")
    m["nt"] = m["neurotransmitter_predicted"].fillna("unknown").str.lower()

    print(f"全脑神经元 {len(m):,}\n")
    print("=" * 70)
    print("感觉输入类型（文献已知）")
    print("=" * 70)
    print(f"{'类型':<10}{'总数':>7}{'L':>7}{'R':>7}   示例 cell_type")
    print("-" * 70)
    sensory_ids: dict[str, list[int]] = {}
    for name, prefixes in SENSORY_GROUPS.items():
        hit = match(m["cell_type"], prefixes)
        ids = m.loc[hit, "fafb_783_id"].astype("int64").tolist()
        sensory_ids[name] = ids
        nl = int(hit.sum() and (m.loc[hit, "side"] == "left").sum())
        nr = int(hit.sum() and (m.loc[hit, "side"] == "right").sum())
        ex = ", ".join(m.loc[hit, "cell_type"].value_counts().head(3).index.tolist())
        print(f"{name:<10}{hit.sum():>7}{nl:>7}{nr:>7}   {ex}")

    print("\n" + "=" * 70)
    print("运动输出类型（具名下行神经元）")
    print("=" * 70)
    print(f"{'类型':<10}{'总数':>7}{'L':>7}{'R':>7}   所属 super_class")
    print("-" * 70)
    motor_ids: dict[str, list[int]] = {}
    for name, prefixes in MOTOR_GROUPS.items():
        hit = match(m["cell_type"], prefixes)
        ids = m.loc[hit, "fafb_783_id"].astype("int64").tolist()
        motor_ids[name] = ids
        nl = int((m.loc[hit, "side"] == "left").sum())
        nr = int((m.loc[hit, "side"] == "right").sum())
        ex = ", ".join(m.loc[hit, "super_class"].value_counts().head(2).index.tolist())
        print(f"{name:<10}{hit.sum():>7}{nl:>7}{nr:>7}   {ex}")

    print("\n" + "=" * 70)
    print("神经递质分布（决定突触符号，pinme.dev 有而我们之前缺）")
    print("=" * 70)
    vc = m["nt"].value_counts()
    for nt, c in vc.head(12).items():
        print(f"  {nt:<20}{c:>8,}  ({100*c/len(m):5.1f}%)  sign={NT_SIGN.get(nt, 0):+d}")

    # ---- 保存 ----
    out = {
        "sensory": {k: v for k, v in sensory_ids.items()},
        "motor": {k: v for k, v in motor_ids.items()},
        "counts": {
            "sensory": {k: len(v) for k, v in sensory_ids.items()},
            "motor": {k: len(v) for k, v in motor_ids.items()},
        },
        "nt_sign": NT_SIGN,
    }
    (DATA / "key_types.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 写出 {DATA/'key_types.json'}")

    # 全部神经元的神经递质索引
    # 注意：不要用 set_index().reindex() 重建 m —— 那会把 nt 列覆盖成 NaN
    # （之前踩过：结果是 nt_index.npz 只剩 1 类递质）。直接按 id 排序取 nt。
    ids = np.sort(m["fafb_783_id"].astype("int64").unique())
    m_sorted = m.sort_values("fafb_783_id").reset_index(drop=True)
    assert np.array_equal(m_sorted["fafb_783_id"].to_numpy(np.int64), ids), "id 排序不一致"
    nt_list = sorted(m_sorted["nt"].unique())
    nt_to_i = {n: i for i, n in enumerate(nt_list)}
    nt_idx = m_sorted["nt"].map(nt_to_i).to_numpy(np.int16)
    signs = np.array([NT_SIGN.get(n, 0) for n in nt_list], dtype=np.int8)
    np.savez_compressed(DATA / "nt_index.npz", node_ids=ids, nt_index=nt_idx,
                        nt_names=np.array(nt_list), nt_sign=signs)
    print(f"✓ 写出 {DATA/'nt_index.npz'}（{len(ids):,} 神经元，{len(nt_list)} 类递质）")

    # 总计关键神经元
    tot_s = sum(len(v) for v in sensory_ids.values())
    tot_m = sum(len(v) for v in motor_ids.values())
    print(f"\n关键感觉神经元合计 {tot_s:,}  关键运动神经元合计 {tot_m:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
