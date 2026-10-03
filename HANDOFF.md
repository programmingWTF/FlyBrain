# HANDOFF —— 接手 Agent 必读

> 写给下一个接手的 Agent。目标：让你在 5 分钟内搞清现状、不踩重复的坑、直接干最有价值的活。
> 最后更新：2026-10-03

---

##  0. 当前状态：新方向（视觉逼近逃避反射）**已经做成**

**必读 `ESCAPE.md`** —— 那是完整的成果报告（含复现命令、限制清单、参考文献）。

一句话：冻结的 FlyWire 连接组确实把 LC4 的逼近活动中继到逃逸指令神经元
**DNp01（= giant fiber, GF）**，阈值 r50≈0.58 满量程，并且**零自由参数地命中了
实验测得的 LC4→DN 靶集合**（Dombrovski, Nature 2023；Fisher 单尾 p=3.7×10⁻¹⁷）。
三个对照（切 102 条直接边 / 目标全局打乱 / 改驱动 LC10a）全部干净归零。
另外还得到：前馈抑制抬高逃逸阈值（r50 0.583→0.500）、左右侧完全分离、
LC4+LPLC2 双通道把连接组内延迟从 260 ms 压到 40–74 ms。

新代码：`src/fpv/looming.py` + `scripts/loom_{probe,transfer,reflex,mechanisms,validate_lit,figure}.py`。
`spiking_brain.py` 只多了一个可选参数 `step(..., clamp=(idx, p))`。

### ⚠️ 两个基座问题（下面第 1~4 节的技术交接里没提，务必先看 ESCAPE.md §1）
1. **`extract_key_types.py` 的前缀匹配污染**：`"LC4"` 前缀 207 个里混了 LC40/41/43/44/45/46；
   精确 `cell_type=="LC4"` 只有 **104 个**。旧脚本和 `key_types.json` 里的群定义都受影响。
2. **`data/spiking_circuit.*` 是截断图**：建的时候用了 `--min-count 3 --max-nodes`，
   只留 59,548/144,837 神经元、245 万/1502 万边（16%）。
   未截断的全脑资产已经建好：**`data/spiking_full.*`**（`--min-count 1`，17 秒建完），
   逃避反射的结论在两个资产上一致。以后做定量请优先用 `spiking_full`。

### 下一步最有价值的三件事（按性价比）
1. **把 dt 从 20 ms 降到 ~1 ms** 再重测延迟。现在模型的 20 ms/tick 让所有时间数字
   只能定性，而真实 GF 通路是 1.4–25 ms —— 这是当前最大的科学短板。
2. **3D 可视化**（原"出路 C"）：把 LC4→DNp01 这条已验证的通路在 `viewer/` 里点亮，
   播放脉冲传播。数据、骨架、渲染管线都是现成的沉睡资产。
3. 用 `spiking_full` 重跑 `diag5_width.py`，看"信息天花板 0.78"在未截断图上变成多少。

---

## 0bis. 历史：FlappyBird 已判定死路（2026-10-02）

**让冻结的脉冲连接组玩 FlappyBird 走不通，而且搞清楚了为什么
—— 不是工程问题，是信息论问题。**

- 全部方案实测 **0 分**（一个管子都没过），教师 MLP 自己能过 31 个管子
- 读出 AUC 随宽度上升但**到顶 0.78**，而 18 维原始状态是 **0.971**
  → 冻结脑对**那种任务**是有损信道
- 已试遍：换教师 / 带载重新标定 / 类别加权 / 加宽读出 4→8192 /
  换感觉编码 / DAgger / ES —— **全部失败**

**👉 完整论证见 `FINDINGS.md`（注意其第八节的事后修正）。**

下面的第一~四节保留的是**技术交接**（环境、资产、已排除的坑），仍然有效；
但第五节之后的"验收标准（让脉冲脑玩 FlappyBird）"已作废。


---

## 一、这个项目在干什么（30 秒版）

**用真实果蝇脑连接组（FlyWire FAFB v783，14.5 万神经元）造一个"冻结的脉冲神经网络"，让它玩 FlappyBird。**

- 工作目录：`D:/Code/FlyBrain/flyflappy/`
- 总纲领：`PLAN.md`（**先读它**）
- 项目长期记忆：`D:/Code/FlyBrain/.workbuddy/memory/MEMORY.md`（**必读，含所有环境坑**）
- 领域调研（极有价值）：`D:/Code/FlyBrain/_recon/REPORT.md`
  —— 这是对 pinme.dev 上一个**已成功的同类 demo** 的逆向报告 + 全社区同类项目的梳理

### 为什么要换这个新方向（重要背景）
项目之前做过一版 **DQN + 连接组稀疏图先验**，18 个 run 跑完 100 万步，结论是**负结果**：
真实 FlyWire 接线反而比稠密 MLP 差。根因分析见 `PLAN.md` 第一节。
**核心洞察**：把连接组当成"可学的稀疏权重矩阵" = 退化成结构奇怪的 MLP，必败。
**pinme.dev 的成功做法**：脑是**冻结的、脉冲的、带兴奋/抑制符号的**，只有一个小小的读出层在学。

---

## 二、当前进度（一句话）

**新范式的地基全部建好了，脉冲脑已经能驱动 FlappyBird 跑起来，但还学不会玩（0 分）。**

| 模块 | 文件 | 状态 |
|---|---|---|
| 关键神经元提取 | `scripts/extract_key_types.py` | ✅ 完成 |
| 神经递质符号 | `data/nt_index.npz` | ✅ 完成 |
| 脉冲连接组资产 | `scripts/build_spiking_asset.py` | ✅ 完成（59,548 神经元 / 245 万突触） |
| LIF 仿真内核 | `src/fpv/spiking_brain.py` | ✅ 完成（3.2 ms/tick） |
| 感觉运动接口 | `src/fpv/spiking_controller.py` | ✅ 完成（504 参数小读出） |
| 端到端玩 | `scripts/play_spiking.py` | ✅ 能跑 / ❌ 0 分 |
| 动力学报工作点 | `scripts/tune_dynamics.py` | ✅ gain=3.0, tonic=0.0 |

---

## 三、当前卡点（你的起跑线）

**脉冲脑输出恒为"不拍翅" → 鸟 28 帧掉地上 → 0 分。**

### 已经被排除的原因（别再重复查）
1. ~~感觉→运动通路不通~~ → 实测 LC4→DNp01 **只有 1 跳**，651 个突触前伙伴
2. ~~读出特征恒为 0~~ → 已修复：**脑必须每帧推进**（早期版本每 2 帧才推一次，膜电位没累积）
3. ~~命名冲突~~ → `self.act = nn.ReLU()` 和 `def act()` **撞名**导致无限递归（伪装成 torch 报错）
4. ~~装饰器递归~~ → `@torch.no_grad()` 在深调用栈下 RecursionError，改用 `with torch.no_grad():`
5. ~~权重全塌缩~~ → log 量化公式不能照搬 pinme.dev（我们的 norm 值远小于 1），要先在 log 域归一化
6. ~~教师太弱~~ → 内置的"鸟在缺口下方就 flap"规则**自己只能活 121 帧 / 0 分**，蒸馏它无意义

### 待判定的关键问题
**4 维下行特征（ESCAPE/TARGET/LOOM/OTHER 群的发放率）信息量够不够学会玩？**

判据脚本：`scripts/diagnose_readout_info.py`
- 用**会玩的 MLP**（`checkpoints/mlp_s0.pt`）当教师，生成专家轨迹
- 测"给定 4 维脉冲特征，能多准预测专家动作"
- 结果写在 `output/diag_readout.txt`

**✅ 已确认（2026-10-02 20:20）：MLP 教师是真会玩的**
```
n_envs=8: 10 局均分 368.2  最高 1657   (明细 [0,102,108,12,117,402,525,588,171,1657])
n_envs=1: 10 局均分 259.5
```
→ 教师能提供高质量动作标签，**蒸馏路线可行**。
⚠️ 教训：诊断脚本 `--max-steps` 别设太小（设 600 时教师只"得 7 分"，
是因为被步数上限截断，不是教师不行）。用 `--max-steps 3000` 以上。

**❌ v1 的结论是错的 —— v2 已推翻（2026-10-02 23:40）**

v1 报：
```
4 维脉冲特征    线性 0.929  1-NN 0.871  [多数类基线 0.926]
结论：脉冲特征 ✅ 信息充足（线性 0.929）
```
**这个结论是假阳性**：v1 的判据是"绝对准确率 > 0.6"，但数据里 flap 只占 7%，
**一个永远输出 noop 的模型就能拿 0.926**。0.929 只比基线高 0.3 个百分点，
实际上等于**零信息**。

v2（`scripts/diag2.py`）改用不受类别不平衡影响的指标，真相是：
```
多数类基线准确率 = 0.9234（永远猜 noop）
特征                acc    平衡acc  flap召回  noop召回   AUC
4维脉冲特征         0.927  0.500    0.000     1.000      0.617
18维原始状态        0.946  0.710    0.434     0.986      0.971
```
**flap 召回 = 0.000** —— 4 维特征完全无法判断"什么时候该拍翅"。
这精确解释了为什么实跑是 0 分（读出永远选 noop）。

**教训（写进方法论）**：类别极度不平衡时，**绝对准确率是无效指标**。
必须用 平衡准确率 / 每类召回 / AUC。v1 的 `acc > 0.6` 判据已废弃。

**🔍 根因定位（不是读出错，是动力学饱和）**
查 4 维特征分布发现：
```
grp0(ESCAPE): mean=0.654  std=0.092  max=0.718  -> 长期顶在上限
flap 时 0.602 vs noop 时 0.658（差异 -0.056，方向还反了）
```
ESCAPE 群只有 2 个神经元（DNp01 左右各 1 + DNp04），在持续感觉注入下
**发放率被钉死在 65%**，饱和了，自然编码不了状态差异。

而 `tune_dynamics.py` 标定的 gain=3/tonic=0 是**静息态**（无感觉驱动）下的
（静息 0% + 区分度 0.611）——接上驱动后工作点完全变了。
**工作点必须在带载条件下标定。**

**✅ 已修复（2026-10-02 23:45）：带载重新标定**
`scripts/diag3_operating_point.py` 扫描 (gain, tonic, inj_scale)×18 组，同时测
"发放率饱和度"和"预测教师动作的 AUC"：
```
 gain  tonic   inj |  DN率  | 4维AUC  4维bal
 3.00  0.00   1.00 | 0.173 | 0.600   0.650   <- 旧默认
 3.00  0.00   0.40 | 0.050 | 0.704   0.729   <- ★ 新工作点
 1.50  0.00   1.00 | 0.042 | 0.696   0.661
 0.30  0.00   0.15 | 0.000 | 0.529   0.500   <- 网络死寂
```
新默认值已写入 `spiking_controller.py`：`BEST_GAIN=3.0, BEST_TONIC=0.0, BEST_INJ=0.4`
（对照：18 维原始状态本身也只有平衡 acc 0.710，说明 0.729 已接近
教师策略的可预测上限 —— 教师本身不是状态的确定性函数。）

**🔧 已实施的两处修复（2026-10-02 20:35）**
1. **读出加"成对比较头"**（`spiking_controller.py`）：
   原来 `q = W2·relu(W1·f)` 是**单调**映射，无法表达"该 flap 当且仅当
   ESCAPE 群领先 OTHER 群"这类相对比较。现并联一路 `cmp_head([f, 所有成对差])`，
   参数量仅 504 → 526，但能表达相对关系。
2. **教师换成真会玩的 MLP**（`play_spiking.py` 的 `load_teacher()`）：
   删除手写规则教师（自己 0 分），改为载入 `checkpoints/mlp_s0.pt` 出动作当标签。

**范围提醒**：上面 0.929 的报告是在**教师按随机初始读出脑的轨迹**上测的，
存在分布偏移风险。若蒸馏后实跑仍 0 分，见"路线 A+"：改为**在线 DAgger**
（用当前读出脑自己跑，教师在线纠正），这是处理分布偏移的标准做法。

---

## 四、两条路线

### 路线 A：信息够 → 用 MLP 教师做蒸馏
```bash
cd /d/Code/FlyBrain/flyflappy
D:\Code\FlyBrain\env\python.exe scripts/play_spiking.py --train --device cpu
# 但注意：play_spiking.py 现在的教师是弱规则，要把它换成 MLP 教师
```
**要改的地方**：`scripts/play_spiking.py` 的"监督预训练"段落（搜索 `教师：鸟在缺口下方`），
把标签来源从规则换成 `checkpoints/mlp_s0.pt` 的动作：
```python
# 现在：
want = 1 if (bird_y > pipe_gap_c + 8 and vy > -0.6) else 0
# 改成（伪代码）：
from fpv.vendor_flappyrl.networks import RainbowNet
teacher = RainbowNet(AgentConfig(...))  # 载入 mlp_s0.pt 的 online 权重
want = int(teacher(s.view(1,-1)).argmax(1))
```
（`scripts/diagnose_readout_info.py` 里已有 `load_teacher()` 可直接抄）

### 路线 B：信息不够 → 扩读出
按性价比排序：
1. **加更多下行神经元群**：当前只用 16 个具名 DN 类型（32 个神经元）。
   可以把**全部 1301 个下行神经元**按解剖/类型分组，扩到 8~16 维读出
2. **拉长积分窗口**：EMA α 从 0.14 调小（如 0.05），或决策时统计最近 10~20 tick 的发放率
3. **降低 gain**：让网络更"线性"一点，信息传递更保真（但要重跑 tune_dynamics 确认区分度）
4. **换读出位置**：不只读下行神经元，也读中枢（central_brain_intrinsic）的特定功能群

---

## 五、完成后的验收（Definition of Done）

> ⚠️ 第 1 条已判定不可达（见 FINDINGS.md）。请按新方向重新定义验收标准。

1. ~~**能跑**：脉冲脑连续通过管道（>10 分）~~ **已判定不可达，放弃**
2. **能证**：三方对照（真图 / 度保持重连 / ER 随机图），冻结 + 同读出 —— 结论可复现
   - 重连工具已有：`src/fpv/rewire.py` 的 `rewire_degree_preserving()`
3. **能看**：导出稀疏脉冲快照 + three.js 网页（复用根目录 `viewer/` 的经验）
4. **能讲**：写 `flyflappy/REPORT.md`
5. **新增（推荐方向）**：用 LC4→DNp01 通路复现视觉逃避反射，
   对照文献的逼近响应曲线（详见 `NEXT_AGENT_PROMPT.md` 出路 A）

---

## 六、环境（血泪教训，照抄即可）

```bash
# 没接 GPU 时（当前状态）——用这个，torch 2.14+cpu，no_grad 没 bug
cd /d/Code/FlyBrain/flyflappy
D:\Code\FlyBrain\env\python.exe scripts/xxx.py --device cpu

# 接上 GPU 后 —— 这个 torch 2.11+cu128 有 CUDA，快 8×
D:\Code\DQN\env\python.exe scripts/xxx.py --device cuda
```

**关键事实：**
- 两个 env 的 `python.exe` 二进制相同，但 **site-packages 不同**（torch 版本不同）
- 两个 env 都是 **Python 3.11.16**（不是 3.13，别被 RecursionError 误导）
- `D:\Code\DQN\env` **没有** pandas / scipy / matplotlib
- `D:\Code\FlyBrain\env` **有** matplotlib（画图用），但 torch 是 CPU 版

---

## 七、数据资产清单

| 文件 | 内容 | 大小 |
|---|---|---|
| `data/fafb_783_meta.feather` | 14.5 万神经元元数据（含 cell_type / 神经递质） | 13.5 MB |
| `data/fafb_783_simple_edgelist.feather` | 1500 万条突触（pre/post/count/norm） | 289 MB |
| `data/key_types.json` | 关键神经元 id 列表（LC4/LPLC2/T4/T5/LC10/DNp01...） | 366 KB |
| `data/nt_index.npz` | 每神经元的神经递质索引 + 符号 | 343 KB |
| `data/spiking_circuit.npz` | **脉冲连接组 CSR**（主资产） | 7.7 MB |
| `data/spiking_circuit.json` | 元数据 + 关键神经元新编号 + 动力学参数 | 215 KB |
| `checkpoints/mlp_s0.pt` | **会玩的 MLP 教师**（1132 分），蒸馏用 | 19.5 MB |

---

## 八、一句话提醒

**别再走"把连接组当可学权重矩阵"的老路** —— 那已被证明是负结果。
这个项目的价值在于：**冻结的真脑 + 脉冲 + 抑制符号 + 极小读出**，
以及最终诚实的**三方对照**（真拓扑到底有没有用）。
