# 用真实果蝇连接组复现「视觉逼近逃避反射」——结果报告

> 完成于 2026-10-03。这是放弃 FlappyBird（见 `FINDINGS.md`）之后的新方向。
> **结论：成了。** 冻结的 FlyWire 连接组确实把 LC4 的逼近信号中继到逃逸指令神经元
> DNp01（即 giant fiber, GF），并且**独立复现了实验测得的 LC4→DN 靶集合**
> （Fisher 单尾 p = 3.7×10⁻¹⁷）。三个对照全部干净地归零。

---

## 0. 一句话结论

把 LC4 群的发放率钳制成已知值 r_in（自变量精确可控），其余 5.9 万 / 14.5 万神经元
照常按 LIF 演化，直接数 DNp01 的脉冲：

| 量 | 测得值 |
|---|---|
| DNp01 激发阈值 r50 | **0.58 满量程**（= 每只 LC4 约 29 Hz；两次独立脚本 0.577 / 0.583） |
| DNp01 最大响应 | 0.081/tick ≈ **4.1 Hz**（r_in=1.0） |
| 传递函数单调性 | Spearman = 0.917（截断图）/ 0.953（全脑图），AUC = 1.000 |
| 基线 | **0 脉冲**（tonic=0，无刺激时 DNp01 完全静默） |
| 特异性 | 472（截断）/ 1301（全脑）个下行神经元里**只有 11 个响应**，6 个型 |

**没有任何可学参数。** 这一步问的不是"能不能学会玩"，而是"连接组本身能不能做这个计算"，
所以 FlappyBird 那套"读出学不动"的问题在这里不存在。

---

## 1. 两个必须先做的修正（否则整条路的科学性是假的）

### 1.1 类型学前缀污染

`scripts/extract_key_types.py` 用 `cell_type.startswith(prefix)` 选神经元，于是
`"LC4"` 前缀匹配 **207** 个 —— 里面混进了 LC40(37) / LC45(24) / LC46(14) / LC41(12) /
LC43(12) / LC44(4)，那些是**别的细胞类型**。

精确 `cell_type == "LC4"` 是 **104 个**（左 54 / 右 50，全部胆碱能）。
这个数和文献里 LC4 约 60 个/半脑（Dombrovski 2023）是同一量级，说明精确匹配才是对的。

→ 新代码 `src/fpv/looming.py:resolve()` 一律用**精确**类型名。

### 1.2 所谓"全脑资产"其实是被截断过的

`data/spiking_circuit.*` 的 meta 里 `min_count=3, radius=3`，即：
建它时用了 `--min-count 3 --max-nodes <~60000>`，于是

| | 原始数据 | 旧资产 | 保留率 |
|---|---|---|---|
| 神经元 | 144,837 | 59,548 | 41% |
| 突触边 | 15,023,799 | 2,450,637 | **16%** |

`min-count>=3` 会**整条删掉**只有 1~2 个突触的边 —— 而连接组里大多数连接正是这种弱连接。
所以 `FINDINGS.md` 里"冻结脑是有损信道"的测量，是在这个截断图上做的。

→ 本报告在**两种资产上都做了同一套实验**（见 §5）。结论：这条通路的定性结论两边一致，
但 DNp01 的上游边数从 963 被砍到 720，阈值类的定量结论应当以全脑资产为准。

---

## 2. 通路审计（实验前提，`scripts/loom_probe.py`）

- **LC4 → DNp01 是真实单跳**：原始边表里 104 条边 / 1000 个突触 / norm 合计 0.1471；
  旧资产里保留 102 条（左细胞 Σw=0.0748，右细胞 0.0732）。BFS 最短跳数 = 1。
- DNp01 的上游（旧资产内 357 / 363 个伙伴）按权重排序：
  **LPLC2 (0.092 / 0.107) > LC4 (0.075 / 0.073) > DNp70 (0.057 / 0.065)** > 其余。
  即"视叶来的两个逼近通道 LPLC2/LC4 是 DNp01 最强的两路输入"——
  与 Gaitanidis 2025 报告的"LC4+LPLC2 占 GF 直接视叶输入的 98.5%"一致。
- **可驱动性判据**：LIF 稳态下每 tick 需要的净输入 = (1−leak)·thr/gain = **0.0604**。
  LC4 满发放能提供 0.0748 → 裕度只有 **1.24×**。这精确解释了为什么 DNp01 的阈值
  高达 r50≈0.58，也解释了为什么它"不到逼近后期不放"。

---

## 3. 结果

### 3.1 中继与阈值（图 A）

`scripts/loom_transfer.py`：钳制 LC4 群发放率，扫 r_in ∈ [0,1]×11 点 ×4 试次。

```
r_in   0.0   0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9   1.0
DNp01  0     0     0     0     0     0     ~0    0.007 0.021 0.034 0.081   (脉冲/tick)
P_fire 0     0     0     0     0     0     0.75  0.75  1.00  1.00  1.00
```

**P_fire 是主判据**（DNp01 全脑只有 2 个细胞，率估计本质是二值的；
把它当成"给这个刺激逃还是不逃"的检出问题，正好对应行为学的逃逸概率）。
logistic 拟合 r50 = 0.577。

### 3.2 三个对照（图 A，全部归零）

| 对照 | 做法 | 结果 |
|---|---|---|
| `cut` | 把 LC4→DNp01 的 102 条直接边权重置 0 | DNp01 **Δ=0**，排名从 6–9 掉到 197–372/472；**而 DNp04 完全不变**（45.8 Hz）→ 证明切断是特异性的，没有破坏整体活动 |
| `shuffled` | 全部边的目标全局洗牌（精确保持每个神经元的出度 + 边数 + 权重分布，只毁掉"谁连到谁"） | **472 个下行神经元全部 0 发放** |
| `ctrl_lc10a` | 同样的驱动打给 LC10a（小目标追踪，与逼近无关） | DNp01/DNp04 恒 0；响应的是 DNa08 / aSP22 —— 完全不同的下游 |

`cut` 这一条尤其重要：**DNp01 的响应 100% 由那 102 条单突触边携带**，
多突触旁路在这个工作点上贡献为零；而 DNp04 的响应在切断后原样保留 → 它是多突触驱动的。
同一个实验里同时给出"单跳"与"多跳"两种机制的分离。

### 3.3 特异性 + 与实验文献的统计学比对（`scripts/loom_validate_lit.py`）

仿真的响应型：**DNp01(GF)、DNp02、DNp03、DNp04、DNp05、DNp11**（11 个细胞 / 472 或 1301）。

文献基准 —— Dombrovski et al., *Nature* 2023（doi:10.1038/s41586-022-05562-8）用 EM 证明
LC4 直接触突 9 个 DN，文中点名 **GF、DNp02、DNp04、DNp06、DNp11**，单个 LC4 对每个靶
1~75 个突触；光遗传激活 GF 引起 >90% 短程逃逸，DNp04/DNp11 引起 15~40% 长程逃逸。

Fisher 精确检验（单尾，"富集"方向），全脑资产 1301 个 DN：

| 口径 | 靶细胞数 | 命中 | p | OR |
|---|---|---|---|---|
| 保守（DNp02/04/06/11，不含身份待定的 GF） | 8 | 6 | **1.9×10⁻¹²** | 773 |
| 含 DNp01（若 DNp01=GF） | 10 | 8 | **3.7×10⁻¹⁷** | 1717 |

也就是说：**把连接组当成一个黑箱、只喂 LC4 的活动，它"挑出来"的运动指令神经元
就是实验上已知的那批 LC4 靶**。这不是拟合出来的，是零自由参数的预测命中。

### 3.4 前馈抑制在抬高逃逸阈值（图 B）

`scripts/loom_mechanisms.py`，判据只看 **DNp01 自己**（上一版把 DNp01+DNp04 混在一起算
"有没有发放"，r50 被更易发的 DNp04 污染成 0.10 —— 那是 DNp04 的阈值，不是 DNp01 的）。

| | r50 | DNp01+DNp04 发放率之和 @r_in=1.0 |
|---|---|---|
| 真实（含抑制） | 0.583 | 1.574 |
| 关掉全部负权 | **0.500** | 1.751 |

Δr50 = −0.083（−14.3%）→ **前馈抑制在抬高逃逸阈值**（门控），同时限制增益（+11% 脉冲）。

### 3.5 侧向完全分离（图 C）

只驱动**左侧** 54 个 LC4：DNp01 左细胞 0.0975/tick（4.9 Hz），**右细胞在全部试次里恒为 0**。
只驱动右侧 50 个 LC4：镜像成立。DNp04 同样。
配对 Wilcoxon：DNp01 p=1.3×10⁻⁴，DNp04 p=5.1×10⁻⁸。

→ 连接组把两个视野半球的逼近信息**毫不串扰地**送到两侧的下行胞体。
（注意：DNp01/GF 的轴突在腹神经索里越中线，所以"同侧胞体"不等于"同侧动作"；
而且 Dombrovski 2023 明确指出 GF 是方位不变的，方向由 LC4→DNp02/DNp11 的
反平行突触梯度决定 —— 我们的数据与此一致，不与之冲突。）

### 3.6 逼近刺激下的触发点（图 D）

`scripts/loom_reflex.py`：把扩张暗盘的角动力学（θ、dθ/dt、剩余碰撞时间 τ）
经 LC4 调谐曲线变成 r_in(t)，刺激**跑到撞击为止**。

单通道（只 LC4，角速度码，s50=30°/s）：
```
v(m/s)   0.2    0.4    0.7    1.0    1.5    2.0
θ_esc°  54.2   39.4   37.2   26.6   38.0   45.6
τ_esc s 0.490  0.355  0.219  0.225  0.102  0.065
brain_ms 390   285    270    175    225    215
```

**双通道（LC4 走角速度 + LPLC2 走角大小，s50=30°）** —— 这正是文献里 GF 的输入结构
（Ache 2019 / von Reyn 2017 / Gaitanidis 2025：GF ≈ 角速度的线性项 + 角大小的项）：
```
条件                        brain_ms 均值   θ_esc 范围      θ_esc 的速度标度律
LC4 单通道 s50=30            ~260 ms        27~54°          v^(-0.11)
LC4+LPLC2 双通道 s50=30      ~74 ms         21~27°          v^(-0.10)
LC4+LPLC2 双通道 s50=60      ~47 ms         27~33°          v^(-0.02)
LC4+LPLC2 双通道 s50=120     ~40 ms         35~45°          v^(+0.10)
只驱动 LPLC2                 ~66 ms         36~116°         v^(+0.55)
```

两个可检验的结论：
1. **LPLC2 那一路不是冗余的，它买的是速度**：连接组内延迟从 260 ms 降到 40–74 ms。
2. 双通道下 **θ_esc 近似与逼近速度无关**（标度律斜率 −0.10 ~ +0.10），
   而 τ_esc ∝ v^(−0.9 ~ −1.1)。也就是说这个模型预测"在固定的视觉角上逃"，
   而不是"在固定的剩余时间上逃"。文献实测：翼抬起 49±4°、起飞 54±5°
   （Fotowat 2009），计算阈值 67.6±2.5°（de Vries & Clandinin 2012）——
   仿真给出的 21~45° 落在同一量级、略偏早。

### 3.7 只驱动 LPLC2 也能触发 DNp01

θ_esc 36~116°，τ_esc 随速度急剧下降。与 §2 的权重审计自洽：
LPLC2→DNp01 的权重（0.092/0.107）本来就比 LC4→DNp01（0.075/0.073）大。

### 3.8 按**真实视野**给刺激，逃逸指令根本不发放（`scripts/loom_retino_coverage.py`）

上面所有实验都是把 104 个 LC4 **一起**钳制成同一发放率。但 LC4 是柱状、
retinotopic 排布的神经元：真实世界里一个威胁只落在视野的一小块，
只该驱动偏好位置与之重叠的那一批。所以 r50≈0.58 这个阈值隐含了一个**不现实的假设**
——整个 LC4 群同时接近最大发放。

把假设拆掉，改用**视野定位驱动**（高斯窗；视野轴不硬指定，
取 LC4 包围盒中心方差最大的那根解剖轴，由数据自己选出来）：

| 驱动方式 | 有效驱动 Σw·p | 对阈值 0.0604 的比值 | DNp01 实测 |
|---|---|---|---|
| 只驱动一半 LC4（52/104，amp=1） | — | — | **0 脉冲** |
| 双通道 窗宽 0.5（计数覆盖 52%） | 0.0193 | **0.32** | 0 脉冲，不触发 |
| 双通道 窗宽 0.7（计数覆盖 67%） | 0.0559 | **0.93** | 0 脉冲，不触发 |
| 双通道 窗宽 1.2 | 0.1177 | **1.95** | 3 脉冲，**触发** |

**解析判据精确预测了 cliff 的位置**：DNp01 每 tick 需要净输入
Σw·p ≥ (1−leak)·thr/gain = 0.0604，实测在比值 0.93 时确实不发放、1.95 时确实发放。

结论（这条比 §3.1–3.7 更能解释 FlappyBird 为什么 0 分）：
> **逃逸指令 DNp01 要求约三分之二的 LC4 群体被同时动员。
> 一个真实尺寸的逼近物体覆盖不了那么多视野，所以指令根本不发放。**
> 这是**几何问题**，不是学习问题——再多训练预算也救不了。

两个附带结果：
- 只加 LC4（角速度）一条通道时，所需覆盖率高达 ~81%；**补上 LPLC2 角大小通道**
  后降到可及范围。与 §3.6"双通道把延迟从 260 ms 压到 40–74 ms"是同一件事的两面：
  第二条通道既买速度也买灵敏度。
- 决定阈值的是**按突触权重加权的覆盖率**，不是细胞个数占比：
  窗宽 0.7 时计数覆盖 67% 但加权覆盖只有 31%——强突触的 LC4 并不都落在视野中心。

已做成可交互演示，见 `demo/README.md` 的「真实视野」模式。

**补记（同日，做演示时发现的重要边界）**：本节说的是**躲避**局部威胁不够用。
但如果把苍蝇的威胁集合改成"取时间到接触最小的碰撞（地面/天花板/管子）"，
那么**维持高度**靠的是地面这个**大面积**逼近物——它恰好能覆盖几乎整个视野，
于是同一颗冻结脑真的能飞起 FlappyBird（演示里实测稳定过 2 根管子）。
这不推翻上面的结论，反而解释了为什么"只把管子当威胁"的那些 Flappy 版本全是 0 分：
它们给逃逸通道的输入，覆盖面积根本不够。

---

## 4. 资产截断的敏感性检验

`scripts/build_spiking_asset.py --out spiking_full --min-count 1`（17 秒）重建出
**144,837 神经元 / 15,023,799 边**的全脑资产，同一套实验重跑：

| | 截断图 (59,548 / 2.45M) | 全脑图 (144,837 / 15.0M) |
|---|---|---|
| DNp01 @r_in=1.0 | 0.0806/tick | 0.0816/tick |
| Spearman | 0.917 | 0.953 |
| 响应的 DN 细胞数 | 11 / 472 | 11 / 1301 |
| 响应型 | DNp01/02/03/04/05/11 | **完全相同** |
| cut / shuffled / LC10a 对照 | 全 0 | 全 0 |

**结论：这条通路的定性结论不依赖资产截断。** 反过来说，`FINDINGS.md` 的
"有损信道"结论至少在这条通路上不是截断造成的伪影 —— 那个结论仍然成立，
只是它说的是 FlappyBird 那种需要长程抽象策略的任务，而不是"连接组不会传递感觉运动信号"。

---

## 5. 复现

```bash
cd /d/Code/FlyBrain/flyflappy
PY=D:/Code/FlyBrain/env/python.exe      # torch 2.14+cpu，有 matplotlib/scipy/pandas

$PY scripts/loom_probe.py                                   # 通路审计（§2）
$PY scripts/loom_transfer.py --controls --reps 4            # 传递函数+对照（§3.1-3.2）
$PY scripts/loom_transfer.py --rank --controls              # DN 特异性排名（§3.3）
$PY scripts/loom_validate_lit.py                            # 文献富集检验（§3.3）
$PY scripts/loom_mechanisms.py                              # 抑制门控 + 侧向（§3.4-3.5）
$PY scripts/loom_reflex.py --controls --codrive --reps 4 --sources dtheta   # 逼近刺激（§3.6）
$PY scripts/build_spiking_asset.py --out spiking_full --min-count 1         # 全脑资产（§4）
$PY scripts/loom_transfer.py --asset spiking_full --controls --rank
$PY scripts/loom_figure.py                                  # 出图 output/loom_escape.png
```

产物：`output/loom_escape.png`、`output/loom_*.csv`、`output/loom_*.txt`。
新代码：`src/fpv/looming.py`（类型解析 / 对照图构造 / 逼近运动学 / 钳制中继），
`src/fpv/spiking_brain.py` 只加了一个可选参数 `step(..., clamp=(idx, p))`
（把指定神经元的发放强制成伯努利(p)，用来精确设定上游发放率）。

---

## 6. 这个模型做不到什么（限制，必须一起读）

1. **时间分辨率差两个数量级。** 模型 dt=20 ms → 发放率上限 50 Hz，单跳最快也要 1 tick。
   而真实 GF 通路的视觉运动延迟是 **22–25 ms**（Fotowat 2009）、
   中枢传导 1.4–4.5 ms（Dombrovski 2023 / Gaitanidis 2025）、GF 跟随频率 ~180 Hz。
   所以本报告的**延迟数字只能定性，不能当预测值**；要复现毫秒级时序必须把 dt 降到 ~1 ms。
   这条也反过来解释了 FlappyBird 为什么需要 10 ms 级决策却拿不到。
2. **LC4 的调谐曲线是外部假设，不是测出来的。** 检索到的文献里
   **LC4 的胞内/在体放电率（Hz）与基线率没有公开数值**（Dombrovski 记录 DNp02/DNp11/DNp04/GF，
   且"除 GF 外都产生了动作电位"）。所以 r50=29 Hz 这个绝对阈值**无法**和实测 LC4 发放率对齐；
   能对齐的是**排序**（GF 比 DNp04 难触发，与 von Reyn 2014"GF 阈值高于并联通路"一致）。
   §3.6 因此把 s50 当显式扫描参数报出依赖关系，而不是假装知道。
3. **刺激是"注入发放率"，不是仿真视网膜到叶的视觉通路。** 检验的是连接组的中继与阈值，
   不是逼近检测本身。
4. **"静止/远离对照"是构造性的**（dθ/dt≤0 → 驱动 0），它检验刺激模型而不是脑；
   检验脑的是 cut / shuffled / 换感觉群那三个。
5. **DNp01 只有 2 个细胞**（左/右各 1），任何"发放率"都是二值噪声的均值，
   所以主判据用 P_fire + Fisher + 试次级配对，而不是细胞级 t 检验。
6. **神经递质预测本身有噪声**：DNp01 左细胞预测 ACh、右细胞预测谷氨酸 ——
   同一对细胞的两侧预测不一致，这是数据的既有限制。
7. **没有腹神经索 / 运动神经元**，所以"逃避"到 DNp01 发放为止，不产生动作。
8. **冻结 = 没有可塑性**，所以文献里的习惯化（Gaitanidis 2025：重复 5–10 Hz 刺激
   在头 50–100 次内出现习惯化，且需要 LC4→GF 突触）这个模型**做不出来**。

---

## 7. 参考文献

**通路与突触（比对基准）**
- Dombrovski M, Peek MY, Park J-Y, … Namiki S, Zipursky SL, Card GM.
  *Synaptic gradients transform object location to action.* **Nature** 2023.
  doi:10.1038/s41586-022-05562-8 — LC4 触突 9 个 DN（点名 GF/DNp02/DNp04/DNp06/DNp11），
  1~75 突触/对，DNp02 前部 vs 后部 44 vs 13 个脉冲，GF>90% / DNp04、DNp11 15~40% 逃逸。
- Gaitanidis A, … Duch C. *The Drosophila escape motor circuit shows differential
  vulnerability to aging linked to functional decay.* **PLoS Biol** 2025.
  doi:10.1371/journal.pbio.3003553 — LC4+LPLC2 = GF 直接视叶输入的 98.5%；GF FF50≈180 Hz。
- Ceballos R, … *iScience* 2026. doi:10.1016/j.isci.2026.115624 — "axo-axonic input to the
  giant fibers (**DNp01**)"（DNp01=GF 的身份依据之一）。
- Mu L, … *J Exp Biol* 2014. PMID 24675562 — GF/GDN 形态，对视觉与机械刺激" reluctant to spike"。

**逼近检测的计算**
- von Reyn CR, … Card GM. *A Neural Circuit for Looming Detection in Drosophila.*
  **Neuron** 2017. doi:10.1016/j.neuron.2017.05.036 — LC4 编码逼近并驱动逃逸。
- Ache JM, … Clark H, Reiff DF. *Two distinct parallel pathways mediate looming-sensitive
  escape responses.* **Curr Biol** 2019. doi:10.1016/j.cub.2019.01.079 — GF = 角速度(LC4) + 角大小(LPLC2)。
- Klapoetke DA, … Card GM, Jayaraman V. *Ultra-selective looming detection from radial
  motion opponency.* **Nature** 2017. doi:10.1038/nature24626 — LPLC2 需要 T4/T5 + LPi 拮抗。
- Zhou W, … Anderson DJ. *Connectome-based characterization of hunting neurons...*
  **eLife** 2022. PMID 35023828 — LPLC2 群体响应随逼近速度单调上升。
- de Vries SEJ, Clandinin GH. *Looming-sensitive neurons in the fly...* **Curr Biol** 2012.
  PMID 22305754 — 计算角阈值 67.6±2.5°。
- Jang C, … *Azimuthal invariance to looming stimuli in the Drosophila giant fiber escape
  circuit.* **J Exp Biol** 2023. doi:10.1242/jeb.244790 — GF 方位不变；r/v = 10–80 ms。

**行为阈值与延迟**
- Fotowat H, … Gabbiani F. *A novel neuronal pathway for visually guided escape.*
  **J Neurophysiol** 2009. doi:10.1152/jn.00073.2009 — 翼抬起 49±4°、起飞 54±5°、
  视觉运动总延迟 22–25 ms，且这些逃逸中 GF 静默。
- von Reyn CR, … Card GM. *Giant fiber-mediated escape initiates a behavioral sequence...*
  **Nat Neurosci** 2014. PMID 24908103 — GF 阈值高于并联通路。

**连接组资源**
- Dorkenwald P, … FlyWire Consortium. *FlyWire: connectome of the adult brain of
  Drosophila.* **Nature** 2024. PMID 39358518 — 139,255 神经元 / 5×10⁷ 突触。
- Scheffer DD, … FlyWire Consortium. *A connectome and analysis of the adult Drosophila
  brain.* **eLife** 2020. PMID 32880371 — FAFB v783（本项目的数据源）。

> ⚠️ 上述文献数字来自本轮的网络检索与原文摘录，DOI 已标注；
> 若要投稿或对外发布，请逐条回原文核对（尤其 DNp01=GF 的身份与 49°/54° 的测量口径）。
