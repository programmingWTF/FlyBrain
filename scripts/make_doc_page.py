#!/usr/bin/env python
"""把算法说明文档写到桌面。

把 scripts/make_doc_figures.py 产出的真实数据配图**内联**进一份自包含 HTML，
这样单文件双击就能看，不依赖任何外部资源、也不需要联网。

用法:
    python scripts/make_doc_page.py                 # 写到桌面
    python scripts/make_doc_page.py --out 某目录
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


CSS = r"""
:root{
  --ink:#16181d; --paper:#fbfaf7; --paper2:#f3f1ea; --rule:#d9d5cc; --rule2:#e8e5dd;
  --lc4:#c2410c; --lplc2:#7c3aed; --dn:#be123c; --phys:#1d4ed8;
  --quiet:#6b6862; --quiet2:#8d8a83;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:"IBM Plex Sans","PingFang SC","Microsoft YaHei",system-ui,sans-serif;
  --cond:"IBM Plex Sans Condensed","PingFang SC","Microsoft YaHei",system-ui,sans-serif;
  --col:44rem;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.72;
  font-feature-settings:"tnum" 1;
  background-image:linear-gradient(var(--rule2) 1px,transparent 1px);
  background-size:100% 5.6rem;
}
.wrap{display:grid; grid-template-columns:15.5rem minmax(0,var(--col)); gap:5rem;
  justify-content:center; padding:0 1.5rem 8rem}
@media (max-width:1000px){ .wrap{grid-template-columns:minmax(0,var(--col)); gap:0} .rail{display:none} }

/* ---------------- 左侧刻度轨：签名装置 ---------------- */
.rail{position:sticky; top:0; align-self:start; height:100vh; padding:2.4rem 0 2rem;
  font-family:var(--mono); font-size:.72rem; color:var(--quiet)}
.rail .k{letter-spacing:.14em; text-transform:uppercase; color:var(--quiet2);
  font-size:.62rem; margin-bottom:.7rem}
.rail ol{list-style:none; margin:0; padding:0}
.rail li{margin:0; padding:.34rem 0 .34rem .9rem; border-left:2px solid var(--rule2);
  line-height:1.35}
.rail li.on{border-left-color:var(--dn); color:var(--ink)}
.rail li a{color:inherit; text-decoration:none}
.rail li a:hover{color:var(--dn)}
.rail .tick{margin-top:1.6rem; padding-top:.9rem; border-top:1px solid var(--rule);
  color:var(--quiet2); font-size:.66rem; line-height:1.6}
.rail .tick b{color:var(--ink); font-weight:600}

/* ---------------- 排版 ---------------- */
main{padding-top:2.4rem; min-width:0}
.eyebrow{font-family:var(--mono); font-size:.68rem; letter-spacing:.18em;
  text-transform:uppercase; color:var(--quiet2)}
h1{font-family:var(--cond); font-weight:700; font-size:clamp(1.85rem,4.1vw,2.6rem);
  line-height:1.14; letter-spacing:-.015em; margin:.5rem 0 1rem}
h1 .thin{font-weight:400; color:var(--quiet)}
h2{font-family:var(--cond); font-weight:700; font-size:1.62rem; letter-spacing:-.01em;
  line-height:1.2; margin:0 0 .8rem}
h3{font-family:var(--cond); font-weight:600; font-size:1.12rem; margin:1.9rem 0 .5rem}
section{padding:3.2rem 0 0; scroll-margin-top:1.5rem}
section+section{border-top:1px solid var(--rule); margin-top:3.2rem; padding-top:3rem}
p{margin:.75rem 0}
.lead{font-size:1.06rem; color:#33363d}
a{color:var(--phys)}
strong{font-weight:600}
code{font-family:var(--mono); font-size:.87em; background:var(--paper2);
  padding:.1em .35em; border:1px solid var(--rule2)}
.mono{font-family:var(--mono)}
.q{color:var(--quiet)}

/* 术语标记：中文 + 英文原词 */
.term{border-bottom:1px dotted var(--quiet2); cursor:help}

/* ---------------- 组件 ---------------- */
.note{margin:1.4rem 0; padding:.9rem 1.1rem; background:var(--paper2);
  border-left:3px solid var(--rule)}
.note.warn{border-left-color:var(--dn)}
.note.ok{border-left-color:var(--lplc2)}
.note .h{font-family:var(--mono); font-size:.68rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--quiet2); margin-bottom:.35rem}
.note p:first-of-type{margin-top:0} .note p:last-child{margin-bottom:0}

figure{margin:1.8rem 0; border:1px solid var(--rule); background:#fff}
figure svg{display:block; width:100%; height:auto}
figcaption{font-family:var(--mono); font-size:.7rem; line-height:1.6; color:var(--quiet);
  padding:.7rem .9rem; border-top:1px solid var(--rule2)}
figcaption b{color:var(--ink); font-weight:600}

table{width:100%; border-collapse:collapse; margin:1.3rem 0; font-size:.9rem}
th,td{text-align:left; padding:.5rem .6rem; border-bottom:1px solid var(--rule2);
  vertical-align:top}
th{font-family:var(--mono); font-size:.68rem; letter-spacing:.08em; text-transform:uppercase;
  color:var(--quiet2); font-weight:500; border-bottom:1px solid var(--rule)}
td.n{font-family:var(--mono); white-space:nowrap}
tbody tr:last-child td{border-bottom:0}

.kv{display:grid; grid-template-columns:auto 1fr; gap:.35rem 1.1rem; margin:1.2rem 0;
  font-size:.9rem}
.kv dt{font-family:var(--mono); color:var(--quiet); white-space:nowrap}
.kv dd{margin:0}

.eq{margin:1.3rem 0; padding:1rem 1.1rem; background:#fff; border:1px solid var(--rule);
  font-family:var(--mono); font-size:.9rem; line-height:1.9; overflow-x:auto}
.eq em{font-style:normal; color:var(--quiet); font-size:.8em}

.chips{display:flex; flex-wrap:wrap; gap:.4rem; margin:1rem 0}
.chip{font-family:var(--mono); font-size:.7rem; padding:.22rem .5rem;
  border:1px solid var(--rule); background:#fff; color:var(--quiet)}
.chip.lc4{border-color:var(--lc4); color:var(--lc4)}
.chip.lplc2{border-color:var(--lplc2); color:var(--lplc2)}
.chip.dn{border-color:var(--dn); color:var(--dn)}

.swatch{display:inline-block; width:.7em; height:.7em; vertical-align:baseline;
  margin-right:.35em; border-radius:50%}

/* 量程尺：解释"缺口可行区间"的结构性装置 */
.ruler{margin:1.6rem 0; font-size:.84rem}
.ruler .bar{position:relative; height:2.1rem; border:1px solid var(--rule);
  background:#fff}
.ruler .band{position:absolute; top:0; bottom:0; background:rgba(124,58,237,.13)}
.ruler .mark{position:absolute; top:-.3rem; bottom:-.3rem; width:0;
  border-left:2px solid var(--dn)}
.ruler .marks{position:relative; height:1.15rem; font-family:var(--mono);
  font-size:.66rem; color:var(--quiet)}
.ruler .marks span{position:absolute; transform:translateX(-50%); white-space:nowrap}
.ruler .why{display:grid; grid-template-columns:6.5rem 1fr; gap:.3rem .9rem;
  margin-top:.9rem; font-size:.82rem}
.ruler .why dt{font-family:var(--mono); font-size:.72rem; color:var(--quiet)}
.ruler .why dd{margin:0}

footer{margin-top:4rem; padding-top:1.4rem; border-top:1px solid var(--rule);
  font-family:var(--mono); font-size:.7rem; color:var(--quiet2); line-height:1.8}

@media (prefers-reduced-motion:no-preference){
  .fade{opacity:0; transform:translateY(6px); transition:opacity .5s, transform .5s}
  .fade.in{opacity:1; transform:none}
}
@media print{
  body{background:#fff; background-image:none} .rail{display:none}
  .wrap{display:block; padding:0} section{break-inside:avoid}
}
"""

JS = r"""
(function(){
  // 章节高亮（滚到哪一节，左侧刻度轨就亮哪一节）
  var items = [].slice.call(document.querySelectorAll('.rail li'));
  var secs  = items.map(function(li){
    var id = li.getAttribute('data-for'); return id ? document.getElementById(id) : null;
  });
  function sync(){
    var best = 0;
    secs.forEach(function(s, i){
      if (s && s.getBoundingClientRect().top <= 140) best = i;
    });
    items.forEach(function(li, i){ li.classList.toggle('on', i === best); });
  }
  addEventListener('scroll', sync, {passive:true}); sync();

  // 侧栏的 20ms 计数：把"这是一条 50Hz 的闭环"变成页面上活着的东西
  var el = document.getElementById('tickno');
  if (el){
    var t0 = performance.now(), reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reduce) setInterval(function(){
      var n = Math.floor((performance.now() - t0) / 20);
      el.textContent = n.toLocaleString();
    }, 220);
  }

  // 进场淡入
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches && 'IntersectionObserver' in window){
    var io = new IntersectionObserver(function(es){
      es.forEach(function(e){ if (e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target); } });
    }, {rootMargin:'-8% 0px -8% 0px'});
    document.querySelectorAll('.fade').forEach(function(n){ io.observe(n); });
  } else {
    document.querySelectorAll('.fade').forEach(function(n){ n.classList.add('in'); });
  }
})();
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default=str(ROOT / "output" / "doc_figs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="果蝇脑玩Flappy·算法说明.html")
    a = ap.parse_args()

    figs = pathlib.Path(a.figs)
    need = ["loop.svg", "raster.svg", "gapgen.svg", "spikes.json"]
    miss = [n for n in need if not (figs / n).is_file()]
    if miss:
        print(f"缺配图 {miss}；先跑 scripts/make_doc_figures.py", file=sys.stderr)
        return 2

    loop_svg = (figs / "loop.svg").read_text(encoding="utf-8")
    raster_svg = (figs / "raster.svg").read_text(encoding="utf-8")
    gap_svg = (figs / "gapgen.svg").read_text(encoding="utf-8")
    stats = json.loads((figs / "spikes.json").read_text(encoding="utf-8"))

    # 图上那几个数字直接从真实数据来，别手写
    n_dn = stats["n_dn"]
    n_flap = stats["n_flap"]
    n_ticks = stats["ticks"]

    out = pathlib.Path(a.out) if a.out else pathlib.Path.home() / "Desktop"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / a.name

    html = HTML_TEMPLATE.format(
        css=CSS, js=JS, loop_svg=loop_svg, raster_svg=raster_svg, gap_svg=gap_svg,
        n_dn=n_dn, n_flap=n_flap, n_ticks=n_ticks,
    )
    dst.write_text(html, encoding="utf-8")
    print(f"→ {dst}  ({dst.stat().st_size/1024:.0f} KB)")
    return 0


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>一只果蝇的脑子怎么玩 Flappy —— 算法与实现说明</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" media="print" onload="this.media='all'"
 href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@400;600;700&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{css}</style>
</head>
<body>
<div class="wrap">

  <aside class="rail" aria-label="章节">
    <div class="k">目录</div>
    <ol>
      <li data-for="s1"><a href="#s1">一句话说清</a></li>
      <li data-for="s2"><a href="#s2">材料：一张真实的脑图</a></li>
      <li data-for="s3"><a href="#s3">神经元怎么算</a></li>
      <li data-for="s4"><a href="#s4">感觉通路</a></li>
      <li data-for="s5"><a href="#s5">逼近反射</a></li>
      <li data-for="s6"><a href="#s6">控制回路</a></li>
      <li data-for="s7"><a href="#s7">游戏物理</a></li>
      <li data-for="s8"><a href="#s8">关卡生成</a></li>
      <li data-for="s9"><a href="#s9">怎么证明不是巧合</a></li>
      <li data-for="s10"><a href="#s10">踩过的坑</a></li>
      <li data-for="s11"><a href="#s11">失败的尝试</a></li>
      <li data-for="s12"><a href="#s12">边界与复现</a></li>
      <li data-for="s13"><a href="#s13">术语对照</a></li>
    </ol>
    <div class="tick">
      页面已在运行 <b id="tickno">0</b> 个 20&nbsp;ms tick<br>
      这只鸟的每一个动作，都是<br>144,837 个神经元算出来的。
    </div>
  </aside>

  <main>

    <header>
      <div class="eyebrow">FlyWire FAFB v783 · 冻结连接组 · 零可学参数</div>
      <h1>一只果蝇的脑子<br>怎么玩 Flappy<span class="thin"> —— 算法与实现说明</span></h1>
      <p class="lead">
        这个项目里没有神经网络训练、没有强化学习、没有任何一个"学出来"的参数。
        我们拿一张<strong>真实的果蝇全脑连接组</strong>（144,837 个神经元、1,502 万条突触）当控制器，
        把"往前飞、别撞管子"这件事交给它。它拿到的分数不是靠调参凑出来的，
        而是<strong>被迫</strong>穿过那条真实的神经通路。
      </p>
      <p class="q">
        这份文档讲三件事：这东西<strong>怎么算</strong>、我们<strong>怎么知道它真的在起作用</strong>、
        以及我们在路上<strong>踩过哪些坑</strong>（包括事后被证明是错的判断）。
      </p>
    </header>

    <!-- ============================================================ 1 -->
    <section id="s1" class="fade">
      <div class="eyebrow">01 · 总览</div>
      <h2>一句话说清它是什么</h2>

      <p>
        想象你从一张果蝇大脑的电子显微镜切片图里，把所有神经细胞和它们之间的连接
        都提取了出来。现在你手里有一个 144,837 个节点的有向图 —— 这就是<span class="term">连接组</span>（connectome）。
        我们不训练它，我们<strong>只给它接上眼睛和翅膀</strong>：
      </p>

      <div class="eq">
        视觉输入 <em>（管子在哪、有多近）</em>
        &nbsp;→&nbsp; <b>连接组内部真实接线</b>
        &nbsp;→&nbsp; 输出 <em>（拍一下翅膀）</em>
      </div>

      <p>
        中间那一层，每根线都是电镜里量出来的、固定的。所以"它会不会飞"这个问题，
        等价于"这张图里到底有没有一套能控制高度的回路"。
      </p>

      <div class="chips">
        <span class="chip">神经元 144,837</span>
        <span class="chip">突触 15,023,799</span>
        <span class="chip">可学参数 <b>0</b></span>
        <span class="chip">时钟 20 ms</span>
        <span class="chip">服务端 50 Hz 锁步</span>
      </div>

      <figure>
        {loop_svg}
        <figcaption>
          <b>20 毫秒的闭环。</b>每 20&nbsp;ms 走一圈：从游戏几何算出感觉输入 →
          灌进感觉神经元 → 脑自己按连接组传播 → 读两根巨纤维的膜电位 →
          决定拍不拍 → 推进物理。下一圈再看新的几何。
          <b>注意这是一条闭环</b>——它不是在"预测"未来，而是每 20&nbsp;ms 重新看一眼。
        </figcaption>
      </figure>
    </section>

    <!-- ============================================================ 2 -->
    <section id="s2" class="fade">
      <div class="eyebrow">02 · 材料</div>
      <h2>材料：一张真实的脑图，以及一个刻意的约束</h2>

      <p>
        数据来自 <strong>FlyWire</strong> 的 FAFB v783 版本 —— 一只成年雌性黑腹果蝇
        （<i>Drosophila melanogaster</i>）的全脑。它是目前最完整的成体昆虫脑连接组之一。
      </p>

      <dl class="kv">
        <dt>神经元</dt><dd>144,837</dd>
        <dt>突触</dt><dd>15,023,799（兴奋 11,043,448 / 抑制 3,152,462 / 符号未知 827,889）</dd>
        <dt>平均出度</dt><dd>103.7（最大 9,773 —— 少数"枢纽"细胞投射极广）</dd>
        <dt>游离神经元</dt><dd>6,963（在这份重建里没有任何突触）</dd>
      </dl>

      <h3>为什么说"零可学参数"是重点</h3>
      <p>
        如果允许改突触权重，那么"脑在控制飞行"这句话就变得没法证伪了 ——
        你总能说"再调调就好了"。所以我们定了一条硬规则：
      </p>

      <div class="note ok">
        <div class="h">约束</div>
        <p>
          连接组的<strong>拓扑与符号</strong>（谁是兴奋、谁是抑制）一个都不改。
          我们唯一能动的是<strong>感觉输入端</strong>：给哪些细胞、灌多强的驱动。
          用类比说：我们负责决定"眼睛看到的东西翻译成多大电流"，
          但<strong>不碰脑内任何一条连线</strong>。
        </p>
      </div>

      <p class="q">
        这个约束的代价是分数上限被压住了；收益是结论非常硬 ——
        后面第 9 节那个"切断一条通路分数直接归零"的实验才有意义。
      </p>
    </section>

    <!-- ============================================================ 3 -->
    <section id="s3" class="fade">
      <div class="eyebrow">03 · 动力学</div>
      <h2>一个神经元怎么算</h2>

      <p>
        用的是最经典的<span class="term">漏积分-发放模型</span>（leaky integrate-and-fire, LIF）。
        每个神经元维持一个膜电位 <span class="mono">G</span>，每 20&nbsp;ms 更新一次：
      </p>

      <div class="eq">
        G ← leak · G + gain · C + tonic<br>
        若 G ≥ threshold &nbsp;→&nbsp; 发放，并令 G ← 0<br>
        <em>leak = exp(−dt/τ) = 0.81873&nbsp;&nbsp;·&nbsp;&nbsp;gain = 3.0&nbsp;&nbsp;·&nbsp;&nbsp;threshold = 1.0</em>
      </div>

      <p>
        其中 <span class="mono">C</span> 是这一个 tick 从所有上游发放细胞传进来的电流，
        由连接组的稀疏矩阵决定（<span class="mono">C = Σ 上游脉冲 × 突触符号</span>）。
        另外还有 <strong>1.2&nbsp;Hz、幅度 0.22 的泊松噪声</strong>，模拟神经元的自发活动 ——
        这一条后来变得很重要，见第 10 节。
      </p>

      <h3>那个决定一切的常数</h3>
      <p>
        因为它是漏积分的，一个"孤立的小刺激"根本推不动它：电位会自己漏掉。
        把上式推到稳态，可以解出让某个神经元<strong>持续发放</strong>所需的最小输入：
      </p>

      <div class="eq">
        need = (1 − leak) · threshold / gain = <b>0.06042</b>&nbsp;&nbsp;<em>每 tick</em>
      </div>

      <p>
        这个 0.06042 是整个项目的"汇率"：所有感觉驱动的强弱，
        都要拿它来换算成"够不够触发"。不够就是 0，够了才开始发放。
        <strong>它不是调出来的，是这套动力学算出来的。</strong>
      </p>

      <div class="note">
        <div class="h">为什么这件小事很关键</div>
        <p>
          它意味着：<em>单次刺激没用，必须持续逼近</em>。
          一个物体在你的视野里"闪一下"不会触发逃逸，但一个正在逼近的物体
          （角尺寸持续变大）会。这正是真实巨型纤维系统的行为特征，
          也是这个项目里"反射"两个字的物理基础。
        </p>
      </div>
    </section>

    <!-- ============================================================ 4 -->
    <section id="s4" class="fade">
      <div class="eyebrow">04 · 感觉通路</div>
      <h2>从"管子逼近了"到"该拍翅膀了"</h2>

      <p>
        视觉信号在果蝇脑里走的是两条并行的叶（lobula plate / lobula）通路。
        我们只用了其中两组细胞，它们加起来占巨型纤维直接视觉输入的 98.5%
        （Ache 2019；Gaitanidis 2025）：
      </p>

      <table>
        <thead><tr><th>细胞群</th><th>数量</th><th>它编码什么</th><th>在本项目里的角色</th></tr></thead>
        <tbody>
          <tr>
            <td><span class="swatch" style="background:var(--lc4)"></span><b>LC4</b></td>
            <td class="n">104</td>
            <td>视野里物体的<strong>角速度</strong>（扩张多快）</td>
            <td>告诉脑"有多急"</td>
          </tr>
          <tr>
            <td><span class="swatch" style="background:var(--lplc2)"></span><b>LPLC2</b></td>
            <td class="n">210</td>
            <td>物体的<strong>角尺寸</strong>（看起来多大）</td>
            <td>告诉脑"有多近"</td>
          </tr>
          <tr>
            <td><span class="swatch" style="background:var(--dn)"></span><b>DNp01</b></td>
            <td class="n">2</td>
            <td>——</td>
            <td><strong>指令读出</strong>：两根巨型纤维</td>
          </tr>
        </tbody>
      </table>

      <h3>怎么把游戏画面变成神经驱动</h3>
      <p>
        游戏里没有"图像"，只有几何：鸟的高度 <span class="mono">y</span>、
        前方最近缺口的中心 <span class="mono">gap</span>。
        把这个几何翻译成"一个正在逼近的物体"的角尺寸，用标准的天文/视觉公式：
      </p>

      <div class="eq">
        θ = 2 · atan(0.55 / dist) &nbsp;<em>（度）</em>
        &nbsp;&nbsp;→&nbsp;&nbsp; amp = gain · θⁿ / (θⁿ + s50ⁿ)
        <em>n = 3, s50 = 30°（Naka-Rushton 饱和曲线）</em>
      </div>

      <p>
        最后把 <span class="mono">amp</span> 当作<strong>强制发放概率</strong>灌给那一群细胞
        （代码里叫 <span class="mono">clamp</span>）：
        也就是说，我们不去手工模拟光感受器的细节，
        而是直接指定"这 104 个 LC4 细胞各自以多大概率发放"。
        这是唯一一个我们<strong>定义</strong>而不是<strong>测量</strong>出来的环节，必须坦白标出来。
      </p>
    </section>

    <!-- ============================================================ 5 -->
    <section id="s5" class="fade">
      <div class="eyebrow">05 · 反射</div>
      <h2>"双向逼近反射"：方向由几何决定，不是控制器</h2>

      <p>
        这是整个算法的核心，也是最容易被误解的地方。
        它<strong>不是一个控制器</strong> —— 没有"目标高度"、没有 PID、没有误差反馈运算。
        它做的事只有一件：<em>视野的哪一半被逼近刺激占住，就把那一半的感觉细胞整片点亮</em>。
      </p>

      <div class="eq">
        缺口在鸟的<strong>上方</strong>超过死区 → 驱动<strong>腹侧</strong>半视野 → 腹侧那群细胞发放 → 拍翅 → 爬升<br>
        缺口在鸟的<strong>下方</strong>超过死区 → 驱动<strong>背侧</strong>半视野 → 背侧那群细胞发放 → 不拍 → 下坠
      </div>

      <p>
        为什么方向能从"哪一半视野"里自然读出来？因为在真实的果蝇视觉里，
        视野的上下两半本来就投射到不同的叶区，而腹侧和背侧两半到 DNp01 的
        突触权重是<strong>不对等</strong>的（腹侧是背侧的 2.6 倍）。
        所以"同一个机制"自然给出了两个方向的行为 —— 这是连接组的事实，不是我们设计的。
      </p>

      <h3>三个不能省的细节</h3>

      <dl class="kv">
        <dt>死区 gap_margin = 18 px</dt>
        <dd>
          缺口中心在鸟上下 18&nbsp;px 之内时<strong>什么都不驱动</strong>。
          第一版没加死区，鸟刚越过缺口中心就翻转成"要爬升"，于是在缺口上下反复横跳，
          实测均分只有 0.81。
        </dd>
        <dt>vy_gate（接近速度门控）</dt>
        <dd>
          地面只在下落时才算"逼近"，天花板只在上升时才算。
          这是纯物理的：你正在远离一个面，那个面并没有朝你逼近。
        </dd>
        <dt>整半招募，不是平滑窗</dt>
        <dd>
          用"整个半视野一起亮"而不是给个平滑的权重斜坡。
          实测平滑斜坡窗只能凑到 need 的 53%，<strong>根本不发放</strong>；
          整半招募能到 need 的 1.79 倍，才推得动 DNp01。
        </dd>
      </dl>

      <figure>
        {raster_svg}
        <figcaption>
          <b>真实的发放栅格</b>（从跑着的仿真里取 {n_ticks} 个 tick，不是示意图）。
          上两行是 LC4 与 LPLC2 的代表细胞，第三行是全部 2 根 DNp01，
          最下面那条是鸟的高度轨迹，<b>红色竖线 = 一次拍翅</b>。
          在这 {n_ticks} 个 tick 里 DNp01 一共发放 {n_dn} 次、触发 {n_flap} 次拍翅 ——
          可以看到拍翅不是"每个 tick 都拍"，而是被那个 0.06042 的阈值筛过的。
          <br><span class="q">注：LC4/LPLC2 是均匀降采样后的代表细胞（每组约 26 个），
          否则一张图会有七千个点；DNp01 只有 2 个，是全画的。</span>
        </figcaption>
      </figure>
    </section>

    <!-- ============================================================ 6 -->
    <section id="s6" class="fade">
      <div class="eyebrow">06 · 回路</div>
      <h2>控制回路：每 20 毫秒闭合一次，而且必须严格 1:1</h2>

      <p>
        一个 tick 里发生的事，顺序不能变：
      </p>

      <div class="eq">
        ① 按<strong>当前</strong>世界几何算驱动<br>
        ② 用这个驱动走<strong>一个</strong>脑 tick<br>
        ③ 读 DNp01（最近 5 个 tick 内发放 ≥ 1 次 → 拍翅）<br>
        ④ 用这个决策走<strong>一步</strong>物理<br>
        <em>四步在同一个进程、同一把锁里完成；没有网络、没有定时器节拍插进来</em>
      </div>

      <p>
        这四步的顺序看起来理所当然，但它是这个项目里花了最长时间才对的东西。
        我们试过四种"驱动方式"，每一种都实测过，每一种都失败：
      </p>

      <table>
        <thead><tr><th>#</th><th>驱动方式</th><th>实测结果</th></tr></thead>
        <tbody>
          <tr><td class="n">①</td><td>物理跟<strong>墙钟</strong>走，每帧消化 1 步</td>
              <td>帧率低于 50&nbsp;fps 就积压几十步，鸟执行的是<em>半秒前</em>的决策 → 完全控不住</td></tr>
          <tr><td class="n">②</td><td>物理跟<strong>渲染帧率</strong>走</td>
              <td>与脑严格 1:1 了，但帧率 11.5&nbsp;fps 时只有 12% 速度 → 鸟慢到飞不起来</td></tr>
          <tr><td class="n">③</td><td>物理跟<strong>脑响应</strong>走</td>
              <td>速度 = HTTP 往返速度（中位 60&nbsp;ms、最快 4&nbsp;ms，差十几倍）→ <strong>时快时慢</strong></td></tr>
          <tr><td class="n">④</td><td>固定节拍 + 决策队列预取</td>
              <td>队列周期性见底，仍然抖</td></tr>
        </tbody>
      </table>

      <div class="note warn">
        <div class="h">根因</div>
        <p>
          <strong>游戏时钟必须挂在某个节奏上，而浏览器里没有可信的节奏。</strong>
          页面内直接测量得到：<span class="mono">setTimeout(4)</span> 实际只有
          <strong>29.8 次/秒</strong>（被压到约 33&nbsp;ms），
          <span class="mono">requestAnimationFrame</span> 只有 <strong>28.2 次/秒</strong>，
          而且页签切到后台会被<strong>完全暂停</strong>。
          只要决策要靠一次跨进程往返拿到，时钟就一定是抖的。
        </p>
      </div>

      <p>
        所以最终的解法是<strong>把整个游戏搬到服务端</strong>：
        脑和物理在同一个进程里按固定 50&nbsp;Hz 跑严格 1:1，
        浏览器只做两件事 —— 按 20&nbsp;Hz 取状态、在两次状态之间插值后画出来。
        测完的结果：
      </p>

      <table>
        <thead><tr><th>指标</th><th>前端自己跑（④）</th><th>服务端锁步（现在）</th></tr></thead>
        <tbody>
          <tr><td>仿真速率</td><td class="n">14–32 步/秒（抖）</td><td class="n"><b>50.0 步/秒</b>（偏差 +0.1%）</td></tr>
          <tr><td>渲染帧率</td><td>与游戏速度耦合</td><td class="n"><b>60 fps</b>，与游戏速度无关</td></tr>
          <tr><td>网络抖动影响</td><td>直接改变游戏速度</td><td>只影响画面新鲜度</td></tr>
        </tbody>
      </table>

      <p class="q">
        顺带一个教训：把仿真搬到服务端之后，前端只剩"取状态 + 画"，
        但那一小块也出过致命的 bug —— 见第 10 节。
      </p>
    </section>

    <!-- ============================================================ 7 -->
    <section id="s7" class="fade">
      <div class="eyebrow">07 · 物理</div>
      <h2>游戏物理与那个"看着撞上了却没死"的 bug</h2>

      <p>
        物理本身是刻意做得和最朴素的 FlappyBird 完全一致的（也让离线评测台
        与网页跑的是同一套）：
      </p>

      <dl class="kv">
        <dt>画布 / 地面</dt><dd>520 × 620，地面高 92 px（地面线 y = 528）</dd>
        <dt>鸟</dt><dd>圆心 x = 120，半径 <b>17 px</b></dd>
        <dt>管子</dt><dd>宽 62 px，缺口高 <b>184 px</b>，间距 300 px，左移速度 240 px/s（= 1 m/s）</dd>
        <dt>重力 / 拍翅</dt><dd><span class="mono">GRAV = 1180 px/s²</span>；<span class="mono">FLAP_V = −340 px/s</span></dd>
        <dt>拍翅冷却</dt><dd>0.14 s（一个间隔内最多约 9 次拍翅）</dd>
      </dl>

      <h3>那个 bug：像素与坐标混用</h3>
      <p>
        最早的碰撞判定用的是"圆心到矩形最近点的距离 ≤ 半径"，数学上很标准。
        但玩家报了一个很具体的现象：<strong>"管子上下两端的体积好像有点偏差，
        我看已经撞上了，但是没有判定为失败。"</strong>
      </p>

      <p>
        根因是<strong>画和判用了两套语义</strong>。<span class="mono">fillRect(y, h)</span>
        覆盖的是像素行 <span class="mono">y … y+h−1</span>（闭区间），
        而判定用的是连续坐标。差一两个像素，肉眼就足够觉得"明明撞上了"。
      </p>

      <div class="eq">
        <em>画：</em> 上管覆盖像素 [0, top+R]&nbsp;&nbsp;下管覆盖 [top+GAP−R−1, …]<br>
        <em>判：</em> 鸟圆周覆盖像素 [⌈y−R⌉, ⌊y+R⌋]<br>
        <em>两组像素行有交集 ⇔ 视觉重叠</em>
      </div>

      <p>
        最终判据化简成两行，并配了一个<strong>逐像素对账</strong>的校验脚本：
        把"画到的像素集合"和"判定为撞的 y 集合"逐个比，
        要求"只画不判"和"只判不画"<strong>同时为 0</strong>。
        历史上错过两次，而且方向相反（<span class="mono">&lt;= / &gt;=</span> 和
        <span class="mono">&lt; / &gt;</span> 各差一段 R）。
      </p>

      <div class="note">
        <div class="h">这一条值得单独记住</div>
        <p>
          "看着撞上了"这类 bug 的根因几乎总是在<strong>表示法边界</strong>上
          （像素 vs 坐标、开区间 vs 闭区间、0-based vs 1-based）。
          而且它只能靠逐项对账抓到，靠读代码是读不出来的。
        </p>
      </div>
    </section>

    <!-- ============================================================ 8 -->
    <section id="s8" class="fade">
      <div class="eyebrow">08 · 关卡</div>
      <h2>关卡生成：一个"物理可达性"约束，和它怎么把生成器逼死</h2>

      <p>
        这是最后修的一个 bug，而且它非常典型：<strong>约束本身是对的，
        但实现方式让它退化了。</strong>
      </p>

      <h3>约束从哪来</h3>
      <p>
        鸟的可持续爬升率大约是 <strong>100 px/s</strong>
        （拍一次翅买约 49&nbsp;px，周期约 0.4&nbsp;s；而且上冲期地面驱动被
        <span class="mono">vy_gate</span> 关掉，膜电位要重新积分）。
        管子间距 300&nbsp;px、速度 240&nbsp;px/s，所以两根管子之间是
        <strong>1.25&nbsp;s</strong> —— 一个间隔内最多净爬约 <strong>125&nbsp;px</strong>。
      </p>
      <p>
        所以如果把相邻缺口的"向上跳变"放得很大（比如 330&nbsp;px），
        <strong>那些关卡物理上根本飞不进去</strong>。那不是难度，是 bug。
        于是有了这条约束：
      </p>

      <div class="eq">
        下一根缺口中心 ≥ 上一根缺口中心 − max_climb &nbsp;&nbsp;<em>（向上受限；向下不设限）</em>
      </div>

      <h3>它怎么把生成器逼死的</h3>
      <p>
        原来的实现是"<strong>从下方均匀抽样</strong>"：
        <span class="mono">uniform(lo, min(hi, prev_c + max_climb − GAP/2))</span>。
        看起来等价，但一旦缺口中心下沉到某个位置，这个上界就缩到<strong>低于下界</strong>，
        于是退化成 <span class="mono">uniform(70, 70)</span> —— 恒等于 70。
      </p>

      <figure>
        {gap_svg}
        <figcaption>
          <b>修前 vs 修后的缺口分布</b>（各 4000 根管子）。
          修前所有管子落在<strong>同一个值</strong>上（标准差 0.0）—— 那不是"不够随机"，是生成器死了。
          而且它只用量程的一半，缺口长期被顶在下边界附近。
          修后铺满整个量程，标准差约 50&nbsp;px。
        </figcaption>
      </figure>

      <h3>量程尺：为什么缺口不能随便摆</h3>
      <div class="ruler">
        <div class="bar">
          <div class="band" style="left:19.7%; right:16.4%"></div>
          <div class="mark" style="left:19.7%"></div>
          <div class="mark" style="left:83.6%"></div>
        </div>
        <div class="marks">
          <span style="left:19.7%">top = 70（上界）</span>
          <span style="left:83.6%">top = 250（下界）</span>
        </div>
        <dl class="why">
          <dt>上方</dt>
          <dd>受执行器带宽约束：鸟的可持续爬升约 100 px/s，一个间隔只有 1.25 s，
              所以相邻缺口最多上移约 125 px。超出这个带宽的关卡<strong>物理上飞不进去</strong>，
              那不是难度，是 bug。</dd>
          <dt>下方</dt>
          <dd>不是可达性问题（自由落体 1.25 s 能掉 900+ px），而是<strong>手感</strong>：
              不想要"连续几十根一路坠到底"的单调关卡，所以给了个与缺口等高的界。</dd>
          <dt>紫色带</dt>
          <dd>缺口 top 的可行区间（画布高 620、缺口高 184 夹出来的）。</dd>
        </dl>
      </div>

      <p>修法是把"从下方抽样"换成<strong>以上一根缺口中心为中心的有界随机游走</strong>：</p>

      <div class="eq">
        next_hi = min(c_hi, prev_c + max_climb) &nbsp;<em>← 向上受执行器带宽限制（110 px）</em><br>
        next_lo = max(c_lo, prev_c − max_drop) &nbsp;<em>← 向下给一个与缺口等高的界（230 px）</em><br>
        c = uniform(next_lo, next_hi) + N(0, 16²) &nbsp;<em>← 再叠一点独立抖动</em><br>
        c = clip(c, next_lo, next_hi, c_lo, c_hi) &nbsp;<em>← 抖动之后必须重新夹，否则约束会被冲掉</em>
      </div>

      <p>
        最后那一步不是形式主义：我第一版把抖动加在夹取<strong>之后</strong>，
        实测最大上升达到 211&nbsp;px，超过了 110 的上限 —— 抖动把"物理可达"这条硬约束冲掉了。
      </p>

      <div class="note warn">
        <div class="h">两个界都必须设</div>
        <p>
          只设向上界会引入一个隐蔽的边界效应：缺口中心只能活在
          <span class="mono">[c_lo, c_hi]</span> 这个死区里，
          一旦被推到上边界，下一根<strong>必须</strong>落在死区内 → 被迫向下掉 216&nbsp;px。
          那不是设计，是边界效应；而且它把缺口长期顶在边界附近。
          两条界都设就没有这个问题。
        </p>
      </div>
    </section>

    <!-- ============================================================ 9 -->
    <section id="s9" class="fade">
      <div class="eyebrow">09 · 证据</div>
      <h2>怎么证明"分数真的穿过了连接组"</h2>

      <p>
        这是整个项目的重点实验。同一套参数、同一批关卡，只改脑的接线，跑 40 局：
      </p>

      <table>
        <thead><tr><th>组</th><th>做法</th><th>均分</th><th>拍翅/局</th></tr></thead>
        <tbody>
          <tr><td><b>real</b></td><td>真实接线</td><td class="n"><b>31.05</b></td><td class="n">80.6</td></tr>
          <tr><td><b>cut</b></td><td>切断 LC4/LPLC2 → DNp01</td><td class="n"><b>1.00 ± 0.00</b></td><td class="n"><b>0.0</b></td></tr>
          <tr><td><b>shuffled</b></td><td>同度分布随机重连</td><td class="n"><b>1.00 ± 0.00</b></td><td class="n"><b>0.0</b></td></tr>
        </tbody>
      </table>

      <p>
        <span class="mono">cut</span> 与 <span class="mono">shuffled</span>
        <strong>一次都不拍翅</strong>（方差为 0，40 局全部死在第 1 根管子）。
        这就是"分数必须穿过真实连接组"的硬证据 ——
        不是因为脑"变笨了"，而是因为<strong>指令神经元收不到驱动了</strong>。
      </p>

      <div class="note">
        <div class="h">那个 1.00 是哪来的</div>
        <p>
          整数 1 分不是脑在做事，是一种<strong>结构性白送</strong>：
          第 1 根管子的起点是 <span class="mono">x = 120 + 0.75×240 − 62 = 118</span>，
          而鸟在 <span class="mono">x = 120</span> —— <em>鸟一出生就在第 1 根管子的缺口里</em>；
          再加上开局 0.8&nbsp;s 的加速期不判碰撞，而第 1 根移出鸟身只需要约 40&nbsp;tick
          （= 0.8&nbsp;s）。所以加速期一结束它就已经"过"了第 1 根。裸分是 1。
        </p>
      </div>

      <h3>还有三个"逐 tick 对账"脚本</h3>
      <p>
        分数能不能信，取决于两套实现是不是同一套物理。所以有两个脚本专门做对账：
      </p>

      <table>
        <thead><tr><th>脚本</th><th>对什么</th><th>结果</th></tr></thead>
        <tbody>
          <tr><td class="mono">verify_server_physics.py</td>
              <td>服务端物理 vs 离线评测台物理，喂同一串拍翅命令，逐 tick 比位置/速度/分数/死亡时刻/管道布局</td>
              <td>60 局 × ≤3000 tick，<b>逐 tick 等价</b></td></tr>
          <tr><td class="mono">verify_server_vs_bench.py</td>
              <td>整体控制回路（含脑）vs 评测台，同种子逐 tick 比 y/vy/发放/拍翅</td>
              <td><b>逐 tick 一致，总分相同</b></td></tr>
          <tr><td class="mono">verify_server_collision.js</td>
              <td>服务端判定的碰撞 vs 前端画出来的管子，逐像素</td>
              <td>12410 个 y 位置，<b>偏差 0 / 0</b></td></tr>
        </tbody>
      </table>

      <p class="q">
        第二个脚本的价值远超预期：它当场抓出了三个真 bug（见下一节）。
        在写它之前，我只能靠猜。
      </p>
    </section>

    <!-- ============================================================ 10 -->
    <section id="s10" class="fade">
      <div class="eyebrow">10 · 坑</div>
      <h2>踩过的坑（这一节大概最有参考价值）</h2>

      <h3>1. 噪声走了全局随机数，于是"同一个种子"毫无意义</h3>
      <p>
        神经元的泊松噪声原来读的是<strong>全局</strong> RNG
        （<span class="mono">np.random.binomial</span> / <span class="mono">torch.rand</span>）。
        后果是：噪声序列取决于<strong>此前消耗了多少随机数</strong>。
      </p>
      <p>
        实测证据（两个进程、同一个种子、喂<strong>完全相同</strong>的驱动）：
      </p>
      <div class="eq">
        DNp01 发放数：服务端 2 / 评测台 1<br>
        膜电位最大差：<b>0.674</b>
      </div>
      <p>
        原因是两个进程初始化时消耗的随机数个数不同（服务端要建 3D 点云抽样、
        可点亮清单等）。表现就是"同一套代码、同一个种子，却飞得完全不同" ——
        这会让所有对照实验都站不住。
      </p>
      <p>
        修法：每个脑持有自己的 <span class="mono">rng</span> / <span class="mono">tgen</span>，
        <span class="mono">reset()</span> 时按种子重置。修完同样对账：
        <strong>膜电位最大差 0.000000、逐 tick 完全一致</strong>。
      </p>
      <div class="note warn">
        <div class="h">代价要说清楚</div>
        <p>
          噪声序列变了，所以历史分数不再逐位可复现。我们留了一个
          <span class="mono">--legacy-noise</span> 开关做对照：
          新噪声流 31.05、旧噪声流 43.65、而文档里长期引用的一个 100.80
          <strong>两条路径都复现不了</strong>（那是更早的代码 + 不同参数下测的）。
          这个修复确实把分数拉低了一截，不能假装没发生。
          <strong>但结论层面完好</strong>：cut/shuffled 依旧是 0 次拍翅。
        </p>
      </div>

      <h3>2. 渲染循环每帧抛异常，被 try/catch 吞掉</h3>
      <p>
        前端从服务端拿"加速期时长"时，引用了一个<strong>从未定义过</strong>的常量
        <span class="mono">WARMUP_S</span>（它只存在于服务端）。于是：
      </p>
      <div class="eq">
        ReferenceError: WARMUP_S is not defined<br>
        &nbsp;&nbsp;at interpolate (app.js)<br>
        &nbsp;&nbsp;at loop (app.js)
      </div>
      <p>
        每帧都抛 → 被 <span class="mono">loop()</span> 的 catch 吞进
        <span class="mono">DBG.err</span> → 后面的
        <span class="mono">drawGame()</span> / <span class="mono">draw3D()</span> / <span class="mono">hud()</span>
        <strong>一帧都没执行</strong>。症状是"游戏卡住 + 3D 面板全黑 + HUD 不动"，
        而 <strong>console 一个错都不报</strong>。
      </p>
      <p>
        两处修法：把那个常量改成<strong>服务端下发</strong>（单一来源），
        并且让这个 catch <strong>把错误显性打到状态栏</strong>。
        更重要的是给验收脚本加了一条检查："渲染循环有没有出错" ——
        正是缺了这一条，这个 bug 才潜伏了很久。
      </p>

      <h3>3. 每帧都去服务端取状态 = 60 req/s，把服务端打爆</h3>
      <p>
        取状态原来写在渲染循环里、<strong>每帧一次</strong>。
        实测 6&nbsp;秒发出 <strong>359 个</strong>请求（= 60&nbsp;req/s），
        把单进程的服务端砸到延迟从 3&nbsp;ms 飙到 <strong>38.9&nbsp;ms</strong>。
        服务端本身只有 50&nbsp;Hz，取 20&nbsp;Hz 完全够（插值补中间的帧）。
      </p>

      <h3>4. 投屏用"保持最新值 + 外推"，画面上会跳</h3>
      <p>
        原来写的是 <span class="mono">y = srvY + vy × (now − srvRx)</span>。
        <span class="mono">(now − srvRx)</span> 每帧都在变，于是<strong>同一个服务端状态
        被复用好几帧、每帧算出的位置都不同</strong>，下一帧收到新状态时基准一换就跳。
      </p>
      <table>
        <thead><tr><th>逐帧位移 dy</th><th>修前</th><th>修后</th></tr></thead>
        <tbody>
          <tr><td>中位</td><td class="n">2.70 px</td><td class="n"><b>0.00 px</b></td></tr>
          <tr><td>p95</td><td class="n">19.47 px</td><td class="n">受帧率限制</td></tr>
          <tr><td>最小（倒跳）</td><td class="n">−52.11 px</td><td class="n">不再出现</td></tr>
        </tbody>
      </table>
      <p>
        修法是维护<strong>最近两次状态</strong>（各带"到达本地的时刻"），在两者之间线性插值。
        用到达时刻而不是服务端时刻，是因为两个时钟之间有未知且会漂移的偏移量；
        而"两次到达的本地时刻之差"与服务端走过的时间是同一段，
        <strong>网络延迟在求商时自然抵消</strong>。
      </p>

      <h3>5. 校验器自己也会骗人</h3>
      <p>
        有一个脚本继承了我们自己的评测台实现，于是它把评测台和评测台比，
        "比值 1.0"永远通过 —— 它在仓库里存在了很久，什么都没验。已删。
      </p>
      <p>
        同样地，我用 <span class="mono">getImageData(drawImage(webglCanvas))</span>
        判断 3D 面板有没有画，结果永远读到全黑 —— 因为 WebGL 默认
        <span class="mono">preserveDrawingBuffer=false</span>，读回来的是空缓冲。
        改用 <span class="mono">renderer.info.render.points</span> 才看清它一直在正常渲染
        （实测 139,237 个点）。<strong>测试方法本身也要被质疑。</strong>
      </p>
      <p class="q">
        最狼狈的一次：我把测试脚本写错了（调生成函数和 append 的顺序反了），
        于是连续几轮都在"修"一个并不存在的越界 bug。同一段代码，
        两个测试给出 0 和 216 两个结论 —— 那时候该怀疑的是测试，不是代码。
      </p>
    </section>

    <!-- ============================================================ 11 -->
    <section id="s11" class="fade">
      <div class="eyebrow">11 · 负结果</div>
      <h2>试过但没用的东西（都如实记着）</h2>

      <p>
        一个项目的可信度，很大程度取决于它愿不愿意报负结果。下面这些全部实测过：
      </p>

      <table>
        <thead><tr><th>尝试</th><th>结果</th><th>为什么</th></tr></thead>
        <tbody>
          <tr><td>1-ply 预测 / 前视（lookahead）</td>
              <td class="n">0.00</td>
              <td>两个分支在 <span class="mono">vy = −316</span> 时行为相同，预判拿不到信息</td></tr>
          <tr><td>提高驱动增益</td>
              <td class="n">饱和在 ~101 px/s</td>
              <td>爬升率 gain 1→2→4 只从 87 涨到 101 就不动了</td></tr>
          <tr><td>上冲期间保留腹侧驱动</td>
              <td class="n">9.45 → 5.12</td>
              <td>它破坏了下潜支</td></tr>
          <tr><td>天花板增益 <span class="mono">ceil_boost</span> 1.4 / 2 / 3</td>
              <td class="n">25.24 / 22.20 / 21.61</td>
              <td>全部比 1.0 更差</td></tr>
          <tr><td>平滑空间窗替代"整半招募"</td>
              <td class="n">不发放</td>
              <td>只到 need 的 53%</td></tr>
          <tr><td>门控类变体（ground_eq 等）</td>
              <td class="n">全部 0 分</td>
              <td>——</td></tr>
        </tbody>
      </table>

      <div class="note">
        <div class="h">一个特别值得看的数</div>
        <p>
          有人做过一组"受控复现"：把物理/脑的速率比设成 <span class="mono">0.694</span>
          —— 正好是早年间页面实测的比例（50 ÷ 60×1.2）—— 分数只有 <strong>2.07</strong>。
          这把"页面为什么远低于评测台"从猜测变成了可复现的实验。
          <strong>现在这个现象不会再出现了</strong>：仿真搬到服务端之后，
          页面与评测台的速率比就是 1.0。
        </p>
      </div>
    </section>

    <!-- ============================================================ 12 -->
    <section id="s12" class="fade">
      <div class="eyebrow">12 · 边界</div>
      <h2>诚实的边界与复现方式</h2>

      <h3>我们没做到 / 说不清的地方</h3>
      <dl class="kv">
        <dt>旋钮是外部的</dt>
        <dd>
          <span class="mono">dors_scale</span> / <span class="mono">vent_gain</span>
          这两个参数是<strong>我们手写的</strong>，不是连接组里的东西。
          它们作用在脑的<strong>输入端</strong>（感觉信号有多强），连接组内部结构没改过。
          但必须承认：它们的存在说明"零参数"这句话限定在<strong>连接组</strong>层面。
        </dd>
        <dt>时钟是 20 ms</dt>
        <dd>
          真实巨型纤维通路的延迟是 1.4–25&nbsp;ms，而我们的 dt 固定 20&nbsp;ms。
          所以<strong>延迟类数字只能定性</strong>。
        </dd>
        <dt>LC4 的灵敏度是假设</dt>
        <dd>
          <span class="mono">s50</span> 取 30° 是外部假设（文献里没找到 LC4 的实测发放率）。
          它可调、会改变触发早晚，但不改变"有阈值、且由那条通路携带"这个结论。
        </dd>
        <dt>颜色与侧别的轴是先剔除半球轴再取的</dt>
        <dd>
          视野上下半的划分用的是"先剔除半球轴、再在同侧内部取方差最大的轴"。
          这个做法解决了早期一个真实错误（把左右半球当成了上下视野），但它是启发式。
        </dd>
      </dl>

      <h3>怎么自己复现</h3>
      <div class="eq">
        <em># 起服务（仓库自包含：脑资产与 three.js 都在库里）</em><br>
        pip install numpy pandas torch --index-url https://download.pytorch.org/whl/cpu<br>
        python demo/server.py --no-browser --port 8620<br>
        <em># 浏览器打开 http://127.0.0.1:8620/</em><br><br>
        <em># 离线评测台（量分数、做消融）</em><br>
        python scripts/flappy_bench.py --modes bidi --games 40 --graph real<br>
        python scripts/flappy_bench.py --modes bidi --games 40 --graph cut<br><br>
        <em># 三个对账脚本（这是"分数能不能信"的依据）</em><br>
        python scripts/verify_server_physics.py<br>
        python scripts/verify_server_vs_bench.py<br>
        node demo/verify_page.js 30
      </div>

      <div class="note warn">
        <div class="h">两个会让你白忙半天的坑</div>
        <p>
          <strong>① 传错资产：</strong>如果用了不含 LC4/LPLC2 的子图，
          分数会<strong>恒为 0 且不报错</strong>。现在服务端会在启动时直接拒绝并说明原因。
          <br>
          <strong>② 缺坐标文件：</strong>3D 面板会静默变空，而游戏照跑 —— 容易被当成"面板就这样"。
        </p>
      </div>
    </section>

    <!-- ============================================================ 13 -->
    <section id="s13" class="fade">
      <div class="eyebrow">13 · 附录</div>
      <h2>术语对照</h2>

      <table>
        <thead><tr><th>中文</th><th>English</th><th>在这个项目里指什么</th></tr></thead>
        <tbody>
          <tr><td>连接组</td><td>connectome</td><td>144,837 个神经元 + 1,502 万条突触的有向图</td></tr>
          <tr><td>漏积分-发放</td><td>LIF, leaky integrate-and-fire</td><td>每个神经元的更新规则（第 3 节）</td></tr>
          <tr><td>巨纤维</td><td>giant fiber (GF)</td><td>果蝇的逃逸指令通路；这里对应 DNp01 那两个细胞</td></tr>
          <tr><td>逼近刺激</td><td>looming</td><td>物体在视野里持续变大；本项目的"威胁"定义</td></tr>
          <tr><td>角尺寸 / 角速度</td><td>angular size / angular velocity</td><td>LPLC2 与 LC4 分别编码的量</td></tr>
          <tr><td>整半招募</td><td>whole-half recruitment</td><td>整个半个视野一起驱动，而不是平滑权重窗</td></tr>
          <tr><td>锁步</td><td>lockstep</td><td>脑 tick 与物理步严格 1:1，同一个进程同一把锁</td></tr>
          <tr><td>关节</td><td>tick</td><td>一次 20 ms 的仿真步；控制回路的最小单位</td></tr>
          <tr><td>逐 tick 对账</td><td>tick-for-tick reconciliation</td><td>两套实现喂同样输入，逐 tick 比全部状态</td></tr>
          <tr><td>逐像素对账</td><td>pixel-exact reconciliation</td><td>判定集合与绘制集合逐个像素比</td></tr>
        </tbody>
      </table>

      <h3>几个关键数字，一页带走</h3>
      <table>
        <thead><tr><th>量</th><th>值</th><th>它为什么重要</th></tr></thead>
        <tbody>
          <tr><td class="n">need = (1−leak)·thr/gain</td><td class="n">0.06042</td><td>持续发放所需的最小输入；所有驱动强弱的"汇率"</td></tr>
          <tr><td class="n">leak = exp(−0.02/0.1)</td><td class="n">0.81873</td><td>漏得多快</td></tr>
          <tr><td class="n">管距 / 管速</td><td class="n">1.25 s</td><td>两根管子之间的可用时间</td></tr>
          <tr><td class="n">可持续爬升率</td><td class="n">约 100 px/s</td><td>所以一个间隔最多净爬约 125 px（关卡约束的依据）</td></tr>
          <tr><td class="n">一个 tick 能爬</td><td class="n">约 2 px</td><td>决策粒度有多细</td></tr>
          <tr><td class="n">服务端仿真速率</td><td class="n">50.0 步/秒</td><td>与真实时间 1:1，且恒定</td></tr>
        </tbody>
      </table>

      <footer>
        数据来源：FlyWire FAFB v783（codex.flywire.ai）· 144,837 神经元 / 15,023,799 突触<br>
        本页所有图形由 <span class="mono">scripts/make_doc_figures.py</span> 从<strong>跑着的仿真</strong>里取数生成，非示意图<br>
        生成时间与数值随代码版本变化 · 只要仓库里跑一次就能复现
      </footer>
    </section>

  </main>
</div>
<script>{js}</script>
</body>
</html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
