"""把算法教材写到桌面 —— 自包含 HTML，配图取自**真实仿真数据**。

用法:
    python scripts/make_doc_page.py                 # 写到桌面
    python scripts/make_doc_page.py --out 某目录
    python scripts/make_doc_page.py --no-fonts      # 不引用在线字体（纯离线）

模板用 `{{TOKEN}}` 手工替换，**不用 str.format** ——
CSS 里的花括号会被 format 当成占位符，上一版就在这儿出过哑巴亏。
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))


# ==================================================================== 样式
CSS = r"""
:root{
  --ink:#14161b; --ink2:#3b3f47; --paper:#fbfaf7; --surface:#f4f2ec;
  --rule:#d7d4cd; --rule2:#e8e5dd; --quiet:#6f6b64; --quiet2:#8f8b83;
  --lc4:#c2410c; --lplc2:#7c3aed; --dn:#be123c; --phys:#1d4ed8;
  --thesis:#0f5c5c;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:"IBM Plex Sans","PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif;
  --cond:"IBM Plex Sans Condensed","PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif;
  --col:46rem;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%; scroll-behavior:smooth}
@media (prefers-reduced-motion:reduce){ html{scroll-behavior:auto} }
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--sans); font-size:16.5px; line-height:1.78;
  font-feature-settings:"tnum" 1; -webkit-font-smoothing:antialiased;
}
.shell{display:grid; grid-template-columns:16.5rem minmax(0,var(--col));
  gap:4.5rem; justify-content:center; padding:0 1.75rem 7rem}
@media (max-width:1080px){
  .shell{grid-template-columns:minmax(0,var(--col)); gap:0; padding:0 1.25rem 5rem}
  .rail{display:none}
}

/* ---------- 进度条 ---------- */
#prog{position:fixed; top:0; left:0; height:2px; width:0; background:var(--thesis);
  z-index:50; transition:width .1s linear}

/* ---------- 左侧目录 ---------- */
.rail{position:sticky; top:0; align-self:start; height:100vh; overflow-y:auto;
  padding:2.2rem 0 2rem; font-family:var(--mono); font-size:.72rem;
  color:var(--quiet); scrollbar-width:thin}
.rail::-webkit-scrollbar{width:5px}
.rail::-webkit-scrollbar-thumb{background:var(--rule); border-radius:3px}
.rail .head{font-size:.6rem; letter-spacing:.2em; text-transform:uppercase;
  color:var(--quiet2); margin:0 0 .5rem}
.rail .grp{margin:1.15rem 0 .4rem; font-size:.6rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--thesis); padding-bottom:.3rem;
  border-bottom:1px solid var(--rule2)}
.rail ol{list-style:none; margin:0; padding:0}
.rail li{margin:0}
.rail a{display:flex; gap:.5rem; padding:.26rem 0 .26rem .55rem; color:inherit;
  text-decoration:none; line-height:1.4; border-left:2px solid transparent;
  transition:color .12s, border-color .12s}
.rail a .n{color:var(--quiet2); flex:0 0 1.5rem}
.rail a:hover{color:var(--ink); border-left-color:var(--rule)}
.rail a.on{color:var(--ink); border-left-color:var(--dn); font-weight:500}
.rail a.on .n{color:var(--dn)}
.rail .meta{margin-top:1.4rem; padding-top:.9rem; border-top:1px solid var(--rule);
  font-size:.66rem; line-height:1.7; color:var(--quiet2)}
.rail .meta b{color:var(--ink); font-weight:500}

/* ---------- 排版 ---------- */
main{padding-top:2.2rem; min-width:0}
.eyebrow{font-family:var(--mono); font-size:.67rem; letter-spacing:.2em;
  text-transform:uppercase; color:var(--quiet2)}
h1{font-family:var(--cond); font-weight:700; font-size:clamp(1.9rem,4.2vw,2.7rem);
  line-height:1.13; letter-spacing:-.015em; margin:.55rem 0 1rem}
h1 .lite{font-weight:400; color:var(--quiet)}
h2{font-family:var(--cond); font-weight:700; font-size:1.72rem; letter-spacing:-.012em;
  line-height:1.2; margin:0 0 .9rem}
h2 .hnum{font-family:var(--mono); font-weight:400; font-size:.95rem; color:var(--dn);
  margin-right:.7rem; letter-spacing:0}
h3{font-family:var(--cond); font-weight:600; font-size:1.14rem; margin:2.1rem 0 .45rem}
h4{font-family:var(--mono); font-weight:500; font-size:.78rem; letter-spacing:.06em;
  margin:1.5rem 0 .35rem; color:var(--ink2)}
section{padding:0; scroll-margin-top:1.5rem}
.chapter{border-top:2px solid var(--ink); margin-top:4.4rem; padding-top:1rem}
.chapter .ctitle{font-family:var(--mono); font-size:.68rem; letter-spacing:.2em;
  text-transform:uppercase; color:var(--thesis)}
.sec{border-top:1px solid var(--rule); margin-top:3.1rem; padding-top:2.4rem}
.sec:first-of-type{border-top:0; margin-top:2rem; padding-top:0}
p{margin:.8rem 0}
.lead{font-size:1.05rem; color:var(--ink2)}
a{color:var(--phys); text-decoration:none; border-bottom:1px solid rgba(29,78,216,.28)}
a:hover{border-bottom-color:var(--phys)}
strong{font-weight:600}
em{font-style:normal; background:linear-gradient(transparent 62%,rgba(124,58,237,.16) 0)}
code{font-family:var(--mono); font-size:.86em; background:var(--surface);
  padding:.08em .34em; border:1px solid var(--rule2); border-radius:2px}
.mono{font-family:var(--mono)}
.q{color:var(--quiet)}
hr{border:0; border-top:1px solid var(--rule2); margin:2rem 0}

/* 线索句：每节开头一句话总结 */
.thread{margin:.9rem 0 0; padding:.85rem 1.1rem; background:var(--surface);
  border-left:3px solid var(--thesis); font-size:.95rem; color:var(--ink2)}
.thread b{color:var(--ink)}

/* ---------- 三层讲解 ---------- */
.layers{margin:1.5rem 0 0}
.layer{display:grid; grid-template-columns:5.4rem 1fr; gap:0 1.1rem;
  padding:.85rem 0; border-top:1px solid var(--rule2)}
.layer:last-child{border-bottom:1px solid var(--rule2)}
.layer .tag{font-family:var(--mono); font-size:.64rem; letter-spacing:.09em;
  text-transform:uppercase; color:var(--quiet2); padding-top:.28rem; line-height:1.5}
.layer .tag b{display:block; color:var(--ink); font-size:.7rem; letter-spacing:.04em}
.layer .body{min-width:0}
.layer .body>:first-child{margin-top:0}
.layer .body>:last-child{margin-bottom:0}
@media (max-width:640px){
  .layer{grid-template-columns:1fr; gap:.2rem}
  .layer .tag{display:flex; gap:.5rem; align-items:baseline}
  .layer .tag b{display:inline}
}

/* ---------- 公式 ---------- */
.eq{margin:1.2rem 0; padding:.95rem 1.15rem; background:#fff;
  border:1px solid var(--rule); font-family:var(--mono); font-size:.9rem;
  line-height:2; overflow-x:auto}
.eq .cmt{color:var(--quiet); font-size:.84em}
.eq .hl{color:var(--dn); font-weight:500}
.calc{font-family:var(--mono); font-size:.86rem; line-height:1.95;
  background:#fff; border:1px solid var(--rule); padding:.85rem 1.05rem;
  margin:1.1rem 0; overflow-x:auto; white-space:pre}
.calc .r{color:var(--dn)} .calc .c{color:var(--quiet)}

/* ---------- 提示块 ---------- */
.box{margin:1.4rem 0; border:1px solid var(--rule); background:#fff;
  padding:1rem 1.15rem}
.box .lab{font-family:var(--mono); font-size:.62rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--quiet2); margin-bottom:.45rem}
.box.work{background:#f7f9ff; border-color:#c9d4f0}
.box.work .lab{color:var(--phys)}
.box.trap{background:#fff6f7; border-color:#f0ccd4}
.box.trap .lab{color:var(--dn)}
.box.key{background:#f3f8f8; border-color:#c4dcdc}
.box.key .lab{color:var(--thesis)}
.box>:first-child{margin-top:0} .box>:last-child{margin-bottom:0}

details{margin:1.1rem 0; border:1px solid var(--rule); background:#fff}
details summary{cursor:pointer; padding:.75rem 1.05rem; font-family:var(--mono);
  font-size:.76rem; letter-spacing:.05em; color:var(--thesis); list-style:none;
  display:flex; align-items:center; gap:.55rem}
details summary::-webkit-details-marker{display:none}
details summary::before{content:"+"; font-size:.95rem; color:var(--quiet2);
  width:.8rem; display:inline-block; transition:transform .15s}
details[open] summary::before{content:"−"}
details summary:hover{background:var(--surface)}
details .ans{padding:.1rem 1.15rem 1.05rem; border-top:1px solid var(--rule2)}
details .ans>:first-child{margin-top:.7rem}
details .ans>:last-child{margin-bottom:0}

/* ---------- 表格 ---------- */
.tw{overflow-x:auto; margin:1.3rem 0}
table{width:100%; border-collapse:collapse; font-size:.89rem}
th,td{text-align:left; padding:.5rem .7rem; border-bottom:1px solid var(--rule2);
  vertical-align:top}
th{font-family:var(--mono); font-size:.65rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--quiet2); font-weight:500;
  border-bottom:1px solid var(--rule); white-space:nowrap}
td.n{font-family:var(--mono); white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover{background:var(--surface)}
td .sw{display:inline-block; width:.6em; height:.6em; border-radius:50%;
  margin-right:.4em}

dl.kv{display:grid; grid-template-columns:auto minmax(0,1fr); gap:.4rem 1.2rem;
  margin:1.2rem 0; font-size:.92rem}
dl.kv dt{font-family:var(--mono); color:var(--quiet); white-space:nowrap; font-size:.85rem}
dl.kv dd{margin:0}

/* ---------- 图 ---------- */
figure{margin:1.9rem 0; border:1px solid var(--rule); background:#fff}
figure svg{display:block; width:100%; height:auto}
figure svg text{font-family:var(--mono)}
figcaption{font-size:.8rem; line-height:1.7; color:var(--quiet);
  padding:.75rem 1rem; border-top:1px solid var(--rule2); background:var(--paper)}
figcaption b{color:var(--ink); font-weight:600}
figcaption .n{font-family:var(--mono); font-size:.72rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--quiet2); display:block; margin-bottom:.2rem}

/* ---------- 量程尺 ---------- */
.ruler{margin:1.6rem 0; font-size:.86rem}
.ruler .bar{position:relative; height:2rem; border:1px solid var(--rule); background:#fff}
.ruler .band{position:absolute; top:0; bottom:0; background:rgba(124,58,237,.14)}
.ruler .mk{position:absolute; top:-.28rem; bottom:-.28rem; width:0;
  border-left:2px solid var(--dn)}
.ruler .scale{position:relative; height:1.1rem; font-family:var(--mono);
  font-size:.66rem; color:var(--quiet)}
.ruler .scale span{position:absolute; transform:translateX(-50%); white-space:nowrap}
.ruler .why{display:grid; grid-template-columns:5.5rem 1fr; gap:.35rem 1rem;
  margin-top:.95rem; font-size:.86rem}
.ruler .why dt{font-family:var(--mono); font-size:.74rem; color:var(--quiet)}
.ruler .why dd{margin:0}

/* ---------- 术语卡 ---------- */
.term{margin:1.5rem 0; border-top:1px solid var(--rule)}
.term .row{display:grid; grid-template-columns:minmax(8.5rem,11rem) minmax(0,1fr);
  gap:0 1.2rem; padding:.8rem 0; border-bottom:1px solid var(--rule2)}
.term .name{font-family:var(--mono); font-size:.82rem; line-height:1.5}
.term .name b{display:block; font-weight:600; color:var(--ink); font-size:.88rem}
.term .name span{color:var(--quiet2); font-size:.72rem}
.term .def{min-width:0; font-size:.9rem}
.term .def>:first-child{margin-top:0} .term .def>:last-child{margin-bottom:0}
@media (max-width:640px){ .term .row{grid-template-columns:1fr; gap:.25rem} }

/* ---------- 速览卡 ---------- */
.card{border:1px solid var(--ink); background:#fff; margin:1.8rem 0}
.card .top{background:var(--ink); color:var(--paper); padding:.5rem 1rem;
  font-family:var(--mono); font-size:.65rem; letter-spacing:.18em;
  text-transform:uppercase}
.card ol{margin:0; padding:1rem 1.15rem 1.15rem 2.4rem}
.card li{margin:.5rem 0; font-size:.92rem; line-height:1.6}
.card li b{color:var(--ink)}

.kbd{font-family:var(--mono); font-size:.8em; border:1px solid var(--rule);
  border-bottom-width:2px; border-radius:3px; padding:.05em .35em; background:#fff}

footer{margin-top:4.5rem; padding-top:1.5rem; border-top:1px solid var(--rule);
  font-family:var(--mono); font-size:.7rem; color:var(--quiet2); line-height:1.9}

@media print{
  body{background:#fff} #prog,.rail{display:none}
  .shell{display:block; padding:0} .sec{break-inside:avoid}
  details{break-inside:avoid} details .ans{display:block !important}
  details{open:true} main{padding-top:0}
}
"""

JS = r"""
(function(){
  // 顶层进度条
  var bar = document.getElementById('prog');
  function prog(){
    var h = document.documentElement.scrollHeight - innerHeight;
    bar.style.width = (h > 0 ? Math.min(100, scrollY / h * 100) : 0) + '%';
  }

  // 目录高亮
  var links = [].slice.call(document.querySelectorAll('.rail a[data-for]'));
  var secs = links.map(function(a){ return document.getElementById(a.getAttribute('data-for')); });
  function sync(){
    var best = 0;
    secs.forEach(function(s, i){
      if (s && s.getBoundingClientRect().top <= 150) best = i;
    });
    links.forEach(function(a, i){ a.classList.toggle('on', i === best); });
  }
  addEventListener('scroll', function(){ prog(); sync(); }, {passive:true});
  prog(); sync();

  // 20ms 计数器：把"这是一条 50Hz 闭环"变成页面上活着的东西
  var tk = document.getElementById('tickno');
  if (tk && !matchMedia('(prefers-reduced-motion: reduce)').matches){
    var t0 = performance.now();
    setInterval(function(){
      tk.textContent = Math.floor((performance.now() - t0) / 20).toLocaleString();
    }, 200);
  }

  // 键盘上一节/下一节
  var items = secs.filter(Boolean);
  addEventListener('keydown', function(e){
    if (e.target.tagName === 'INPUT' || e.metaKey || e.ctrlKey) return;
    var cur = 0;
    items.forEach(function(s, i){ if (s.getBoundingClientRect().top <= 150) cur = i; });
    if (e.key === 'j' || e.key === 'ArrowDown' && e.shiftKey){}
    if (e.key === 'n'){ var t = items[Math.min(cur+1, items.length-1)]; if (t) t.scrollIntoView(); }
    if (e.key === 'p'){ var u = items[Math.max(cur-1, 0)]; if (u) u.scrollIntoView(); }
  });

  // 打印时把所有自测答案展开
  addEventListener('beforeprint', function(){
    document.querySelectorAll('details').forEach(function(d){ d.open = true; });
  });
})();
"""


def build(args) -> int:
    figs = pathlib.Path(args.figs)
    need = ["loop.svg", "raster.svg", "gapgen.svg", "membrane.svg", "spikes.json"]
    miss = [n for n in need if not (figs / n).is_file()]
    if miss:
        print(f"缺配图 {miss}；先跑 scripts/make_doc_figures.py", file=sys.stderr)
        return 2
    svg = {k: (figs / f"{k}.svg").read_text(encoding="utf-8")
           for k in ("loop", "raster", "gapgen", "membrane")}
    st = json.loads((figs / "spikes.json").read_text(encoding="utf-8"))

    # ---- 从代码里取常数，保证文档永远不会和实现脱节
    # ⚠️ 有些常数是**模块级**的（GROUND/PX_PER_M/MAX_CLIMB），
    #    有些是 **GameWorld 的类属性**（GAP/PIPE_W/GRAV/FLAP_V/TICK_S/WARM_S）。
    #    所以分两处取，不要假设都在模块级 —— 我第一次就 import 错了。
    from server import (BIDI, FIRST_GAP_EXTRA, GAIN, GROUND_Y, MAX_CLIMB,
                        PX_PER_M, SIM_HZ, WINDOW, GameWorld)
    import math as _m
    GW = GameWorld
    TICK_S = GW.TICK_S
    LEAK = _m.exp(-TICK_S / 0.1)
    NEED = (1 - LEAK) * 1.0 / GAIN
    C = dict(
        LEAK=f"{LEAK:.6f}", NEED=f"{NEED:.6f}", TICK_MS=int(TICK_S * 1000),
        GAIN=f"{GAIN:.0f}", WINDOW=WINDOW, SIM_HZ=f"{SIM_HZ:.0f}",
        MAX_CLIMB=f"{MAX_CLIMB:.0f}", GAP=f"{GW.GAP:.0f}",
        PIPE_W=f"{GW.PIPE_W:.0f}", GRAV=f"{GW.GRAV:.0f}",
        FLAP_V=f"{GW.FLAP_V:.0f}",
        SPACING=f"{GW.SPACING:.0f}", WARM=f"{GW.WARM_S}",
        GROUND=f"{GW.G_H - GW.GROUND:.0f}" if False else f"{GROUND_Y:.0f}",
        GROUND_H="92", PX_PER_M=f"{PX_PER_M:.0f}",
        BIRD_R=f"{GW.BIRD_R:.0f}", BIRD_X=f"{GW.BIRD_X:.0f}",
        G_W=f"{GW.G_W:.0f}", G_H=f"{GW.G_H:.0f}",
        TWEEN=f"{GW.SPACING / PX_PER_M:.3f}",
        N_TICKS=st["ticks"], N_DN=st["n_dn"], N_FLAP=st["n_flap"],
        MEM_GAP=st["mem_gap"], MEM_FIRST=st["mem_fires"][0] + 1,
    )

    # 顶部是否引用在线字体
    font_link = "" if args.no_fonts else (
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" media="print" onload="this.media=\'all\'"\n'
        ' href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
        '&family=IBM+Plex+Sans+Condensed:wght@400;600;700'
        '&family=IBM+Plex+Sans:wght@400;600&display=swap">')

    body = CONTENT
    for k, v in C.items():
        body = body.replace("{{" + k + "}}", str(v))
    body = (body.replace("{{SVG_LOOP}}", svg["loop"])
                .replace("{{SVG_RASTER}}", svg["raster"])
                .replace("{{SVG_GAPGEN}}", svg["gapgen"])
                .replace("{{SVG_MEMBRANE}}", svg["membrane"]))

    html = PAGE.replace("{{CSS}}", CSS).replace("{{JS}}", JS) \
               .replace("{{FONTS}}", font_link).replace("{{BODY}}", body)

    left = [t for t in ("{{", "}}") if t in html]
    # 只报告"看起来像占位符"的残留
    import re
    rest = sorted(set(re.findall(r"\{\{[A-Z_0-9]+\}\}", html)))
    if rest:
        print(f"⚠️ 未替换的占位符: {rest}", file=sys.stderr)

    out = pathlib.Path(args.out) if args.out else pathlib.Path.home() / "Desktop"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / args.name
    dst.write_text(html, encoding="utf-8")
    print(f"→ {dst}  ({dst.stat().st_size/1024:.0f} KB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default=str(ROOT / "output" / "doc_figs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="果蝇脑玩Flappy·算法教材.html")
    ap.add_argument("--no-fonts", action="store_true")
    return build(ap.parse_args())


# ==================================================================== 页面骨架
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>一只果蝇的脑子怎么玩 Flappy · 算法教材</title>
{{FONTS}}
<style>{{CSS}}</style>
</head>
<body>
<div id="prog"></div>
<div class="shell">
<aside class="rail" aria-label="目录">
  <p class="head">算法教材</p>
  <div class="grp">第一部 · 全局</div>
  <ol>
    <li><a href="#s01" data-for="s01"><span class="n">01</span><span>五分钟看懂</span></a></li>
    <li><a href="#s02" data-for="s02"><span class="n">02</span><span>总图与数据</span></a></li>
  </ol>
  <div class="grp">第二部 · 计算基础</div>
  <ol>
    <li><a href="#s03" data-for="s03"><span class="n">03</span><span>神经元怎么算</span></a></li>
    <li><a href="#s04" data-for="s04"><span class="n">04</span><span>那个门槛 0.06042</span></a></li>
  </ol>
  <div class="grp">第三部 · 感觉与指令</div>
  <ol>
    <li><a href="#s05" data-for="s05"><span class="n">05</span><span>两条视觉通道</span></a></li>
    <li><a href="#s06" data-for="s06"><span class="n">06</span><span>把几何变成驱动</span></a></li>
    <li><a href="#s07" data-for="s07"><span class="n">07</span><span>逼近期望与判据</span></a></li>
  </ol>
  <div class="grp">第四部 · 反射</div>
  <ol>
    <li><a href="#s08" data-for="s08"><span class="n">08</span><span>双向逼近反射</span></a></li>
    <li><a href="#s09" data-for="s09"><span class="n">09</span><span>回路闭合</span></a></li>
  </ol>
  <div class="grp">第五部 · 游戏世界</div>
  <ol>
    <li><a href="#s10" data-for="s10"><span class="n">10</span><span>物理与像素</span></a></li>
    <li><a href="#s11" data-for="s11"><span class="n">11</span><span>关卡生成</span></a></li>
  </ol>
  <div class="grp">第六部 · 证据</div>
  <ol>
    <li><a href="#s12" data-for="s12"><span class="n">12</span><span>消融实验</span></a></li>
    <li><a href="#s13" data-for="s13"><span class="n">13</span><span>逐 tick 对账</span></a></li>
    <li><a href="#s14" data-for="s14"><span class="n">14</span><span>可复现性与噪声</span></a></li>
  </ol>
  <div class="grp">第七部 · 工程</div>
  <ol>
    <li><a href="#s15" data-for="s15"><span class="n">15</span><span>为什么必须服务端</span></a></li>
    <li><a href="#s16" data-for="s16"><span class="n">16</span><span>六个真实 bug</span></a></li>
    <li><a href="#s17" data-for="s17"><span class="n">17</span><span>负结果</span></a></li>
  </ol>
  <div class="grp">附录</div>
  <ol>
    <li><a href="#s18" data-for="s18"><span class="n">18</span><span>常见误解</span></a></li>
    <li><a href="#s19" data-for="s19"><span class="n">19</span><span>术语表</span></a></li>
    <li><a href="#s20" data-for="s20"><span class="n">20</span><span>一页带走</span></a></li>
  </ol>
  <div class="meta">
    页面已运行 <b id="tickno">0</b> 个 20&nbsp;ms tick<br>
    这只鸟的每个动作，<br>都是 144,837 个神经元算出来的。<br><br>
    <span class="q">按 n / p 跳上/下一节</span>
  </div>
</aside>
<main>
{{BODY}}
</main>
</div>
<script>{{JS}}</script>
</body>
</html>
"""


# ==================================================================== 正文
CONTENT = r"""
<header>
  <div class="eyebrow">FlyWire FAFB v783 · 冻结连接组 · 零可学参数</div>
  <h1>一只果蝇的脑子<br>怎么玩 Flappy<span class="lite"> —— 算法教材</span></h1>
  <p class="lead">
    这个项目里<strong>没有训练、没有强化学习、没有任何一个"学出来"的参数</strong>。
    我们拿一张真实的果蝇全脑连接组 —— 144,837 个神经元、1,502 万条突触 ——
    当控制器，把"往前飞、别撞管子"交给它。
    它拿到的分数不是调参凑出来的，是<strong>被迫</strong>穿过那条真实的神经通路。
  </p>
  <p class="q">
    这份文档的目标是让你<strong>真的掌握</strong>这套东西，而不只是"看过"。
    每一节都按三层写：<span class="mono">直觉</span>（用人话讲）、
    <span class="mono">形式</span>（专业定义）、
    <span class="mono">本项目</span>（具体到数字和代码）。
    中间有可动手核对的计算、自测题，末尾有常见误解与速查表。
  </p>
  <div class="box key">
    <div class="lab">怎么读这份文档</div>
    <p>
      ① 想先建立整体感 → 读 <a href="#s01">01</a>、<a href="#s02">02</a>，再看
      <a href="#s20">20（一页带走）</a>。<br>
      ② 想真的学会 → 顺着读，<strong>每道自测题都自己先答一遍</strong>再展开答案。
      遇到公式不要跳过：<a href="#s04">04</a> 那个 0.06042 是整套东西的"汇率"，
      不懂它就理解不了后面为什么"驱动必须够强"。<br>
      ③ 想复现 → 直接看 <a href="#s13">13</a>、<a href="#s14">14</a>、<a href="#s17">17</a>。
    </p>
  </div>
</header>

<!-- ============================================================ 01 -->
<div class="chapter"><span class="ctitle">第一部 · 全局</span></div>
<section class="sec" id="s01">
  <div class="eyebrow">01</div>
  <h2>五分钟看懂：这东西到底在做什么</h2>
  <div class="thread">
    一句话：<b>把游戏画面翻译成"有东西正在朝我逼近"的神经信号，交给一张真实的果蝇脑图去反应，再把它的反应接回游戏。</b>
  </div>

  <div class="layers">
    <div class="layer">
      <div class="tag"><b>直觉</b>人话版</div>
      <div class="body">
        <p>
          想象你从一张果蝇大脑的电子显微镜切片里，把每个神经细胞和它们之间的连接
          全部提取出来，得到一个 14 万节点的有向图。
        </p>
        <p>
          现在你<strong>不训练它</strong>，只给它接上"眼睛"和"翅膀"：
          左边灌进去"前面那根管子离我多近"，右边读出来"要不要拍一下翅膀"。
          中间那一层，每根线都是电镜里量出来的、固定死的。
        </p>
        <p>
          于是"它会不会飞"这个问题，就变成了一个关于那张图的<strong>客观问题</strong>：
          <em>这张图里到底有没有一套能控制高度的回路？</em>
        </p>
      </div>
    </div>
    <div class="layer">
      <div class="tag"><b>形式</b>专业表述</div>
      <div class="body">
        <p>
          这是一个<strong>固定拓扑、固定符号的脉冲神经网络</strong>
          （fixed-topology spiking neural network）作为闭环控制器，
          权重与连接由连接组数据给定，<strong>不参与任何优化</strong>。
        </p>
        <p>
          可调的部分只有<strong>感觉编码层</strong>（sensory encoding）：
          即"外部状态 → 感觉神经元注入电流"这个映射。
          换句话说，<span class="mono">f: 游戏几何 → 驱动电流</span> 是可设计的，
          <span class="mono">g: 驱动电流 → 运动输出</span> 完全由生物学固定。
        </p>
      </div>
    </div>
    <div class="layer">
      <div class="tag"><b>本项目</b>具体数字</div>
      <div class="body">
        <div class="tw"><table>
          <thead><tr><th>量</th><th>值</th><th>说明</th></tr></thead>
          <tbody>
            <tr><td>神经元</td><td class="n">144,837</td><td>FAFB v783 全脑重建</td></tr>
            <tr><td>突触</td><td class="n">15,023,799</td><td>兴奋 11,043,448 / 抑制 3,152,462 / 未知符号 827,889</td></tr>
            <tr><td>仿真步长</td><td class="n">{{TICK_MS}} ms</td><td>即 50 Hz，与真实时间 1:1</td></tr>
            <tr><td>可学参数</td><td class="n">0</td><td>连接组权重全部冻结</td></tr>
            <tr><td>动作</td><td class="n">拍翅 / 不拍</td><td>单一二值输出</td></tr>
          </tbody>
        </table></div>
      </div>
    </div>
  </div>

  <details>
    <summary>自测：如果用一句话向别人解释这个项目，最不能漏掉的是哪个限定词？</summary>
    <div class="ans">
      <p>
        <strong>"零可学参数"</strong>（或者"连接组冻结"）。
      </p>
      <p>
        漏掉它，这个项目就退化成"又一个用神经网络玩游戏的 demo"——
        因为只要允许调权重，"脑在控制飞行"这句话就无法证伪：
        你总能说"再调调就好了"。
        正是这个约束，才让 <a href="#s12">第 12 节</a>那个
        "切断一条通路分数直接归零"的实验有了意义。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 02 -->
<section class="sec" id="s02">
  <div class="eyebrow">02</div>
  <h2>总图：一条 20 毫秒的闭环</h2>
  <div class="thread">
    整个系统只有一个节奏：<b>每 20 毫秒走一圈</b>。
    所有设计上的争论，最后都归结为"这一圈里顺序是什么"。
  </div>

  <figure>
    {{SVG_LOOP}}
    <figcaption>
      <span class="n">图 1 · 控制回路</span>
      <b>每 20 ms 闭合一次。</b>从游戏几何算出感觉输入 → 灌进感觉神经元 →
      脑按连接组自己传播 → 读两根巨纤维 → 决定拍不拍 → 推进物理。下一圈再看新的几何。
      <br>注意这是一条<strong>闭环</strong>：它不是在"预测"未来，而是每 20 ms 重新看一眼。
      颜色即语义：<span style="color:#c2410c">LC4 角速度</span>、
      <span style="color:#7c3aed">LPLC2 角大小</span>、
      <span style="color:#be123c">DNp01 指令</span>、
      <span style="color:#1d4ed8">物理</span>。全文沿用这套配色。
    </figcaption>
  </figure>

  <h3>一个 tick 里发生的事（顺序不能变）</h3>
  <div class="eq">
    ① 按<strong>当前</strong>世界几何算驱动<br>
    ② 用这个驱动走<strong>恰好一个</strong>脑 tick<br>
    ③ 读 DNp01（最近 5 个 tick 内发放 ≥ 1 次 → 拍翅）<br>
    ④ 用这个决策走<strong>恰好一步</strong>物理<br>
    <span class="cmt">四步在同一个进程、同一把锁里完成；没有网络往返、没有定时器节拍插进来</span>
  </div>

  <div class="box key">
    <div class="lab">为什么"恰好一个 / 恰好一步"是重点</div>
    <p>
      因为<strong>决策与物理必须严格 1:1</strong>。
      如果脑走一步而物理走两步（或相反），你执行的决策就对应了错误的世界状态，
      控制器立刻失稳。这个坑我们踩过四次，详见 <a href="#s15">第 15 节</a>。
    </p>
  </div>

  <details>
    <summary>自测：为什么这个设计叫"闭环"，而不是"开环 + 预测"？</summary>
    <div class="ans">
      <p>
        因为它<strong>每一圈都重新测量</strong>世界状态（缺口在哪、鸟在哪），
        而不是在某一时刻算出一条完整轨迹然后照着执行。
      </p>
      <p>
        差别很实际：开环方案需要准确预测"管子什么时候到、我什么时候该拍"，
        一旦预测有偏差就无法纠正；闭环方案对模型误差容忍得多——
        反正 20 ms 后你会重新看一眼。这也是为什么它能直接吃"当前几何"这种
        极其简化的输入而仍然飞得起来。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 03 -->
<div class="chapter"><span class="ctitle">第二部 · 计算基础</span></div>
<section class="sec" id="s03">
  <div class="eyebrow">03</div>
  <h2>一个神经元怎么算：漏积分-发放模型</h2>
  <div class="thread">
    核心只有一句：<b>电位一边被灌、一边在漏；漏着还能涨过阈值，就发放。</b>
  </div>

  <div class="layers">
    <div class="layer">
      <div class="tag"><b>直觉</b>人话版</div>
      <div class="body">
        <p>
          想象一个<strong>底部有小孔的水桶</strong>。上游神经元每发一次脉冲，
          就往桶里倒一点水；同时桶底一直在漏水（漏掉的比例固定）。
        </p>
        <p>
          水位超过桶壁上那条线，桶就"发"一次 —— 发完<strong>立刻倒空</strong>，
          重新开始接水。
        </p>
        <p>
          这个模型的两个直接推论，是本项目后面一切的地基：
        </p>
        <p>
          <strong>推论一：孤立的一瓢水没用。</strong>倒一下，它漏得比积得快，
          水位还没到线就下去了。<br>
          <strong>推论二：持续倒水才有用。</strong>而且倒的速率必须超过漏水速率，
          水位才能持续上涨。
        </p>
      </div>
    </div>
    <div class="layer">
      <div class="tag"><b>形式</b>专业表述</div>
      <div class="body">
        <p>
          <strong>漏积分-发放</strong>（leaky integrate-and-fire, LIF）是最经典的
          脉冲神经元模型。离散化形式（步长 <span class="mono">Δt</span>）：
        </p>
        <div class="eq">
          G ← leak · G + gain · C + tonic &nbsp;&nbsp;<span class="cmt">// 积分 + 漏</span><br>
          若 G ≥ threshold &nbsp;→&nbsp; 发放，且 G ← 0 &nbsp;&nbsp;<span class="cmt">// 发放 + 重置</span>
        </div>
        <p>
          其中 <span class="mono">leak = exp(−Δt/τ)</span> 是<strong>膜时间常数</strong>
          <span class="mono">τ</span> 离散化的结果，<span class="mono">C</span> 是
          本 tick 从所有上游发放细胞汇入的电流（<span class="mono">C = Σ 上游脉冲 × 突触符号</span>），
          <span class="mono">gain</span> 是突触增益，<span class="mono">tonic</span> 是恒定的紧张性输入。
        </p>
        <p>
          这是一个<strong>无状态记忆的一阶低通滤波器 + 硬阈值</strong>，
          本质上是一阶无限脉冲响应（IIR）系统串联一个比较器。
        </p>
      </div>
    </div>
    <div class="layer">
      <div class="tag"><b>本项目</b>具体数字</div>
      <div class="body">
        <div class="calc">Δt        = {{TICK_MS}} ms = 0.02 s
τ         = 0.1 s          <span class="c"># 膜时间常数</span>
leak      = exp(−0.02/0.1) = exp(−0.2) = <span class="r">{{LEAK}}</span>
gain      = 3.0
threshold = 1.0
tonic     = 0.0            <span class="c"># 没有额外的基础兴奋</span></div>
        <p>
          另外每个神经元还有<strong>泊松自发噪声</strong>
          （约 1.2 Hz、幅度 0.22），模拟真实神经元不等输入也会偶尔发放。
          但<strong>每个脑有自己的随机流</strong>——这一点后来变得极其重要，
          见 <a href="#s14">第 14 节</a>。
        </p>
      </div>
    </div>
  </div>

  <h3>关键常数逐项拆开</h3>
  <div class="tw"><table>
    <thead><tr><th>常数</th><th>值</th><th>它的物理含义</th><th>改大/改小会怎样</th></tr></thead>
    <tbody>
      <tr><td class="mono">Δt</td><td class="n">{{TICK_MS}} ms</td>
          <td>时间分辨率。真实巨型纤维通路延迟是 1.4–25 ms，而我们固定 20 ms</td>
          <td>更小更接近生物，但计算量线性上升</td></tr>
      <tr><td class="mono">τ</td><td class="n">0.1 s</td>
          <td>膜电位漏掉 63% 所需时间</td>
          <td>τ 越大越"记仇"（漏得慢），越容易被单个刺激推动</td></tr>
      <tr><td class="mono">leak</td><td class="n">{{LEAK}}</td>
          <td>每个 tick 保留 81.9% 的旧电位</td>
          <td>——</td></tr>
      <tr><td class="mono">gain</td><td class="n">3.0</td>
          <td>一条突触输入能把电位抬多高</td>
          <td>它进了 <code>need</code> 的分母：gain 越大，越容易触发</td></tr>
      <tr><td class="mono">threshold</td><td class="n">1.0</td>
          <td>发放线</td>
          <td>——</td></tr>
    </tbody>
  </table></div>

  <div class="box trap">
    <div class="lab">容易搞错的地方</div>
    <p>
      <code>leak</code> 是<strong>保留</strong>比例，不是漏掉比例。
      每个 tick <strong>漏掉</strong>的是 <span class="mono">1 − leak =
      0.181269</span>，也就是 18.127%。这两个数在后面反复出现，
      写反了会得出完全错误的结论（我就写反过一次）。
    </p>
  </div>

  <details>
    <summary>自测：如果我把 τ 从 0.1 s 改成 0.2 s，leak 变成多少？这对"神经元有多难被推动"意味着什么？</summary>
    <div class="ans">
      <p><span class="mono">leak = exp(−0.02/0.2) = exp(−0.1) = 0.904837</span>。</p>
      <p>
        漏掉的比例从 18.1% 降到 <span class="mono">1 − 0.9048 = 9.5%</span>，
        也就是<strong>每个 tick 漏得更少了</strong>，电位累积得更容易。
      </p>
      <p>
        量化地看：持续发放所需的输入
        <span class="mono">need = (1−leak)·threshold/gain</span>
        从 0.060423 降到 <span class="mono">0.095163/3 = 0.031721</span>，
        <strong>差不多只要原来一半的驱动就能让神经元发放</strong>。
      </p>
      <p class="q">
        换句话说 <span class="mono">need</span> 与 <span class="mono">1/τ</span> 成正比：
        膜时间常数越大，神经元越"黏"，越容易记住历史输入。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 04 -->
<section class="sec" id="s04">
  <div class="eyebrow">04</div>
  <h2>那个门槛：need = 0.06042</h2>
  <div class="thread">
    这是全套东西的<b>"汇率"</b>。所有感觉驱动的强弱，最后都要拿它换算成
    "够不够触发"——不够就是 0，够了才开始发放。
  </div>

  <h3>怎么推出来的</h3>
  <p>
    假设某个神经元<strong>每个 tick 都收到恒定的输入</strong> <span class="mono">C</span>。
    那么电位会收敛到一个稳态。令稳态时不再增长（即 <span class="mono">G</span>
    在本 tick 前后相等），解出：
  </p>
  <div class="eq">
    G<sub>∞</sub> = leak · G<sub>∞</sub> + gain · C<br>
    G<sub>∞</sub>(1 − leak) = gain · C<br>
    G<sub>∞</sub> = gain · C / (1 − leak)
    &nbsp;&nbsp;<span class="cmt">// 漏积分带来 1/(1−leak) ≈ 5.5 倍的放大</span>
  </div>
  <p>
    要能发放，就得 <span class="mono">G<sub>∞</sub> ≥ threshold</span>，于是解出
    所需的最小恒定输入：
  </p>
  <div class="eq">
    <span class="hl">need = (1 − leak) · threshold / gain = 0.181269 × 1.0 / 3.0 = {{NEED}}</span>
  </div>

  <div class="box key">
    <div class="lab">这个数的地位</div>
    <p>
      <strong>它不是调出来的，是这套动力学算出来的。</strong>
      只要你接受 leak / gain / threshold 这三个数，0.06042 就是唯一的结论。
    </p>
    <p>
      它的含义是：<em>一个神经元要被"持续驱动"到发放，上游每 tick 至少要送来
      0.06042 的等效电流。</em>低于这个值，无论你灌多久，它都只会越来越接近发放线
      但永远不发放。
    </p>
  </div>

  <div class="box work">
    <div class="lab">手算示例 · 自己核对一遍</div>
    <p>
      给一个 DNp01 神经元灌<strong>恒定驱动</strong>，看它的电位怎么走。
      递推式 <span class="mono">G ← 0.818731·G + 3·C</span>。
    </p>
    <div class="calc"><span class="c">情形 A：驱动 0.05（太弱）</span>
tick1  0.818731×0 + 3×0.05 = 0.150
tick2  0.818731×0.150 + 0.15 = 0.273
tick3  0.818731×0.273 + 0.15 = 0.373
tick4  0.456      tick5  0.523      tick6  0.578
...
稳态  3×0.05/(1−0.818731) = 0.8275   <span class="r">→ 永远不发放</span>

<span class="c">情形 B：驱动恰好 = need = 0.060423</span>
tick3  0.451      tick6  0.699      tick9  0.835
稳态  3×0.060423/0.181269 = 1.0000   <span class="r">→ 仍然永远不发放</span>

<span class="c">情形 C：真实工作点（见第 6、7 节怎么算出来）</span>
驱动 = amp 0.97 × 腹侧总权重 0.1080 = 0.1048
tick1  0.314   tick2  0.572   tick3  0.782   tick4  0.955
tick5  1.096 ≥ 1.0  <span class="r">→ 第 5 个 tick 发放，电位归零</span>
之后每 5 个 tick 复现一次         <span class="c">（周期 = 100 ms）</span></div>
    <p>
      情形 C 的完整轨迹画成了图（<a href="#s05">第 5 节</a>最后一张），
      你可以拿纸笔把上面那几行逐个核对。
    </p>
  </div>

  <div class="box trap">
    <div class="lab">情形 B 是个陷阱，值得单独记一句</div>
    <p>
      <strong>恰好等于 need 时，神经元永远不发放。</strong>
      因为 <span class="mono">need</span> 是从"稳态<strong>等于</strong>阈值"解出来的，
      而渐近线只能无限逼近、不能达到。要真的发放，驱动必须
      <strong>严格大于</strong> need。
    </p>
    <p>
      我第一版文档就是拿"恰好 = need"当算例的，结果写出来的序列
      <strong>永远不发放</strong>，自己看着都别扭 —— 那是个会把人教糊涂的例子，
      已换成上面的情形 C。
    </p>
  </div>

  <details>
    <summary>自测：把 gain 从 3.0 提到 6.0，need 变成多少？对那些"离触发只差一点"的细胞意味着什么？</summary>
    <div class="ans">
      <p><span class="mono">need = 0.181269 × 1.0 / 6.0 = 0.030211</span>，正好减半。</p>
      <p>
        意味着<strong>同样的感觉驱动会触发更多细胞、也更早触发</strong>。
        在我们的控制里，这直接改变"什么时候开始拍翅膀"，
        进而改变鸟的平均高度。所以 <span class="mono">gain</span> 是个
        敏感参数，不是随便设的。
      </p>
      <p class="q">
        注意区分：<span class="mono">gain</span> 是<strong>突触增益</strong>（连接组之外的常数），
        而感觉驱动 <span class="mono">amp</span> 是<strong>外部输入强度</strong>。
        两者都在 <span class="mono">need</span> 的换算里起作用，但可调性是不同层级的。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 05 -->
<div class="chapter"><span class="ctitle">第三部 · 感觉与指令</span></div>
<section class="sec" id="s05">
  <div class="eyebrow">05</div>
  <h2>两条视觉通道：LC4 与 LPLC2</h2>
  <div class="thread">
    果蝇用<b>两组不同的细胞分别编码"多快"和"多大"</b>，
    再把它们一起送给两根巨纤维。我们只是接上这个真实的接口。
  </div>

  <div class="layers">
    <div class="layer">
      <div class="tag"><b>直觉</b>人话版</div>
      <div class="body">
        <p>
          一个东西朝你飞过来时，你脑子里同时得到两条信息：
          <strong>"它变大的速度有多快"</strong>和<strong>"它现在看起来有多大"</strong>。
        </p>
        <p>
          果蝇把这两件事交给了两组不同的细胞去做：
        </p>
        <p>
          一组专门测<strong>"画面在膨胀得多快"</strong>（相当于角速度）；
          另一组专门测<strong>"那个东西现在占多大视野"</strong>（相当于角尺寸）。
        </p>
        <p>
          有意思的是，这两组细胞最后<strong>汇到同样两根输出神经元上</strong>。
          所以"逼近"这个判断不是某一组细胞单独做的，
          而是这两路信息在里面<strong>合起来</strong>做的。
        </p>
      </div>
    </div>
    <div class="layer">
      <div class="tag"><b>形式</b>专业表述</div>
      <div class="body">
        <p>
          视觉运动信息在果蝇的<strong>小叶板</strong>（lobula plate）与
          <strong>小叶</strong>（lobula）里被并行处理：
        </p>
        <p>
          <strong>LC4</strong>（lobula columnar 4）是<strong>角速度</strong>
          敏感细胞（angular-velocity sensitive），对视野内物体的扩张速率响应，
          属于小叶柱状细胞（lobula columnar, LC）家族。
        </p>
        <p>
          <strong>LPLC2</strong>（lobula plate/lobula columnar 2）是
          <strong>逼近敏感</strong>细胞（looming sensitive），
          编码角尺寸及其变化，是公认的逼近检测器。
        </p>
        <p>
          二者共同投射到<strong>巨纤维系统</strong>（giant fiber system, GF）——
          果蝇的逃逸指令通路。在 DNp01（两根巨纤维）的直接视觉输入中，
          这两组加起来占 <strong>98.5%</strong>（Ache 2019；Gaitanidis 2025）。
          这个比例是这个项目选择它们、而不是别的细胞群的<strong>唯一理由</strong>。
        </p>
      </div>
    </div>
    <div class="layer">
      <div class="tag"><b>本项目</b>具体数字</div>
      <div class="body">
        <div class="tw"><table>
          <thead><tr><th>细胞群</th><th>数量</th><th>编码</th><th>在本项目里的角色</th></tr></thead>
          <tbody>
            <tr>
              <td><span class="sw" style="background:#c2410c"></span><b>LC4</b></td>
              <td class="n">104</td>
              <td>角速度（扩张多快）</td>
              <td>告诉脑"有多急"</td>
            </tr>
            <tr>
              <td><span class="sw" style="background:#7c3aed"></span><b>LPLC2</b></td>
              <td class="n">210</td>
              <td>角尺寸（看起来多大）</td>
              <td>告诉脑"有多近"</td>
            </tr>
            <tr>
              <td><span class="sw" style="background:#be123c"></span><b>DNp01</b></td>
              <td class="n">2</td>
              <td>——</td>
              <td><strong>指令读出</strong>：两根巨纤维</td>
            </tr>
          </tbody>
        </table></div>
        <p>
          注意 <strong>DNp01 只有 2 个细胞</strong>。所以整个控制器的输出维度是 2，
          而我们只读它们"最近 100 ms 内有没有发放过"这一个比特。
        </p>
      </div>
    </div>
  </div>

  <figure>
    {{SVG_MEMBRANE}}
    <figcaption>
      <span class="n">图 2 · 漏积分的实际形状</span>
      <b>这是按公式手算出来的，不是仿真截图</b>——所以你可以拿纸笔核对每一个点。
      恒定驱动 0.1048 时的电位轨迹：每个 tick 先漏掉 18.127%，再加上 3 × 0.1048。
      第 {{MEM_FIRST}} 个 tick 越过 1.0 发放并归零，之后每 {{MEM_GAP}} 个 tick 复现一次。
      <br>蓝色虚线是渐近值 <span class="mono">gain·C/(1−leak) = 1.734</span>；
      红色虚线是阈值 1.0。<strong>两条线的相对位置决定它发不发放</strong>——
      渐近值低于阈值就永远不发（对比 <a href="#s04">04 节</a>的情形 A/B）。
    </figcaption>
  </figure>

  <details>
    <summary>自测：为什么 DNp01 只有 2 个细胞，却足以当"指令"？</summary>
    <div class="ans">
      <p>
        因为<strong>它的功能角色是汇聚点，不是表示容量的来源</strong>。
      </p>
      <p>
        巨纤维系统在生物学上就是一个<strong>逃逸扳机</strong>：
        它的任务是把"很多路感觉证据"压缩成一个二值决定——逃，还是不逃。
        这种"高汇聚、低维度"的结构正是指令神经元（command neuron）的定义特征。
      </p>
      <p>
        反过来说，如果它有两千个细胞，我们反而不该只读一个比特，
        因为那意味着信息在输出端还保持着高维结构。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 06 -->
<section class="sec" id="s06">
  <div class="eyebrow">06</div>
  <h2>把游戏几何变成神经驱动</h2>
  <div class="thread">
    这是整套系统里<b>唯一"我们定义"而不是"生物学测量"</b>的一环 ——
    所以它必须被明确标出来，并接受质疑。
  </div>

  <div class="layers">
    <div class="layer">
      <div class="tag"><b>直觉</b>人话版</div>
      <div class="body">
        <p>
          游戏里没有"图像"，只有三个数字：鸟的高度、前方缺口的中心在哪、
          离地面多远。可神经元需要的是"电流"。
        </p>
        <p>
          所以我们要造一个翻译器：<strong>把"距离"翻译成"看起来有多大"，
          再翻译成"该灌多少电流"</strong>。
        </p>
        <p>
          翻译"距离 → 看起来多大"用的是很朴素的几何：东西离你越近，它占的视野越大。
          而"看起来多大 → 电流"用的是一条<strong>饱和曲线</strong>：
          小的时候涨得快，大了就涨不动了（因为已经占满视野了）。
        </p>
      </div>
    </div>
    <div class="layer">
      <div class="tag"><b>形式</b>专业表述</div>
      <div class="body">
        <p>第一步，用标准的小角度/天文测距公式把线距离换成视角：</p>
        <div class="eq">θ = 2 · atan(r / D) &nbsp;&nbsp;<span class="cmt">// r = 物体特征半径，D = 距离</span></div>
        <p>第二步，用 <strong>Naka-Rushton</strong> 饱和函数把角尺寸换成归一化响应：</p>
        <div class="eq">
          amp = gain · θ<sup>n</sup> / (θ<sup>n</sup> + s<sub>50</sub><sup>n</sup>)
          &nbsp;&nbsp;<span class="cmt">// 希尔型饱和，n = 3，s<sub>50</sub> = 30°</span>
        </div>
        <p>
          这条曲线是视觉生理学里的标准工具：<span class="mono">s<sub>50</sub></span>
          是半饱和点（响应达到 50% 时的刺激强度），
          <span class="mono">n</span> 控制拐点陡峭程度。
          <span class="mono">n = 3</span> 意味着它<strong>接近开关</strong>：
          在 s<sub>50</sub> 附近响应变化极快。
        </p>
        <p>
          第三步，把 <span class="mono">amp</span> 当作<strong>强制发放概率</strong>
          灌给那一群细胞（代码里的 <span class="mono">clamp</span>）：
          我们不去手工模拟光感受器的级联细节，而是直接指定
          "这 104 个 LC4 各自以多大概率发放"。
        </p>
      </div>
    </div>
    <div class="layer">
      <div class="tag"><b>本项目</b>具体数字</div>
      <div class="body">
        <div class="calc">r     = 0.55 m      <span class="c"># 物体特征半径</span>
PX_PER_M = {{PX_PER_M}} px/m   <span class="c"># 像素/米换算</span>
n     = 3.0         s50 = 30.0°

<span class="c"># 距离 → 角尺寸 → 驱动</span>
D = 2.00 m (480 px)   θ = 30.75°   amp = 0.5186
D = 1.00 m (240 px)   θ = 57.62°   amp = 0.8763
D = 0.50 m (120 px)   θ = 95.45°   amp = 0.9699
D = 0.25 m ( 60 px)   θ = 131.11°  amp = 0.9882</div>
        <p>
          看最后两行：距离减半，<span class="mono">amp</span> 只从 0.9700 涨到 0.9882。
          <strong>饱和了。</strong>这意味着在很近的距离上，
          "更近一点"带来的额外神经驱动<strong>几乎没有</strong>——
          脑必须靠别的东西判断该不该拍（那就是 <a href="#s08">第 8 节</a>的方向机制）。
        </p>
      </div>
    </div>
  </div>

  <div class="box trap">
    <div class="lab">这一环的诚实声明</div>
    <p>
      <strong>s<sub>50</sub> = 30° 是一个外部假设。</strong>
      我们没有找到 LC4/LPLC2 的实测发放率数据可以反推它。
      它可调、会改变触发的早晚，但<strong>不改变</strong>"存在一个阈值、
      而且这个阈值由那条真实通路携带"这个核心结论。
    </p>
    <p>
      同时也要说清：<span class="mono">clamp</span>（强制指定发放概率）
      是一种工程近似。严格做法是把光照→光感受器→层状神经元→LC/LPLC2
      的整条级联都模拟出来，但那会引入<strong>远比这个近似更多的未知参数</strong>。
      我们选择把不确定性集中在一个显式的地方。
    </p>
  </div>

  <details>
    <summary>自测：如果把 s50 从 30° 改成 60°，同一距离下的 amp 会变大还是变小？</summary>
    <div class="ans">
      <p>
        <strong>变小。</strong>
        <span class="mono">s<sub>50</sub></span> 在分母上：
        <span class="mono">amp = θ³/(θ³ + 60³)</span>。
      </p>
      <p>
        物理含义：<span class="mono">s<sub>50</sub></span> 变大 =
        这个神经元"要更大的东西才兴奋" = <strong>更不敏感</strong>。
        所以同一距离下驱动更小，鸟要更靠近才触发拍翅 →
        平均飞得更低、更容易撞下管。
      </p>
      <p class="q">
        算例核对：θ = 57.62°（1 m）时，
        s50=30 给 0.8763；s50=60 给
        <span class="mono">57.62³/(57.62³+60³) = 191,300/407,300 = 0.4697</span>。
        差了将近一倍。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 07 -->
<section class="sec" id="s07">
  <div class="eyebrow">07</div>
  <h2>逼近期望与那个必须算清的判据</h2>
  <div class="thread">
    要让 DNp01 发放，灌进去的驱动必须<b>换算成等效权重之后超过 need</b>。
    这一节就是把"够不够"算成一个数。
  </div>

  <h3>为什么要"换算成等效权重"</h3>
  <p>
    我们不是直接给 DNp01 灌电流，而是给<strong>上游的感觉细胞</strong>灌。
    感觉细胞发放后，要经过连接组的突触才能影响 DNp01。
    所以关键是这条链的<strong>总等效权重</strong>：
  </p>
  <div class="eq">
    W = Σ<sub>i</sub> p<sub>i</sub> · w<sub>i</sub>
    &nbsp;&nbsp;<span class="cmt">// p<sub>i</sub> = 驱动概率（= amp），w<sub>i</sub> = 突触权重</span>
  </div>
  <p>
    因为我们是"整半招募"（整个半视野一起点亮，见 <a href="#s08">第 8 节</a>），
    所以 <span class="mono">p<sub>i</sub></span> 在这一半里几乎相同，
    于是 <span class="mono">W ≈ amp · Σw</span>，只取决于这一半的<strong>权重和</strong>。
  </p>

  <div class="box work">
    <div class="lab">算一遍：半边视野到底推不推得动 DNp01</div>
    <p>从连接组里把 LC4 / LPLC2 → DNp01 的突触权重按视野上下半分别求和：</p>
    <div class="calc">LC4   → DNp01 全群权重和 = 0.0744    腹侧半 = 0.0328
LPLC2 → DNp01 全群权重和 = 0.1043    腹侧半 = 0.0752

<span class="c"># 腹侧（触发拍翅的那一半）</span>
Σw = 0.0328 + 0.0752 = 0.1080
W  = amp × Σw = 0.97 × 0.1080 = 0.1048
need = {{NEED}}
<span class="r">0.1048 / 0.06042 = 1.73 倍 need  → 推得动，且有余量</span>

<span class="c"># 背侧（触发下坠的那一半）</span>
Σw = 0.0416 + 0.0291 = 0.0707
<span class="r">0.0707 / 0.06042 = 1.17 倍 need  → 也能推动，但余量小得多</span></div>
    <p>
      这个 <strong>1.73 倍 vs 1.17 倍的不对称</strong>不是巧合，
      它正是"两个方向行为不同"的物理来源：腹侧到 DNp01 的权重本来就是背侧的
      2.6 倍。这是连接组的事实，不是我们设计的。
    </p>
  </div>

  <div class="box trap">
    <div class="lab">为什么不能用"平滑空间窗"</div>
    <p>
      最直觉的做法是给每个细胞一个平滑权重（离缺口中心越近的细胞驱动越强）。
      实测结果：<strong>只到 need 的 53%，根本不发放。</strong>
    </p>
    <p>
      原因不难理解：平滑窗把总权重<strong>摊薄</strong>了——
      每个细胞只分到一点点，谁都过不了阈值。
      而"整半招募"把 <span class="mono">Σw</span> 完整地用在一半细胞上，
      实测到 1.79 倍 need，才推得动。
    </p>
    <p>
      这是个很典型的教训：<strong>在阈值系统里，均匀分布不如集中</strong>。
      阈值是非线性的，"稍微给所有细胞都加一点"和"给一半细胞加满"
      产生的效果完全不同。
    </p>
  </div>

  <details>
    <summary>自测：如果腹侧权重和因为数据更新变成了 0.0900（比现在小），会发生什么？</summary>
    <div class="ans">
      <p>
        <span class="mono">W = 0.97 × 0.0900 = 0.0873</span>，
        <span class="mono">0.0873 / 0.06042 = 1.44 倍 need</span> ——
        <strong>还是会发放</strong>，但余量从 73% 降到 44%。
      </p>
      <p>会发生的具体后果：</p>
      <p>
        ① 电位上升变慢 → 拍翅<strong>响应变迟钝</strong>（从"5 个 tick 内触发"
        拖到更久），鸟对缺口的反应滞后；<br>
        ② 在噪声大的时候，<strong>漏发</strong>的概率上升（信号裕度变小）；
        ③ 综合效果是平均飞行高度下降、撞下管的概率上升。
      </p>
      <p class="q">
        这解释了为什么 <span class="mono">need</span> 这个数值得单独花一节讲：
        它是所有"够不够"判断的标尺，也是理解控制裕度的入口。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 08 -->
<div class="chapter"><span class="ctitle">第四部 · 反射</span></div>
<section class="sec" id="s08">
  <div class="eyebrow">08</div>
  <h2>双向逼近反射：方向由几何决定，不是控制器算出来的</h2>
  <div class="thread">
    这是整个算法最容易误解的地方：<b>它没有目标高度、没有误差运算、没有 PID。</b>
    它只做一件事——视野的哪一半被逼近刺激占住，就点亮那一半。
  </div>

  <div class="layers">
    <div class="layer">
      <div class="tag"><b>直觉</b>人话版</div>
      <div class="body">
        <p>
          一只果蝇感觉有东西冲过来时会逃。但"逃"是有方向的：
          威胁在上方，它往下躲；威胁在下方，它往上飞。
        </p>
        <p>
          关键在于：<strong>这个方向不是脑子"算"出来的，而是几何本身给的。</strong>
        </p>
        <p>
          缺口（安全通道）在鸟的上方，就意味着"危险"来自下方——
          视野下半部分被逼近占住了。于是点亮下半部分的细胞，
          而这一半恰好连到"拍翅爬升"。
        </p>
        <p>
          反过来，缺口在下方 → 视野上半部分被占住 → 点亮上半部分的细胞 →
          不拍翅、下坠。
        </p>
      </div>
    </div>
    <div class="layer">
      <div class="tag"><b>形式</b>专业表述</div>
      <div class="body">
        <p>规则本身非常短：</p>
        <div class="eq">
          Δ = 缺口中心 − 鸟的高度<br>
          若 Δ &gt; gap_margin &nbsp;→&nbsp; 驱动<strong>腹侧</strong>半视野（拍翅爬升）<br>
          若 Δ &lt; −gap_margin &nbsp;→&nbsp; 驱动<strong>背侧</strong>半视野（不拍，下坠）<br>
          否则 &nbsp;→&nbsp; <strong>什么都不驱动</strong>（死区）
        </div>
        <p>
          注意这里的"腹侧/背侧"指的是<strong>视野</strong>（visual field）的上下半，
          不是鸟身体的上下。果蝇视野上下半投射到不同的叶区，
          而腹侧与背侧两半到 DNp01 的突触权重<strong>本来就不对等</strong>——
          所以同一套机制自然给出两个方向的行为。
        </p>
        <p>
          更形式化地，这是一个<strong>双向逼近反射</strong>：
          刺激的空间位置直接映射到不同的运动程序，
          映射关系写在连接组里，无需显式的误差反馈。
        </p>
      </div>
    </div>
    <div class="layer">
      <div class="tag"><b>本项目</b>具体参数</div>
      <div class="body">
        <div class="calc">gap_margin  = 18 px    <span class="c"># 死区半宽</span>
vy_gate     = 1        <span class="c"># 接近速度门控：地面只在下落时算逼近</span>
ceil_boost  = 1.0
dors_scale  = 0.35     <span class="c"># 背侧驱动缩放</span>
vent_gain   = 2.0      <span class="c"># 腹侧驱动增益</span>
vent_dev    = 120.0 px <span class="c"># 腹侧招募的半宽</span>
groups      = ["LC4", "LPLC2"]</div>
        <p>
          其中 <span class="mono">dors_scale</span> 与 <span class="mono">vent_gain</span>
          是<strong>我们手写的</strong>，不是连接组里的东西。
          它们作用在脑的<strong>输入端</strong>（感觉信号有多强），
          连接组内部结构一个都没改。
        </p>
      </div>
    </div>
  </div>

  <h3>三个不能省的细节</h3>
  <dl class="kv">
    <dt>死区 · 18 px</dt>
    <dd>
      缺口中心在鸟上下 18 px 之内时<strong>什么都不驱动</strong>。
      第一版没有死区，鸟刚越过缺口中心就翻转成"要爬升"，
      于是在缺口上下反复横跳 —— 实测均分只有 <span class="mono">0.81</span>。
      死区的作用是提供<strong>滞环</strong>（hysteresis），消除抖动。
    </dd>
    <dt>vy_gate · 门控</dt>
    <dd>
      地面只在下落时才算"逼近"，天花板只在上升时才算。
      这是纯物理的：你正在远离一个面，那个面并没有朝你逼近。
      去掉它，鸟在上冲时会把地面当成威胁，行为立刻崩坏。
    </dd>
    <dt>整半招募</dt>
    <dd>
      整个半视野一起亮，而不是给平滑的权重斜坡。
      原因见 <a href="#s07">第 7 节</a>：平滑窗只到 need 的 53%，根本不发放。
    </dd>
  </dl>

  <figure>
    {{SVG_RASTER}}
    <figcaption>
      <span class="n">图 3 · 真实发放栅格</span>
      <b>从跑着的仿真里取的 {{N_TICKS}} 个 tick，不是示意图。</b>
      上两行是 LC4 与 LPLC2 的代表细胞，第三行是全部 2 根 DNp01，
      最下面那条是鸟的高度轨迹，<strong>红竖线 = 一次拍翅</strong>。
      <br>在这 {{N_TICKS}} 个 tick 里，DNp01 一共发放 {{N_DN}} 次、
      触发 {{N_FLAP}} 次拍翅。可以看到拍翅<strong>不是每个 tick 都拍</strong>，
      而是被那个 0.06042 的阈值筛过的——两次拍翅之间往往隔着好几个 tick 的重积分。
      <br><span class="q">注：LC4/LPLC2 是均匀降采样后的代表细胞（每组 26 个），
      否则一张图会有七千个点；DNp01 只有 2 个，是全画的。</span>
    </figcaption>
  </figure>

  <details>
    <summary>自测：为什么"死区"能解决抖动？用滞环的概念解释。</summary>
    <div class="ans">
      <p>
        没有死区时，判定是"缺口在上方 → 爬"，这是一个<strong>无记忆的二值判断</strong>：
        鸟只要越过缺口中心哪怕 0.1 px，判断就翻转。
        于是鸟在缺口中心附近会：拍一下 → 越过去 → 判断变成"下坠" → 掉下来 →
        又变成"爬" → 拍一下 …… <strong>自激振荡</strong>。
      </p>
      <p>
        加了 ±18 px 的死区后，<strong>切换方向需要的位移变大了</strong>：
        要越过中心 18 px 才会翻转。这就在"爬"和"下坠"之间造出了一个
        <strong>状态重叠区</strong>——这就是滞环。
      </p>
      <p>
        代价是控制精度下降（鸟不再精确对准缺口中心，而是对准一个 ±18 px 的带），
        但收益是<strong>稳定性</strong>。实测 0.81 → 可用水平，这个交换非常划算。
      </p>
      <p class="q">
        这是控制工程里的标准手段：<strong>用少量精度换稳定性</strong>。
        继电器、温控器、施密特触发器都是同一个思路。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 09 -->
<section class="sec" id="s09">
  <div class="eyebrow">09</div>
  <h2>回路闭合：从驱动到拍翅，一直到物理</h2>
  <div class="thread">
    这一节把前面所有环节<b>串成一个可执行的循环</b>，
    并给出每一步的实测数字。
  </div>

  <div class="eq">
    <span class="cmt">// Session._game_tick() 的骨架（顺序不能变）</span><br>
    ① 读世界：y, vy, 最近缺口中心 gap, 离地高度<br>
    ② 算驱动：<span class="hl">amp</span> = NakaRushton(角度(gap − y))<br>
    ③ 定方向：Δ &gt; +18 px → 腹侧；Δ &lt; −18 px → 背侧；否则不驱动<br>
    ④ 灌脑：给这一半的 LC4/LPLC2 以概率 <span class="hl">amp</span> 强制发放<br>
    ⑤ 走脑：<span class="mono">brain.step()</span> —— 144,837 个神经元各更新一次<br>
    ⑥ 读出：DNp01 最近 5 个 tick 内发放 ≥ 1 次？<br>
    ⑦ 门槛：拍翅冷却 0.14 s 到了没？<br>
    ⑧ 走物理：<span class="mono">vy = −340</span>（若拍）→ <span class="mono">vy += 1180·0.02</span> → <span class="mono">y += vy·0.02</span>
  </div>

  <h3>为什么要看"最近 5 个 tick"而不是当前那一个</h3>
  <p>
    因为它是<strong>漏积分</strong>的。DNp01 发放之后电位归零，
    要重新积起来需要几个 tick。如果我们只看当前这一个 tick 有没有发放，
    那么在一次发放之后紧接着的几帧就一定是"不拍"，
    控制器会变得对相位极其敏感（dispersion）。
  </p>
  <p>
    用"最近 100 ms 的窗口里发放过就算"（代码里 <code>WINDOW = 5</code>），
    等价于在输出端加了一个<strong>短时记忆</strong>，
    把"发放发生在哪个 tick"这件事抹平了，控制因此稳定得多。
  </p>

  <div class="box work">
    <div class="lab">实测：一轮完整的驱动 → 发放 → 拍翅</div>
    <div class="calc"><span class="c"># 取自图 3 那一局的前若干 tick（seed=7）</span>
tick    y      gap     Δ      驱动方向    amp     DNp01  拍翅
─────────────────────────────────────────────────────────────
  0   300.0   ...     ...     —          ...     —      <span class="c">(加速期，不判定)</span>
  ── 加速期 0.8 s = 40 tick 结束 ──
 41   ...    ...     +...    腹侧       0.94    <span class="r">● 发放</span>   否
 42   ...    ...     +...    腹侧       0.95    ·       <span class="r">是</span>
 ...
<span class="c"># 关键观察：DNp01 不是每 tick 都发，因为电位要重新积分；</span>
<span class="c"># 而"拍翅"通过 5-tick 窗口把它变成了一段稳定的指令。</span></div>
    <p class="q">
      完整的逐 tick 数据可以用
      <span class="mono">scripts/verify_server_gameplay.py --trace</span>
      打出来，或者看 <span class="mono">output/doc_figs/spikes.json</span>。
    </p>
  </div>

  <details>
    <summary>自测：如果把 WINDOW 从 5 改成 1（只看当前 tick），控制器会怎样？</summary>
    <div class="ans">
      <p>
        会变得<strong>对相位过度敏感</strong>，而且更容易漏发。
      </p>
      <p>
        具体机制：DNp01 发放的周期大约是 5 个 tick（图 2 算出来的）。
        如果用窗口 1，那么"这一 tick 恰好有发放"只覆盖了 1/5 的相位，
        其余 4/5 的相位里它一律输出"不拍"。
      </p>
      <p>
        后果：<strong>鸟的拍翅频率会掉到原来的约 1/5</strong>，
        爬升能力大幅下降；而且因为相位敏感，
        感觉驱动强度的微小变化会导致行为跳变（不连续）。
      </p>
      <p class="q">
        这类"输出端加短时记忆"的手法在神经工程里很常见，
        本质上是一个<strong>速率编码 → 二值解码</strong>的转换器。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 10 -->
<div class="chapter"><span class="ctitle">第五部 · 游戏世界</span></div>
<section class="sec" id="s10">
  <div class="eyebrow">10</div>
  <h2>游戏物理与那个"看着撞上了却没死"的 bug</h2>
  <div class="thread">
    物理本身故意做得最朴素；真正的坑在<b>表示法边界</b>——
    像素 vs 连续坐标，闭区间 vs 开区间。
  </div>

  <h3>几何与物理参数</h3>
  <div class="tw"><table>
    <thead><tr><th>量</th><th>值</th><th>说明</th></tr></thead>
    <tbody>
      <tr><td>画布</td><td class="n">{{G_W}} × {{G_H}}</td><td>地面高 {{GROUND}} px</td></tr>
      <tr><td>鸟</td><td class="n">x = {{BIRD_X}}, r = {{BIRD_R}} px</td><td>只有圆心位置变，水平不动</td></tr>
      <tr><td>管子</td><td class="n">宽 {{PIPE_W}} px，缺口高 {{GAP}} px</td><td>间距 {{SPACING}} px</td></tr>
      <tr><td>左移速度</td><td class="n">{{PX_PER_M}} px/s</td><td>= 1 m/s，所以 1 m = {{PX_PER_M}} px</td></tr>
      <tr><td>重力</td><td class="n">{{GRAV}} px/s²</td><td>严重超标（真实 9.8 m/s²），是游戏手感</td></tr>
      <tr><td>拍翅速度</td><td class="n">−340 px/s</td><td>直接置位，不是加力</td></tr>
      <tr><td>拍翅冷却</td><td class="n">0.14 s</td><td>一个管距间隔内最多 8 次</td></tr>
      <tr><td>加速期</td><td class="n">{{WARM}} s</td><td>开局不判定碰撞</td></tr>
    </tbody>
  </table></div>

  <div class="box work">
    <div class="lab">手算：拍一次翅能爬多高、爬多久</div>
    <div class="calc">拍翅瞬间   vy = −340 px/s
每个 tick  vy += 1180 × 0.02 = +23.6 px/s
回到最高点  需要 340/1180 = 0.288 s = 14.4 tick
最高上升    vy²/(2·g) = 340²/(2×1180) = 49.0 px

<span class="c"># 所以"一次拍翅买 49 px 高度、花 0.288 s"</span>
<span class="c"># 而两条管子之间是 {{SPACING}}/{{PX_PER_M}} = {{TWEEN}} s = 62 tick</span>
<span class="c"># 一个间隔里理论上能拍 62/7 ≈ 8 次（也受冷却 0.14 s 限制 → 8 次）</span></div>
    <p>
      但实际<strong>可持续爬升率只有约 100 px/s</strong>，远低于"8 次 × 49 px / 1.25 s = 314 px/s"
      的朴素估计。原因见 <a href="#s11">第 11 节</a>：
      上冲期间地面驱动被 <span class="mono">vy_gate</span> 关掉，
      膜电位要从零重新积分，所以拍翅周期被钉在约 0.4 s。
    </p>
  </div>

  <h3>那个 bug：像素与坐标混用</h3>
  <div class="layers">
    <div class="layer">
      <div class="tag"><b>现象</b></div>
      <div class="body">
        <p>玩家报了一个非常具体的现象：</p>
        <p class="q">
          "管子上下两端的体积好像有点偏差，我看已经撞上了，但是没有判定为失败。"
        </p>
      </div>
    </div>
    <div class="layer">
      <div class="tag"><b>根因</b></div>
      <div class="body">
        <p>
          <strong>画和判用了两套语义。</strong>
          <span class="mono">fillRect(y, h)</span> 覆盖的是像素行
          <span class="mono">y … y+h−1</span>（闭区间），
          而判定最初用的是连续坐标下的"圆到矩形最近点距离 ≤ 半径"。
        </p>
        <p>差一两个像素，肉眼就足够觉得"明明撞上了"。</p>
      </div>
    </div>
    <div class="layer">
      <div class="tag"><b>解法</b></div>
      <div class="body">
        <p>把两者都归到<strong>像素行集合</strong>上，然后要求集合有交集：</p>
        <div class="eq">
          <span class="cmt">// 画：</span>上管覆盖 [0, top+R]　下管覆盖 [top+GAP−R−1, …]<br>
          <span class="cmt">// 判：</span>鸟圆周覆盖 [⌈y−R⌉, ⌊y+R⌋]<br>
          <span class="hl">两组像素行有交集 ⇔ 视觉重叠</span>
        </div>
        <p>
          并配了一个<strong>逐像素对账</strong>脚本：把"画到的像素集合"与"判定为撞的
          y 集合"逐个比，要求"只画不判"和"只判不画"<strong>同时为 0</strong>。
        </p>
      </div>
    </div>
  </div>

  <div class="box trap">
    <div class="lab">这类 bug 的一般规律</div>
    <p>
      "看着撞上了却没判"的根因几乎总是在<strong>表示法边界</strong>上：
      像素 vs 坐标、开区间 vs 闭区间、0-based vs 1-based、
      <span class="mono">&lt;=</span> vs <span class="mono">&lt;</span>。
    </p>
    <p>
      它<strong>只能靠逐项对账抓到</strong>，靠读代码读不出来 ——
      因为两边的写法各自都"看起来对"。我们历史上错过两次，
      而且方向相反（一次是 <span class="mono">&lt;= / &gt;=</span> 多算了一段 R，
      一次是 <span class="mono">&lt; / &gt;</span> 少算了一段 R）。
    </p>
  </div>

  <details>
    <summary>自测：为什么"圆心到矩形最近点距离 ≤ 半径"这个判定在数学上是对的，却仍然不够？</summary>
    <div class="ans">
      <p>
        因为它<strong>在连续坐标下是对的</strong>，但渲染不是连续的。
      </p>
      <p>
        屏幕是离散像素阵列。<span class="mono">fillRect</span> 会把某个像素整块涂满，
        所以一个"数学上没碰到"的位置，可能因为像素取整而<strong>视觉上已经重叠</strong>。
        反过来也有：数学上碰到了，但取整之后那一像素没被涂。
      </p>
      <p>
        要消除这个歧义，只有两条路：
        ① 让判定也走像素网格（我们的选择）；
        ② 让渲染也走连续坐标（抗锯齿、亚像素绘制）—— 但那就不是像素风了。
      </p>
      <p class="q">
        推广地说：<strong>当判定与呈现使用不同的离散化方式时，
        必须显式地做一次对账</strong>，否则差异会以"偶发的不合理"形式长期潜伏。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 11 -->
<section class="sec" id="s11">
  <div class="eyebrow">11</div>
  <h2>关卡生成：一个把生成器逼死的约束</h2>
  <div class="thread">
    约束是对的，<b>但实现方式让它退化了</b>。
    这一节也是一个通用的教训：<em>在边界处，随机数会悄悄变成常数。</em>
  </div>

  <h3>约束从哪来</h3>
  <p>
    鸟的可持续爬升率约 <strong>100 px/s</strong>（不是理论上限 314 px/s）。
    管子间距 {{SPACING}} px、速度 {{PX_PER_M}} px/s，
    所以两根管子之间只有 <strong>{{TWEEN}} s</strong>。
    一个间隔内最多净爬约 <strong>125 px</strong>。
  </p>
  <div class="eq">
    下一根缺口中心 ≥ 上一根缺口中心 − max_climb
    &nbsp;&nbsp;<span class="cmt">// 向上受限；向下不设限</span>
  </div>
  <p>
    如果把"向上跳变"放得很大（比如 330 px），<strong>那些关卡物理上根本飞不进去</strong>。
    那不是难度，是 bug。
  </p>

  <h3>它怎么把生成器逼死的</h3>
  <p>原来的实现是"从下方均匀抽样"：</p>
  <div class="eq">
    uniform(lo, min(hi, prev_c + max_climb − GAP/2))
  </div>
  <p>
    看起来和上面的约束等价，但一旦缺口中心下沉到某个位置，
    这个上界就缩到<strong>低于下界</strong>，于是退化成
    <span class="mono">uniform(70, 70)</span> —— 恒等于 70。实测 200 根管子：
  </p>
  <div class="calc">top 范围 205 .. 205     标准差 0.0     上升次数 0 / 199
<span class="r">→ 所有管子一模一样。那不是"不够随机"，是生成器死了。</span></div>

  <figure>
    {{SVG_GAPGEN}}
    <figcaption>
      <span class="n">图 4 · 缺口分布对照</span>
      <b>各 4000 根管子的实际输出。</b>
      修前所有管子落在<strong>同一个值</strong>上（标准差 0.0），
      而且只用量程的一半、长期被顶在下边界附近。
      修后铺满整个量程，标准差约 50 px。
    </figcaption>
  </figure>

  <h3>量程尺：为什么缺口不能随便摆</h3>
  <div class="ruler">
    <div class="bar">
      <div class="band" style="left:19.7%; right:16.4%"></div>
      <div class="mk" style="left:19.7%"></div>
      <div class="mk" style="left:83.6%"></div>
    </div>
    <div class="scale">
      <span style="left:19.7%">top = 70（贴顶）</span>
      <span style="left:83.6%">top = 250（贴近地面）</span>
    </div>
    <dl class="why">
      <dt>上方</dt>
      <dd>受<strong>执行器带宽</strong>约束：一个间隔只有 {{TWEEN}} s，
          而鸟的可持续爬升约 100 px/s，所以相邻缺口最多上移约 125 px
          （我们取 <span class="mono">max_climb = {{MAX_CLIMB}}</span> 留余量）。
          超出这个带宽的关卡<strong>物理上飞不进去</strong>——那不是难度，是 bug。</dd>
      <dt>下方</dt>
      <dd>不是可达性问题（自由落体 {{TWEEN}} s 能掉 900+ px），而是<strong>手感</strong>：
          不想要"连续几十根一路坠到底"的单调关卡，所以给了个与缺口等高的界
          <span class="mono">max_drop = 230</span>。</dd>
      <dt>紫色带</dt>
      <dd>缺口 top 的可行区间（画布高 {{G_H}}、缺口高 {{GAP}} 夹出来的）。</dd>
    </dl>
  </div>

  <h3>改法：有界随机游走</h3>
  <div class="eq">
    next_hi = min(c_hi, prev_c + max_climb) &nbsp;<span class="cmt">// 向上受带宽限制</span><br>
    next_lo = max(c_lo, prev_c − max_drop) &nbsp;<span class="cmt">// 向下给一个界</span><br>
    c = uniform(next_lo, next_hi) + N(0, 16²) &nbsp;<span class="cmt">// 再叠独立抖动</span><br>
    c = clip(c, next_lo, next_hi, c_lo, c_hi) &nbsp;<span class="cmt">// 抖动后必须重新夹</span>
  </div>
  <p>
    坐标用"缺口<strong>中心</strong>"而不是 top：top 被 GAP 和画面边界夹住，
    直接对 top 做游走会在边界处再次退化成常数（原来就是这么死的）。
  </p>

  <div class="box trap">
    <div class="lab">两个我亲手踩的坑</div>
    <p>
      <strong>① 抖动必须夹在约束之内。</strong>
      我第一版把抖动加在夹取<strong>之后</strong>，实测最大上升达到 211 px，
      超过了 {{MAX_CLIMB}} 的上限 —— 抖动把"物理可达"这条硬约束冲掉了。
      约束是硬性的，抖动只能在约束内抖。
    </p>
    <p>
      <strong>② 只设向上界不够。</strong>
      那样缺口中心只能活在 <span class="mono">[c_lo, c_hi]</span> 这个死区里，
      一旦被推到边界，下一根<strong>必须</strong>落在死区内 → 被迫向下掉 216 px。
      那不是设计，是边界效应。
    </p>
  </div>

  <div class="box key">
    <div class="lab">代价要说清楚</div>
    <p>
      随机性上去了，但<strong>单局均分从约 36 降到约 31</strong>（20 局口径，在方差内）。
      另外，离线评测台留下的历史扫描曾显示
      <span class="mono">max_climb = 40</span> 最好（均分 22.23）；
      但那很可能不是因为 40 物理上更合理，而是因为
      <strong>40 恰好让旧生成器退化得最彻底</strong>（关卡最单调、最容易）。
      这个疑点已写进 <span class="mono">ESCAPE.md</span>，没有藏起来。
    </p>
    <p class="q">
      参数扫描脚本 <span class="mono">scripts/tune_levels.py</span> 可复跑，
      口径都写在脚本里。
    </p>
  </div>

  <details>
    <summary>自测：为什么"从下方均匀抽样"看起来满足了约束，实际上却在边界处变成常数？</summary>
    <div class="ans">
      <p>
        因为 <span class="mono">uniform(a, b)</span> 在
        <span class="mono">b &lt; a</span> 时<strong>没有定义良好的行为</strong>，
        实现通常返回 <span class="mono">a</span>。
      </p>
      <p>
        旧代码的上界 <span class="mono">min(hi, prev_c + max_climb − GAP/2)</span>
        随 <span class="mono">prev_c</span> 一起下降，而下界
        <span class="mono">lo = 70</span> 是<strong>固定的</strong>。
        所以必然存在一个 <span class="mono">prev_c</span> 以下，
        上界 &lt; 下界 → 每次返回 70。
      </p>
      <p>
        更糟的是它是<strong>吸收态</strong>：一旦落到 70，下一轮的
        <span class="mono">prev_c</span> 就固定在 161 附近，
        上界又 &lt; 下界，于是永远停在 70。<strong>再也出不来。</strong>
      </p>
      <p class="q">
        通用教训：<strong>用"从某个区间均匀抽样"来实现随机游走时，
        区间的端点必须是固定的</strong>。让端点跟着状态走，
        迟早会撞上"端点交叉"这个退化。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 12 -->
<div class="chapter"><span class="ctitle">第六部 · 证据</span></div>
<section class="sec" id="s12">
  <div class="eyebrow">12</div>
  <h2>怎么证明"分数真的穿过了连接组"</h2>
  <div class="thread">
    这是整个项目的重点实验：<b>同一套参数、同一批关卡，只改脑的接线。</b>
  </div>

  <div class="tw"><table>
    <thead><tr><th>组</th><th>做法</th><th>均分</th><th>拍翅/局</th></tr></thead>
    <tbody>
      <tr><td><b>real</b></td><td>真实接线</td><td class="n"><b>31.05</b></td><td class="n">80.6</td></tr>
      <tr><td><b>cut</b></td><td>切断 LC4/LPLC2 → DNp01</td>
          <td class="n"><b>1.00 ± 0.00</b></td><td class="n"><b>0.0</b></td></tr>
      <tr><td><b>shuffled</b></td><td>同度分布随机重连</td>
          <td class="n"><b>1.00 ± 0.00</b></td><td class="n"><b>0.0</b></td></tr>
    </tbody>
  </table></div>

  <p>
    <span class="mono">cut</span> 与 <span class="mono">shuffled</span>
    <strong>一次都不拍翅</strong>（方差为 0，40 局全部死在第 1 根管子）。
    这就是"分数必须穿过真实连接组"的硬证据 ——
    不是因为脑"变笨了"，而是因为<strong>指令神经元收不到驱动了</strong>。
  </p>

  <div class="box trap">
    <div class="lab">那 1.00 分是哪来的（必须解释，否则会被当成"脑在做事"）</div>
    <p>
      整数 1 分不是脑在做事，是一种<strong>结构性白送</strong>。
    </p>
    <div class="calc">第 1 根管子起点 x = {{BIRD_X}} + 0.75×{{PX_PER_M}} − {{PIPE_W}}
                  = {{BIRD_X}} + 180 − {{PIPE_W}} = 238
鸟在 x = {{BIRD_X}}，管宽 {{PIPE_W}} → 管子占 [238, 300]，鸟占 [103, 137]

管子以 {{PX_PER_M}} px/s 左移，到覆盖鸟只需 (238−137)/{{PX_PER_M}} = 0.42 s
而加速期是 {{WARM}} s = 40 tick
→ <span class="r">加速期内管子已经穿过鸟身，而这期间不判定碰撞</span>
→ 加速期结束时，第 1 根已经被记为 "passed"，白送 1 分</div>
    <p>
      所以真正的"零分基线"是 0；那个 1.00 是我们自己的 <span class="mono">warmup</span>
      机制留下的痕迹。<strong>裸分是 1，不是脑挣的。</strong>
    </p>
  </div>

  <details>
    <summary>自测：为什么"随机重连（shuffled）"比"直接切断（cut）"是更强的对照？</summary>
    <div class="ans">
      <p>
        因为 <span class="mono">cut</span> 只证明了"这两条通路有用"，
        而 <span class="mono">shuffled</span> 证明了<strong>"具体的接线方式有用"</strong>。
      </p>
      <p>
        <span class="mono">shuffled</span> 保留了两个关键统计量：
        每个神经元的<strong>出度</strong>（连多少下游）和
        <strong>入度</strong>（被多少上游连）的分布。
        也就是说，网络的"规模结构"完全没变，只是<strong>谁连谁</strong>被打乱了。
      </p>
      <p>
        如果分数只取决于"有多少连接"，那么 shuffled 应该和 real 差不多。
        结果是 <strong>0 次拍翅</strong> —— 说明起作用的不是连接的数量，
        而是连接的<strong>具体拓扑</strong>。
      </p>
      <p class="q">
        这是网络神经科学里区分"结构决定功能"与"规模决定功能"的标准手段。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 13 -->
<section class="sec" id="s13">
  <div class="eyebrow">13</div>
  <h2>逐 tick 对账：为什么分数可以信</h2>
  <div class="thread">
    分数能不能信，取决于<b>两套实现是不是同一套物理</b>。
    所以我们不"抽查"，而是<em>逐 tick 比全部状态</em>。
  </div>

  <p>
    这个项目有两套独立的实现：网页用的服务端（<span class="mono">demo/server.py</span>）
    和离线评测台（<span class="mono">scripts/flappy_bench.py</span>）。
    如果它们不完全一致，那么"网页分数"和"消融实验分数"就没法放在一起讨论。
  </p>

  <div class="tw"><table>
    <thead><tr><th>脚本</th><th>比什么</th><th>判据</th></tr></thead>
    <tbody>
      <tr>
        <td class="mono">verify_server_physics.py</td>
        <td>只用物理：喂同一串拍翅命令、<strong>同一串缺口序列</strong>，
            逐 tick 比位置/速度/分数/死亡时刻/管道布局</td>
        <td>60 局 × ≤3000 tick<br><b>逐 tick 等价</b></td>
      </tr>
      <tr>
        <td class="mono">verify_server_vs_bench.py</td>
        <td>整体控制回路（<strong>含脑</strong>）：同种子逐 tick 比
            y / vy / 发放 / 拍翅</td>
        <td>3 种子 × 1200 tick<br><b>逐 tick 一致、总分相同</b></td>
      </tr>
      <tr>
        <td class="mono">verify_server_collision.js</td>
        <td>服务端判定的碰撞 vs 前端画出来的管子，<strong>逐像素</strong></td>
        <td>12410 个 y 位置<br><b>偏差 0 / 0</b></td>
      </tr>
    </tbody>
  </table></div>

  <div class="box key">
    <div class="lab">第二个脚本的价值远超预期</div>
    <p>
      它<strong>当场抓出了三个真 bug</strong>：噪声走了全局随机数、
      <span class="mono">deque</span> 窗口起点与评测台不一致、拍翅冷却放错了层。
      在写它之前，我只能靠猜和局部打印。
    </p>
    <p>
      一个具体的例子：服务端 <span class="mono">vy = −292.8</span>、
      评测台 <span class="mono">vy = −316.4</span>，差 23.6。
      一眼看不出问题，但差值恰好等于<strong>一个 tick 的重力增量</strong>
      （1180 × 0.02 = 23.6），这直接指向"服务端把拍翅吞掉了"——
      原来我把冷却门写进了物理层，而评测台把冷却放在 harness 层。
    </p>
  </div>

  <div class="box trap">
    <div class="lab">一个"什么都没验"的校验器</div>
    <p>
      我们曾经有一个脚本，它<strong>继承了我们自己的评测台实现</strong>，
      于是它把评测台和评测台比，"比值 1.0"永远通过。
      它在仓库里存在了很久，什么都没验。已删。
    </p>
    <p>
      教训：<strong>校验器必须与被校验对象有独立的实现路径</strong>，
      否则它只是一段会打印"✅"的装饰。
    </p>
  </div>

  <div class="box trap">
    <div class="lab">另一个：改了 A 之后 B 的校验挂了，先想清楚校验在验什么</div>
    <p>
      我们改关卡生成算法之后，<strong>两个对账脚本都失败了</strong> ——
      而且是正确失败：评测台还是旧生成算法，于是它们比的是
      "关卡生成差异"，不再是"物理差异"。
    </p>
    <p>
      修法是把两侧的关卡来源都替换成<strong>同一串预生成缺口序列</strong>，
      从而严格隔离出要验的那个变量。
    </p>
  </div>

  <details>
    <summary>自测：如果两个实现都错了同一个地方，逐 tick 对账能发现吗？</summary>
    <div class="ans">
      <p><strong>不能。</strong>这是对账类校验的根本局限。</p>
      <p>
        逐 tick 对账证明的是"两个实现一致"，不是"实现是对的"。
        它抓不到<strong>共同错误</strong>（common-mode error）——
        比如两边都把重力常数写成了 1180 而真实意图是 1200，
        对账会一路绿灯。
      </p>
      <p>所以对账必须配合<strong>独立证据</strong>：</p>
      <p>
        ① <strong>解析结果</strong>：像"拍一次翅上升 49 px"这种可以用公式算出来的量，
        算一遍和测一遍比对；<br>
        ② <strong>逐像素对账</strong>：它把"判定"和"渲染"这两条<em>本来无关</em>的
        路径绑在一起，比两个物理实现互相比更有独立性；<br>
        ③ <strong>功能性实验</strong>：消融实验（第 12 节）验证的是
        "分数是否依赖那条通路"，与实现细节无关。
      </p>
      <p class="q">
        一句话：<strong>对账抓分歧，不抓共识错误。</strong>
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 14 -->
<section class="sec" id="s14">
  <div class="eyebrow">14</div>
  <h2>可复现性与噪声：一个让所有对照实验失效的 bug</h2>
  <div class="thread">
    "同一个种子却飞得不一样" —— 这个 bug 的根因是<b>随机数走了全局流</b>，
    而它的后果比看起来严重得多。
  </div>

  <h3>现象</h3>
  <p>
    神经元的泊松噪声原来读的是<strong>全局</strong>随机数发生器
    （<span class="mono">np.random.binomial</span> / <span class="mono">torch.rand</span>）。
    后果是：噪声序列取决于<strong>此前消耗了多少随机数</strong>。
  </p>
  <div class="calc"><span class="c"># 两个进程、同一个种子、喂完全相同的驱动</span>
DNp01 发放数：服务端 2 / 评测台 1
膜电位最大差：<span class="r">0.674</span></div>
  <p>
    原因是两个进程初始化时消耗的随机数个数不同 ——
    服务端要建 3D 点云抽样、可点亮清单等，评测台不用。
    于是"同一个种子"这件事<strong>完全失去意义</strong>。
  </p>

  <h3>为什么这个 bug 特别致命</h3>
  <div class="box trap">
    <div class="lab">它让所有的 A/B 对照都站不住</div>
    <p>
      消融实验的本质是"除了我想测的那个变量，其他全都不变"。
      如果噪声流取决于代码路径，那么"改一行无关代码"也会改变噪声序列，
      从而改变结果 —— 你测到的是<strong>噪声抖动</strong>，不是你想测的效应。
    </p>
  </div>

  <h3>修法与代价</h3>
  <p>
    每个脑持有自己的 <span class="mono">rng</span> / <span class="mono">tgen</span>，
    <span class="mono">reset()</span> 时按种子重置。修完同样对账：
  </p>
  <div class="calc">膜电位最大差 <span class="r">0.000000</span>；逐 tick 完全一致</div>
  <div class="box trap">
    <div class="lab">代价必须说清</div>
    <p>
      噪声序列变了，所以<strong>历史分数不再逐位可复现</strong>。
      我们留了一个 <span class="mono">--legacy-noise</span> 开关做对照：
    </p>
    <div class="calc">新噪声流（当前）       31.05 ± 25.47
旧噪声流（--legacy-noise） 43.65 ± 27.56
文档里长期引用的那个数     100.80 ± 37.57   <span class="r">← 两条路径都复现不了</span></div>
    <p>
      那个 100.80 是更早的代码 + 不同参数下测的，<strong>已经无法复现</strong>。
      我们选择把它留在文档里并标注清楚，而不是悄悄删掉。
    </p>
    <p>
      这个修复确实把分数拉低了一截，不能假装没发生。
      但<strong>结论层面完好</strong>：cut/shuffled 依旧是 0 次拍翅，
      "分数必须穿过连接组"这个结论不受影响。
    </p>
  </div>

  <details>
    <summary>自测：为什么"每个脑有自己的 RNG"就解决了这个问题？什么情况下它仍然不能复现？</summary>
    <div class="ans">
      <p>
        因为噪声序列从此只取决于<strong>这个脑自己的种子和它自己走过的 tick 数</strong>，
        与"进程此前消耗了多少随机数"无关。所以只要种子相同、驱动序列相同，
        噪声序列就逐位相同。
      </p>
      <p><strong>仍然不能复现的情况</strong>：</p>
      <p>
        ① <strong>浮点累加顺序变了</strong>：比如把某个求和从 CPU 换成多线程、
        或者改变张量拼接顺序，浮点舍入就会不同。这不是 RNG 问题，是数值问题。<br>
        ② <strong>用了非确定性算子</strong>：某些 GPU kernel（如原子加）
        本身不保证顺序。<br>
        ③ <strong>驱动序列被上游改变了</strong>：如果关卡生成也用了随机，
        那么关卡不同 → 驱动不同 → 结果当然不同（这也是为什么我们对账脚本
        要把关卡固定成同一串）。
      </p>
      <p class="q">
        目前 ① ② 在我们的 CPU 单线程路径上没有发生，
        所以"逐 tick 完全一致"才能成立。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 15 -->
<div class="chapter"><span class="ctitle">第七部 · 工程</span></div>
<section class="sec" id="s15">
  <div class="eyebrow">15</div>
  <h2>为什么游戏必须跑在服务端</h2>
  <div class="thread">
    一句话：<b>浏览器里没有可信的时钟</b>，
    而这条回路要求决策与物理严格 1:1。
  </div>

  <h3>四种驱动方式，全部实测失败</h3>
  <div class="tw"><table>
    <thead><tr><th>#</th><th>驱动方式</th><th>实测结果</th></tr></thead>
    <tbody>
      <tr><td class="n">①</td><td>物理跟<strong>墙钟</strong>走，每帧消化 1 步</td>
          <td>帧率低于 50 fps 就积压几十步，鸟执行的是<em>半秒前</em>的决策 → 完全控不住</td></tr>
      <tr><td class="n">②</td><td>物理跟<strong>渲染帧率</strong>走</td>
          <td>与脑严格 1:1 了，但帧率 11.5 fps 时只有 <strong>12%</strong> 速度
              → 鸟慢到飞不起来</td></tr>
      <tr><td class="n">③</td><td>物理跟<strong>脑响应</strong>走</td>
          <td>速度 = HTTP 往返速度（中位 60 ms、最快 4 ms，差十几倍）
              → <strong>时快时慢</strong></td></tr>
      <tr><td class="n">④</td><td>固定节拍 + 决策队列预取</td>
          <td>队列周期性见底，仍然抖</td></tr>
    </tbody>
  </table></div>

  <div class="box key">
    <div class="lab">根因：浏览器里没有可信的节奏</div>
    <p>页面内直接测量得到：</p>
    <div class="calc"><span class="c">// 这些是实测值，不是规格书上的承诺</span>
setTimeout(4)        实际只有 <span class="r">29.8 次/秒</span>（被压到约 33 ms）
requestAnimationFrame 实际只有 <span class="r">28.2 次/秒</span>
页签切到后台        <span class="r">完全暂停</span>
HTTP 往返（本机）   中位 60 ms，p90 86 ms，最快 4 ms</div>
    <p>
      只要决策要靠一次跨进程往返拿到，时钟就一定是抖的。
      而这条回路要求<strong>决策与物理严格 1:1</strong>——
      抖的时钟直接违反这个前提。
    </p>
  </div>

  <h3>解法与效果</h3>
  <p>
    把整个游戏搬到服务端：脑和物理在<strong>同一个进程、同一把锁</strong>里
    按固定 50 Hz 跑，浏览器只做两件事 ——
    按 20 Hz 取状态、在两次状态之间插值后画出来。
  </p>
  <div class="tw"><table>
    <thead><tr><th>指标</th><th>前端自己跑（④）</th><th>服务端锁步（现在）</th></tr></thead>
    <tbody>
      <tr><td>仿真速率</td><td class="n">14–32 步/秒（抖）</td>
          <td class="n"><b>50.0 步/秒</b>（偏差 +0.1%）</td></tr>
      <tr><td>渲染帧率</td><td>与游戏速度耦合</td>
          <td class="n"><b>60 fps</b>，与游戏速度无关</td></tr>
      <tr><td>网络抖动影响</td><td>直接改变游戏速度</td><td>只影响画面新鲜度</td></tr>
    </tbody>
  </table></div>

  <div class="box trap">
    <div class="lab">还有一个历史证据，后来变成了可复现实验</div>
    <p>
      早年间网页分数远低于评测台，一直只是猜测。后来有人做了受控复现：
      把物理/脑的速率比设成 <span class="mono">0.694</span>
      （正好是当时页面实测的比例 50 ÷ 60×1.2），分数只有 <strong>2.07</strong>。
    </p>
    <p>
      这把"页面为什么差"从传闻变成了可复现的实验。
      <strong>现在这个现象不会再出现了</strong>：搬到服务端后，
      页面与评测台的速率比就是 1.0。
    </p>
  </div>

  <details>
    <summary>自测：为什么"物理跟渲染帧率走"（方式 ②）虽然保证了 1:1，却还是不行？</summary>
    <div class="ans">
      <p>
        因为它保证了<strong>比例</strong>正确，却牺牲了<strong>速率</strong>正确。
      </p>
      <p>
        脑和物理确实 1:1 了，但两者一起被绑到了渲染帧率上。
        当帧率只有 11.5 fps 时，整个仿真只跑了标称速度的 12%——
        游戏变成慢动作，鸟的爬升率相对于管子的接近速度严重失衡，
        控制律完全失效。
      </p>
      <p>
        <strong>关键区分：</strong>
        ① "决策与物理的比例"（1:1）—— 保证控制律成立；<br>
        ② "仿真与真实时间的比例"（1.0）—— 保证游戏手感与难度。
      </p>
      <p>
        正确的做法是<strong>两个都要满足</strong>，而这要求一个
        <strong>独立于渲染帧率、也独立于网络往返</strong>的时钟。
        浏览器给不了，所以只能搬到服务端。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 16 -->
<section class="sec" id="s16">
  <div class="eyebrow">16</div>
  <h2>六个真实 bug（这一节大概最有参考价值）</h2>
  <div class="thread">
    每个 bug 都写清：<b>现象 → 根因 → 怎么发现的 → 修法 → 通用教训</b>。
  </div>

  <h3>1 · 渲染循环每帧抛异常，被 try/catch 吞掉</h3>
  <div class="layers">
    <div class="layer"><div class="tag"><b>现象</b></div><div class="body">
      <p>游戏卡住 + 3D 面板全黑 + HUD 不动，而 <strong>console 一个错都不报</strong>。</p>
    </div></div>
    <div class="layer"><div class="tag"><b>根因</b></div><div class="body">
      <p>前端引用了一个<strong>从未定义过</strong>的常量 <span class="mono">WARMUP_S</span>
      （它只存在于服务端）。每帧都抛：</p>
      <div class="calc">ReferenceError: WARMUP_S is not defined
    at interpolate (app.js)
    at loop (app.js)</div>
      <p>这个异常被 <span class="mono">loop()</span> 的 catch 吞进
      <span class="mono">DBG.err</span>，于是后面的
      <span class="mono">drawGame() / draw3D() / hud()</span>
      <strong>一帧都没执行</strong>。</p>
    </div></div>
    <div class="layer"><div class="tag"><b>怎么发现的</b></div><div class="body">
      <p>
        验收脚本一直报绿，因为它只看 console 异常和速率——
        而异常被吞了、速率也"正常"（服务端在跑）。
        最后是靠<strong>在页面内部读 <span class="mono">DBG.err</span></strong>
        才看到它每帧都有值。
      </p>
    </div></div>
    <div class="layer"><div class="tag"><b>修法与教训</b></div><div class="body">
      <p>
        两处：① 那个常量改成<strong>服务端下发</strong>（单一来源）；
        ② 让 catch <strong>把错误显性打到状态栏</strong>；
        ③ 给验收脚本加一条"渲染循环有没有出错"的检查。
      </p>
      <p class="q">
        教训：<strong>最致命的 bug 是"静默"的那种。</strong>
        任何"每帧同一个异常"的状态都不该只进日志。
        另外，一个只检查"外部指标"（速率、console）的验收脚本，
        抓不到"内部整个循环已经死了"。
      </p>
    </div></div>
  </div>

  <h3>2 · 每帧都去服务端取状态 = 60 req/s，把服务端打爆</h3>
  <p>
    取状态原来写在渲染循环里、<strong>每帧一次</strong>。
    实测 6 秒发出 <strong>359 个</strong>请求（= 60 req/s），
    把单进程的服务端（还要跟 50 Hz 仿真线程抢 GIL）砸到
    <span class="mono">/api/game/state</span> 延迟从约 3 ms 飙到 <strong>38.9 ms</strong>。
  </p>
  <p>
    服务端本身只有 50 Hz，取 20 Hz 完全够（插值补中间帧）。
    限流后：6 秒 113 个请求、延迟中位 2.4 ms。
  </p>
  <p class="q">
    教训：<strong>前端取数据的频率应该由数据源的变化频率决定，而不是由渲染频率决定。</strong>
  </p>

  <h3>3 · 投屏用"保持最新值 + 外推"，画面上会跳</h3>
  <p>原来写的是 <span class="mono">y = srvY + vy × (now − srvRx)</span>。</p>
  <div class="tw"><table>
    <thead><tr><th>逐帧位移 dy</th><th>修前</th><th>修后</th></tr></thead>
    <tbody>
      <tr><td>中位</td><td class="n">2.70 px</td><td class="n"><b>0.00 px</b></td></tr>
      <tr><td>p95</td><td class="n">19.47 px</td><td class="n">受帧率限制</td></tr>
      <tr><td>最小（倒跳）</td><td class="n">−52.11 px</td><td class="n">不再出现</td></tr>
    </tbody>
  </table></div>
  <p>
    <strong>根因</strong>：<span class="mono">(now − srvRx)</span> 每帧都在变，
    于是<strong>同一个服务端状态被复用好几帧、每帧算出的位置都不同</strong>；
    下一帧收到新状态时基准一换，位置就跳一下。
  </p>
  <p>
    <strong>修法</strong>：维护<strong>最近两次状态</strong>（各带"到达本地的时刻"），
    在两者之间线性插值。用到达时刻而不是服务端时刻，是因为两个时钟之间有
    未知且会漂移的偏移量；而"两次到达的本地时刻之差"与服务端走过的时间是同一段，
    <strong>网络延迟在求商时自然抵消</strong>。
  </p>
  <p class="q">
    教训：插值必须<strong>夹在两个已知状态之间</strong>，不能"保持一个状态然后外推"。
    后者误差随时间线性增长，且在状态更新时产生跳变。
  </p>

  <h3>4 · 噪声走全局随机数流</h3>
  <p>见 <a href="#s14">第 14 节</a>，单独用一节讲，因为它的影响面最大。</p>

  <h3>5 · 校验器自己也会骗人</h3>
  <p>
    ① 有个脚本继承了我们自己的评测台实现，"比值 1.0"永远通过，什么都没验（已删）。
  </p>
  <p>
    ② 我用 <span class="mono">getImageData(drawImage(webglCanvas))</span>
    判断 3D 面板有没有画，结果<strong>永远读到全黑</strong> ——
    因为 WebGL 默认 <span class="mono">preserveDrawingBuffer=false</span>，
    读回来的是空缓冲。改用 <span class="mono">renderer.info.render.points</span>
    才看清它一直在正常渲染（实测 139,237 个点）。
  </p>
  <p class="q"><strong>测试方法本身也要被质疑。</strong></p>

  <h3>6 · 我自己把测试脚本写错，连续几轮"修"一个不存在的 bug</h3>
  <p>
    在查关卡生成越界时，我调生成函数和 append 的顺序反了，
    于是读到的"上一根管子"比预期晚一步。
    结果是：<strong>同一段代码，两个测试给出 0 和 216 两个结论</strong>，
    而我花了好几轮在"修"一个并不存在的越界 bug。
  </p>
  <p class="q">
    教训：<strong>当两个测试互相矛盾时，先怀疑测试。</strong>
    那时候正确的动作是打印出违规那一步的<em>原始输入</em>，
    而不是继续改被测代码。
  </p>

  <details>
    <summary>自测：这六个 bug 里，哪两个属于"验证方法本身有问题"？</summary>
    <div class="ans">
      <p><strong>5 和 6。</strong></p>
      <p>
        这两个不是产品代码的错，而是<strong>我们用来判断"对不对"的手段错了</strong>：
        ⑤ 一个继承了自己实现的校验器（循环论证）；
        以及一个读到空缓冲的探测方法（工具失效）。
        ⑥ 一个顺序写反的测试（实验设计错）。
      </p>
      <p>
        它们比产品 bug 更危险，因为<strong>产品 bug 会让你看到错误的结果，
        而验证 bug 会让你看到"正确"的结果</strong>——
        你会带着错误的确信继续往下走。
      </p>
      <p class="q">
        实践建议：任何自动校验脚本，都要问一句
        "<em>它在什么情况下会误报通过？</em>"
        并且<strong>故意制造一次失败</strong>来确认它真的会失败
        （mutation test 的思路）。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 17 -->
<section class="sec" id="s17">
  <div class="eyebrow">17</div>
  <h2>负结果：试过但没用的东西</h2>
  <div class="thread">
    一个项目的可信度，很大程度取决于它<b>愿不愿意报负结果</b>。
    下面这些全部实测过，全部没用。
  </div>

  <div class="tw"><table>
    <thead><tr><th>尝试</th><th>结果</th><th>为什么不行</th></tr></thead>
    <tbody>
      <tr><td>1-ply 预测 / 前视</td><td class="n">0.00</td>
          <td>两个分支在 <span class="mono">vy = −316</span> 时行为相同，预判拿不到信息</td></tr>
      <tr><td>提高驱动增益</td><td class="n">饱和在 ~101 px/s</td>
          <td>爬升率在 gain 1→2→4 时只从 87 涨到 101 就不动了</td></tr>
      <tr><td>上冲期间保留腹侧驱动</td><td class="n">9.45 → 5.12</td>
          <td>它破坏了下潜支</td></tr>
      <tr><td>天花板增益 1.4 / 2 / 3</td><td class="n">25.24 / 22.20 / 21.61</td>
          <td>全部比 1.0 更差</td></tr>
      <tr><td>平滑空间窗替代整半招募</td><td class="n">不发放</td>
          <td>只到 need 的 53%</td></tr>
      <tr><td>门控类变体</td><td class="n">全部 0 分</td><td>——</td></tr>
    </tbody>
  </table></div>

  <div class="box key">
    <div class="lab">从负结果里能学到什么</div>
    <p>
      <strong>"提高爬升率"这条路是被物理封死的，不是被调参封死的。</strong>
      我们试过 gain 1→2→4，爬升率只从 87 涨到 101 px/s 就饱和了。
    </p>
    <p>
      为什么？因为爬升率不是由"驱动强度"决定的，而是由
      <strong>拍翅周期的下界</strong>决定的：一次拍翅买 49 px，
      而膜电位要从零重新积分到发放，加上 <span class="mono">vy_gate</span>
      在上冲期关掉了地面驱动，周期被钉在约 0.4 s。
      再大的 gain 也只是让每次积分更快一点，改变不了"每个周期只能买 49 px"这件事。
    </p>
    <div class="calc">理论上限  49 px / 0.4 s = 122 px/s
实测        约 100 px/s
→ 所以关卡约束按 ~125 px/间隔 来定是合理的</div>
    <p class="q">
      这个认识直接决定了<a href="#s11">第 11 节</a>那个关卡约束的数值，
      所以负结果不是白做的。
    </p>
  </div>

  <details>
    <summary>自测：为什么"1-ply 预测"在这个系统里天然无效？</summary>
    <div class="ans">
      <p>
        因为<strong>两个动作在当前的物理状态下会导致同一个中间结果</strong>。
      </p>
      <p>
        具体地：鸟在 <span class="mono">vy = −316</span>（正在快速上升）时，
        "拍翅"和"不拍"经过一个 tick 后的位置几乎相同 ——
        因为拍翅只是把 <span class="mono">vy</span> 置为 −340，
        而它已经接近这个值了。
      </p>
      <p>
        所以前视一步得到的两条轨迹<strong>不可区分</strong>，
        基于它做的决策等于没有信息。
      </p>
      <p class="q">
        更一般的教训：<strong>预测的价值取决于"动作能多快改变状态"</strong>。
        当执行器已经饱和（或接近饱和）时，短程预测就是浪费。
        这也解释了为什么这里必须靠<strong>闭环反馈</strong>而不是规划。
      </p>
    </div>
  </details>
</section>

<!-- ============================================================ 18 -->
<div class="chapter"><span class="ctitle">附录</span></div>
<section class="sec" id="s18">
  <div class="eyebrow">18</div>
  <h2>常见误解</h2>
  <div class="thread">这些是读到这里最容易产生的错误理解，逐条纠正。</div>

  <div class="box trap">
    <div class="lab">误解 1 · "这就是让 AI 学会玩 Flappy Bird"</div>
    <p>
      <strong>没有学习。</strong>没有梯度、没有奖励、没有试错。
      连接组的权重是电镜量出来的，全程不变。
      我们称它为"控制器"而不是"模型"，就是为了避免这个联想。
    </p>
  </div>

  <div class="box trap">
    <div class="lab">误解 2 · "脑在算目标高度"</div>
    <p>
      <strong>没有目标高度这个变量。</strong>
      整个控制律是"视野哪一半被逼近占住 → 点亮那一半"，
      没有任何地方计算"我应该飞到哪个 y"。方向信息来自几何本身。
    </p>
  </div>

  <div class="box trap">
    <div class="lab">误解 3 · "分数高说明脑聪明"</div>
    <p>
      分数是<strong>整个闭环</strong>（感觉编码 + 连接组 + 物理 + 关卡）的性质，
      不是"脑的智力"。我们特意设计消融实验来区分这两者：
      <span class="mono">cut</span> / <span class="mono">shuffled</span>
      证明分数<strong>依赖</strong>真实拓扑，但下一个 100.8 分不代表脑有 100 的智力。
    </p>
  </div>

  <div class="box trap">
    <div class="lab">误解 4 · "0.06042 是调出来的超参数"</div>
    <p>
      <strong>它是算出来的。</strong>
      <span class="mono">need = (1−leak)·threshold/gain</span>，
      由三个更基本的常数决定。你改不了 <span class="mono">need</span> 本身，
      只能改 <span class="mono">leak</span> / <span class="mono">threshold</span> /
      <span class="mono">gain</span>。
    </p>
  </div>

  <div class="box trap">
    <div class="lab">误解 5 · "既然是固定连接组，那 20ms 步长不重要"</div>
    <p>
      <strong>很重要。</strong>
      <span class="mono">leak = exp(−Δt/τ)</span> 直接依赖 Δt。
      把 Δt 从 20 ms 改成 10 ms，<span class="mono">leak</span> 就从
      0.8187 变成 0.9048，<span class="mono">need</span> 跟着从 0.06042
      降到 0.03172 —— <strong>整个系统的触发门槛减半</strong>，
      行为会完全不同。
    </p>
  </div>

  <div class="box trap">
    <div class="lab">误解 6 · "平滑的输入一定比硬开关好"</div>
    <p>
      <strong>在阈值系统里恰恰相反。</strong>
      实测：平滑空间窗只到 need 的 53%（不发放），整半招募到 1.79 倍（能发放）。
      原因是阈值非线性，"给所有细胞都加一点"会被逐个阈值筛掉，
      而"集中给一半细胞"能真正推过线。
    </p>
  </div>

  <div class="box trap">
    <div class="lab">误解 7 · "两个实现逐 tick 一致，就说明实现是对的"</div>
    <p>
      <strong>不能说明。</strong>逐 tick 对账只能抓<strong>分歧</strong>，
      抓不到<strong>共识错误</strong>（两边都写错的同一个常数）。
      必须配合解析验算（如"拍一次翅上升 49 px"）和功能性实验。
      详见 <a href="#s13">第 13 节</a>自测题的答案。
    </p>
  </div>
</section>

<!-- ============================================================ 19 -->
<section class="sec" id="s19">
  <div class="eyebrow">19</div>
  <h2>术语表</h2>
  <div class="thread">左边中文 / 英文，右边是"在这个项目里它具体指什么"。</div>

  <div class="term">
    <div class="row"><div class="name"><b>连接组</b><span>connectome</span></div>
      <div class="def"><p>144,837 个神经元 + 1,502 万条突触构成的有向图。本项目里它<strong>冻结不变</strong>。</p></div></div>
    <div class="row"><div class="name"><b>漏积分-发放</b><span>LIF, leaky integrate-and-fire</span></div>
      <div class="def"><p>每个神经元的更新规则：<span class="mono">G ← leak·G + gain·C</span>，越过阈值则发放并归零。</p></div></div>
    <div class="row"><div class="name"><b>膜时间常数</b><span>membrane time constant τ</span></div>
      <div class="def"><p>0.1 s。决定电位漏得多快；<span class="mono">leak = exp(−Δt/τ)</span>。</p></div></div>
    <div class="row"><div class="name"><b>驱动门槛</b><span>need</span></div>
      <div class="def"><p><span class="mono">0.06042</span>。让一个神经元持续发放所需的最小恒定输入。整套系统的"汇率"。</p></div></div>
    <div class="row"><div class="name"><b>巨纤维</b><span>giant fiber, GF</span></div>
      <div class="def"><p>果蝇的逃逸指令通路。本项目里对应 <strong>DNp01</strong> 那 2 个细胞。</p></div></div>
    <div class="row"><div class="name"><b>逼近刺激</b><span>looming</span></div>
      <div class="def"><p>物体在视野里持续变大。本项目的"威胁"定义；由 LPLC2 编码。</p></div></div>
    <div class="row"><div class="name"><b>角尺寸 / 角速度</b><span>angular size / angular velocity</span></div>
      <div class="def"><p>LPLC2 与 LC4 分别编码的量。前者是"多大"，后者是"变多快"。</p></div></div>
    <div class="row"><div class="name"><b>整半招募</b><span>whole-half recruitment</span></div>
      <div class="def"><p>整个半个视野一起驱动，而不是给平滑权重窗。原因见第 7 节。</p></div></div>
    <div class="row"><div class="name"><b>死区 / 滞环</b><span>dead zone / hysteresis</span></div>
      <div class="def"><p>18 px 的带，方向判定在里面不翻转。用来消除自激振荡。</p></div></div>
    <div class="row"><div class="name"><b>锁步</b><span>lockstep</span></div>
      <div class="def"><p>脑 tick 与物理步严格 1:1，同一进程同一把锁。保证控制律成立。</p></div></div>
    <div class="row"><div class="name"><b>tick</b><span>tick</span></div>
      <div class="def"><p>一次 20 ms 的仿真步；控制回路的最小单位。</p></div></div>
    <div class="row"><div class="name"><b>逐 tick 对账</b><span>tick-for-tick reconciliation</span></div>
      <div class="def"><p>两套实现喂同样输入，逐 tick 比全部状态。抓分歧，不抓共识错误。</p></div></div>
    <div class="row"><div class="name"><b>逐像素对账</b><span>pixel-exact reconciliation</span></div>
      <div class="def"><p>判定集合与绘制集合逐个像素比，要求两个方向的差异同时为 0。</p></div></div>
    <div class="row"><div class="name"><b>消融实验</b><span>ablation study</span></div>
      <div class="def"><p>只改一个变量（这里是脑的接线），看结果如何变化。本项目的核心证据。</p></div></div>
    <div class="row"><div class="name"><b>同度随机重连</b><span>degree-preserving shuffle</span></div>
      <div class="def"><p>保持每个节点出入度分布不变、打乱连接对象。用来区分"规模"与"拓扑"。</p></div></div>
    <div class="row"><div class="name"><b>Naka-Rushton</b><span>希尔型饱和函数</span></div>
      <div class="def"><p><span class="mono">amp = θⁿ/(θⁿ+s₅₀ⁿ)</span>，视觉生理学的标准饱和曲线。</p></div></div>
    <div class="row"><div class="name"><b>强制发放</b><span>clamp</span></div>
      <div class="def"><p>直接指定某群细胞以多大概率发放，跳过上游级联。本项目唯一"定义而非测量"的环节。</p></div></div>
  </div>
</section>

<!-- ============================================================ 20 -->
<section class="sec" id="s20">
  <div class="eyebrow">20</div>
  <h2>一页带走</h2>
  <div class="thread">如果只记一页，记这页。</div>

  <div class="card">
    <div class="top">十条核心</div>
    <ol>
      <li><b>零可学参数。</b>连接组冻结，只有感觉编码层可设计。这是全部结论的前提。</li>
      <li><b>need = (1−leak)·thr/gain = {{NEED}}。</b>所有"够不够"判断的标尺；
          恰好等于它时<strong>永远不发放</strong>，必须严格大于。</li>
      <li><b>leak = 0.818731</b> 是<strong>保留</strong>比例；每 tick <strong>漏掉</strong> 18.127%。</li>
      <li><b>方向不由控制器算，由几何给。</b>缺口在上 → 视野下被逼近 → 腹侧 → 拍翅。</li>
      <li><b>死区 18 px 提供滞环</b>，消除自激振荡（没有它均分只有 0.81）。</li>
      <li><b>整半招募 &gt; 平滑窗。</b>阈值系统里"集中"胜过"均匀"（53% vs 179% of need）。</li>
      <li><b>决策与物理必须严格 1:1</b>，而且仿真与真实时间也要 1:1。浏览器给不了这两个，所以跑服务端。</li>
      <li><b>拍一次翅买 49 px</b>，可持续爬升约 100 px/s，一个管距间隔最多净爬约 125 px ——
          这是关卡约束的依据。</li>
      <li><b>cut / shuffled 都是 0 次拍翅</b>（方差 0）。分数必须穿过真实连接组。
          那个裸分 1 是 warmup 白送的，不是脑挣的。</li>
      <li><b>逐 tick 对账抓分歧，不抓共识错误。</b>校验器也要被质疑——
          我们删过一个"把评测台和评测台比"的脚本。</li>
    </ol>
  </div>

  <div class="card">
    <div class="top">关键数字速查</div>
    <div class="tw"><table>
      <thead><tr><th>量</th><th>值</th><th>它为什么重要</th></tr></thead>
      <tbody>
        <tr><td class="n">need</td><td class="n">{{NEED}}</td><td>持续发放所需的最小输入</td></tr>
        <tr><td class="n">leak</td><td class="n">{{LEAK}}</td><td>每 tick 保留 81.9%</td></tr>
        <tr><td class="n">Δt</td><td class="n">{{TICK_MS}} ms</td><td>= 50 Hz，与控制回路同源</td></tr>
        <tr><td class="n">管距 / 管速</td><td class="n">{{TWEEN}} s</td><td>两根管子之间的可用时间</td></tr>
        <tr><td class="n">可持续爬升率</td><td class="n">约 100 px/s</td><td>一个间隔最多净爬约 125 px</td></tr>
        <tr><td class="n">拍翅上升高度</td><td class="n">49.0 px</td><td>vy²/2g = 340²/(2×1180)</td></tr>
        <tr><td class="n">拍翅上升时长</td><td class="n">0.288 s</td><td>340/1180</td></tr>
        <tr><td class="n">腹侧 Σw</td><td class="n">0.1080</td><td>→ 1.73 × need（有余量）</td></tr>
        <tr><td class="n">背侧 Σw</td><td class="n">0.0707</td><td>→ 1.17 × need（余量小）</td></tr>
        <tr><td class="n">服务端仿真速率</td><td class="n">50.0 步/秒</td><td>与真实时间 1:1 且恒定</td></tr>
        <tr><td class="n">消融 real / cut</td><td class="n">31.05 / 1.00</td><td>cut 的 1.00 是 warmup 白送</td></tr>
      </tbody>
    </table></div>
  </div>

  <div class="card">
    <div class="top">想动手：五条命令</div>
    <div class="calc"><span class="c"># 1. 起服务（仓库自包含：脑资产与 three.js 都在库里）</span>
pip install numpy pandas torch --index-url https://download.pytorch.org/whl/cpu
python demo/server.py --no-browser --port 8620
<span class="c">#   然后浏览器打开 http://127.0.0.1:8620/</span>

<span class="c"># 2. 离线评测台：量分数、做消融</span>
python scripts/flappy_bench.py --modes bidi --games 40 --graph real
python scripts/flappy_bench.py --modes bidi --games 40 --graph cut

<span class="c"># 3. 三个对账脚本（"分数能不能信"的依据）</span>
python scripts/verify_server_physics.py
python scripts/verify_server_vs_bench.py
node demo/verify_server_collision.js

<span class="c"># 4. 关卡生成参数扫描</span>
python scripts/tune_levels.py --games 10

<span class="c"># 5. 重建这份文档的配图与页面</span>
python scripts/make_doc_figures.py --out output/doc_figs
python scripts/make_doc_page.py</div>
    <div class="box trap" style="margin:1rem 1.15rem 1.15rem">
      <div class="lab">两个会让你白忙半天的坑</div>
      <p>
        <strong>① 传错资产：</strong>如果用了不含 LC4/LPLC2 的子图，
        分数会<strong>恒为 0 且不报错</strong>。
        现在服务端会在启动时直接拒绝并说明原因。<br>
        <strong>② 缺坐标文件：</strong>3D 面板会静默变空，而游戏照跑 ——
        容易被当成"面板就这样"。
      </p>
    </div>
  </div>

  <footer>
    数据来源 · FlyWire FAFB v783（codex.flywire.ai）· 144,837 神经元 / 15,023,799 突触<br>
    本页所有图形由 <span class="mono">scripts/make_doc_figures.py</span>
    从<strong>跑着的仿真</strong>取数生成，非示意图；
    正文里的常数由 <span class="mono">make_doc_page.py</span> 直接从
    <span class="mono">demo/server.py</span> 读取，避免文档与实现脱节。<br>
    数值随代码版本变化 · 在仓库里跑一次即可复现
  </footer>
</section>
"""


if __name__ == "__main__":
    raise SystemExit(main())
