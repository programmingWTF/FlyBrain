// 精确量"画出来的鸟"的外接范围，与碰撞判据用的 R=11 对账。
//
// 为什么要它：碰撞判据是 |y_center - y_pipe| < 11（中心 ±11px），
// 而鸟的精灵是椭圆(17,13)+翅膀(旋转0.9rad)+眼睛+喙，还会随速度旋转。
// 如果精灵的外接范围 > 11，就会出现"看着碰到了却不判定"——用户报的就是这个。
//
// 做法：直接从 app.js 里抽 drawBirdSprite() 的源码，灌一个记录型 ctx，
// 把 ellipse/arc/moveTo/lineTo 都算成世界坐标下的实际几何（含 translate/rotate），
// 然后给出精灵相对鸟中心的最小/最大 x、y。
//
// 用法: node demo/verify_bird_volume.js
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');
const i0 = src.indexOf('function drawBirdSprite');
if (i0 < 0) throw new Error('抽不到 drawBirdSprite');
// 函数体末尾 = 从起点往后第一个**行首**的 '}'（app.js 顶层函数都是这个缩进风格）。
// 不能用"下一个 \nfunction "来找末尾 —— birdRot() 就紧挨在 drawBirdSprite 前面，
// 那样切出来的范围是错的（踩过）。
const endM = /\n\}/.exec(src.slice(i0));
if (!endM) throw new Error('找不到 drawBirdSprite 的结尾');
const fnSrc = src.slice(i0, i0 + endM.index + 2);
// birdRot() 被 drawBirdSprite 调用，必须一起抽出来。
const j0 = src.indexOf('function birdRot');
const j1 = j0 >= 0 ? /\n\}/.exec(src.slice(j0)) : null;
const rotSrc = (j0 >= 0 && j1) ? src.slice(j0, j0 + j1.index + 2) : 'function birdRot(v){return v/430;}';

// 收集精灵形状所需的常量（若 drawBirdSprite 引用了它们，也要一并抽出来）
function grabConst(name) {
  const m = src.match(new RegExp(`const\\s+${name}\\s*=\\s*([^;]+);`));
  return m ? m[1] : null;
}
const consts = {};
for (const n of ['BIRD_R', 'BIRD_SPRITE_R', 'SPRITE_SCALE']) {
  const v = grabConst(n);
  if (v) consts[n] = v;
}

const R = Number(consts.BIRD_R) || 11;

// ---- 记录型 canvas ctx：把所有形状变换到世界坐标后记下来
function makeCtx() {
  const shapes = [];
  const stack = [];
  let m = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };   // 当前变换（2x3）
  const mul = (m1, m2) => ({                          // m1 之后再应用 m2
    a: m1.a * m2.a + m1.c * m2.b, b: m1.b * m2.a + m1.d * m2.b,
    c: m1.a * m2.c + m1.c * m2.d, d: m1.b * m2.c + m1.d * m2.d,
    e: m1.a * m2.e + m1.c * m2.f + m1.e, f: m1.b * m2.e + m1.d * m2.f + m1.f,
  });
  const apply = (x, y) => ({ x: m.a * x + m.c * y + m.e, y: m.b * x + m.d * y + m.f });

  // 轴对齐椭圆经当前变换后的外接范围（保守但精确到椭圆公式）：
  // 变换后的椭圆半轴投影 = sqrt((a*cosφ... )) —— 这里直接用旋转+缩放的组合：
  // 局部椭圆 (rx, ry) 在局部旋转 θ 后，再经线性部分 L，得到世界外接半宽/半高。
  function ellipseExtent(cx, cy, rx, ry, theta) {
    const size = Math.hypot(m.a, m.b), sizeY = Math.hypot(m.c, m.d);
    // 取线性部分的两个基向量长度做保守估计（旋转不变，足够判定是否越界）
    const sx = size, sy = sizeY;
    // 椭圆旋转 theta 后的轴对齐外接半宽/半高
    const ct = Math.cos(theta), st = Math.sin(theta);
    const ex = Math.hypot(rx * ct * sx, ry * st * sx);
    const ey = Math.hypot(rx * st * sy, ry * ct * sy);
    const c = apply(cx, cy);
    return { x0: c.x - ex, x1: c.x + ex, y0: c.y - ey, y1: c.y + ey };
  }

  const rec = (x0, y0, x1, y1) => shapes.push({ x0, y0, x1, y1 });
  const ctx = {
    save() { stack.push({ ...m }); },
    restore() { if (stack.length) m = stack.pop(); },
    translate(x, y) { m = mul(m, { a: 1, b: 0, c: 0, d: 1, e: x, f: y }); },
    rotate(t) { const c = Math.cos(t), s = Math.sin(t);
                m = mul(m, { a: c, b: s, c: -s, d: c, e: 0, f: 0 }); },
    scale(x, y) { m = mul(m, { a: x, b: 0, c: 0, d: y, e: 0, f: 0 }); },
    ellipse(cx, cy, rx, ry, theta) { const e = ellipseExtent(cx, cy, rx, ry, theta || 0); rec(e.x0, e.y0, e.x1, e.y1); },
    arc(cx, cy, r) { const e = ellipseExtent(cx, cy, r, r, 0); rec(e.x0, e.y0, e.x1, e.y1); },
    moveTo(x, y) { const p = apply(x, y); ctx._px = p.x; ctx._py = p.y;
                   ctx._minx = Math.min(ctx._minx ?? p.x, p.x); ctx._maxx = Math.max(ctx._maxx ?? p.x, p.x);
                   ctx._miny = Math.min(ctx._miny ?? p.y, p.y); ctx._maxy = Math.max(ctx._maxy ?? p.y, p.y); },
    lineTo(x, y) { const p = apply(x, y);
                   ctx._minx = Math.min(ctx._minx ?? p.x, p.x); ctx._maxx = Math.max(ctx._maxx ?? p.x, p.x);
                   ctx._miny = Math.min(ctx._miny ?? p.y, p.y); ctx._maxy = Math.max(ctx._maxy ?? p.y, p.y); },
    closePath() { if (ctx._minx !== undefined) rec(ctx._minx, ctx._miny, ctx._maxx, ctx._maxy); },
    beginPath() { ctx._minx = ctx._maxx = ctx._miny = ctx._maxy = undefined; },
    fill() {}, stroke() {}, clip() {}, setLineDash() {},
    fillRect() {}, strokeRect() {}, fillText() {},
    _shapes: shapes,
  };
  // 每次 fill/stroke 之前把路径也计入（有些形状只走 lineTo 不 closePath）
  const wrap = (name) => { const orig = ctx[name];
    ctx[name] = function () { if (ctx._minx !== undefined) rec(ctx._minx, ctx._miny, ctx._maxx, ctx._maxy);
                              return orig && orig.apply(ctx, arguments); }; };
  wrap('fill'); wrap('stroke');
  return ctx;
}

const G = { birdX: 120, y: 300, vy: 0, wingPhase: 0 };
// 用真实的 SPRITE_ROT_MAX / SPRITE_SCALE（从 app.js 抽），否则量的是另一套数
const consts2 = {};
for (const n of ['SPRITE_ROT_MAX', 'SPRITE_SCALE']) {
  const v = grabConst(n);
  consts2[n] = v === null ? null : Number(v);
}
if (consts2.SPRITE_ROT_MAX === null || consts2.SPRITE_SCALE === null) {
  throw new Error('app.js 里找不到 SPRITE_ROT_MAX / SPRITE_SCALE');
}
// 在**沙箱里把 drawBirdSprite 的 g.scale(...) 换成我们指定的值**：
// 这样可以用 scale=1 量原始形状，再用真实 scale 复核。不去改磁盘上的 app.js。
const fnNoScale = fnSrc.replace(/g\.scale\(SPRITE_SCALE,\s*SPRITE_SCALE\)/,
  'g.scale(__SCALE__, __SCALE__)');
if (fnNoScale === fnSrc) throw new Error('drawBirdSprite 里找不到 g.scale(SPRITE_SCALE, SPRITE_SCALE)');

/** 抽出 drawBirdSprite，并允许指定缩放。 */
const buildDraw = (scale) => new Function('G', 'SPRITE_ROT_MAX', 'SPRITE_SCALE', '__SCALE__',
  rotSrc + '\n' + fnNoScale + '; return drawBirdSprite;')(
  G, consts2.SPRITE_ROT_MAX, consts2.SPRITE_SCALE, scale);

/** = app.js 的 birdRot()，用于打印。 */
function birdRotOf(vy) {
  return Math.max(-consts2.SPRITE_ROT_MAX, Math.min(consts2.SPRITE_ROT_MAX, vy / 430));
}

/** 量一次精灵在给定缩放下的外接半宽/半高。 */
function measure(scale) {
  const draw = buildDraw(scale);
  let hw = 0, hh = 0;
  const rows = [];
  for (const vy of [-340, -200, 0, 200, 400, 500, 800]) {
    for (const wp of [0, 1, 2]) {
      G.vy = vy; G.wingPhase = wp;
      const ctx = makeCtx();
      draw(ctx);
      const sh = ctx._shapes;
      const a = Math.max(Math.abs(Math.min(...sh.map(s => s.x0)) - G.birdX),
                         Math.abs(Math.max(...sh.map(s => s.x1)) - G.birdX));
      const b = Math.max(Math.abs(Math.min(...sh.map(s => s.y0)) - G.y),
                         Math.abs(Math.max(...sh.map(s => s.y1)) - G.y));
      hw = Math.max(hw, a); hh = Math.max(hh, b);
      if (wp === 0) rows.push({ vy, rot: birdRotOf(vy),
        x0: Math.min(...sh.map(s => s.x0)) - G.birdX,
        x1: Math.max(...sh.map(s => s.x1)) - G.birdX,
        y0: Math.min(...sh.map(s => s.y0)) - G.y,
        y1: Math.max(...sh.map(s => s.y1)) - G.y, a, b });
    }
  }
  return { hw, hh, rows };
}

console.log(`碰撞判据用的半径 R = ${R} px（|y_center − y_pipe| < R）`);
console.log(`app.js 实际参数: SPRITE_ROT_MAX=${consts2.SPRITE_ROT_MAX}  `
          + `SPRITE_SCALE=${consts2.SPRITE_SCALE}`);
console.log('先量**原始形状**（scale=1），据此算应有缩放：\n');
console.log('  vy      rot     左     右   半宽  |    上     下   半高');
console.log('  ' + '-'.repeat(70));

const raw = measure(1);
for (const r of raw.rows) {
  console.log(`  ${String(r.vy).padStart(5)}  ${r.rot.toFixed(2).padStart(5)}  `
    + `${r.x0.toFixed(1).padStart(6)} ${r.x1.toFixed(1).padStart(6)} ${r.a.toFixed(1).padStart(6)}  | `
    + `${r.y0.toFixed(1).padStart(6)} ${r.y1.toFixed(1).padStart(6)} ${r.b.toFixed(1).padStart(6)}`);
}
console.log('  ' + '-'.repeat(70));
const rawMax = Math.max(raw.hw, raw.hh);
const want = R / rawMax;
console.log(`原始外接：半宽上界 ${raw.hw.toFixed(2)}px  半高上界 ${raw.hh.toFixed(2)}px`
          + `  → 外接半径上界 ${rawMax.toFixed(2)}px`);
console.log(`判定半径 R = ${R}px，所以 SPRITE_SCALE 应 ≤ ${want.toFixed(3)}`);
console.log(`当前 SPRITE_SCALE = ${consts2.SPRITE_SCALE} → 缩放后外接 `
          + `${(rawMax * consts2.SPRITE_SCALE).toFixed(2)}px\n`);

const sc = measure(consts2.SPRITE_SCALE);
const scWorst = Math.max(sc.hw, sc.hh);
if (scWorst > R + 1e-6) {
  console.log(`❌ 缩放后外接 ${scWorst.toFixed(2)}px > 判定半径 ${R}px（超出 ${(scWorst - R).toFixed(2)}px）`);
  console.log('   → 精灵与管子视觉重叠时中心还差这些像素，判定不触发 = "看着碰到了却没事"。');
  console.log(`   把 app.js 的 SPRITE_SCALE 改成 ≤ ${want.toFixed(3)}`
            + `（建议 ${(want * 0.98).toFixed(3)}，留 2% 余量）`);
  process.exit(1);
}
console.log(`✅ 缩放后外接 ${scWorst.toFixed(2)}px ≤ 判定半径 ${R}px：`
          + `画出来的鸟完全落在判定体积内，看着碰到就一定判定到了。`);
process.exit(0);
