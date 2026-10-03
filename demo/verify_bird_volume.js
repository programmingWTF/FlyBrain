// 校验"画出来的鸟"与碰撞判据严格等价。
//
// 碰撞判据：`|y_center − y_pipe| < BIRD_R`（见 physics()）。这在几何上等价于
//   "**半径 BIRD_R 的圆**（圆心=鸟中心）碰到了管子边界"。
//
// 所以唯一自洽的画法就是：鸟身画成一个**半径 BIRD_R 的圆**，且眼睛/喙/翅膀
// 都不许探出这个圆（否则视觉又比判定大）。这个脚本把 drawBirdSprite 的所有形状
// （arc/ellipse/路径，含 translate/rotate）变换到世界坐标，逐条核对：
//   1) 外接半径必须**恰好** == BIRD_R（大一点点就是"画得比判定大"）
//   2) 不能小于 BIRD_R（小一点点就是"画得比判定小"，看着没碰到却判定撞了）
//
// 用法: node demo/verify_bird_volume.js
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');

function grabConst(name) {
  const m = src.match(new RegExp(`const\\s+${name}\\s*=\\s*([^;]+);`));
  return m ? Number(m[1]) : null;
}
const R = grabConst('BIRD_R');
const ROT_MAX = grabConst('SPRITE_ROT_MAX');
if (!R || ROT_MAX === null) throw new Error('app.js 里找不到 BIRD_R / SPRITE_ROT_MAX');

/** 从 app.js 抽出 [function 名 .. 行首 }] 的源码。 */
function extractFn(name) {
  const i = src.indexOf(`function ${name}(`);
  if (i < 0) throw new Error(`抽不到 ${name}`);
  const m = /\n\}/.exec(src.slice(i));
  if (!m) throw new Error(`找不到 ${name} 的结尾`);
  return src.slice(i, i + m.index + 2);
}
const rotSrc = extractFn('birdRot');
const drawSrc = extractFn('drawBirdSprite');
if (/SPRITE_SCALE/.test(drawSrc)) {
  throw new Error('drawBirdSprite 里还引用了 SPRITE_SCALE —— 鸟现在应该是不缩放的圆');
}

// ---- 记录型 canvas ctx：把形状变换到世界坐标后记下外接框
function makeCtx() {
  const shapes = [];
  const stack = [];
  let m = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
  const mul = (m1, m2) => ({
    a: m1.a * m2.a + m1.c * m2.b, b: m1.b * m2.a + m1.d * m2.b,
    c: m1.a * m2.c + m1.c * m2.d, d: m1.b * m2.c + m1.d * m2.d,
    e: m1.a * m2.e + m1.c * m2.f + m1.e, f: m1.b * m2.e + m1.d * m2.f + m1.f,
  });
  const apply = (x, y) => ({ x: m.a * x + m.c * y + m.e, y: m.b * x + m.d * y + m.f });
  const rec = (x0, y0, x1, y1) => shapes.push({ x0, y0, x1, y1 });

  function ellipseExtent(cx, cy, rx, ry, theta) {
    // 线性部分的两个基向量长度（旋转+缩放都含在内）
    const sx = Math.hypot(m.a, m.b), sy = Math.hypot(m.c, m.d);
    const ct = Math.cos(theta || 0), st = Math.sin(theta || 0);
    const ex = Math.hypot(rx * ct * sx, ry * st * sx);
    const ey = Math.hypot(rx * st * sy, ry * ct * sy);
    const c = apply(cx, cy);
    return { x0: c.x - ex, x1: c.x + ex, y0: c.y - ey, y1: c.y + ey };
  }

  const ctx = {
    save() { stack.push({ ...m }); },
    restore() { if (stack.length) m = stack.pop(); },
    translate(x, y) { m = mul(m, { a: 1, b: 0, c: 0, d: 1, e: x, f: y }); },
    rotate(t) { const c = Math.cos(t), s = Math.sin(t);
                m = mul(m, { a: c, b: s, c: -s, d: c, e: 0, f: 0 }); },
    scale(x, y) { m = mul(m, { a: x, b: 0, c: 0, d: y, e: 0, f: 0 }); },
    ellipse(cx, cy, rx, ry, theta) { const e = ellipseExtent(cx, cy, rx, ry, theta); rec(e.x0, e.y0, e.x1, e.y1); },
    arc(cx, cy, r) { const e = ellipseExtent(cx, cy, r, r, 0); rec(e.x0, e.y0, e.x1, e.y1); },
    beginPath() { ctx._p = null; },
    moveTo(x, y) { const p = apply(x, y); ctx._p = { x0: p.x, y0: p.y, x1: p.x, y1: p.y }; },
    lineTo(x, y) { if (!ctx._p) return; const p = apply(x, y);
                   ctx._p.x0 = Math.min(ctx._p.x0, p.x); ctx._p.x1 = Math.max(ctx._p.x1, p.x);
                   ctx._p.y0 = Math.min(ctx._p.y0, p.y); ctx._p.y1 = Math.max(ctx._p.y1, p.y); },
    closePath() {},
    fill() {}, stroke() {}, clip() {}, setLineDash() {},
    fillRect() {}, strokeRect() {}, fillText() {},
    _shapes: shapes, _flushPath() { if (ctx._p) { rec(ctx._p.x0, ctx._p.y0, ctx._p.x1, ctx._p.y1); ctx._p = null; } },
  };
  // fill/stroke 之前把路径计入（有些形状只走 lineTo 不 closePath）
  for (const name of ['fill', 'stroke']) {
    const orig = ctx[name];
    ctx[name] = function () { ctx._flushPath(); return orig && orig.apply(ctx, arguments); };
  }
  return ctx;
}

const G = { birdX: 120, y: 300, vy: 0, wingPhase: 0 };
const newDraw = new Function('G', 'BIRD_R', 'SPRITE_ROT_MAX',
  rotSrc + '\n' + drawSrc + '; return drawBirdSprite;')(G, R, ROT_MAX);

console.log(`碰撞判据：|y_center − y_pipe| < BIRD_R = ${R}px`);
console.log('  → 几何等价于"半径 R 的圆碰到了管子边界"，所以鸟必须画成半径 R 的圆。\n');
console.log('  vy      rot   画出来的外接半径  与 R 之差');
console.log('  ' + '-'.repeat(52));

let maxR = 0, minR = Infinity, worst = null;
for (const vy of [-340, -200, 0, 200, 400, 500, 800]) {
  for (const wp of [0, 1, 2]) {
    G.vy = vy; G.wingPhase = wp;
    const ctx = makeCtx();
    newDraw(ctx);
    const sh = ctx._shapes;
    const x0 = Math.min(...sh.map(s => s.x0)) - G.birdX;
    const x1 = Math.max(...sh.map(s => s.x1)) - G.birdX;
    const y0 = Math.min(...sh.map(s => s.y0)) - G.y;
    const y1 = Math.max(...sh.map(s => s.y1)) - G.y;
    // 用"到中心的最大距离"衡量（圆形判定关心的是各方向上的伸展）
    const rad = Math.max(Math.abs(x0), Math.abs(x1), Math.abs(y0), Math.abs(y1));
    const rot = Math.max(-ROT_MAX, Math.min(ROT_MAX, vy / 430));
    if (rad > maxR) { maxR = rad; worst = { vy, wp, rad }; }
    minR = Math.min(minR, rad);
    if (wp === 0) {
      console.log(`  ${String(vy).padStart(5)}  ${rot.toFixed(2).padStart(5)}   `
        + `${rad.toFixed(2).padStart(8)}px        ${(rad - R >= 0 ? '+' : '')}${(rad - R).toFixed(2)}`);
    }
  }
}
console.log('  ' + '-'.repeat(52));
console.log(`画出来的外接半径：最大 ${maxR.toFixed(2)}px  最小 ${minR.toFixed(2)}px   判定半径 ${R}px`);

const EPS = 0.01;
if (maxR > R + EPS) {
  console.log(`\n❌ 画出来的鸟比判定体积**大** ${(maxR - R).toFixed(2)}px`
            + `（vy=${worst.vy}, wingPhase=${worst.wp}）`);
  console.log('   → 精灵与管子视觉重叠了却还没判定 = "看着碰到了却不失败"。');
  process.exit(1);
}
if (maxR < R - 0.5) {
  console.log(`\n❌ 画出来的鸟比判定体积**小** ${(R - maxR).toFixed(2)}px —— `
            + `看着还没碰到却已经判定撞了（这更难玩，也不公平）。`);
  process.exit(1);
}
console.log(`\n✅ 画出来的鸟外接半径 ${maxR.toFixed(2)}px == 判定半径 ${R}px：`
          + `画出来的形状与碰撞判据严格等价（看着碰到就一定判定到，反之亦然）。`);
process.exit(0);
