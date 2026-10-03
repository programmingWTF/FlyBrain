// 严格对账：判定 vs 渲染，**扫管子的 x 位置**（只用水平确实够得到的圆心），
// 并把 drawPipe 的所有矩形（含描边）都算进来。
//
// 判据（= D:/Code/DQN 的 FlappySim._collides，圆 vs 轴对齐矩形）：
//   矩形1 = [x, 0] .. [x+PIPE_W, top]      矩形2 = [x, top+GAP] .. [x+PIPE_W, ground]
//   圆心 (birdX, y)、半径 R。矩形竖直连续 → 化简为
//     y - R <= top   或   y + R >= top + GAP   （且水平要够到）
//
// ⚠️ 上一版这个脚本把管子固定在 x=160（鸟在 x=120、R=17 → 水平永远够不到），
//    于是"判定 vs 画面"两侧都 false，报"不一致 0"其实什么都没测到（踩过）。
//    所以这里必须扫 x。
//
// 用法: node demo/verify_collision_exact.js
const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');

function grabConst(name) {
  const m = src.match(new RegExp('(?:^|[,;{(\\s])' + name + '\\s*=\\s*(-?[0-9.]+)'));
  if (!m) throw new Error(`找不到常量 ${name}`);
  return Number(m[1]);
}
const GAP = grabConst('GAP'), R = grabConst('BIRD_R'), PIPE_W = grabConst('PIPE_W');
const G_H = 620;
const BIRD_X = 120;                      // app.js: G = { W: 520, H: 620, birdX: 120 }

const i0 = src.indexOf('function drawPipe(');
const fnSrc = src.slice(i0, i0 + /\n\}/.exec(src.slice(i0)).index + 2);
function makeCtx(rec) {
  const grad = { addColorStop() {} };
  return { fillStyle: null, strokeStyle: null, lineWidth: 1,
    createLinearGradient: () => grad,
    fillRect: (x, y, w, h) => rec.push(['fill', x, y, w, h]),
    strokeRect: (x, y, w, h) => rec.push(['stroke', x, y, w, h]),
    beginPath() {}, arc() {}, fill() {}, stroke() {}, save() {}, restore() {},
    ellipse() {}, moveTo() {}, lineTo() {}, closePath() {}, setLineDash() {},
    fillText() {}, strokeText() {}, clip() {}, translate() {}, rotate() {}, scale() {} };
}
const Gstub = { drawnRects: [] };
const drawPipe = new Function('PIPE_W', 'G', fnSrc + '; return drawPipe;')(PIPE_W, Gstub);

/** drawGame() 里的两处调用，逐字照抄。返回 [[x,y,w,h], ...]。 */
function drawnRects(px, top) {
  const rec = [];
  const ctx = makeCtx(rec);
  drawPipe(ctx, px, 0, top + R + 1);
  drawPipe(ctx, px, top + GAP - R - 1, G_H);
  return rec.filter(([, , , , h]) => h > 0).map(([, x, y, w, h]) => [x, y, w, h]);
}

/** 把"块索引"矩形换算成**几何坐标**矩形 [x0,y0,x1,y1]。
 *  ⚠️ 这是本脚本最容易错的地方：像素块从索引 h 起、高 len，覆盖的是**坐标**
 *  [h, h+len)。之前直接把块索引当坐标喂进距离公式，于是凭空多出 1px 的
 *  "画面重叠"、报了一堆假不一致（踩过）。 */
function toGeom(rec) {
  return rec.map(([x, y, w, h]) => [x, y, x + w, y + h]);
}

/** 判定：逐字照抄 physics()。 */
function judged(px, top, y) {
  const cx = Math.max(px, Math.min(BIRD_X, px + PIPE_W));
  if ((BIRD_X - cx) ** 2 > R ** 2) return false;
  return Math.ceil(y + R) > top + R || Math.floor(y - R) < top + GAP - R;
}
/** 画面：圆心到任一矩形（**几何坐标**）的最短距离 <= R */
function visual(geom, y) {
  for (const [x0, y0, x1, y1] of geom) {
    const cx = Math.max(x0, Math.min(BIRD_X, x1));
    const cy = Math.max(y0, Math.min(y, y1));
    if ((BIRD_X - cx) ** 2 + (y - cy) ** 2 <= R * R + 1e-9) return true;
  }
  return false;
}

console.log(`几何: GAP=${GAP}  BIRD_R=${R}  PIPE_W=${PIPE_W}  birdX=${BIRD_X}`);
console.log('判定: 圆 vs 矩形，竖直化简为 y−R<=top 或 y+R>=top+GAP\n');

// ---- A. 渲染矩形是否越出判定矩形
console.log('=== A. 渲染矩形的 x 范围 vs 判定用的 [x, x+PIPE_W] ===');
let aBad = 0;
for (const top of [70, 200, 300, 340]) {
  const rects = drawnRects(100, top);
  let xMin = Infinity, xMax = -Infinity;
  for (const [x, , w] of rects) { xMin = Math.min(xMin, x); xMax = Math.max(xMax, x + w); }
  const ok = xMin >= 100 - 1e-9 && xMax <= 100 + PIPE_W + 1e-9;
  if (!ok) aBad++;
  console.log(`  top=${String(top).padEnd(4)} 渲染 x∈[${xMin}, ${xMax}]  `
    + `判定 x∈[100, ${100 + PIPE_W}]  ` + (ok ? '✅' : `❌ 越出 ${xMax - (100 + PIPE_W)}px`));
}

// ---- B. 扫管子的 x，逐个圆心 y 对账（**分级**：漏判必须为 0）
//
// 为什么分级：`fillRect(y, h)` 覆盖的是半开区间 [y, y+h)，而判定用的是"≤"（碰上就算）。
// 二者之间必然有**半像素错位**，表现出来就是边界处 1px 的"判定撞了但画面还差一点"
// 或"画面压到了但判定还没"。这不是逻辑错误，是像素与坐标的表示差异。
//
// 但两个方向严重性完全不同：
//   · 画面重叠 ⇒ 判定没撞  = **漏判**（用户报的"碰到柱子不死"）→ 必须为 0
//   · 判定撞   ⇒ 画面没重叠 = 多判 1px 羽化 → 允许，且必须非常窄
console.log('\n=== B. 扫管子 x（水平够不到 / 刚好够到 / 完全覆盖）===');
let passThrough = 0, feather = 0, bTested = 0, bReach = 0;
const xs = [BIRD_X - PIPE_W - R, BIRD_X - PIPE_W - R + 1, BIRD_X - PIPE_W,
            BIRD_X - R, BIRD_X, BIRD_X + R, BIRD_X + PIPE_W - R,
            BIRD_X + PIPE_W, BIRD_X + PIPE_W + R];
for (const top of [70, 200, 300]) {
  for (const px of xs) {
    const geom = toGeom(drawnRects(px, top));
    let pt = 0, ft = 0, firstPt = null, reach = 0;
    for (let y = 0; y <= G_H - 1; y++) {
      const j = judged(px, top, y), v = visual(geom, y);
      if (j || v) reach++;
      if (v && !j) { pt++; if (!firstPt) firstPt = y; }   // 漏判：画面重叠却没撞
      if (j && !v) ft++;                                   // 羽化：撞了但画面差不到 1px
    }
    bTested++; bReach += reach; passThrough += pt; feather += ft;
    console.log(`  top=${String(top).padEnd(4)} pipeX=${String(px).padStart(4)}  `
      + `涉及 y 数=${String(reach).padStart(3)}  漏判=${pt}  羽化=${ft}  `
      + (pt ? `❌ 首个漏判 y=${firstPt}` : '✅'));
  }
}

// ---- C. 边界明细（管子完全盖住鸟的 x）
console.log('\n=== C. 边界明细（top=300, pipeX=120 完全覆盖鸟的 x）===');
{
  const top = 300, px = 120;
  const rects = drawnRects(px, top);
  const geom = toGeom(rects);
  console.log(`  上管判定边界 y<=${top + R}   下管判定边界 y>=${top + GAP - R}`);
  for (const y of [top + R - 1, top + R, top + R + 1, top + GAP - R - 1, top + GAP - R]) {
    const j = judged(px, top, y), v = visual(geom, y);
    console.log(`    y=${String(y).padStart(4)}  判定=${String(j).padEnd(5)} 画面重叠=${String(v).padEnd(5)}  `
      + (j === v ? '一致' : (v && !j ? '❌ 漏判' : '羽化(允许)')));
  }
  console.log('  渲染矩形（块索引 -> 几何坐标）:');
  rects.forEach((r, i) => console.log(`    块 x=${r[0]} y=${r[1]} w=${r[2]} h=${r[3]}`
    + `  ->  几何 y∈[${geom[i][1]}, ${geom[i][3]})`));
}

console.log('\n' + '-'.repeat(60));
console.log(`B 段测了 ${bTested} 个 (top,pipeX) 组合，累计涉及 ${bReach} 个 y`);
console.log(`  漏判（画面重叠却没撞）= ${passThrough}   ← 必须为 0`);
console.log(`  羽化（撞了但画面差不到 1px）= ${feather}   ← 允许，半像素表示差异`);
const bad = aBad + passThrough;
console.log(bad === 0
  ? '✅ 渲染矩形不越出判定矩形；且**没有任何漏判**：画面一旦重叠就一定判死'
  : `❌ 有 ${bad} 处问题（A 段越界 ${aBad}，漏判 ${passThrough}）`);
process.exit(bad === 0 ? 0 : 1);
