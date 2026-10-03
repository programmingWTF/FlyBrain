// 校验管子的**形状不变量**（与判定无关的部分），避免与 verify_collision_pixels.js
// 的口径打架：
//   1) 管子不越出 [x, x+PIPE_W]  —— 旧版在管体两侧各凸出 5px 画粗管口，那 10px
//      "有画面、没判定"，而当时的校验器把比较范围裁在管宽内所以没抓到。
//   2) 管子覆盖的像素行是**连续**的（中间不能有空洞）。
//   3) 两根管子的外沿分别贴住画面顶与画面底。
//
// 竖直边界与判定的对账**不在这里** —— 那需要同时考虑鸟半径与像素闭区间，
// 由 demo/verify_collision_pixels.js 逐像素负责（它读的是 app.js 里的真代码）。
//
// 用法: node demo/verify_pipe_visual.js
const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');

function grabConst(name) {
  const m = src.match(new RegExp('(?:^|[,;{(\\s])' + name + '\\s*=\\s*(-?[0-9.]+)'));
  if (!m) throw new Error(`app.js 里找不到常量 ${name}`);
  return Number(m[1]);
}
const GAP = grabConst('GAP'), R = grabConst('BIRD_R'), PIPE_W = grabConst('PIPE_W');
const G_H = 620;

const i0 = src.indexOf('function drawPipe(');
if (i0 < 0) throw new Error('抽不到 drawPipe');
const fnSrc = src.slice(i0, i0 + /\n\}/.exec(src.slice(i0)).index + 2);

function makeCtx(rec) {
  const grad = { addColorStop() {} };
  return { fillStyle: null, strokeStyle: null, lineWidth: 1,
    createLinearGradient: () => grad,
    fillRect: (x, y, w, h) => rec.push([x, y, w, h]),
    strokeRect: (x, y, w, h) => rec.push([x, y, w, h]),
    beginPath() {}, arc() {}, fill() {}, stroke() {}, save() {}, restore() {},
    ellipse() {}, moveTo() {}, lineTo() {}, closePath() {}, setLineDash() {},
    fillText() {}, strokeText() {}, clip() {}, translate() {}, rotate() {}, scale() {} };
}
const Gstub = { drawnRects: [] };
const drawPipe = new Function('PIPE_W', 'G', fnSrc + '; return drawPipe;')(PIPE_W, Gstub);

/** drawGame() 的两处调用，逐字照抄。返回像素行集合与 x 范围。 */
function pipe(px, top) {
  const rec = [];
  const ctx = makeCtx(rec);
  drawPipe(ctx, px, 0, top + R + 1);                 // 上管
  drawPipe(ctx, px, top + GAP - R - 1, G_H);         // 下管
  const solid = rec.filter(r => r[3] > 0);
  let x0 = Infinity, x1 = -Infinity;
  for (const [x, , w] of solid) { x0 = Math.min(x0, x); x1 = Math.max(x1, x + w); }
  // 每根管子单独一行集合。每根记两条（fillRect + strokeRect），
  // 所以按块索引区分：上管从 y=0 起，下管从 y>0 起。
  const rowsOf = (r) => { const s = new Set();
    for (let y = r[1]; y <= r[1] + r[3] - 1; y++) s.add(y); return s; };
  const ups = solid.filter(r => r[1] === 0);
  const dns = solid.filter(r => r[1] > 0);
  return { x0, x1, up: rowsOf(ups[0]), dn: rowsOf(dns[0]) };
}
const contiguous = (s) => {
  const a = [...s].sort((x, y) => x - y);
  for (let i = 1; i < a.length; i++) if (a[i] !== a[i - 1] + 1) return false;
  return true;
};

let bad = 0;
console.log(`几何: GAP=${GAP} BIRD_R=${R} PIPE_W=${PIPE_W}\n`);
console.log('top   | x 范围        | 上管行数(连续?) | 下管行数(连续?) | 贴顶/贴底 | 结果');
console.log('-'.repeat(80));

for (const top of [70, 120, 200, 250, 300, 340, G_H - GAP - 150]) {
  const q = pipe(100, top);
  const okX = q.x0 >= 100 - 1e-9 && q.x1 <= 100 + PIPE_W + 1e-9;
  const okUp = contiguous(q.up), okDn = contiguous(q.dn);
  const touchesTop = q.up.has(0);
  const touchesBottom = q.dn.has(G_H - 1);
  const ok = okX && okUp && okDn && touchesTop && touchesBottom;
  if (!ok) bad++;
  console.log(`${String(top).padEnd(6)}| [${q.x0},${q.x1}]`.padEnd(24)
    + `| ${String(q.up.size).padStart(5)} ${okUp ? '✅' : '❌'}      `
    + `| ${String(q.dn.size).padStart(5)} ${okDn ? '✅' : '❌'}      `
    + `| ${touchesTop ? '✅' : '❌'}/${touchesBottom ? '✅' : '❌'}     | ${ok ? '✅' : '❌'}`);
}
console.log('-'.repeat(80));
console.log(bad === 0
  ? '✅ 管子不越出管宽、像素行连续、外沿分别贴住画面顶与底'
  : `❌ 有 ${bad} 个 top 值不满足形状不变量`);
console.log('\n（竖直边界与判定的对账在 demo/verify_collision_pixels.js —— 逐像素、读真代码）');
process.exit(bad === 0 ? 0 : 1);
