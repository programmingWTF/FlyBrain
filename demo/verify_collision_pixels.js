// 纯解析对账：把"画到的像素集合"与"判定为撞的 y 集合"逐个比。
// 这里**逐像素**比较（用 fillRect 的实际像素覆盖），不引入任何几何换算，
// 避免我又在"坐标 vs 像素"上绕圈。
//
// 用法: node demo/verify_collision_pixels.js
const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');

function grabConst(name) {
  const m = src.match(new RegExp('(?:^|[,;{(\\s])' + name + '\\s*=\\s*(-?[0-9.]+)'));
  if (!m) throw new Error(`找不到常量 ${name}`);
  return Number(m[1]);
}
const GAP = grabConst('GAP'), R = grabConst('BIRD_R'), PIPE_W = grabConst('PIPE_W');
const G_H = 620, BIRD_X = 120;

// 抽出 drawPipe 与 physics 里那两行判定，保证测的就是真代码
const i0 = src.indexOf('function drawPipe(');
const fnSrc = src.slice(i0, i0 + /\n\}/.exec(src.slice(i0)).index + 2);

// 抽出 physics 里的判定表达式（严格/非严格都在这一行）
// 允许用环境变量覆盖判据，方便先试公式再改代码
// 判据跨两行，且以 `if (` 开头、`|| ` 续行。按行抽取，不能用 [\s\S]*?
// （那会把函数体里 for (...) { 之类也吞进去，实测报 Unexpected token '{'）。
const lines = src.split('\n');
const iIf = lines.findIndex(l => l.includes('p.top + BIRD_R') && l.trim().startsWith('if ('));
if (iIf < 0) throw new Error('抽不到判定行');
let raw = lines[iIf].trim().replace(/^if \(/, '');
let k = iIf;
while (!/\)\s*return die\('撞上管子'\);\s*$/.test(raw) && k + 1 < lines.length) {
  raw += ' ' + lines[++k].trim();
}
const m = [null, raw.replace(/\)\s*return die\('撞上管子'\);\s*$/, '')];
const OVERRIDE = process.env.JUDGE;
if (!m) throw new Error('抽不到判定行');
const judgedExpr = (OVERRIDE || m[1]).replace(/\s+/g, ' ').trim();
console.log('从 app.js 抽到的判定：');
console.log('  ' + judgedExpr + '\n');

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
const drawPipe = new Function('PIPE_W', fnSrc + '; return drawPipe;')(PIPE_W);

/** 这根管子（drawGame 的调用，逐字照抄）在 x 覆盖鸟时，**画到的像素行**集合。
 *  fillRect(y, h) 覆盖像素 y … y+h-1（闭区间）。 */
function paintedRows(top, px) {
  const rec = [];
  const ctx = makeCtx(rec);
  drawPipe(ctx, px, 0, top + R + 1);                 // 上管
  drawPipe(ctx, px, top + GAP - R - 1, G_H);         // 下管
  const rows = new Set();
  for (const [x, y, w, h] of rec) {
    if (h <= 0) continue;
    // x 方向：鸟心 birdX 时圆周最左/最右；只要这一列被画到就算
    const l = Math.max(x, BIRD_X), r = Math.min(x + w, BIRD_X + 1);
    if (r - l < 1) continue;
    for (let yy = y; yy <= y + h - 1; yy++) rows.add(yy);
  }
  return rows;
}

/** 判定：把抽到的表达式套进真实变量求值。 */
const judged = new Function('G', 'p', 'BIRD_R', 'GAP',
  'return (' + judgedExpr + ');');

let bad = 0;
console.log(`几何: GAP=${GAP} BIRD_R=${R} PIPE_W=${PIPE_W} birdX=${BIRD_X}\n`);
console.log('管子 x 盖住鸟（px=100）时：');
console.log('top   | 判定为撞的 y 数 | 画到的像素行数 | 只判不画 | 只画不判');
console.log('-'.repeat(72));

for (const top of [70, 120, 200, 300, 340]) {
  const rows = paintedRows(top, 100);
  // 鸟心可达范围：受上下边界限制
  const yLo = R + 1, yHi = G_H - 14 - R;
  let judgedN = 0, onlyJudge = 0, onlyPaint = 0;
  for (let y = yLo; y <= yHi; y++) {
    const G = { y, birdX: BIRD_X };
    const p = { x: 100, top };
    const j = judged(G, p, R, GAP);
    // 画面重叠：圆周覆盖到的像素行里，有任意一行属于 paintedRows
    const cLo = Math.ceil(y - R), cHi = Math.floor(y + R);
    let vis = false;
    for (let yy = cLo; yy <= cHi; yy++) if (rows.has(yy)) { vis = true; break; }
    if (j) judgedN++;
    if (j && !vis) onlyJudge++;
    if (!j && vis) onlyPaint++;
  }
  if (process.env.SHOW_BAD && (onlyJudge || onlyPaint)) {
    const bads = [];
    for (let y = yLo; y <= yHi; y++) {
      const G = { y, birdX: BIRD_X };
      const p2 = { x: 100, top };
      const j2 = judged(G, p2, R, GAP);
      const cLo = Math.ceil(y - R), cHi = Math.floor(y + R);
      let vis2 = false;
      for (let yy = cLo; yy <= cHi; yy++) if (rows.has(yy)) { vis2 = true; break; }
      if (j2 !== vis2) bads.push(`${y}(判${j2 ? 1 : 0}/画${vis2 ? 1 : 0})`);
    }
    console.log('   不一致的 y: ' + bads.slice(0, 20).join(' ') + (bads.length > 20 ? ` … 共${bads.length}` : ''));
  }
  const ok = onlyJudge === 0 && onlyPaint === 0;
  if (!ok) bad++;
  console.log(`${String(top).padEnd(6)}| ${String(judgedN).padStart(15)} | ${String(rows.size).padStart(14)} | `
    + `${String(onlyJudge).padStart(8)} | ${String(onlyPaint).padStart(9)}  ${ok ? '✅' : '❌'}`);
}
console.log('-'.repeat(72));
console.log(bad === 0
  ? '✅ 判定集合 == 画面重叠集合（逐像素，无几何换算）'
  : `❌ 有 ${bad} 个 top 值不一致 —— "只画不判"就是用户报的碰到不死`);
process.exit(bad === 0 ? 0 : 1);
