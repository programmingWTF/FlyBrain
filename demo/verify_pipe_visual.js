// 画出来的管子 vs 碰撞判定用的管子 —— 逐像素核对，防止"看起来撞到了却不算"
// 用法: node demo/verify_pipe_visual.js
//
// 背景：原来管帽（capH=30）画在**缺口那一侧**，画出来的管子整整探进缺口 30px，
// 而 physics() 的判定只认管体 [top, top+GAP]。于是"压着帽子飞过去不算撞"。
// 这个脚本用一个只记录 fillRect/strokeRect 的**假 canvas** 复现绘制调用，
// 把每个像素是否被画到算出来，再和碰撞判定逐像素比对。
const fs = require('fs');
const path = require('path');

// ---- 从 app.js 里抽出 drawPipeSprite 的源码，避免手抄漂移
//     允许用 argv[2] 指定另一个文件（用来验证"这个测试确实能抓到旧 bug"）
const srcPath = process.argv[2] || path.join(__dirname, 'app.js');
const src = fs.readFileSync(srcPath, 'utf8');
const start = src.indexOf('function drawPipeSprite');
const end = src.indexOf('\nfunction drawGround');
if (start < 0 || end < 0) throw new Error('找不到 drawPipeSprite / drawGround');
const spriteSrc = src.slice(start, end);

const PIPE_W = 62, GAP = 168, G_H = 620, GROUND = 92;

// 假 canvas：记录矩形覆盖
function makeCtx(rec) {
  const grad = { addColorStop() {} };
  const ctx = {
    fillStyle: null, strokeStyle: null, lineWidth: 1,
    createLinearGradient: () => grad,
    fillRect(x, y, w, h) { rec.push([x, y, w, h]); },
    strokeRect(x, y, w, h) { rec.push([x, y, w, h]); },
    beginPath() {}, arc() {}, fill() {}, stroke() {}, save() {}, restore() {},
    ellipse() {}, moveTo() {}, lineTo() {}, closePath() {}, setLineDash() {},
    fillText() {}, strokeText() {}, clip() {}, translate() {}, rotate() {},
  };
  return ctx;
}

// 用 Function 构造出被测函数（与 app.js 同一份源码）
const drawPipeSprite = new Function('PIPE_W', spriteSrc + '; return drawPipeSprite;')(PIPE_W);

function drawnAt(top, W = 520, H = 620) {
  const rec = [];
  const ctx = makeCtx(rec);
  // 与 drawGame() 里的两处调用完全一致
  drawPipeSprite(ctx, 100, 0, top, true);                       // 上管
  drawPipeSprite(ctx, 100, top + GAP, H - GROUND - top - GAP, true);  // 下管
  // 把矩形展开成像素集合（只关心 y；x 用管体区间 [100, 100+PIPE_W)）
  const px = new Set();
  for (const [x, y, w, h] of rec) {
    if (x > 100 + PIPE_W - 0.5 || x + w < 100 + 0.5) continue;   // 只算与管体 x 重叠的部分
    for (let yy = Math.ceil(y); yy < Math.ceil(y + h); yy++) px.add(yy);
  }
  return px;
}

// 碰撞判定：鸟（中心 y、半径 11）在管体 x 区间内时，哪些 y 算撞
const BIRD_R = 11;
function collides(y, top) { return (y - BIRD_R < top) || (y + BIRD_R > top + GAP); }

let bad = 0;
console.log('top   | 画到(缺口内)的像素                        | 判定为"不撞"却画到的 y 区间');
console.log('-'.repeat(100));
for (const top of [70, 120, 200, 250, 300, 380, 402]) {
  const px = drawnAt(top);
  // 缺口 [top, top+GAP] 内被画到的 y
  const inGap = [...px].filter(y => y >= top && y < top + GAP).sort((a, b) => a - b);
  // 其中判定为"不撞"的（鸟中心落在这些 y 时不该撞，但却有管子的像素）
  const falsePaint = inGap.filter(y => !collides(y, top));
  if (falsePaint.length) bad++;
  const rng = falsePaint.length
    ? `${falsePaint[0]}..${falsePaint[falsePaint.length - 1]} (${falsePaint.length}px)`
    : '—';
  console.log(`${String(top).padEnd(6)}| ${String(inGap.length).padEnd(9)} px` +
              `${''.padEnd(24)}| ${rng}`);
}
console.log('-'.repeat(100));
console.log(bad === 0
  ? '✅ 画出来的管子与碰撞判定逐像素一致（缺口内没有任何被画到的像素）'
  : `❌ 有 ${bad} 个 top 值下，缺口内被画到但不判定碰撞 —— 就是那个"碰到不算"的 bug`);
process.exit(bad === 0 ? 0 : 1);
