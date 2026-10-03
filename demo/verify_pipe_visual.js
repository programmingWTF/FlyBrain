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

function rects(top, W = 520, H = 620) {
  const rec = [];
  const ctx = makeCtx(rec);
  // ⚠️ 必须与 drawGame() 里的两处调用**逐字一致**：边界内缩一个鸟半径
  //    （physics() 判的是鸟中心点，所以管子要画到"鸟身体真正碰到的位置"）。
  const R = 11;
  drawPipeSprite(ctx, 100, top + R, 0, R);              // 上管
  drawPipeSprite(ctx, 100, top + GAP - R + 1, H, R);    // 下管
  return rec.filter(([x, y, w, h]) => x < 100 + PIPE_W && x + w > 100 && h > 0);
}

/** 把矩形裁到**管体宽度** [100, 100+PIPE_W] 再展开成 y —— 只有这部分是
 *  "玩家可能撞到的东西"。管口左右各多出 5px（capW = PIPE_W+10）是装饰性收边：
 *  鸟身宽 22px、中心固定在 birdX，整个身体都落在管体 x 区间内，那 5px 碰不到。 */
function solidY(rec) {
  const ys = new Set();
  const x0 = 100, x1 = 100 + PIPE_W;
  for (const [x, y, w, h] of rec) {
    const l = Math.max(x, x0), r = Math.min(x + w, x1);
    if (r - l < 1) continue;                       // 与管体宽度没有重叠
    for (let yy = Math.ceil(y); yy < Math.ceil(y + h); yy++) ys.add(yy);
  }
  return [...ys].sort((a, b) => a - b);
}

// 碰撞判定：**逐字照抄 app.js physics() 的那一行**
//     if (G.y - 11 < p.top || G.y + 11 > p.top + GAP) return die('撞上管子');
// 注意鸟身有 2R 高，所以判定会**越过**管子边界各 R 像素。
const BIRD_R = 11;
const isBlocked = (y, top) => (y - BIRD_R < top) || (y + BIRD_R > top + GAP);

// 鸟**实际可达**的 y 上界：physics() 里 `G.y > G.H - 14` 就撞地面，
// 所以再往下早就是"撞到地面"了，不在"管子该不该挡"的讨论范围。
const Y_MAX_REACHABLE = G_H - 14;   // 606

let bad = 0;
console.log('判据：在**鸟实际可达且在画布内**的 y 范围里，'
          + '"被画到"的像素集合 必须恰好等于"判定挡住"的集合');
console.log('（可达上界 = G.H-14（先判撞地面）；判定照抄 G.y-11 < top || G.y+11 > top+GAP）\n');
console.log('top   | 比较域 | 画到 | 应挡 | 只画不挡 | 只挡不画');
console.log('-'.repeat(62));
for (const top of [70, 120, 200, 250, 300, 380, 402]) {
  const drawn = new Set(solidY(rects(top)));
  const lo = 0, hi = Math.min(Math.max(...drawn), Y_MAX_REACHABLE, G_H - 1);
  let onlyPaint = 0, onlyBlock = 0, shouldBlock = 0, painted = 0;
  for (let y = lo; y <= hi; y++) {
    const blocked = isBlocked(y, top);
    if (blocked) shouldBlock++;
    if (drawn.has(y)) painted++;
    if (drawn.has(y) && !blocked) onlyPaint++;
    if (!drawn.has(y) && blocked) onlyBlock++;
  }
  const ok = onlyPaint === 0 && onlyBlock === 0;
  if (!ok) bad++;
  console.log(`${String(top).padEnd(6)}| ${String(lo).padStart(3)}..${String(hi).padStart(3)} | `
            + `${String(painted).padStart(4)} | ${String(shouldBlock).padStart(4)} | `
            + `${String(onlyPaint).padStart(8)} | ${String(onlyBlock).padStart(8)} `
            + (ok ? '✅' : '❌'));
}
console.log('-'.repeat(62));
console.log(bad === 0
  ? '✅ 管口朝缺口、边界正好落在判定线上：画出来的管子 = 判定用的管子'
  : `❌ 有 ${bad} 个 top 值不一致：画出来的管子与碰撞判定已经漂移`);
process.exit(bad === 0 ? 0 : 1);
