// 校验"画出来的管子"与"判定挡住的范围"逐像素一致 —— 覆盖**整幅画面的 x 范围**。
//
// ⚠️ 这里踩过一个致命盲区：旧版校验器把像素比较范围裁在 `[x, x+PIPE_W]`（管体宽度）内，
//    而当时的 drawPipeSprite 在管体两侧各凸出 5px 画了一圈粗管口 ——
//    凸出的 10px **有画面、没判定**，鸟压上去不死，而校验器因为裁剪压根没检查。
//    现在管子就是一个矩形，且校验器**不再裁剪 x**，还额外断言管子不越出
//    `[x, x+PIPE_W]`，从形状上杜绝这类问题。
//
// 判定（= D:/Code/DQN 的 FlappySim._collides，圆 vs 轴对齐矩形化简而来）：
//     if (G.y - BIRD_R <= p.top || G.y + BIRD_R >= p.top + GAP) → 撞上管子
// 用法: node demo/verify_pipe_visual.js
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');

// ⚠️ 正则要能匹配**多变量声明**里的名字：GAP 写在
//    `const GRAV = 1180, FLAP_V = -340, PIPE_W = 62, GAP = 184;` 里。
function grabConst(name) {
  const m = src.match(new RegExp('(?:^|[,;{(\\s])' + name + '\\s*=\\s*(-?[0-9.]+)'));
  if (!m) throw new Error(`app.js 里找不到常量 ${name}`);
  return Number(m[1]);
}
const GAP = grabConst('GAP');
const BIRD_R = grabConst('BIRD_R');
const PIPE_W = grabConst('PIPE_W');
const G_H = 620;
//: 最后一个还没撞地面的整数 y（判据 `y + BIRD_R >= G_H - 14` → 撞地面）
const Y_MAX = G_H - 14 - BIRD_R - 1;

// ---- 抽 drawPipe
const i0 = src.indexOf('function drawPipe(');
if (i0 < 0) throw new Error('抽不到 drawPipe');
const endM = /\n\}/.exec(src.slice(i0));
if (!endM) throw new Error('找不到 drawPipe 的结尾');
const fnSrc = src.slice(i0, i0 + endM.index + 2);

function makeCtx(rec) {
  const grad = { addColorStop() {} };
  return {
    fillStyle: null, strokeStyle: null, lineWidth: 1,
    createLinearGradient: () => grad,
    fillRect(x, y, w, h) { rec.push([x, y, w, h]); },
    strokeRect(x, y, w, h) { rec.push([x, y, w, h]); },
    beginPath() {}, arc() {}, fill() {}, stroke() {}, save() {}, restore() {},
    ellipse() {}, moveTo() {}, lineTo() {}, closePath() {}, setLineDash() {},
    fillText() {}, strokeText() {}, clip() {}, translate() {}, rotate() {}, scale() {},
  };
}
const drawPipe = new Function('PIPE_W', fnSrc + '; return drawPipe;')(PIPE_W);

/** 画一根 pipe（x 固定 100），返回其矩形列表。**保留全部矩形，不裁剪 x**。 */
function rects(top) {
  const rec = [];
  const ctx = makeCtx(rec);
  // ⚠️ 必须与 drawGame() 里的两处调用逐字一致
  drawPipe(ctx, 100, 0, top + BIRD_R + 1);
  drawPipe(ctx, 100, top + GAP - BIRD_R, G_H);
  return rec;
}

const isBlocked = (y, top) => (y - BIRD_R <= top) || (y + BIRD_R >= top + GAP);

let bad = 0;
console.log(`从 app.js 读到的几何: GAP=${GAP}  BIRD_R=${BIRD_R}  PIPE_W=${PIPE_W}`);
console.log('判据（= DQN 的圆-矩形测试）: y−R <= top  或  y+R >= top+GAP');
console.log('比较范围: 全 x —— **不裁剪**（旧版裁在管体宽度内，漏掉了凸出的管口）\n');
console.log('top   | 画到 | 应挡 | 只画不挡 | 只挡不画 | 越出管宽');
console.log('-'.repeat(60));

for (const top of [70, 120, 200, 250, 300, 340, G_H - GAP - 150]) {
  const rec = rects(top);

  // 1) 断言：管子不许越出 [x, x+PIPE_W]
  let overhang = 0;
  for (const [x, y, w, h] of rec) {
    if (h <= 0) continue;
    if (x < 100 - 1e-9 || x + w > 100 + PIPE_W + 1e-9) overhang = Math.max(overhang, 1);
  }

  // 2) 逐像素：画到的集合 == 判定挡住的集合
  const drawn = new Set();
  for (const [x, y, w, h] of rec) {
    if (h <= 0) continue;
    for (let yy = Math.ceil(y); yy < Math.ceil(y + h); yy++) drawn.add(yy);
  }
  let onlyPaint = 0, onlyBlock = 0, painted = 0, should = 0;
  for (let y = 0; y <= Y_MAX; y++) {
    const b = isBlocked(y, top);
    if (b) should++;
    if (drawn.has(y)) painted++;
    if (drawn.has(y) && !b) onlyPaint++;
    if (!drawn.has(y) && b) onlyBlock++;
  }
  const ok = onlyPaint === 0 && onlyBlock === 0 && overhang === 0;
  if (!ok) bad++;
  console.log(`${String(top).padEnd(6)}| ${String(painted).padStart(4)} | `
    + `${String(should).padStart(4)} | ${String(onlyPaint).padStart(8)} | `
    + `${String(onlyBlock).padStart(8)} | ${overhang ? '是 ❌' : '否'}  ` + (ok ? '✅' : '❌'));
}
console.log('-'.repeat(60));
console.log(bad === 0
  ? '✅ 管子 = 一个矩形，与判定逐像素一致，且不越出管宽'
  : `❌ 有 ${bad} 个 top 值不一致`);
process.exit(bad === 0 ? 0 : 1);
