// 校验"画出来的管子"与"判定挡住的范围"是否严格一致 —— 以**几何边界**对账。
//
// 判定（与 demo/app.js 的 physics() 逐字一致）：
//     撞上管 ⟺ y - BIRD_R < p.top  或  y + BIRD_R > p.top + GAP
// 即圆心被挡的区间是 (-∞, p.top+R) ∪ (p.top+GAP-R, +∞)。
//
// 画出来的管子是像素块，几何范围是 [0, X) 与 [Y, G_H)。一致性条件是：
//     X == p.top + BIRD_R + 1        （上管的几何下沿 = 判定边界）
//     Y == p.top + GAP - BIRD_R - 1  （下管的几何上沿 = 判定边界）
// 并且管子不越出 [x, x+PIPE_W]。
//
// ⚠️ 历史盲区（别再犯）：旧版校验器把像素比较范围裁在 `[x, x+PIPE_W]` 内，
//    而当时的 drawPipeSprite 在管体两侧**各凸出 5px** 画粗管口 —— 凸出的 10px
//    **有画面、没判定**，而校验器因为裁剪压根没检查。现在管子是纯矩形，
//    且这里断言不越出管宽。
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
const GAP = grabConst('GAP'), BIRD_R = grabConst('BIRD_R'), PIPE_W = grabConst('PIPE_W');
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

/** drawGame() 里的两处调用，逐字照抄。返回各块的几何范围。 */
function pipeGeom(px, top) {
  const rec = [];
  const ctx = makeCtx(rec);
  drawPipe(ctx, px, 0, top + BIRD_R + 1);
  drawPipe(ctx, px, top + GAP - BIRD_R - 1, G_H);
  // ⚠️ rec 里是 **4 元组** [x, y, w, h]（drawPipe 记录的就是这个）。
  //    之前 filter/destructure 写成 5 元，h 恒为 undefined、solid 永远为空，
  //    报出 7 个假的"不一致"（踩过）。
  const solid = rec.filter(r => r[3] > 0);
  let x0 = Infinity, x1 = -Infinity;
  for (const [x, , w] of solid) { x0 = Math.min(x0, x); x1 = Math.max(x1, x + w); }
  // 每根管子会记两条（fillRect + strokeRect），而且顺序是"上管两条、下管两条"。
  // 用块索引区分：上管从 y=0 起，下管从 y>0 起。取"最高的下管记录"= 几何上沿。
  const ups = solid.filter(r => r[1] === 0);
  const dns = solid.filter(r => r[1] > 0);
  const up = ups[0];
  const dn = dns.reduce((a, b) => (a === undefined || b[1] < a[1] ? b : a), undefined);
  return {
    x0, x1,
    upBottom: up ? up[1] + up[3] : null,  // 上管几何下沿 = 块索引 + 高
    dnTop: dn ? dn[1] : null,             // 下管几何上沿 = 块索引
  };
}

let bad = 0;
console.log(`从 app.js 读到: GAP=${GAP}  BIRD_R=${BIRD_R}  PIPE_W=${PIPE_W}`);
console.log('判定边界: 上管几何下沿应为 top+R+1 ；下管几何上沿应为 top+GAP-R-1\n');
console.log('top   | 上管下沿 / 应为 | 下管上沿 / 应为 | x 范围      | 结果');
console.log('-'.repeat(74));

for (const top of [70, 120, 200, 250, 300, 340, G_H - GAP - 150]) {
  const g = pipeGeom(100, top);
  const wantUp = top + BIRD_R + 1;
  const wantDn = top + GAP - BIRD_R - 1;
  const okUp = g.upBottom === wantUp;
  const okDn = g.dnTop === wantDn;
  const okX = g.x0 >= 100 - 1e-9 && g.x1 <= 100 + PIPE_W + 1e-9;
  const ok = okUp && okDn && okX;
  if (!ok) bad++;
  console.log(`${String(top).padEnd(6)}| ${String(g.upBottom).padStart(5)} / ${String(wantUp).padStart(5)} `
    + `${okUp ? '✅' : '❌'} | ${String(g.dnTop).padStart(5)} / ${String(wantDn).padStart(5)} `
    + `${okDn ? '✅' : '❌'} | [${g.x0},${g.x1}]`.padEnd(14)
    + `| ${ok ? '✅' : '❌' + (okX ? '' : ' 越出管宽')}`);
}
console.log('-'.repeat(74));
console.log(bad === 0
  ? '✅ 管子是纯矩形、不越出管宽，且上下两个几何边界与判定边界严格相等'
  : `❌ 有 ${bad} 个 top 值不一致`);
process.exit(bad === 0 ? 0 : 1);
