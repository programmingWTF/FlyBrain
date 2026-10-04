// 逐像素对账：**服务端判定的碰撞** vs **前端画出来的管子**。
//
// 为什么需要这个脚本
// ------------------
// 用户报过："管子上下两端的体积好像还是有点偏差，我看已经撞上了，但是没有判定为失败"。
// 根因是"画到的像素集合"与"判定为撞的 y 集合"不相等 —— 差一两个像素就足够
// 让人肉眼觉得"明明撞上了"。当时写了 `verify_collision_pixels.js` 守着，
// 但它读的是 `app.js` 里的 `physics()`。
//
// 现在物理搬到服务端了（见 server.py 的 GameWorld），判据也一起搬了过去。
// 如果校验还盯着 app.js，它就会**静默失效**（app.js 里已经没有那个判据了）——
// 这类"校验悄悄不再校验"的坑很危险，所以这里改成读 server.py。
//
// 用法: node demo/verify_server_collision.js
const fs = require('fs');
const path = require('path');

const appSrc = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');
const srvSrc = fs.readFileSync(path.join(__dirname, 'server.py'), 'utf8');

// ---- 从 server.py 的 GameWorld 里取常数
function grabPy(name) {
  const m = srvSrc.match(new RegExp('^\\s*' + name + '\\s*=\\s*(-?[0-9.]+)', 'm'));
  if (!m) throw new Error(`server.py 里找不到常量 ${name}`);
  return Number(m[1]);
}
const GAP = grabPy('GAP'), R = grabPy('BIRD_R'), PIPE_W = grabPy('PIPE_W');
const G_H = grabPy('G_H'), BIRD_X = grabPy('BIRD_X');

// ---- 从 server.py 抽出那条判定（跨两行的 or）
const lines = srvSrc.split('\n');
const iIf = lines.findIndex(l => l.includes('self.BIRD_R <= p["top"] + self.BIRD_R')
  && l.trim().startsWith('if ('));
if (iIf < 0) throw new Error('server.py 里抽不到碰撞判定行');
let raw = lines[iIf].trim().replace(/^if \(/, '');
let k = iIf;
while (!/撞上管子/.test(raw) && k + 1 < lines.length) {
  raw += ' ' + lines[++k].trim();
}
// 去掉结尾的 `): return self.die("撞上管子")`
raw = raw.replace(/\)\s*:\s*return self\.die\("撞上管子"\)\s*$/, '');
// Python → JS：换属性写法、`or` → `||`
const judgedExpr = raw
  .replace(/self\./g, '')
  .replace(/p\["top"\]/g, 'p.top')
  .replace(/\bor\b/g, '||')
  .replace(/\s+/g, ' ')
  .trim();
console.log('从 server.py 抽到的判定：');
console.log('  ' + judgedExpr + '\n');

// ---- 从 app.js 抽 drawPipe 与它的调用（画法必须照旧用真代码）
const i0 = appSrc.indexOf('function drawPipe(');
if (i0 < 0) throw new Error('app.js 里找不到 drawPipe');
const fnSrc = appSrc.slice(i0, i0 + /\n\}/.exec(appSrc.slice(i0)).index + 2);

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

/** drawGame 里的两处调用（逐字照抄）：fillRect(y,h) 覆盖像素 y … y+h-1。 */
function paintedRows(top, px) {
  const rec = [];
  const ctx = makeCtx(rec);
  drawPipe(ctx, px, 0, top + R + 1);                 // 上管
  drawPipe(ctx, px, top + GAP - R - 1, G_H);         // 下管
  const rows = new Set();
  for (const [x, y, w, h] of rec) {
    if (h <= 0) continue;
    const l = Math.max(x, BIRD_X), r = Math.min(x + w, BIRD_X + 1);
    if (r - l < 1) continue;
    for (let yy = y; yy <= y + h - 1; yy++) rows.add(yy);
  }
  return rows;
}

/** 鸟圆周覆盖的像素行： [ceil(y-R), floor(y+R)]（闭区间）。 */
const birdRows = (y) => {
  const a = Math.ceil(y - R), b = Math.floor(y + R);
  return a > b ? null : [a, b];
};

const judged = new Function('y', 'p', 'BIRD_R', 'GAP', 'return (' + judgedExpr + ');');

console.log(`几何: GAP=${GAP} BIRD_R=${R} PIPE_W=${PIPE_W} birdX=${BIRD_X}`);
console.log('管子 x 盖住鸟时，逐像素比较"画到的行"与"判定为撞的 y"：\n');
console.log('  top | 画到的行数 | 判定为撞的行数 | 只画不判 | 只判不画');

let mismatch = 0, checked = 0;
// ⚠️ top 只扫**真实游戏能生成的量程**（server.py 的 `_next_gap_top`：70..286）。
//    我第一版扫到了 470，于是"下管起点 470+184-17-1 = 636 > 画布 620，
//    下管根本没画"被算成 32 处"只画不判"—— 又是一个**测试范围错了**导致的假报。
//    真实量程的上界 286 对应下管起点 469 < 620，所以下管**永远在画面内**，
//    这个几何性质是成立的（下面那条断言就是在守它）。
const TOP_MIN = 70, TOP_MAX = 286;                   // = server.py 的 _next_gap_top 量程
console.log(`top 扫描范围 ${TOP_MIN}..${TOP_MAX}（真实关卡生成量程）`);
const bottomStart = (top) => top + GAP - R - 1;
if (bottomStart(TOP_MAX) >= G_H) {
  console.log(`❌ 下管起点 ${bottomStart(TOP_MAX)} >= 画布高 ${G_H}：`
    + ' 会出现"画面外的管子"，玩家看不见墙却会撞死');
  process.exit(1);
}
console.log(`  量程上界的下管起点 ${bottomStart(TOP_MAX)} < 画布高 ${G_H}  ✅ 下管永远可见\n`);
for (const top of [70, 85, 101, 120, 150, 180, 210, 240, 260, 286]) {
  const rows = paintedRows(top, 100);
  let paintedHit = 0, judgedHit = 0, onlyPaint = 0, onlyJudge = 0;
  // ⚠️ y 只扫**物理上可达**的范围。
  //    第一版我扫了 -20..640，于是 y∈[-20,-3] 这些"鸟整个在画面外"的位置
  //    被算成"判定为撞但没画到" → 假报了 14 处。实际鸟心最高只能到 18
  //    （再高就先判"撞到天花板"），最低 606（地面线 606-14=606）。
  //    测试范围错了会把**好代码判成坏代码** —— 这一段就是踩过的记录。
  for (let y = 0; y <= G_H; y += 0.5) {
    const br = birdRows(y);
    if (!br) continue;
    const painted = (() => {
      for (let yy = br[0]; yy <= br[1]; yy++) if (rows.has(yy)) return true;
      return false;
    })();
    const jd = !!judged(y, { top }, R, GAP);
    if (painted) paintedHit++;
    if (jd) judgedHit++;
    if (painted && !jd) onlyPaint++;
    if (!painted && jd) onlyJudge++;
    checked++;
  }
  mismatch += onlyPaint + onlyJudge;
  console.log(`  ${String(top).padStart(4)} | ${String(paintedHit).padStart(10)} | `
    + `${String(judgedHit).padStart(14)} | ${String(onlyPaint).padStart(8)} | ${String(onlyJudge).padStart(8)}`);
}

console.log(`\n共比较 ${checked} 个 y 位置`);
if (mismatch === 0) {
  console.log('✅ 服务端判定的碰撞与前端画出来的管子**逐像素等价**');
  console.log('   （看着碰到就一定判定到，反之亦然 —— 用户报的那个洞被守住了）');
  process.exit(0);
} else {
  console.log(`❌ 有 ${mismatch} 处只画不判 / 只判不画 —— 玩家会看到"撞上了却没死"`);
  process.exit(1);
}
