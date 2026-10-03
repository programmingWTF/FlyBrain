/**
 * 果蝇逃跑反射可交互演示 —— 前端
 *
 * 左边：游戏。管子 = 迎面撞来的物体，反射触发 = 拍翅。
 * 右边：按 FlyWire 骨架**真实解剖坐标**摆出来的 5.9 万个神经元，随脑活动闪烁。
 *
 * 所有神经计算都在后端跑真代码（src/fpv/spiking_brain.py + looming.py），
 * 这里只负责物理、取数、画。
 */
import * as THREE from 'three';

const $ = (id) => document.getElementById(id);
const R50 = 0.577;                 // 脚本量到的 DNp01 激发阈值（满量程比例）
const PX_PER_M = 240;              // 世界尺度：1 米 = 240 像素
const TICK_S = 0.02;               // 脑的时钟：20ms/tick
const FLAP_COOLDOWN = 0.14;        // 两次拍翅最小间隔（秒）

// 严格模式（?strict=1）：与评测台同构的"脑决策 → 1 步物理"驱动。
// 用途是**判定页面与评测台的差异到底在时序还是在算法**，不是给玩家用的模式。
const STRICT = /[?&]strict=1\b/.test(location.search);
//: 严格模式跑多少个 tick 就停（用于量分）。5000 tick = 100s 游戏时间。
const STRICT_TICKS = 5000;
// ?batch=N：一次向脑请求 N 个 tick（用于量"批大小 vs 速率/分数"的取舍）
const BATCH_Q = Number((location.search.match(/[?&]batch=(\d+)/) || [])[1] || 0);

const P = { s50: 15, s50size: 30, rad: 0.05, spd: 1.0, need: 1, graph: 'real',
            mode: 'flappy', tmax: 40, show3d: true };
// ---------------------------------------------------------------- 双向逼近反射
// 页面上的 Flappy 用的是"双向逼近反射 + 按执行器带宽生成关卡"这一档
// （ESCAPE.md §6/§7）。它比原来那套"最近碰撞 + LC4 角速度"高一个数量级：
//   原来 1.54 分（86% 的局 ≤2 分） → 现在 29.5 分（中位 21，19% 的局 ≤2 分）
// 神经计算仍然**全在后端**，这里只负责把"望远镜几何"算出来送过去：
//   · 方向由**缺口在视野里的高低**决定（缺口比鸟高 → 腹侧支爬升；
//     比鸟低超过一个死区 → 背侧支下潜）—— 纯几何，没有方向控制器
//   · vy_gate：地面只在下落时逼近、天花板只在上升时逼近（纯物理）
// 关卡生成加上"相邻缺口的向上跳变 ≤ max_climb"的可达性约束：
// 鸟的可持续爬升率只有 ~100 px/s（一次拍翅买 49px、周期 ~0.4s），
// 管距 1.25s 只能爬 ~125px，而缺口的随机跳变有 330px 量程 ——
// 超出这个带宽的关卡在给定物理下无论如何都飞不进去。
const FLAPPY = { maxClimb: 40, gapMargin: 18, vyGate: 1, ceilBoost: 1.0,
                 groups: ['LC4', 'LPLC2'], s50size: 30, n: 3,
                 // 本轮调出的两个旋钮（**外部介入**，不是连接组事实）：
                 //   dorsScale=0.35：背侧（下潜）支按 0.35 缩 —— 它会把鸟推去撞天花板；
                 //                   实测越强越差（1.0 时 80 局里 24 局撞天花板）
                 //   ventGain=2.0 ：缺口明显在上方时放大腹侧（爬升）驱动 ——
                 //                   撞管死亡 100% 是"偏低没爬够"
                 dorsScale: 0.35, ventGain: 2.0, ventDev: 120,
                 //: 一次向脑请求几个 tick。见主循环里的说明。
                 //
                 // 实测（无头 Edge，42~90 秒窗口，同一台机）：
                 //   batch=1 → 14.3 tick/s，最高 6 分
                 //   batch=4 → 20.4 tick/s，最高 19 分   ← 取它
                 //   batch=8 → 15.1 tick/s，最高 10 分
                 // batch=4 明显最好：批太小喂不饱脑（受 fetch 往返 ~70ms 限制），
                 // 批太大则视觉输入滞后太多、控制变钝。4 是这两者的折中。
                 batch: 4 };
const S = { info: null, coords: null, ready: false };

// ---------------------------------------------------------------- 游戏状态
const G = {
  W: 520, H: 620, birdX: 120, y: 300, vy: 0,
  pipes: [], score: 0, best: 0, dead: false, deadT: 0, cause: '',
  acc: 0, last: 0, cooldown: 0, spawnT: 0,
  threatD: null, dodged: false, miss: 0, dodgeT: 0, cov: 0, hold: 0, geo: null,
  scroll: 0, wingPhase: 0, started: false, manual: false,
  resp: null, hist: [], flashT: 0, dnFlash: 0,
};
const GRAV = 1180, FLAP_V = -340, PIPE_W = 62, GAP = 168;

/** 双向逼近反射的"望远镜几何"：只算纯几何量，交给后端决定驱动哪些细胞。
 *
 *  方向：缺口比鸟高 → 威胁在下方（地面）→ 腹侧半视野；缺口比鸟低 → 天花板 → 背侧。
 *  vy_gate 在后端做（地面只在下落时逼近、天花板只在上升时逼近）。
 *  距离沿用同一套"等效半宽 0.55"的写法，与 scripts/flappy_bench.py 一致。
 */
function bidiBody(th, ticks) {
  const gapC = nearestPipe() ? nearestPipe().top + GAP / 2 : G.H / 2;
  const up = gapC < G.y - FLAPPY.gapMargin;
  const hPx = Math.max(up ? (G.H - GROUND) - G.y : G.y, 1);
  const dist = Math.max(hPx / PX_PER_M, 0.02);
  const thetaDeg = 180 / Math.PI * 2 * Math.atan(0.55 / dist);
  const n = FLAPPY.n, s50 = FLAPPY.s50size;
  const x = Math.pow(Math.max(thetaDeg, 0), n);
  const amp = x / (x + Math.pow(s50, n));
  G.geo = { theta_deg: thetaDeg,
            dtheta_dps: 180 / Math.PI * (0.55 / Math.max(dist, 1e-3)),
            tau_s: dist };
  G.cov = amp;
  return { ticks, need_spikes: P.need,
           drives: [{ type: 'bidi', y: G.y, vy: G.vy, gap: gapC,
                      ground_y: G.H - GROUND, s50size: s50, n: n,
                      gap_margin: FLAPPY.gapMargin, vy_gate: FLAPPY.vyGate,
                      ceil_boost: FLAPPY.ceilBoost, groups: FLAPPY.groups,
                      dors_scale: FLAPPY.dorsScale, vent_gain: FLAPPY.ventGain,
                      vent_dev: FLAPPY.ventDev }] };
}

/** 当前威胁的几何：返回 {distM, speed, radius} 或 null。
 *  reflex 模式：一个正对苍蝇飞来的暗盘（这就是文献里的逼近刺激）。
 *  flappy  模式：见 bidiBody()（双向反射不看"最近碰撞"，看缺口在视野的哪一半）。 */
function threat() {
  if (P.mode === 'fly3d') return geom3d();
  if (P.mode !== 'flappy') {
    if (G.threatD == null) return null;
    return { distM: Math.max(G.threatD, 0.02), speed: P.spd, radius: P.rad };
  }
  const vpx = P.spd * PX_PER_M;
  const cands = [];
  const near = nearestPipe();
  if (near) cands.push({ dpx: near.x + PIPE_W - G.birdX, vpx: vpx, rad: P.rad });
  const groundY0 = G.H - GROUND;
  if (G.vy > 0) cands.push({ dpx: groundY0 - G.y, vpx: G.vy, rad: 0.55 });
  if (G.vy < 0) cands.push({ dpx: G.y, vpx: -G.vy, rad: 0.55 });
  let best = null;
  for (const c of cands) {
    const ttc = c.dpx / Math.max(c.vpx, 1e-6);
    if (c.dpx > 0 && (!best || ttc < best.ttc)) best = { ...c, ttc };
  }
  if (!best) return null;
  return { distM: Math.max(best.dpx / PX_PER_M, 0.02),
           speed: Math.max(best.vpx / PX_PER_M, 0.02), radius: best.rad };
}

function disp() {
  const r = S.resp; if (!r) return null;
  return Object.assign({}, G.geo || {}, r);
}

function coverage(width) {
  // mean_{u in [0,1]} exp(-((u-0.5)/(width/2.355))^2/2)，数值积分
  let sum = 0;
  for (let i = 0; i < 101; i++) {
    const u = i / 100;
    sum += Math.exp(-Math.pow((u - 0.5) / Math.max(width / 2.355, 1e-6), 2) / 2);
  }
  return sum / 101;
}

const HOLD_S = 0.7;      // 逼近到最近点后**保持**多久
function newThreat() {
  G.threatD = 2.2;                       // 从 2.2 米外开始逼近
  G.hold = 0;
  G.dodged = false;
}

function resetGame() {
  G.started = false; G.readyT = performance.now();
  G.y = 300; G.vy = 0; G.pipes = []; G.dead = false;
  G.deadT = 0; G.acc = 0; G.cooldown = 0; G.spawnT = 0; G.hist = [];
  // 每一局都要归零。之前一次改动把这行弄丢了，导致撞死后分数跨局累加、
  // "最高"也跟着变成累计值 —— 看起来就是得分算错。
  G.score = 0;
  if (P.mode !== 'flappy') {
    G.y = 310; newThreat();
  } else {
    // 开局就得有管子、而且不能太远：第一版栽在两个自锁上 ——
    //  (a) 没管子 -> 没刺激 -> 反射不触发 -> 摔死 -> 重置 -> 还是没管子；
    //  (b) 管子从 1.4 米外开始，而鸟 0.72 秒落地，反射来不及救它。
    G.pipes.push({ x: G.birdX + 0.75 * PX_PER_M - PIPE_W, top: 250, passed: false });
  }
}

/** 关卡生成：加上"相邻缺口的向上跳变 ≤ maxClimb"的可达性约束。
 *
 *  为什么必须加（ESCAPE.md §6.1~6.3 实测）：鸟的可持续爬升率只有 ~100 px/s，
 *  管距 300px / 240px·s⁻¹ = 1.25s，一个间隔最多爬 ~125px；而原来
 *  `70 + rand*(G.H-GAP-150)` 的缺口中心跳变有 330px 量程 —— 超出的关卡
 *  在给定物理下无论如何都飞不进去（早死局实测 42/53 是"鸟偏低没爬够"）。
 *  **向下不设限**：自由落体快得多（1.25s 可掉 921px，远超量程）。
 *  管宽、缺口高、管速、重力一个都没改。
 */
function spawnPipe() {
  const lo = 70, hi = G.H - GAP - 150;          // 与原来同一个分布范围
  const last = G.pipes[G.pipes.length - 1];
  let gapTop, x;
  if (!last) {
    // ⚠️ 第 1 根必须**和评测台逐字一致**：top=250（缺口中心 334），
    // 而且开局就在 x = birdX + 0.75*PX_PER_M - PIPE_W = 238，
    // 不是生成在屏幕右缘（550）等 5.2 秒才到。
    //
    // 原来这里是"随机 gapTop + 生成在右缘"。随机让缺口中心可能远在鸟上方
    // （中心 154~386 vs 鸟起始 300），而 ventGain=2.0 会为追它猛爬、
    // 冲过缺口后拉不回来 —— 实测页面 100% 死在管 1/管 2（死亡 y 都在 190~232，
    // 即"爬过头"）。评测台用固定 250，开局是"鸟比缺口中心高 34px"这种温和局面。
    gapTop = 250;
    x = G.birdX + 0.75 * PX_PER_M - PIPE_W;
  } else {
    const prevC = last.top + GAP / 2;           // 上一根缺口中心
    const topMax = Math.min(hi, prevC + FLAPPY.maxClimb - GAP / 2);
    gapTop = topMax > lo ? lo + Math.random() * (topMax - lo) : lo;
    x = G.W + 30;
  }
  G.pipes.push({ x, top: gapTop, passed: false });
}

// ---------------------------------------------------------------- 后端通信
async function post(path, body) {
  const r = await fetch(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  return r.json();
}

let inFlight = false;
function stepBrain(ticks) {
  DBG.sbCalls = (DBG.sbCalls || 0) + 1;      // 排障：被调用次数
  if (inFlight) { DBG.sbBusy = (DBG.sbBusy || 0) + 1; return false; }
  // Flappy 走双向反射（方向由缺口在视野里的高低决定），不需要"最近碰撞"那套几何。
  const th = (P.mode === 'flappy') ? { bidi: true } : threat();
  if (!th) { DBG.sbNoThreat = (DBG.sbNoThreat || 0) + 1; return false; }
  inFlight = true;
  // 几何量是前端算的（后端只收"每个群发多少"），所以显示要自己留一份
  if (!th.bidi) {
    G.geo = {
      theta_deg: 180 / Math.PI * 2 * Math.atan(th.radius / Math.max(th.distM, 1e-3)),
      dtheta_dps: 180 / Math.PI * (2 * th.radius * th.speed /
                                   (th.distM ** 2 + th.radius ** 2)),
      tau_s: th.distM / Math.max(th.speed, 1e-6),
    };
  }
  let body;
  if (P.mode === 'fly3d') {
    // 整面墙都驱动，只有洞那块不驱动：把洞的视网膜方位/角半径换算成视野坐标
    const g = threat();
    if (!g) return false;
    const c = Math.min(0.98, Math.max(0.02, 0.5 + g.elevDeg / FOV_DEG));
    const width = Math.max(0.03, 2 * g.thetaDeg / FOV_DEG);
    const ampV = Math.min(1, Math.max(0, Math.pow(g.dtheta_dps, 3) /
                   (Math.pow(g.dtheta_dps, 3) + Math.pow(P.s50, 3))));
    const ampS = Math.min(1, Math.max(0, Math.pow(g.thetaDeg * 2, 3) /
                   (Math.pow(g.thetaDeg * 2, 3) + Math.pow(P.s50size, 3))));
    G.cov = 1 - coverage(width);
    body = { ticks, need_spikes: P.need,
             drives: [{ group: 'LC4', amp: ampV, center: c, width, hole: 1 },
                      { group: 'LPLC2', amp: ampS, center: c, width, hole: 1 }] };
  } else if (P.mode === 'retino') {
    // 威胁只覆盖它视野里该覆盖的那一块：按角尺寸换算成视野占比 width，
    // 用高斯窗只驱动偏好位置与之重叠的那批 LC4。
    // 一个威胁只落在视野的一小块上 -> 只有偏好位置重叠的那批细胞被驱动。
    // 两条逼近通道一起给：LC4=角速度、LPLC2=角大小（GF 的真实输入结构）。
    const width = Math.min(1.6, Math.max(0.05, P.tmax / 180));
    G.cov = coverage(width);
    body = { ticks, need_spikes: P.need,
             drives: [{ group: 'LC4', amp: loomAmp(th), center: 0.5, width },
                      { group: 'LPLC2', amp: sizeAmp(th), center: 0.5, width }] };
  } else {
    // Flappy：双向逼近反射（见 bidiBody 的说明）。物理常数与判据都没变，
    // 只是把"驱动哪些细胞"从"最近碰撞"换成"缺口落在视野的哪一半"。
    body = bidiBody(th, ticks);
  }
  post('/api/step', body).then(r => {
    DBG.sbOk = (DBG.sbOk || 0) + 1;
    if (r.error) { $('stat').textContent = '后端错误：' + r.error; return; }
    S.resp = r;
    G.hist.push(r.drive ?? r.drive_max ?? 0);
    if (G.hist.length > 200) G.hist.shift();
    if (r.flap && G.cooldown <= 0 && !G.dead) {
      G.cooldown = FLAP_COOLDOWN; G.flashT = 0.12;
      if (P.mode === 'fly3d') { F3.vz = FLAP3; G.flashT = 0.12; }
      else if (P.mode !== 'flappy') {
        if (!G.dodged) { G.dodged = true; G.score++; G.best = Math.max(G.best, G.score); }
        G.dodgeT = 0.3;
      } else {
        G.vy = FLAP_V;
      }
    }
    if (r.dn01_recent > 0) G.dnFlash = 0.2;
  }).catch(e => {
    DBG.sbErr = (DBG.sbErr || 0) + 1;
    DBG.sbErrMsg = String(e).slice(0, 120);
    $('stat').textContent = '后端连接断了：' + e;
  }).finally(() => { inFlight = false; DBG.sbFin = (DBG.sbFin || 0) + 1; });
  return true;
}

function loomAmp(th) {
  const d = Math.max(th.distM, 1e-3);
  const dtheta = 180 / Math.PI * (2 * th.radius * th.speed / (d * d + th.radius ** 2));
  const n = 3;
  return Math.min(1, Math.max(0, Math.pow(dtheta, n) / (Math.pow(dtheta, n) + Math.pow(P.s50, n))));
}

/** 角大小通道（LPLC2）：文献里 LPLC2 编码的是角尺寸而不是扩张速度，
 *  且对平移不响应（Klapoetke 2017 / Ache 2019）。 */
function sizeAmp(th) {
  const d = Math.max(th.distM, 1e-3);
  const theta = 180 / Math.PI * 2 * Math.atan(th.radius / d);
  const n = 3;
  return Math.min(1, Math.max(0, Math.pow(theta, n) /
         (Math.pow(theta, n) + Math.pow(P.s50size, n))));
}

function nearestPipe() {
  let best = null;
  for (const p of G.pipes) if (p.x + PIPE_W > G.birdX - 6 && (!best || p.x < best.x)) best = p;
  return best;
}

// ---------------------------------------------------------------- 物理
function physics(dt) {
  if (P.mode === 'fly3d') {
    G.cooldown = Math.max(0, G.cooldown - dt);
    G.flashT = Math.max(0, G.flashT - dt);
    physics3d(dt);
    return;
  }
  G.cooldown = Math.max(0, G.cooldown - dt);
  G.flashT = Math.max(0, G.flashT - dt);
  G.dodgeT = Math.max(0, G.dodgeT - dt);
  G.dnFlash = Math.max(0, G.dnFlash - dt);

  if (P.mode !== 'flappy') {
    // 悬停 + 逼近盘；反射触发就往上闪避。没有重力，所以这个模式一定看得见效果。
    // 经典逼近范式：扩张到最近点后**保持一段时间**再撤。
    // 不保持的话，最高驱动区只持续 2~3 个 tick（40~60ms），
    // 而脑的泄漏积分需要 ~8 个 tick 才越过阈值 —— 那是时钟分辨率问题，
    // 不是"脑不躲"。
    if (G.threatD != null && G.threatD > 0.05) G.threatD -= P.spd * dt;
    else if (G.threatD != null) {
      G.hold += dt;
      if (G.hold > HOLD_S) {
        if (!G.dodged) { G.miss++; G.cause = '没逃掉'; G.deadT = 0.5; }
        newThreat();
      }
    }
    const bob = Math.sin(performance.now() / 260) * 5;
    const dodge = G.dodgeT > 0 ? Math.sin((0.3 - G.dodgeT) / 0.3 * Math.PI) * 95 : 0;
    G.y = 310 + bob - dodge;
    return;
  }

  if (G.dead) { G.deadT += dt; if (G.deadT > 1.6) resetGame(); return; }
  if (!G.started) {                      // 前 1.2 秒悬空展示，然后交给反射
    G.scroll += 60 * dt; G.y = 300 + Math.sin(performance.now() / 300) * 10;
    G.wingT = (G.wingT || 0) + dt; G.wingPhase = Math.floor(G.wingT * 8) % 3;
    if (G.readyT === undefined) G.readyT = performance.now();
    if (performance.now() - G.readyT > 1200) { G.started = true; delete G.readyT; }
    return;
  }
  G.scroll += P.spd * PX_PER_M * dt;
  G.wingT = (G.wingT || 0) + dt; G.wingPhase = Math.floor(G.wingT * 9) % 3;
  G.vy += GRAV * dt;
  G.y += G.vy * dt;
  const vpx = P.spd * PX_PER_M;
  // 按**间距**而不是时间生成：否则管速一慢，鸟会在下一根管子出现前摔死，
  // 而"没管子 = 没逼近刺激 = 反射不触发"，看起来就像脑子的错。
  const spacing = 300;
  const lastP = G.pipes[G.pipes.length - 1];
  if (!lastP || lastP.x < G.W - spacing) spawnPipe();
  for (const p of G.pipes) {
    p.x -= vpx * dt;
    if (!p.passed && p.x + PIPE_W < G.birdX) { p.passed = true; G.score++; G.best = Math.max(G.best, G.score); }
  }
  G.pipes = G.pipes.filter(p => p.x > -PIPE_W - 10);
  if (G.y > G.H - 14 || G.y < 6) return die(G.y > G.H - 14 ? '撞到地面' : '撞到天花板');
  for (const p of G.pipes) {
    if (G.birdX + 11 > p.x && G.birdX - 11 < p.x + PIPE_W) {
      if (G.y - 11 < p.top || G.y + 11 > p.top + GAP) return die('撞上管子');
    }
  }
}
function die(cause) { if (!G.dead) { G.dead = true; G.cause = cause; G.deadT = 0; } }

// ============================================================ 经典 FlappyBird 画面
const GROUND = 92;                       // 地面高度（像素）
function drawSky(g) {
  const gr = g.createLinearGradient(0, 0, 0, G.H - GROUND);
  gr.addColorStop(0, '#4ec0ca'); gr.addColorStop(0.7, '#8fd8de');
  gr.addColorStop(1, '#cdeef0');
  g.fillStyle = gr; g.fillRect(0, 0, G.W, G.H - GROUND);
  // 云：两层视差
  g.fillStyle = 'rgba(255,255,255,.85)';
  for (let i = 0; i < 5; i++) {
    const x = ((i * 173 - G.scroll * 0.25) % (G.W + 140) + G.W + 140) % (G.W + 140) - 70;
    const y = 60 + (i % 3) * 52;
    g.beginPath();
    g.ellipse(x, y, 34, 15, 0, 0, 7); g.ellipse(x + 26, y + 5, 24, 12, 0, 0, 7);
    g.ellipse(x - 24, y + 6, 20, 11, 0, 0, 7); g.fill();
  }
  g.fillStyle = 'rgba(255,255,255,.45)';
  for (let i = 0; i < 4; i++) {
    const x = ((i * 231 - G.scroll * 0.12) % (G.W + 200) + G.W + 200) % (G.W + 200) - 100;
    g.beginPath(); g.ellipse(x, 150 + (i % 2) * 70, 46, 18, 0, 0, 7); g.fill();
  }
}

/** 画一根管子。**管口（粗的那圈"帽"）朝着缺口**，与标准 FlappyBird 一致：
 *      上管：帽在**下端**（朝下，贴着 top）    下管：帽在**上端**（朝上，贴着 top+GAP）
 *
 *  参数用**两个边界**描述，避免"长度算两次"：
 *      nearY = 贴缺口那一端 = 碰撞判定用的边界（上管 top / 下管 top+GAP）
 *      farY  = 出画那一端（上管 0 / 下管画面底）
 *
 *  实现刻意写得最笨：管体先按"从 farY 到 nearY **整段**"画满，再把**靠近缺口
 *  的 capH 那一段**换成管口。这样"管子的实心范围"永远恰好是 [farY, nearY)，
 *  和碰撞判定用的区间逐像素对齐 —— 不会再出现前几版那种"帽让位让错方向、
 *  管子比判定短 30px（缺口里于是有一条撞得到却画不出的缝）"。
 *  `demo/verify_pipe_visual.js` 逐像素守着这一点。
 *
 *  @param nearY    包住缺口那一侧的**实心末端**（= 画到碰撞判定真正挡住的位置）
 *  @param farY     出画那一端（上管 0 / 下管画面底）
 *  @param capInset 管口（粗的那圈）再往里挪多少像素 —— 让管口正好落在缺口侧边缘，
 *                  与标准 FlappyBird 的"管口朝着缺口"一致。传 0 就是紧贴 nearY。
 */
function drawPipeSprite(g, x, nearY, farY, capInset) {
  const capW = PIPE_W + 10, capH = 30;
  const ins = capInset || 0;
  const down = farY > nearY;                        // 管子从 nearY 往 +y 长（下管）
  const bodyStart = Math.min(nearY, farY);
  const bodyLen = Math.abs(nearY - farY);
  const body = g.createLinearGradient(x, 0, x + PIPE_W, 0);
  body.addColorStop(0, '#8ce350'); body.addColorStop(0.25, '#74bf2e');
  body.addColorStop(0.85, '#4e8a1c'); body.addColorStop(1, '#3d6d16');
  g.fillStyle = body; g.fillRect(x, bodyStart, PIPE_W, bodyLen);
  g.strokeStyle = '#2f5212'; g.lineWidth = 2; g.strokeRect(x, bodyStart, PIPE_W, bodyLen);
  // 管口：占靠近 nearY 的 capH，再往里挪 ins（绝不越过 nearY）
  const capY = down ? (nearY + ins) : (nearY - capH - ins);
  const cg = g.createLinearGradient(x - 5, 0, x + capW - 5, 0);
  cg.addColorStop(0, '#96ee58'); cg.addColorStop(0.3, '#74bf2e');
  cg.addColorStop(1, '#3d6d16');
  g.fillStyle = cg; g.fillRect(x - 5, capY, capW, capH);
  g.strokeRect(x - 5, capY, capW, capH);
  g.lineWidth = 1;
}

function drawGround(g) {
  const y = G.H - GROUND;
  g.fillStyle = '#ded895'; g.fillRect(0, y, G.W, GROUND);
  g.fillStyle = '#73bf2e'; g.fillRect(0, y, G.W, 14);
  g.fillStyle = '#5a9c22';
  for (let x = -40; x < G.W + 40; x += 24) {
    const sx = x - (G.scroll % 24);
    g.beginPath(); g.moveTo(sx, y + 14); g.lineTo(sx + 12, y + 14);
    g.lineTo(sx + 6, y + 22); g.fill();
  }
  g.strokeStyle = 'rgba(120,110,60,.5)';
  for (let x = -40; x < G.W + 40; x += 18) {
    const sx = x - (G.scroll % 18);
    g.beginPath(); g.moveTo(sx, y + 24); g.lineTo(sx + 10, G.H); g.stroke();
  }
}

function drawBirdSprite(g) {
  const r = Math.max(-0.55, Math.min(1.1, G.vy / 430));
  g.save(); g.translate(G.birdX, G.y); g.rotate(r);
  g.fillStyle = '#f7d51d'; g.strokeStyle = '#5c4708'; g.lineWidth = 2;
  g.beginPath(); g.ellipse(0, 0, 17, 13, 0, 0, 7); g.fill(); g.stroke();
  g.fillStyle = '#fdf3c0'; g.beginPath(); g.ellipse(-2, 5, 11, 6, 0, 0, 7); g.fill();
  // 翅膀：三相位扇动
  const ph = [0.9, 0.1, -0.7][G.wingPhase];
  g.save(); g.translate(-4, -1); g.rotate(ph);
  g.fillStyle = '#f0a81c'; g.strokeStyle = '#5c4708';
  g.beginPath(); g.ellipse(-6, 0, 10, 6, 0, 0, 7); g.fill(); g.stroke(); g.restore();
  g.fillStyle = '#fff'; g.beginPath(); g.arc(8, -4, 5.5, 0, 7); g.fill();
  g.fillStyle = '#222'; g.beginPath(); g.arc(9.5, -4, 2.4, 0, 7); g.fill();
  g.fillStyle = '#f07f18'; g.strokeStyle = '#a4530b'; g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(14, -1); g.lineTo(25, 2); g.lineTo(14, 6); g.closePath();
  g.fill(); g.stroke();
  g.restore();
}

function drawScoreBig(g) {
  if (!G.started) return;
  g.font = 'bold 46px "Trebuchet MS", system-ui'; g.textAlign = 'center';
  g.lineWidth = 6; g.strokeStyle = '#5c4708'; g.fillStyle = '#fff';
  g.strokeText(String(G.score), G.W / 2, 74); g.fillText(String(G.score), G.W / 2, 74);
  g.textAlign = 'left'; g.lineWidth = 1;
}

function drawReady(g) {
  g.fillStyle = 'rgba(0,0,0,.25)'; g.fillRect(0, 0, G.W, G.H - GROUND);
  g.textAlign = 'center';
  g.font = 'bold 34px "Trebuchet MS", system-ui';
  g.lineWidth = 6; g.strokeStyle = '#5c4708'; g.fillStyle = '#fff';
  g.strokeText('冻结果蝇脑 · Flappy', G.W / 2, 190);
  g.fillText('冻结果蝇脑 · Flappy', G.W / 2, 190);
  g.font = '15px system-ui'; g.lineWidth = 0; g.fillStyle = '#0b2a2e';
  g.fillText('脑完全冻结、零可学参数 —— 拍翅只来自 LC4→DNp01 逼近逃逸反射', G.W / 2, 224);
  g.fillStyle = '#fff'; g.font = 'bold 20px system-ui';
  g.fillText('按空格 / 点画面 你可以亲自接管拍翅试试', G.W / 2, 300);
  g.textAlign = 'left';
}
function drawOverCard(g) {
  const w = 300, h = 150, x = (G.W - w) / 2, y = 200;
  g.fillStyle = '#ded895'; g.strokeStyle = '#5c4708'; g.lineWidth = 3;
  g.fillRect(x, y, w, h); g.strokeRect(x, y, w, h);
  g.textAlign = 'center'; g.fillStyle = '#7a5b12'; g.font = 'bold 26px system-ui';
  g.fillText('GAME OVER', G.W / 2, y + 40);
  g.font = '15px system-ui'; g.fillStyle = '#4a3a0c';
  g.fillText('得分  ' + G.score + '      最高  ' + G.best, G.W / 2, y + 76);
  g.font = '13px system-ui'; g.fillStyle = '#6b5416';
  g.fillText(G.cause, G.W / 2, y + 100);
  g.fillText('1.2 秒后自动重来', G.W / 2, y + 124);
  g.textAlign = 'left'; g.lineWidth = 1;
}

// ---------------------------------------------------------------- 画游戏
function drawGame() {
  const c = $('game'), g = c.getContext('2d');
  g.fillStyle = '#02040a'; g.fillRect(0, 0, G.W, G.H);
  g.strokeStyle = 'rgba(120,140,170,.25)';
  for (let x = 0; x < G.W; x += 40) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, G.H); g.stroke(); }

  if (P.mode === 'fly3d') { drawCorridor(g); drawDeath(g); return; }
  if (P.mode === 'flappy') {
    drawSky(g);
    for (const p of G.pipes) {
      // 管口朝着缺口（标准 FlappyBird 的形状）。
      //
      // ⚠️ 边界要内缩一个**鸟半径**：physics() 判的是鸟**中心点**
      //    （`G.y - 11 < top || G.y + 11 > top + GAP`），所以对中心点而言
      //    管子真正挡住的是 `y < top+11` 那一段。如果管子只画到 top，
      //    就等于"看着碰到了（鸟身压着管口）却不算撞" —— 这正是之前那个
      //    "碰到管子不判定失败"的根源。画到"鸟身体真正会碰到的位置"才对得上。
      // 注意边界是**坐标**，而"最后一个不该画的像素"是坐标-1：
      //   y+11 > top+GAP 不成立的最大整数 y 是 top+GAP-11，所以下管从 top+GAP-11 起画。
      // 差 1 像素就会在缺口里多出一条"看着撞了却不算"的线，所以这里对齐到像素。
      const R = 11;                                  // = physics() 里的鸟半径
      drawPipeSprite(g, p.x, p.top + R, 0, R);                    // 上管
      drawPipeSprite(g, p.x, p.top + GAP - R + 1, G.H, R);        // 下管
    }
    drawGround(g);
    drawBirdSprite(g);
    drawScoreBig(g);
    if (!G.started) drawReady(g);
  } else if (P.mode !== 'retino') {
    drawThreat(g); drawBirdSprite(g);
  } else {
    drawThreat(g); drawBirdSprite(g);
  }


  if (G.flashT > 0) { $('flash').className = 'on'; $('flash').innerHTML = '<span>跳!</span>'; }
  else $('flash').className = '';
  drawDeath(g);
}

function drawDeath(g) {
  if (!G.dead) return;
  if (P.mode === 'flappy') { g.fillStyle = 'rgba(0,0,0,.35)';
    g.fillRect(0, 0, G.W, G.H); drawOverCard(g); return; }
  g.fillStyle = 'rgba(0,0,0,.55)'; g.fillRect(0, 0, G.W, G.H);
  g.fillStyle = '#fff'; g.font = 'bold 26px system-ui'; g.textAlign = 'center';
  g.fillText(G.cause, G.W / 2, G.H / 2 - 8);
  g.font = '13px system-ui'; g.fillStyle = '#9aa7b8';
  g.fillText('本局 ' + G.score + ' 分 · 反射在下一局继续', G.W / 2, G.H / 2 + 18);
  g.textAlign = 'left';
}

/** 反射镜模式：一个正对苍蝇扩张的暗盘 —— 文献里的 looming 刺激本体 */
function drawThreat(g) {
  const r = disp(); if (!r) return;
  const rad = Math.min(G.W, (r.theta_deg / 160) * G.W);
  g.fillStyle = '#000';
  g.beginPath(); g.arc(G.birdX + 150, 310, Math.max(6, rad), 0, 7); g.fill();
  g.strokeStyle = 'rgba(255,140,26,.5)'; g.setLineDash([5, 4]);
  g.beginPath(); g.arc(G.birdX + 150, 310, Math.max(6, rad), 0, 7); g.stroke();
  g.setLineDash([]);
  g.fillStyle = '#9aa7b8'; g.font = '12px system-ui';
  g.fillText('θ = ' + r.theta_deg.toFixed(1) + '°   dθ/dt = ' + r.dtheta_dps.toFixed(0)
             + ' °/s   τ = ' + r.tau_s.toFixed(2) + ' s', 16, 24);
  g.fillStyle = G.miss ? '#ff6b6b' : '#455263';
  g.fillText('逃掉 ' + G.score + ' 次 · 没逃掉 ' + G.miss + ' 次', 16, 44);
}

/** 苍蝇视角：把最近的管子画成正在扩张的暗盘 */
function drawFlyView(g, ox, oy, R) {
  const r = disp(); if (!r) return;
  const cx = ox, cy = oy;
  g.save(); g.beginPath(); g.arc(cx, cy, R, 0, 7); g.clip();
  g.fillStyle = '#7fd0d8'; g.fillRect(cx - R, cy - R, R * 2, R * 2);
  const rr = Math.max(3, Math.min(R * 1.9, (r.theta_deg / 160) * R * 1.9));
  g.fillStyle = '#000'; g.beginPath(); g.arc(cx, cy, rr, 0, 7); g.fill();
  g.restore();
  g.strokeStyle = 'rgba(120,140,170,.55)'; g.beginPath(); g.arc(cx, cy, R, 0, 7); g.stroke();
  g.fillStyle = '#8d9aab'; g.font = '10px system-ui'; g.textAlign = 'center';
  g.fillText('苍蝇看到 θ=' + r.theta_deg.toFixed(0) + '°', cx, cy + R + 11);
  g.textAlign = 'left';
}

function drawScope() {
  const cv = $('scope'); if (!cv) return;
  const g = cv.getContext('2d');
  g.fillStyle = '#02040a'; g.fillRect(0, 0, cv.width, cv.height);
  g.strokeStyle = 'rgba(120,140,170,.22)'; g.strokeRect(0.5, 0.5, cv.width - 1, cv.height - 1);
  drawFlyView(g, 58, 56, 42);
  drawDrive(g, 118, 8, cv.width - 130, cv.height - 16);
}

/** 驱动 vs 阈值：这条线是整个 demo 的主角 */
function drawDrive(g, x0 = 14, y0 = G.H - 118, w = G.W - 28, h = 96) {
  g.fillStyle = 'rgba(6,10,18,.85)'; g.fillRect(x0, y0, w, h);
  g.strokeStyle = 'rgba(120,140,170,.35)'; g.strokeRect(x0, y0, w, h);
  const yTh = y0 + h - R50 * h;
  g.strokeStyle = '#ff2d55'; g.setLineDash([4, 3]); g.beginPath();
  g.moveTo(x0, yTh); g.lineTo(x0 + w, yTh); g.stroke(); g.setLineDash([]);
  g.fillStyle = '#ff2d55'; g.font = '10px system-ui';
  g.fillText('DNp01 反射阈值 r50 = 0.58', x0 + 6, yTh - 4);
  g.strokeStyle = '#ff8c1a'; g.lineWidth = 1.6; g.beginPath();
  G.hist.forEach((v, i) => {
    const x = x0 + (i / 199) * w, y = y0 + h - Math.min(1, v) * h;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke(); g.lineWidth = 1;
  g.fillStyle = '#ff8c1a'; g.fillText('LC4 群发放率（驱动）', x0 + 6, y0 + 12);
  // 驱动从没越过红线 = 反射根本没机会触发，这几乎一定是感觉参数不对，
  // 不是"脑子不行"。直接把可调的旋钮告诉用户。
  if (P.mode === 'retino') {
    // 仪表显示的是**连接组算出来的有效突触驱动** Σw·p 对阈值 (1-leak)·thr/gain 的比值。
    // 这比"细胞个数占比"诚实：决定逃逸的是被驱动的那批细胞里有多少强突触。
    const cov = Math.min(1.2, (S.resp?.eff ?? 0) / Math.max(S.resp?.need ?? 0.0604, 1e-9));
    const needTxt = '需 Σw·p ≥ ' + ((S.resp?.need ?? 0.0604)).toFixed(3);
    const bw = 26, bx = x0 + w - bw - 10, by = y0 + 10, bh = h - 20;
    g.fillStyle = '#0d1420'; g.fillRect(bx, by, bw, bh);
    g.fillStyle = cov >= 1 ? '#3ddc84' : '#ff6b6b';
    g.fillRect(bx, by + bh * (1 - Math.min(1, cov)), bw, bh * Math.min(1, cov));
    const yl = by + bh * (1 - 1);
    g.strokeStyle = '#ffd400'; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(bx - 4, yl); g.lineTo(bx + bw + 4, yl); g.stroke();
    g.setLineDash([]);
    g.fillStyle = '#ffd400'; g.font = '10px system-ui';
    g.fillText('阈值线', bx - 4, yl - 4);
    g.fillText(needTxt, x0 + 6, y0 + h + 24);
    g.fillStyle = '#cfd8e3';
    g.fillText('驱动/阈值', bx - 16, by + bh + 12);
    g.fillText((cov * 100).toFixed(0) + '%', bx + 4, by + bh + 24);
  }
  if (!G.hist.length || Math.max(...G.hist) < R50) {
    g.fillStyle = '#ffb300'; g.font = '11px system-ui';
    g.fillText('驱动从未越过阈值 → 反射没机会触发：把 s50 调小 / 管速或威胁半径 R 调大',
               x0 + 6, y0 + h - 8);
    g.font = '10.5px system-ui';
  }
}

// ================================================================ 第一人称走廊
// 摄像机就是苍蝇的眼睛：往前飞，一整面带洞的墙迎面压过来。
// 侧视角的管子只占视野一小条（§3.8 证明那达不到逃逸阈值）；墙才是
// 覆盖大部分视野的逼近物。这里把"墙"按视网膜坐标换算成驱动图样。
const FOV_DEG = 180;            // 前半视野跨 180°，映射到视野坐标 u∈[0,1]
const F_PX = 300;               // 透视焦距（像素/弧度）
const GRAV3 = 3.0;              // 米/秒²
const FLAP3 = 2.2;              // 一次拍翅给的垂直速度

const F3 = { alt: 0.0, vz: 0.0, walls: [], spawnD: 6.0, nextHole: 0.0 };

function reset3d() {
  F3.alt = 0.0; F3.vz = 0.0; F3.walls = []; F3.spawnD = 5.0;
  G.score = 0; G.miss = 0; G.dead = false; G.deadT = 0;
}

function spawnWall() {
  const hr = Math.max(0.12, P.tmax / 400);            // 洞半径（米）
  F3.walls.push({ d: F3.spawnD, hole: (Math.random() * 2 - 1) * 0.9, r: hr,
                  passed: false });
}

/** 最近那面墙的视网膜几何：洞的方位、角半径、边界扩张速率 */
function geom3d() {
  const w = F3.walls.find(x => x.d > 0.02);
  if (!w) return null;
  const dy = w.hole - F3.alt;
  const elev = Math.atan2(dy, w.d) * 180 / Math.PI;      // 洞在视野里的高低角
  const theta = Math.atan(w.r / w.d) * 180 / Math.PI;    // 洞的角半径
  // 逼近一面墙时，LC4 收到的不是"洞边界扩张"（远看几乎为 0），而是
  // **视网膜上任意纹理点的移动速度**：偏心 45° 处 = v/d 弧度/秒。
  // 单位仍是 °/s，所以能直接复用同一个 s50，不用另加自由参数。
  const flow_dps = 180 / Math.PI * (P.spd / Math.max(w.d, 1e-3));
  return { wall: w, distM: w.d, speed: P.spd, radius: w.r,
           elevDeg: elev, thetaDeg: theta, dtheta_dps: flow_dps };
}

function physics3d(dt) {
  if (G.dead) { G.deadT += dt; if (G.deadT > 1.2) reset3d(); return; }
  F3.vz -= GRAV3 * dt;
  F3.alt += F3.vz * dt;
  if (F3.alt < -2.2) { F3.alt = -2.2; F3.vz = 0; }
  if (F3.alt > 2.2) { F3.alt = 2.2; F3.vz = 0; }
  for (const w of F3.walls) {
    w.d -= P.spd * dt;
    if (w.d <= 0.02 && !w.passed) {
      w.passed = true;
      if (Math.abs(w.hole - F3.alt) < w.r) { G.score++; G.best = Math.max(G.best, G.score); }
      else { die('撞墙'); }
    }
  }
  F3.walls = F3.walls.filter(w => w.d > -0.5);
  const last = F3.walls[F3.walls.length - 1];
  if (!last || last.d < F3.spawnD - 3.2) spawnWall();
}

function drawCorridor(g) {
  const cx = G.W / 2, cy = G.H / 2;
  // 地面/天花板的汇聚线：给出"正在往前飞"的深度感
  const vp = cy - F3.alt * F_PX * 0.35;
  g.fillStyle = '#04070e'; g.fillRect(0, 0, G.W, G.H);
  g.strokeStyle = 'rgba(90,130,180,.28)'; g.lineWidth = 1;
  for (let i = -4; i <= 4; i++) {
    const y = vp + i * 150;
    g.beginPath(); g.moveTo(cx, vp); g.lineTo(cx + i * 260, G.H > y ? G.H : -G.H); g.stroke();
  }
  for (let k = 1; k <= 6; k++) {                       // 深度环
    const d = k * 1.6;
    const rr = F_PX * 2.6 / d;
    g.strokeStyle = `rgba(60,90,130,${0.30 - k * 0.04})`;
    g.beginPath(); g.arc(cx, vp, rr, 0, 7); g.stroke();
  }
  // 墙：整面暗色 + 一个亮的洞
  for (const w of [...F3.walls].sort((a, b) => a.d - b.d)) {
    if (w.d <= 0.05) continue;
    const rr = Math.max(6, F_PX * w.r / w.d);
    const oy = vp - F_PX * (w.hole - F3.alt) / w.d;
    const alpha = Math.min(0.92, 1.4 / w.d);
    g.fillStyle = `rgba(2,4,8,${alpha})`; g.fillRect(0, 0, G.W, G.H);
    g.save();
    g.globalCompositeOperation = 'destination-out';
    g.beginPath(); g.arc(cx, oy, rr, 0, 7); g.fill();
    g.restore();
    g.strokeStyle = w.d < 1.2 ? '#ff2d55' : '#39d0ff';
    g.lineWidth = 2.5; g.beginPath(); g.arc(cx, oy, rr, 0, 7); g.stroke();
    g.lineWidth = 1;
  }
  g.fillStyle = '#cfd8e3'; g.font = '12px system-ui';
  g.fillText('苍蝇的第一人称视野 · 洞 = 可通过的亮区', 16, G.H - 14);
  g.fillText('高度 ' + F3.alt.toFixed(2) + ' m   下一个洞 '
             + ((F3.walls.find(x => x.d > 0.02) || {}).hole ?? 0).toFixed(2) + ' m',
             16, 20);
}

// ---------------------------------------------------------------- 3D 脑
let three = null;
function init3D() {
  const cv = $('brain');
  const renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: false });
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070c);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
  camera.position.set(0, 0, 2.6);

  const coords = S.coords;                       // (N,3) 归一化解剖坐标
  const N = coords.length / 3;
  // 背景点抽稀：全量 5.9 万个半透明点每帧混合排序，在核显上只要 13fps。
  // 这里固定抽一个子集当"背景"，但**关键神经元（LC4/DN*）一个不丢** ——
  // 抽稀只影响装饰性的灰点，不影响任何被解读的对象。
  const BG_MAX = 14000;
  const key = new Set(S.info.lit.map(c => c.asset));
  const keep = [];
  for (let i = 0; i < N; i++) {
    if (Math.abs(coords[i * 3]) + Math.abs(coords[i * 3 + 1]) + Math.abs(coords[i * 3 + 2]) > 1e-6)
      keep.push(i);
  }
  const stride = Math.max(1, Math.ceil(keep.length / BG_MAX));
  const kept2 = keep.filter((a, i) => key.has(a) || i % stride === 0);
  S.bgShown = kept2.length;
  const keepFinal = kept2;
  const pos = new Float32Array(keep.length * 3);
  const col = new Float32Array(keep.length * 3);
  const asset2pt = new Int32Array(N).fill(-1);
  const base = new THREE.Color(0x455263), tmp = new THREE.Color();
  keepFinal.forEach((a, k) => {
    asset2pt[a] = k;
    pos[k * 3] = coords[a * 3]; pos[k * 3 + 1] = coords[a * 3 + 1]; pos[k * 3 + 2] = coords[a * 3 + 2];
    base.toArray(col, k * 3);
  });
  // 关键群预先上色（静态身份），发放时再打亮
  const TYPE_COL = { LC4: 0xff8c1a, DNp01: 0xff2d55, DNp04: 0x39d0ff,
                     DNp02: 0x39d0ff, DNp11: 0x39d0ff, LC10a: 0x8a5bd6 };
  S.info.lit.forEach((c, i) => {
    const k = asset2pt[c.asset]; if (k < 0) return;
    tmp.setHex(TYPE_COL[c.type] ?? 0x888888).multiplyScalar(0.85);
    tmp.toArray(col, k * 3);
  });

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const mat = new THREE.PointsMaterial({ size: 0.011, vertexColors: true,
                                         transparent: true, opacity: 0.92 });
  const pts = new THREE.Points(geo, mat);
  const grp = new THREE.Group(); grp.add(pts); scene.add(grp);

  // 发放中的神经元用**独立的小覆盖点云**画，而不是每帧改写整张底色缓冲。
  // 底色缓冲是 59,548×3 float ≈ 2.1MB；每帧重传就是 ~130MB/s 的上传，
  // 这才是卡顿的来源。覆盖层每帧只传实际发放的那几百个点（约 24KB）。
  function makeOverlay(max, size, color) {
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(max * 3), 3));
    pg.setDrawRange(0, 0);
    const pm = new THREE.PointsMaterial({ size, color, transparent: true,
                                          opacity: 0.95, depthWrite: false,
                                          blending: THREE.AdditiveBlending });
    const po = new THREE.Points(pg, pm);
    grp.add(po);
    return { geo: pg, max, po };
  }
  const ovLit = makeOverlay(400, 0.055, 0xffffff);
  const ovViz = makeOverlay(1500, 0.022, 0x8fd0ff);

  // 自写轨道控制（不依赖 addons，避免 import 路径问题）
  const orbit = { rx: -0.3, ry: 0.4, zoom: 2.6, drag: null };
  cv.addEventListener('pointerdown', e => { orbit.drag = [e.clientX, e.clientY]; });
  addEventListener('pointerup', () => orbit.drag = null);
  addEventListener('pointermove', e => {
    if (!orbit.drag) return;
    orbit.ry += (e.clientX - orbit.drag[0]) * 0.006;
    orbit.rx += (e.clientY - orbit.drag[1]) * 0.006;
    orbit.rx = Math.max(-1.4, Math.min(1.4, orbit.rx));
    orbit.drag = [e.clientX, e.clientY];
  });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    orbit.zoom = Math.max(0.6, Math.min(6, orbit.zoom * (1 + e.deltaY * 0.0012)));
  }, { passive: false });

  three = { renderer, scene, camera, grp, geo, col, base, asset2pt, orbit, ovLit, ovViz,
            coords: S.coords,
            litAsset: S.info.lit.map(c => c.asset), litType: S.info.lit.map(c => c.type) };
  resize();
}

function resize() {
  const box = $('right').getBoundingClientRect();
  three.renderer.setSize(box.width, box.height, false);
  three.camera.aspect = box.width / box.height;
  three.camera.updateProjectionMatrix();
}

function fillOverlay(ov, assetIdxList) {
  const T = three, arr = ov.geo.attributes.position.array;
  let n = 0;
  for (const a of assetIdxList) {
    if (n >= ov.max) break;
    if (a == null || a < 0) continue;
    arr[n * 3] = T.coords[a * 3]; arr[n * 3 + 1] = T.coords[a * 3 + 1];
    arr[n * 3 + 2] = T.coords[a * 3 + 2];
    n++;
  }
  ov.geo.setDrawRange(0, n);
  ov.geo.attributes.position.needsUpdate = true;
}

function draw3D() {
  const T = three, r = S.resp;
  if (r) {
    // 底色云完全不动，只挪覆盖层
    fillOverlay(T.ovViz, r.viz_spike);
    fillOverlay(T.ovLit, (r.lit_spike || []).map(i => T.litAsset[i]));
  } else {
    T.ovLit.geo.setDrawRange(0, 0); T.ovViz.geo.setDrawRange(0, 0);
  }
  T.grp.rotation.set(T.orbit.rx, T.orbit.ry, 0);
  T.camera.position.z = T.orbit.zoom;
  T._n = (T._n || 0) + 1;
  if (T._n % 2 === 0) T.renderer.render(T.scene, T.camera);   // 3D 半速即可
}

// ---------------------------------------------------------------- HUD
const _hudCache = {};
function setTxt(id, v) {                       // 值没变就不碰 DOM（避免每帧重排）
  if (_hudCache[id] === v) return;
  _hudCache[id] = v; $(id).textContent = v;
}
function hud() {
  const r = disp();
  setTxt('h-theta', r ? r.theta_deg.toFixed(1) + '°' : '–');
  setTxt('h-dtheta', r ? r.dtheta_dps.toFixed(0) + ' °/s' : '–');
  setTxt('h-tau', r ? r.tau_s.toFixed(2) + ' s' : '–');
  setTxt('h-lc4', r ? (r.lc4_rate * 50).toFixed(1) + ' Hz' : '–');
  setTxt('h-dn', r ? r.dn01_recent + ' 个脉冲' : '–');
  const dn = $('h-dn'), c = r && r.dn01_recent > 0 ? '#ff2d55' : '';
  if (dn.style.color !== c) dn.style.color = c;
  setTxt('h-score', G.score + ' / ' + G.best);
}

// ---------------------------------------------------------------- UI
function initUI() {
  const bind = (id, key, out, fmt) => {
    const el = $(id);
    el.addEventListener('input', () => {
      P[key] = parseFloat(el.value);
      $(out).textContent = fmt ? fmt(P[key]) : P[key];
    });
  };
  bind('s50', 's50', 'o-s50'); bind('rad', 'rad', 'o-rad', v => v.toFixed(3));
  bind('spd', 'spd', 'o-spd', v => v.toFixed(1)); bind('need', 'need', 'o-need');
  bind('tmax', 'tmax', 'o-tmax');
  const cb = $('cb3d');
  if (cb) cb.addEventListener('change', () => {
    P.show3d = cb.checked;
    $('brain').style.display = cb.checked ? 'block' : 'none';
    $('legend').style.display = cb.checked ? 'block' : 'none';
  });
  document.querySelectorAll('input[name=mode]').forEach(el => {
    el.addEventListener('change', () => {
      P.mode = el.value; G.score = 0; G.miss = 0;
      if (el.value === 'fly3d') reset3d(); else resetGame();
    });
  });
  document.querySelectorAll('input[name=graph]').forEach(el => {
    // 干预状态存在服务端，刷新页面不会回滚 —— 必须把单选框同步成实际值，
    // 否则会出现"界面写着真实接线、脑其实是 cut"这种要命的误导。
    if (el.value === S.info.graph) el.checked = true;
    el.addEventListener('change', async () => {
      $('stat').textContent = '正在重建脑（' + el.value + '）…';
      const r = await post('/api/config', { graph: el.value });
      // 换干预 = 换被试：计分必须清零，否则看不出是干预造成的差别
      G.score = 0; G.miss = 0; G.best = 0; resetGame();
      $('stat').textContent = r.error ? '失败：' + r.error
        : '干预 = ' + el.value + (el.value === 'real' ? '（正常反射）' : '（对照）');
    });
  });
  $('btn-reset').addEventListener('click', () => { post('/api/reset', {}); resetGame(); });
  // 手动拍翅：让人亲自试一下这个游戏有多难，体感比看数字直观
  const manual = () => {
    if (P.mode !== 'flappy' || G.dead) return;
    G.started = true; delete G.readyT; G.vy = FLAP_V; G.manual = true;
  };
  addEventListener('keydown', e => { if (e.code === 'Space') { e.preventDefault(); manual(); } });
  $('game').addEventListener('pointerdown', manual);
}

// ---------------------------------------------------------------- 主循环
// ⚠️ 排障计数器**必须**在这里初始化。
// 踩过：`DBG.brainTicks++` 在 DBG 没有该字段时是 `undefined++` = **NaN**，
// 而诊断脚本写的是 `d.brainTicks || 0`，NaN 就被显示成 0 —— 于是看起来像
// "脑一个 tick 都没推进"，我照着这个假象查了很久的时序和服务器队列问题。
const DBG = { frames: 0, brainTicks: 0, err: null, lastTs: null, G, P, S };
window.__dbg = DBG;                 // 排障用：控制台读 __dbg.frames / __dbg.err
window.G = G;                       // 排障用：控制台看当前分数 / 死因
const PERF = { f: 0, last: 0, fps: 0 };
function loop(ts) {
  requestAnimationFrame(loop);
  DBG.frames++; DBG.lastTs = ts;
  PERF.f++;
  if (ts - PERF.last > 500) {                 // 每半秒报一次真实帧率
    PERF.fps = PERF.f * 1000 / (ts - PERF.last); PERF.f = 0; PERF.last = ts;
    const el = $('h-fps');
    if (el) el.textContent = PERF.fps.toFixed(0) + ' fps';
  }
  try {
    const now = ts / 1000;
    const dt = G.last ? Math.min(0.05, now - G.last) : 0;
    G.last = now;

    // ---- 严格模式（?strict=1）：与评测台同构 —— "脑决策之后走恰好 1 步物理"。
    //
    // 为什么需要这个开关：页面实测 1~4 分、评测台 111 分，差两个数量级。
    // 而 `scripts/flappy_page_parity.py` 号称验证过两边一致 —— 但它其实是
    // `class PageWorld(fb.World)`，**继承了评测台**，比的是评测台自己。
    // 所以"页面算法是否等价"从来没被真正验证过。这个开关就是那个对照：
    //   开了还低分 → 算法本身与评测台不等价（真 bug，去查算法）
    //   开了就正常 → 只是墙钟时序问题（去改时序）
    if (STRICT) {
      if (G.t < STRICT_TICKS * TICK_S) {
        if (!brainBusy() && stepBrain(1)) { DBG.brainTicks++; physics(TICK_S); }
      } else if (!DBG.strictDone) {
        DBG.strictDone = true;
        DBG.strictScore = G.score;
        DBG.strictCause = G.cause || '存活到上限';
      }
    } else {
      // 固定步长：物理按"脑 tick 数"推进，不按墙上时钟。
      const ticks = Math.floor((G.acc = Math.min(G.acc + dt, 0.3)) / TICK_S);
      // 批大小：一次请求向脑要几个 tick。
      //
      // 为什么可以 >1（推翻我之前的判断）：一次请求带回 N 个 tick 时，N 步物理
      // **各自**都用同一个拍翅决策 —— 也就是控制回路本来就是按脑的决策率离散运行的，
      // 并不是"物理跑在脑前面"。唯一代价是视觉输入最多滞后 N 个 tick。
      // 而实测浏览器只能做到 14.3 req/s（需要 50 才 1:1 实时），
      // 所以批大小是唯一能把速率提上来的旋钮 —— 到底值不值，直接量。
      if (ticks > 0 && stepBrain(FLAPPY.batch)) {
        const n = Math.min(ticks, FLAPPY.batch);
        G.acc -= n * TICK_S; DBG.brainTicks += n;
        for (let i = 0; i < n; i++) physics(TICK_S);
      } else if (ticks === 0) {
        physics(TICK_S);          // 不足一个 tick 也给一帧物理，保持画面连续
      }
    }
    // 排障遥测：每 ~0.5s 记一行
    if (PERF && (ts - (DBG.telTs || 0)) > 500) {
      DBG.telTs = ts;
      const np = nearestPipe();
      (DBG.tel = DBG.tel || []).push({
        t: +(G.t || 0).toFixed(2), sc: G.score, y: Math.round(G.y),
        vy: Math.round(G.vy), gapC: np ? Math.round(np.top + GAP / 2) : null,
        px: np ? Math.round(np.x) : null, np: G.pipes.length,
        br: (G.geo && G.geo.branch) || (DBG.lastBranch || '-'), dead: G.dead,
      });
      if (DBG.tel.length > 120) DBG.tel.shift();
    }
    drawGame(); drawScope();
    if (P.show3d) draw3D();
    hud();
  } catch (e) {
    DBG.err = (e && e.stack) ? e.stack : String(e);
  }
}

async function boot() {
  try {
    const info = await (await fetch('/api/info')).json();
    if (info.error) { $('stat').textContent = '后端错误 ' + info.error; return; }
    S.info = info;
    // 投射参数**从后端读**，前端不另存一份默认值 —— 否则页面数字会和
    // 命令行评测台（scripts/flappy_bench.py）悄悄失配，那就等于实验不可复现。
    if (info.bidi) {
      FLAPPY.s50size = info.bidi.s50size ?? FLAPPY.s50size;
      FLAPPY.n = info.bidi.n ?? FLAPPY.n;
      FLAPPY.gapMargin = info.bidi.gap_margin ?? FLAPPY.gapMargin;
      FLAPPY.vyGate = info.bidi.vy_gate ?? FLAPPY.vyGate;
      FLAPPY.ceilBoost = info.bidi.ceil_boost ?? FLAPPY.ceilBoost;
      FLAPPY.groups = info.bidi.groups ?? FLAPPY.groups;
      FLAPPY.dorsScale = info.bidi.dors_scale ?? FLAPPY.dorsScale;
      FLAPPY.ventGain = info.bidi.vent_gain ?? FLAPPY.ventGain;
      FLAPPY.ventDev = info.bidi.vent_dev ?? FLAPPY.ventDev;
    }
    if (BATCH_Q > 0) FLAPPY.batch = BATCH_Q;
    if (info.max_climb != null) FLAPPY.maxClimb = info.max_climb;
    const buf = await (await fetch('/api/coords.bin')).arrayBuffer();
    S.coords = new Float32Array(buf);
    $('stat').innerHTML =
      `冻结脑 <b>${info.n_neurons.toLocaleString()}</b> 神经元 / ` +
      `<b>${(info.n_synapses / 1e6).toFixed(2)}M</b> 突触 · dt=${info.dt_ms}ms · ` +
      `gain=${info.gain} tonic=${info.tonic} · 有坐标可点亮 ${info.lit.length} 个关键神经元` +
      `<br>LC4 ${info.groups.LC4} · DNp01 ${info.groups.DNp01} · DNp04 ${info.groups.DNp04} · ` +
      `零可学参数 —— 分数完全来自反射` +
      `<br><b>Flappy = 双向逼近反射</b>：` +
      `腹侧/背侧半视野 + 接近速度门控（死区 ${FLAPPY.gapMargin}px）· ` +
      `关卡按执行器带宽生成（向上跳变 ≤ ${FLAPPY.maxClimb}px）`;
    init3D(); initUI(); resetGame();
    requestAnimationFrame(loop);
  } catch (e) {
    // 之前这里静默失败，页面永远停在"载入中"
    $('stat').textContent = '启动失败：' + (e && e.stack ? e.stack : e);
    $('stat').className = 'warn';
  }
}
addEventListener('resize', () => three && resize());
boot();
