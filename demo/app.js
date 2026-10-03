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

const P = { s50: 15, s50size: 30, rad: 0.05, spd: 1.0, need: 1, graph: 'real',
            mode: 'flappy', tmax: 40 };
// DNp01 发放需要每 tick 净输入 Σw·p >= (1-leak)*thr/gain = 0.0604。
// 这个"有效驱动 / 阈值"由后端按每个细胞到 DNp01 的真实突触权重算出来（resp.eff / resp.need），
// 就是仪表里那根黄色阈值线。scripts/loom_retino_coverage.py 验证过它能精确预测 cliff。
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

/** 当前威胁的几何：返回 {distM, speed, radius} 或 null。
 *  reflex 模式：一个正对苍蝇飞来的暗盘（这就是文献里的逼近刺激）。
 *  flappy  模式：最近的那根管子被当作迎面撞来的物体。 */
function threat() {
  if (P.mode === 'fly3d') return geom3d();
  if (P.mode !== 'flappy') {
    if (G.threatD == null) return null;
    return { distM: Math.max(G.threatD, 0.02), speed: P.spd, radius: P.rad };
  }
  // 苍蝇该对**最 imminent 的那个碰撞**报警：地面 / 天花板 / 前方管子，
  // 取时间到接触(TTC)最小的。只盯管子的话，苍蝇下落时没有任何东西逼它拍翅，
  // 而地面逼近是一个货真价实的 looming 刺激。
  const vpx = P.spd * PX_PER_M;
  const cands = [];
  const near = nearestPipe();
  if (near) cands.push({ dpx: near.x + PIPE_W - G.birdX, vpx: vpx, rad: P.rad });
  const groundY = G.H - GROUND;
  if (G.vy > 0) cands.push({ dpx: groundY - G.y, vpx: G.vy, rad: 0.55 });
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
  if (P.mode !== 'flappy') {
    G.y = 310; newThreat();
  } else {
    // 开局就得有管子、而且不能太远：第一版栽在两个自锁上 ——
    //  (a) 没管子 -> 没刺激 -> 反射不触发 -> 摔死 -> 重置 -> 还是没管子；
    //  (b) 管子从 1.4 米外开始，而鸟 0.72 秒落地，反射来不及救它。
    G.pipes.push({ x: G.birdX + 0.75 * PX_PER_M - PIPE_W, top: 250, passed: false });
  }
}

function spawnPipe() {
  const gapTop = 70 + Math.random() * (G.H - GAP - 150);
  G.pipes.push({ x: G.W + 30, top: gapTop, passed: false });
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
  if (inFlight) return false;
  const th = threat();
  if (!th) return false;
  inFlight = true;
  // 几何量是前端算的（后端只收"每个群发多少"），所以显示要自己留一份
  G.geo = {
    theta_deg: 180 / Math.PI * 2 * Math.atan(th.radius / Math.max(th.distM, 1e-3)),
    dtheta_dps: 180 / Math.PI * (2 * th.radius * th.speed /
                                 (th.distM ** 2 + th.radius ** 2)),
    tau_s: th.distM / Math.max(th.speed, 1e-6),
  };
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
    body = { ticks, dist: th.distM, speed: th.speed, radius: th.radius,
             s50: P.s50, n: 3, source: 'dtheta', need_spikes: P.need,
             drive: ['LC4'] };
  }
  post('/api/step', body).then(r => {
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
    $('stat').textContent = '后端连接断了：' + e;
  }).finally(() => { inFlight = false; });
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

function drawPipeSprite(g, x, yTop, hgt) {
  const capW = PIPE_W + 10, capH = 30;
  const body = g.createLinearGradient(x, 0, x + PIPE_W, 0);
  body.addColorStop(0, '#8ce350'); body.addColorStop(0.25, '#74bf2e');
  body.addColorStop(0.85, '#4e8a1c'); body.addColorStop(1, '#3d6d16');
  g.fillStyle = body; g.fillRect(x, yTop, PIPE_W, hgt);
  g.strokeStyle = '#2f5212'; g.lineWidth = 2; g.strokeRect(x, yTop, PIPE_W, hgt);
  const cy = yTop >= 0 ? yTop : yTop + hgt;          // 管帽贴在缺口那一侧
  const capY = yTop >= 0 ? cy - capH : cy;
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
      drawPipeSprite(g, p.x, 0, p.top);
      drawPipeSprite(g, p.x, p.top + GAP, G.H - GROUND - p.top - GAP);
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


  if (G.flashT > 0) { $('flash').className = 'on'; $('flash').innerHTML = '<span>逃!</span>'; }
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
  const keep = [];
  for (let i = 0; i < N; i++) {
    if (Math.abs(coords[i * 3]) + Math.abs(coords[i * 3 + 1]) + Math.abs(coords[i * 3 + 2]) > 1e-6)
      keep.push(i);
  }
  const pos = new Float32Array(keep.length * 3);
  const col = new Float32Array(keep.length * 3);
  const asset2pt = new Int32Array(N).fill(-1);
  const base = new THREE.Color(0x455263), tmp = new THREE.Color();
  keep.forEach((a, k) => {
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

  three = { renderer, scene, camera, grp, geo, col, base, asset2pt, orbit,
            litAsset: S.info.lit.map(c => c.asset), litType: S.info.lit.map(c => c.type) };
  resize();
}

function resize() {
  const box = $('right').getBoundingClientRect();
  three.renderer.setSize(box.width, box.height, false);
  three.camera.aspect = box.width / box.height;
  three.camera.updateProjectionMatrix();
}

const HOT = new THREE.Color();
function draw3D() {
  const T = three, r = S.resp;
  const arr = T.geo.attributes.color.array;
  // 每帧先回到"身份色"：LC4 橙 / DNp01 红 / 其余灰蓝
  arr.set(T.col);
  if (r) {
    for (const a of r.viz_spike) {
      const k = T.asset2pt[a]; if (k < 0) continue;
      HOT.setRGB(0.55, 0.75, 1.0); HOT.toArray(arr, k * 3);
    }
    r.lit_spike.forEach(li => {
      const a = T.litAsset[li]; if (a === undefined) return;
      const k = T.asset2pt[a]; if (k < 0) return;
      const ty = T.litType[li];
      const b = ty === 'DNp01' ? [1.6, 0.25, 0.4] : ty === 'LC4' ? [1.5, 0.8, 0.2] : [0.4, 1.5, 1.9];
      HOT.setRGB(b[0], b[1], b[2]); HOT.toArray(arr, k * 3);
    });
  }
  T.geo.attributes.color.needsUpdate = true;
  T.grp.rotation.set(T.orbit.rx, T.orbit.ry, 0);
  T.camera.position.z = T.orbit.zoom;
  T.renderer.render(T.scene, T.camera);
}

// ---------------------------------------------------------------- HUD
function hud() {
  const r = disp();
  $('h-theta').textContent = r ? r.theta_deg.toFixed(1) + '°' : '–';
  $('h-dtheta').textContent = r ? r.dtheta_dps.toFixed(0) + ' °/s' : '–';
  $('h-tau').textContent = r ? r.tau_s.toFixed(2) + ' s' : '–';
  $('h-lc4').textContent = r ? (r.lc4_rate * 50).toFixed(1) + ' Hz' : '–';
  const dn = $('h-dn');
  dn.textContent = r ? r.dn01_recent + ' 个脉冲' : '–';
  dn.style.color = r && r.dn01_recent > 0 ? '#ff2d55' : '';
  $('h-score').textContent = G.score + ' / ' + G.best;
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
const DBG = { frames: 0, err: null, lastTs: null, G, P, S };
window.__dbg = DBG;                 // 排障用：控制台读 __dbg.frames / __dbg.err
function loop(ts) {
  requestAnimationFrame(loop);
  DBG.frames++; DBG.lastTs = ts;
  try {
    const now = ts / 1000;
    const dt = G.last ? Math.min(0.05, now - G.last) : 0;
    G.last = now;
    physics(dt);
    G.acc = Math.min(G.acc + dt, 0.3);      // 卡住时不要攒出一大坨补帧
    // 只有请求真的发出去了才消耗累积的 tick。之前是先减后发、
    // 而 stepBrain 可能因上一请求未完成直接 return -> 那些 tick 被丢掉，
    // 脑就长期跑不满实时（实测只剩 24%）。
    const ticks = Math.floor(G.acc / TICK_S);
    if (ticks > 0 && stepBrain(Math.min(ticks, 12))) G.acc -= ticks * TICK_S;
    drawGame(); drawScope(); draw3D(); hud();
  } catch (e) {
    DBG.err = (e && e.stack) ? e.stack : String(e);
  }
}

async function boot() {
  try {
    const info = await (await fetch('/api/info')).json();
    if (info.error) { $('stat').textContent = '后端错误 ' + info.error; return; }
    S.info = info;
    const buf = await (await fetch('/api/coords.bin')).arrayBuffer();
    S.coords = new Float32Array(buf);
    $('stat').innerHTML =
      `冻结脑 <b>${info.n_neurons.toLocaleString()}</b> 神经元 / ` +
      `<b>${(info.n_synapses / 1e6).toFixed(2)}M</b> 突触 · dt=${info.dt_ms}ms · ` +
      `gain=${info.gain} tonic=${info.tonic} · 有坐标可点亮 ${info.lit.length} 个关键神经元` +
      `<br>LC4 ${info.groups.LC4} · DNp01 ${info.groups.DNp01} · DNp04 ${info.groups.DNp04} · ` +
      `零可学参数 —— 分数完全来自反射`;
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
