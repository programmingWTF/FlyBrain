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

// 注：原来有个 `?strict=1` 的"严格模式"，用来判定页面与评测台的差异是在
// 时序还是在算法。**现在默认模式就是那个形状了**（脑响应驱动、决策与物理步严格
// 1:1 配对，见 stepDriven / driveLoop），所以这个开关连同它的死代码一起删掉。
// ?batch=N：一次向脑请求 N 个 tick（用于量"批大小 vs 速率/分数"的取舍）
const BATCH_Q = Number((location.search.match(/[?&]batch=(\d+)/) || [])[1] || 0);

// 只留 Flappy 一个模式：`mode` 仍保留（少数读数按它取值），但**钉死不变**，
// 界面上已无模式选择器 —— 其余模式的代码路径已整体删除。
const MODE = 'flappy';
const P = { s50size: 30, spd: 1.0, need: 1, graph: 'real',
            mode: MODE, show3d: true };
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
// pending: 当前在飞的 /api/step promise（由 driveLoop 消费，并配对刚好 1 步物理）
const S = { info: null, coords: null, ready: false, pending: null,
            resp: null, lit: null };

// ---------------------------------------------------------------- 游戏状态
const G = {
  W: 520, H: 620, birdX: 120, y: 300, vy: 0,
  pipes: [], score: 0, best: 0, dead: false, deadT: 0, cause: '',
  acc: 0, last: 0, cooldown: 0, spawnT: 0,
  threatD: null, dodged: false, miss: 0, dodgeT: 0, cov: 0, hold: 0, geo: null,
  scroll: 0, wingPhase: 0, started: false, manual: false,
  resp: null, hist: [], flashT: 0, dnFlash: 0,
};
const GRAV = 1180, FLAP_V = -340, PIPE_W = 62, GAP = 184;
//: 管子间距（px）。**按距离**生成，不按"上一根在哪"—— 物理子步数变化时后者会漏生成
//: （没管子 = 没逼近刺激 = 反射不触发，看起来像脑子的错）。与评测台 SPACING 一致。
const SPACING = 300;
//: 碰撞判定用的鸟半径（px）。physics() 的判据是
//:   `G.y - BIRD_R < p.top || G.y + BIRD_R > p.top + GAP`
//: 即"判定体积 = 鸟中心 ± BIRD_R"。它必须与**画出来的鸟**一致，否则就会出现
//: "看着碰到管子却不判定失败"。`demo/verify_bird_volume.js` 逐形状量着守它。
const BIRD_R = 17;
//: 鸟的最大旋转角（弧度）。**只用于画眼睛/喙的朝向提示** —— 鸟身是一个正圆，
//: 旋转不会改变它的外接范围（这正是用圆的好处之一）。
const SPRITE_ROT_MAX = 0.30;

/** 双向逼近反射的"望远镜几何"：只算纯几何量，交给后端决定驱动哪些细胞。
 *
 *  方向：缺口比鸟高 → 威胁在下方（地面）→ 腹侧半视野；缺口比鸟低 → 天花板 → 背侧。
 *  vy_gate 在后端做（地面只在下落时逼近、天花板只在上升时逼近）。
 *  距离沿用同一套"等效半宽 0.55"的写法，与 scripts/flappy_bench.py 一致。
 */
// ============================================================ 与服务端同步
//
// ⚠️ 这一整块是 **服务端权威** 的接入点：前端**不再自己算物理、也不再自己喂脑**。
//
// 为什么必须这么改（我把前端的四种驱动方式都试遍了，每次都实测）：
//   ① 物理跟墙钟走 + pend 每帧 1 步  → 帧率 <50fps 就积压几十步、鸟控不住
//   ② 物理跟帧率走                    → 帧率 11.5fps 只有 12% 速度、飞不起来
//   ③ 物理跟脑响应走                  → 速度 = HTTP 往返速度 → 时快时慢
//   ④ 固定节拍 + 决策队列预取          → 队列周期性见底，仍然抖
// 根因是结构性的：**游戏时钟必须挂在某个节奏上**，而浏览器里两个可选项实测都不行：
//      setTimeout(4)          → 29.8 次/s（被压到 ~33ms）
//      requestAnimationFrame  → 28.2 次/s（后台还会完全暂停）
// 再加上 HTTP 往返中位 60ms、p90 86ms、最快 4ms（差十几倍）——
// 只要决策要靠一次跨进程往返拿到，时钟就一定是抖的。
//
// 所以把它搬到服务端：脑与物理在**同一个进程、同一个锁**里按固定 50 Hz 跑 1:1
// （见 server.py 的 GameWorld / _game_tick）。前端只做两件事：
//      ① 按自己的帧率**取状态**（取晚了只影响画面新鲜度，不影响游戏速度）
//      ② 把两次状态**插值**后画出来（所以画面是 60fps 流畅的）
// 实测服务端速率：**50.0 步/秒，偏差 +0.1%**。

/** 从服务端同步来的权威状态（后端的 _game_tick 每 20ms 更新一次）。 */
function applyServerState(st) {
  if (!st || st.error) return;
  const nowS = performance.now() / 1000;
  // ---- 维护**最近两次状态**（各自带"到达本地的时刻"），插值就在这两者之间做。
  //     为什么用到达时刻而不是服务端时刻：两者是同一个时钟吗？不是。
  //     本地墙钟与服务端时钟之间有一个未知且会漂移的偏移，直接算会持续抖；
  //     而"两次到达的本地时刻之差"与服务端走过的时间是同一段，
  //     网络延迟在求商时自然抵消。详见 interpolate 的注释。
  const prev = G.srvB;
  G.srvA = prev ? { local: prev.local, t: prev.t, y: prev.y, vy: prev.vy,
                    pipes: prev.pipes } : null;
  // 管子存 [top, x] 的二元组（按 top 配对，不依赖数组下标）
  G.srvB = { local: nowS, t: st.t, y: st.y, vy: st.vy,
             pipes: (st.pipes || []).map(p => [p[1], p[0]]) };
  G.t = st.t;
  G.y = st.y;
  G.vy = st.vy;
  G.srvY = st.y;
  G.score = st.score; G.best = st.best;
  G.dead = !!st.dead; G.cause = st.cause || '';
  G.flapT = st.flap_t;
  // 管道：服务端发的是 [x, top]。`x`/`top` 在这里作为**当前权威值**，
  // 显示位置由 interpolate 按两次状态算出（见那里的说明）。
  const ps = st.pipes || [];
  if (!G.pipes || G.pipes.length !== ps.length) {
    G.pipes = ps.map(([x, top]) => ({ x, top, passed: false }));
  } else {
    for (let i = 0; i < ps.length; i++) {
      G.pipes[i].x = ps[i][0];
      G.pipes[i].top = ps[i][1];
    }
  }
  G.srvT = st.t;
  // 加速期的显示由**服务端时间**推导（服务端才是权威，别再本地积分一遍）
  G.warmupT = Math.max(G.warmupT || 0, st.t);
  // 加速期时长由服务端下发（唯一来源，见 interpolate 的说明）
  if (st.warm_s !== undefined) G.warmS = st.warm_s;
  G.srvRx = performance.now() / 1000;
  // ---- 画图要用的脑侧量。
  // 原来这些来自前端自己那次 /api/step 的响应；改成服务端权威后必须从
  // 状态快照里拿，否则**3D 脑图会静默变空、驱动曲线也不动**（不报错，很隐蔽）。
  S.resp = {
    eff: st.eff, need: st.need,
    dn01_recent: st.dn01_recent,
    drive: (st.plan && st.plan.amp) || 0,
    plan: st.plan || {},
    viz_spike: st.viz_spike || [],
    lit_spike: st.lit_spike || [],
  };
  // 驱动曲线：原来由前端脑循环 push，现在从快照里取（否则曲线永远是一条平线）。
  // 画的是"这一步的角尺寸驱动量" amp，与阈值 r50 比较 —— 见 drawDrive。
  G.hist.push((st.plan && st.plan.amp) || 0);
  if (G.hist.length > 200) G.hist.shift();
  G.srvStats = { rate: st.rate, ticks: st.ticks, flaps: st.flaps, sim_hz: st.sim_hz };
  if (st.dn01_recent > 0) G.dnFlash = 0.2;
}

/** 取状态：**单飞**（上一次没回来就不发新的），避免请求堆积。 */
let fetching = false;
async function pollGame() {
  if (fetching) return;
  fetching = true;
  try {
    const r = await fetch('/api/game/state', { cache: 'no-store' });
    applyServerState(await r.json());
  } catch (e) {
    DBG.err = '取状态失败：' + e;
  } finally {
    fetching = false;
  }
}

/** 渲染用插值：服务端 50 Hz、屏幕 60 fps。只影响显示，判定完全在服务端。
 *
 *  ## 为什么必须在**两个已知状态之间**插值
 *
 *  原来写的是"保持最新值 + 从它往前外推"：
 *      y_display = srvY + vy * (now − srvRx)
 *  **这会跳。** `(now − srvRx)` 每帧都在变，于是同一个服务端状态被复用好几帧、
 *  每帧算出的位置都不同；下一帧收到新状态时基准一换，位置就跳一下。
 *  实测逐帧位移：中位 2.70 px、p95 19.47 px、**最小 −52.11 px**（倒跳），
 *  与公式预测的 2.7 / 20 / 倒跳完全吻合。视觉上就是"过了一根管子后跳掉一块"。
 *
 *  ## 为什么用「到达时刻」而不是「服务端时刻」
 *
 *  服务端在它自己的时间轴上给出 (prev_t, t)。但前端的墙钟和它不是一个钟，
 *  两者之间有一个未知且会漂移的偏移量 —— 直接用服务器时间算 `u` 会持续抖。
 *  所以这里用**两次状态各自到达本地的时刻**做插值：
 *      A 在 aLocal 到（服务端 aT），B 在 bLocal 到（服务端 bT）
 *      u = (now − aLocal) / (bLocal − aLocal)
 *  因为 A→B 之间服务端和本地走过的时间是同一段，**网络延迟在求商时自然抵消**，
 *  不需要对齐时钟，也不需要知道延迟是多少。
 */
function interpolate() {
  if (G.srvStats === undefined) return;
  const A = G.srvA, B = G.srvB;          // {local, t, y, vy, pipes} 两次相邻状态
  if (!A || !B || !(B.local > A.local)) return;
  const now = performance.now() / 1000;
  let u = (now - A.local) / (B.local - A.local);   // 0 = A, 1 = B
  if (!(u >= 0)) u = 0;
  if (u > 1.3) u = 1.3;                            // 网络变慢时最多多推一点
  const uc = Math.min(u, 1);
  // 鸟的 y：两状态线性插值；超出部分用当前 vy 稍微外推，避免"卡住感"
  G.y = A.y + (B.y - A.y) * uc + (u > 1 ? (u - 1) * (B.local - A.local) * G.vy : 0);
  // 管子：按**缺口 top** 在 A/B 之间配对（不能按下标 —— 生成/回收时下标会错位，
  // 会算出"管子往右跑"这种不可能的值）。管速恒定，所以 x 可以精确插值。
  if (!G.dead && G.srvT >= (G.warmS || 0.8) && G.pipes) {
    const aMap = new Map(A.pipes);      // top -> x
    const bMap = new Map(B.pipes);
    for (const q of G.pipes) {
      const xa = aMap.get(q.top), xb = bMap.get(q.top);
      if (xa === undefined || xb === undefined) continue;   // 刚生成/刚回收：这帧不画它
      q.x = xa + (xb - xa) * uc + (u > 1 ? -PX_PER_M * P.spd * (u - 1) * (B.local - A.local) : 0);
    }
  }
}

function nearestPipe() {
  let best = null;
  for (const p of G.pipes) if (p.x + PIPE_W > G.birdX - 6 && (!best || p.x < best.x)) best = p;
  return best;
}


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
/** 画一根管子：**就是一个轴对齐矩形**（管宽 PIPE_W，从 y0 到 y1）。
 *
 *  为什么不再画"管口帽"：旧版在管体两侧各凸出 5px 画了一圈粗管口，而碰撞用的是
 *  `[x, x+PIPE_W]` 的矩形 —— 凸出的那 10px **有画面、没判定**，鸟压上去不死。
 *  （我当时的逐像素校验器把范围裁在管体宽度内，所以没抓到，是校验器的盲区。）
 *  现在管子与判定是同一个矩形，形状上就不可能不一致。
 *
 *  `demo/verify_pipe_visual.js` 覆盖**整幅画面的 x 范围**逐像素对账，
 *  并断言管子不越出 `[x, x+PIPE_W]`。 */
function drawPipe(g, x, y0, y1) {
  const top = Math.min(y0, y1), h = Math.abs(y1 - y0);
  if (h <= 0) return;
  const body = g.createLinearGradient(x, 0, x + PIPE_W, 0);
  body.addColorStop(0, '#8ce350'); body.addColorStop(0.25, '#74bf2e');
  body.addColorStop(0.85, '#4e8a1c'); body.addColorStop(1, '#3d6d16');
  g.fillStyle = body; g.fillRect(x, top, PIPE_W, h);
  g.strokeStyle = '#2f5212'; g.lineWidth = 2; g.strokeRect(x, top, PIPE_W, h);
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

/** 鸟精灵的旋转角 = f(竖直速度)。**画图与体积校验共用这一个函数**，避免两处漂移。 */
function birdRot(vy) {
  return Math.max(-SPRITE_ROT_MAX, Math.min(SPRITE_ROT_MAX, vy / 430));
}

/** 画鸟：**就是一个半径 BIRD_R 的圆**（加眼睛和喙做朝向提示）。
 *
 *  为什么用圆（这一版才想通）：判定是 `|y_center − y_pipe| < BIRD_R`，
 *  也就是"圆心到管子边界的最小距离 < 半径"。**只有圆**能让画出来的形状与这个
 *  判据严格等价 —— 圆心在 `top − BIRD_R` 时圆周正好与 `y = top` 相切，
 *  再近一点就真重叠。换成椭圆/长喙都得靠"内缩多少"去凑，那是调出来的不是推出来的。
 *
 *  眼睛与喙都画在圆**内部**（不改变外接），只用来表示朝向；
 *  它们不影响碰撞，判定仍然只是半径 BIRD_R 的圆。
 *  demo/verify_bird_volume.js 校验"画出来的外接半径 == BIRD_R"。 */
function drawBirdSprite(g) {
  const R = BIRD_R;
  g.save(); g.translate(G.birdX, G.y); g.rotate(birdRot(G.vy));
  // 身体 = 判定体积本身，半径**正好**是 BIRD_R。
  // 描边（lineWidth 2）会向外晕出 1px：圆周边界的判定是严格 `<`，所以那 1px
  // 是最外侧的羽化边缘，**亮着但不算撞** —— 与"画得比判定大"是相反方向的问题，
  // 也正好让"看着擦到了"偏向公平。校验器量的是路径半径（应 == BIRD_R）。
  g.fillStyle = '#f7d51d'; g.strokeStyle = '#5c4708'; g.lineWidth = 2;
  g.beginPath(); g.arc(0, 0, R, 0, 7); g.fill(); g.stroke();
  // 腹部高光
  g.fillStyle = '#fdf3c0'; g.beginPath(); g.ellipse(-1.5, 3, 5.5, 3.5, 0, 0, 7); g.fill();
  // 翅膀：三相位扇动，全部留在圆内
  const ph = [0.5, 0.1, -0.4][G.wingPhase];
  g.save(); g.translate(-1, -1); g.rotate(ph);
  g.fillStyle = '#f0a81c'; g.strokeStyle = '#5c4708';
  g.beginPath(); g.ellipse(-3, 0, 5, 3.2, 0, 0, 7); g.fill(); g.stroke(); g.restore();
  // 眼睛
  g.fillStyle = '#fff'; g.beginPath(); g.arc(4.5, -3, 3.2, 0, 7); g.fill();
  g.fillStyle = '#222'; g.beginPath(); g.arc(5.3, -3, 1.5, 0, 7); g.fill();
  // 喙：也要在圆内（尖端离圆心 8.8 < R-1）
  g.fillStyle = '#f07f18'; g.strokeStyle = '#a4530b'; g.lineWidth = 1.2;
  g.beginPath(); g.moveTo(6.5, -1); g.lineTo(8.8, 1); g.lineTo(6.5, 3); g.closePath();
  g.fill(); g.stroke();
  g.restore();
}

function drawScoreBig(g) {
  g.font = 'bold 46px "Trebuchet MS", system-ui'; g.textAlign = 'center';
  g.lineWidth = 6; g.strokeStyle = '#5c4708'; g.fillStyle = '#fff';
  g.strokeText(String(G.score), G.W / 2, 74); g.fillText(String(G.score), G.W / 2, 74);
  g.textAlign = 'left'; g.lineWidth = 1;
}

/** 开局加速期的提示。这段时间**脑已经在飞**（见 WARMUP_S），所以不是"准备中"，
 *  而是"反射正在接管"—— 把这件事如实画出来。 */
function drawWarmup(g) {
  g.textAlign = 'center';
  g.font = 'bold 22px system-ui'; g.lineWidth = 0;
  g.fillStyle = 'rgba(255,255,255,.85)';
  g.fillText('反射接管中…', G.W / 2, 150);
  g.font = '13px system-ui'; g.fillStyle = 'rgba(255,255,255,.55)';
  g.fillText('冻结脑正在把逼近驱动积分到阈值', G.W / 2, 176);
  g.textAlign = 'left'; g.lineWidth = 1;
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

  // 只有 Flappy 一个模式（mode 已钉死），所以这里不再按模式分支。
  drawSky(g);
  for (const p of G.pipes) {
    // 管子 = 矩形，边界与判定逐像素对齐（见 drawPipe 的说明）。
    // `p.top` 是坐标；判定用 `y - R <= top`（碰上就算），所以上管画到 top+R+1、
    // 下管从 top+GAP-R 起画。"最后一个被挡住的像素"比坐标大 1。
    // demo/verify_pipe_visual.js 覆盖全 x 范围逐像素守着这一点。
    drawPipe(g, p.x, 0, p.top + BIRD_R + 1);              // 上管
    // 下管画到 top+GAP-R-1：与严格不等式 `y+R > top+GAP` 的判定边界严格对齐
    // （少这 1px 会多画一格，偏向"提前判死"；多画不会漏判，但两边就不是严格相等了）
    drawPipe(g, p.x, p.top + GAP - BIRD_R - 1, G.H);      // 下管
  }
  drawGround(g);
  drawBirdSprite(g);
  drawScoreBig(g);
  if ((G.warmupT || 0) < (G.warmS || 0.8)) drawWarmup(g);

  if (G.flashT > 0) { $('flash').className = 'on'; $('flash').innerHTML = '<span>跳!</span>'; }
  else $('flash').className = '';
  drawDeath(g);
}

function drawDeath(g) {
  if (!G.dead) return;
  g.fillStyle = 'rgba(0,0,0,.35)'; g.fillRect(0, 0, G.W, G.H); drawOverCard(g);
}

/** 反射状态圆盘：显示当前驱动的是哪一半视野、以及 Σw·p / 阈值。
 *  （原来这里是"把管子画成扩张暗盘"的视网膜视图，那套几何只对已删除的模式有意义。） */
function drawFlyView(g, ox, oy, R) {
  const cx = ox, cy = oy;
  const br = (G.geo && G.geo.branch) || '-';
  const ratio = Math.min(1.2, (S.resp?.eff ?? 0) / Math.max(S.resp?.need ?? 0.0604, 1e-9));
  g.save(); g.beginPath(); g.arc(cx, cy, R, 0, 7); g.clip();
  g.fillStyle = '#0d1420'; g.fillRect(cx - R, cy - R, R * 2, R * 2);
  // 上半 = 腹侧（爬升），下半 = 背侧（下潜）
  const vent = br === 'ventral', dors = br === 'dorsal';
  g.fillStyle = vent ? '#3ad07a' : '#26313f';
  g.fillRect(cx - R, cy - R, R * 2, R);
  g.fillStyle = dors ? '#ff8c1a' : '#26313f';
  g.fillRect(cx - R, cy, R * 2, R);
  // 驱动条：越接近阈值越满
  g.fillStyle = ratio >= 1 ? '#ff2d55' : '#39d0ff';
  g.fillRect(cx - R, cy + R - Math.min(1, ratio) * R * 0.4, R * 2, Math.min(1, ratio) * R * 0.4);
  g.restore();
  g.strokeStyle = 'rgba(120,140,170,.55)'; g.beginPath(); g.arc(cx, cy, R, 0, 7); g.stroke();
  g.fillStyle = '#8d9aab'; g.font = '10px system-ui'; g.textAlign = 'center';
  g.fillText(br === 'ventral' ? '腹侧→爬升' : br === 'dorsal' ? '背侧→下潜' : '已对准',
             cx, cy + R + 11);
  g.fillText('Σw·p/阈值 = ' + ratio.toFixed(2), cx, cy - R - 4);
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
  // 驱动从未越过红线 = 反射没机会触发。现在只有 Flappy 一个模式，
  // 可调的旋钮就剩"管速"和"触发脉冲数"。
  if (!G.hist.length || Math.max(...G.hist) < R50) {
    g.fillStyle = '#ffb300'; g.font = '11px system-ui';
    g.fillText('驱动从未越过阈值 → 反射没机会触发：把管速调快或触发脉冲数调小',
               x0 + 6, y0 + h - 8);
    g.font = '10.5px system-ui';
  }
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
  // 排障用：控制台 `__three` 可以看面板真实状态（点数 / drawRange / 渲染尺寸）——
  // 3D 面板是"静默失败"的重灾区：它挂了页面照跑、不报错，只是那块是黑的。
  // 别只在某个副本里加这一行，否则两端会不一致（我踩过：探针只在运行目录里，
  // 同步时被规范副本覆盖掉了，于是怎么测都说"3D 没渲染"）。
  window.__three = three;
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
  // 这里**不再做半速**：限频已经由调用方（loop 里的 DRAW3D_MS = 30fps）负责。
  // 原来这里还有一层 `T._n % 2`，两层叠加会让 3D 只剩 ~15fps（实测 707/1022 帧），
  // 看起来像"脑图卡卡的"。限频只保留一处，免得以后又叠出第三层。
  T.renderer.render(T.scene, T.camera);
}

// ---------------------------------------------------------------- HUD
const _hudCache = {};
function setTxt(id, v) {                       // 值没变就不碰 DOM（避免每帧重排）
  if (_hudCache[id] === v) return;
  _hudCache[id] = v; $(id).textContent = v;
}
function hud() {
  // 只报告 Flappy 真正用到的量：反射方向、驱动/阈值、发放率、脉冲数。
  const r = S.resp;
  const br = (G.geo && G.geo.branch) || (r && r.plan && r.plan.branch) || null;
  const brTxt = br === 'ventral' ? '腹侧 → 爬升'
              : br === 'dorsal'  ? '背侧 → 下潜'
              : br === 'aligned' ? '已对准' : '–';
  const ratio = r ? (r.eff ?? 0) / Math.max(r.need ?? 0.0604, 1e-9) : null;
  setTxt('h-theta', brTxt);
  setTxt('h-dtheta', ratio === null ? '–' : ratio.toFixed(2) + '×');
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
  bind('spd', 'spd', 'o-spd', v => v.toFixed(1)); bind('need', 'need', 'o-need');
  const cb = $('cb3d');
  if (cb) cb.addEventListener('change', () => {
    P.show3d = cb.checked;
    $('brain').style.display = cb.checked ? 'block' : 'none';
    $('legend').style.display = cb.checked ? 'block' : 'none';
  });
  document.querySelectorAll('input[name=graph]').forEach(el => {
    // 干预状态存在服务端，刷新页面不会回滚 —— 必须把单选框同步成实际值，
    // 否则会出现"界面写着真实接线、脑其实是 cut"这种要命的误导。
    if (el.value === S.info.graph) el.checked = true;
    el.addEventListener('change', async () => {
      $('stat').textContent = '正在重建脑（' + el.value + '）…';
      const r = await post('/api/config', { graph: el.value });
      // 换干预 = 换被试：计分必须清零，否则看不出是干预造成的差别。
      // ⚠️ 重置世界也必须在**服务端**做（前端只清显示值没用）。
      G.score = 0; G.best = 0;
      post('/api/game/reset', {});
      $('stat').textContent = r.error ? '失败：' + r.error
        : '干预 = ' + el.value + (el.value === 'real' ? '（正常反射）' : '（对照）');
    });
  });
  $('btn-reset').addEventListener('click', () => {
    // 重开由**服务端**做（世界与脑都归零），前端不再自己 resetGame
    post('/api/game/reset', {}).then(() => { S.pendingReset = true; });
  });
  // 手动拍翅：让人亲自试一下这个游戏有多难，体感比看数字直观。
  // ⚠️ 现在是**服务端权威**：拍翅必须让服务端执行，前端自己改 G.vy 会被下一次
  //    状态同步直接覆盖掉（这一点是"改前端"和"改服务端"最容易踩的坑）。
  const manual = () => {
    if (G.dead) return;
    post('/api/game/flap', {});
  };
  addEventListener('keydown', e => { if (e.code === 'Space') { e.preventDefault(); manual(); } });
  $('game').addEventListener('pointerdown', manual);
}

// ---------------------------------------------------------------- 主循环
// 排障计数器**必须**在这里显式初始化。
// 踩过：`DBG.brainTicks++` 在 DBG 没有该字段时是 `undefined++` = **NaN**，
// 而诊断脚本写的是 `d.brainTicks || 0`，NaN 就被显示成 0 —— 于是看起来像
// "脑一个 tick 都没推进"，我照着这个假象查了很久的时序和服务器队列问题。
//
// 注：`brainTicks / physSteps / sbOk / pendMax` 这些是**前端自己跑仿真**时期
// 的计数器。现在仿真在服务端，它们没有意义了，已删掉 —— 免得又拿一个恒为 0
// 的数去判断"是不是没在跑"。要看真实速率请读 `G.srvStats`（来自服务端）。
const DBG = { frames: 0, err: null, lastTs: null, G, P, S };
window.__dbg = DBG;                 // 排障用：控制台读 __dbg.frames / __dbg.err
window.G = G;                       // 排障用：控制台看当前分数 / 死因
const PERF = { f: 0, last: 0, fps: 0 };

//: 取状态的间隔（毫秒）。服务端是 50 Hz，取 **20 Hz** 足够 ——
//: 中间的帧由 `interpolate()` 补上。**不要设成 0（每帧都取）**：
//: 60 req/s 会把单进程的服务端打爆（实测延迟从 3 ms 飙到 38.9 ms、页面卡死）。
const POLL_MS = 50;
//: 3D 脑图的绘制间隔（毫秒）。它有 13.9 万个点 + 每帧上传颜色缓冲，
//: 是页面上最贵的一步，**限到 30fps** 就够看，省下的时间留给游戏画面。
const DRAW3D_MS = 33;
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

    // ---- 前端**不跑仿真**。这里做三件事：
    //      ① **按固定频率**去服务端取状态（见下面的节流说明）
    //      ② 在两次状态之间**插值**，画面才平滑
    //      ③ 画游戏 + 画 3D 脑图
    //      游戏速度完全由服务端的固定 50 Hz 决定，与渲染帧率、与网络都无关。
    //
    // ⚠️ **`pollGame()` 必须节流，否则会把服务端打爆。**
    //    原来这里无条件每帧调用一次 = 浏览器以 **60 req/s** 去砸 Python 的
    //    `ThreadingHTTPServer`（单进程，还要跟一个 50 Hz 的仿真线程抢 GIL）。
    //    实测后果：6 秒发出 **359 个**请求，`/api/game/state` 延迟从 ~3 ms
    //    飙到 **38.9 ms**，页面卡顿、跳一下就停住、3D 面板几乎没机会更新。
    //    服务端本身只有 50 Hz，取 20 Hz 完全够（插值补上中间的帧），请求量降到 1/3。
    if (ts - (DBG.pollTs || 0) >= POLL_MS) {
      DBG.pollTs = ts;
      pollGame();
    }
    interpolate();
    // 死亡后的重开由**服务端**负责（撞了停 1.6s 再重开，见 server.py 的 _game_loop）。
    // 前端只跟着显示，不再自己计时重开。
    if (G.dead) G.deadT = 0;                  // 本地不再累加（避免旧逻辑误触发 reset）
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
    // 只负责画。
    drawGame(); drawScope();
    // 3D 脑图单独限频（见 DRAW3D_MS 的说明）—— 它是页面上最贵的一步
    if (P.show3d && ts - (DBG.d3Ts || 0) >= DRAW3D_MS) {
      DBG.d3Ts = ts;
      draw3D();
    }
    hud();
  } catch (e) {
    DBG.err = (e && e.stack) ? e.stack : String(e);
    // ⚠️ **必须让它看得见。** 这个 try/catch 原来只写 DBG.err，而 DBG.err
    //    除了控制台没人看 —— 于是"每帧都抛同一个异常"这种致命状态可以完全静默：
    //    渲染循环每帧在异常处中断，游戏卡住、3D 面板全黑、HUD 不动，
    //    而 console 一个错都不报。我就被一个**从未定义过的 `WARMUP_S`**
    //    这样坑了很久（同一个函数名在别处有定义，所以看代码也容易漏）。
    //    现在把错误显性打到状态栏；只打几次，避免每帧重排 DOM。
    DBG.errN = (DBG.errN || 0) + 1;
    if (DBG.errN === 1 || DBG.errN === 60 || DBG.errN % 600 === 0) {
      const el = $('stat');
      if (el) {
        el.className = 'warn';
        el.textContent = `⚠️ 渲染循环出错（第 ${DBG.errN} 次）：`
          + String(DBG.err).split('\n').slice(0, 3).join(' | ');
      }
    }
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
    init3D(); initUI();
    // 游戏世界由服务端持有（进页面时它已经在跑了），前端直接开始取状态即可。
    // 先同步取一次，避免头几帧没有 srvStats 而画不出东西。
    await pollGame();
    requestAnimationFrame(loop);
  } catch (e) {
    // 之前这里静默失败，页面永远停在"载入中"
    $('stat').textContent = '启动失败：' + (e && e.stack ? e.stack : e);
    $('stat').className = 'warn';
  }
}
addEventListener('resize', () => three && resize());
boot();
