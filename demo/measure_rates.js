// 测量页面的**真实速率与得分**。改了控制回路或页面循环之后跑一遍 ——
// 很容易出现"看着能飞"但实际跑在 12% 速度上的情况，这个脚本就是把两件事拆开。
//
// 判据（健康的标准）：
//   · 物理步/秒 == 脑 tick/秒   → 决策与物理严格 1:1（与评测台同构）
//   · pendMax == 0             → 没有决策积压
//   · 物理步/秒 与帧率**解耦**  → 慢帧率也能跑满（物理由脑响应驱动）
//
// 用法: node demo/measure_rates.js [秒数] [URL]
const { spawn } = require('child_process');
const os = require('os'), path = require('path'), fs = require('fs');

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const SECS = Number(process.argv[2] || 90);
const URL = process.argv[3] || 'http://127.0.0.1:8620/';
const PORT = 9960 + Math.floor(Math.random() * 30);
const prof = path.join(os.tmpdir(), 'dshm_' + Date.now());
fs.mkdirSync(prof, { recursive: true });
const child = spawn(EDGE, ['--headless=new', '--disable-gpu', '--no-first-run',
  `--user-data-dir=${prof}`, `--remote-debugging-port=${PORT}`, 'about:blank'],
  { stdio: 'ignore' });
function killTree() {
  try { spawn('taskkill', ['/F', '/T', '/PID', String(child.pid)], { stdio: 'ignore' }); }
  catch (_) { try { child.kill(); } catch (_) {} }
}
process.on('exit', killTree);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

(async () => {
  let u = null;
  for (let i = 0; i < 80 && !u; i++) {
    try {
      const l = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const p = l.find(t => t.type === 'page' && t.webSocketDebuggerUrl);
      if (p) u = p.webSocketDebuggerUrl;
    } catch (_) {}
    if (!u) await sleep(300);
  }
  if (!u) throw new Error('连不上 CDP');
  const ws = new WebSocket(u); let id = 0; const pend = new Map(); const errs = [];
  const send = (m, p = {}) => new Promise(r => {
    const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method: m, params: p }));
  });
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); return; }
    if (m.method === 'Runtime.exceptionThrown')
      errs.push(m.params.exceptionDetails.text);
  };
  await new Promise(r => { ws.onopen = r; });
  await send('Runtime.enable'); await send('Page.enable');
  await send('Page.navigate', { url: URL });
  const ev = async (e) => {
    const r = await send('Runtime.evaluate', { expression: e, returnByValue: true });
    return r.result?.result?.value;
  };
  const probe = `(function(){var d=window.__dbg||{},g=d.G||{};
    return {t:performance.now(), f:d.frames||0, ps:d.physSteps||0, bt:d.brainTicks||0,
      ok:d.sbOk||0, busy:d.sbBusy||0, pendMax:d.pendMax||0,
      sc:g.score||0, best:g.best||0, y:Math.round(g.y||0), dead:g.dead?1:0,
      cause:g.cause||'-', err:d.err?String(d.err).slice(0,120):null}})()`;

  await sleep(6000);                      // 让脑加载完
  const a = await ev(probe);
  console.log(`观察 ${SECS}s   URL=${URL}`);
  console.log('  秒   score  best  本段过管  局数');
  let prevSc = a.sc, lives = 0, totalPass = 0;
  const STEP = 10;
  for (let t = STEP; t <= SECS; t += STEP) {
    await sleep(STEP * 1000);
    const b = await ev(probe);
    if (b.sc < prevSc) lives++;
    const gain = b.sc - prevSc;
    if (gain > 0) totalPass += gain;
    prevSc = b.sc;
    console.log(`  ${String(t).padStart(3)}  ${String(b.sc).padStart(5)}  ${String(b.best).padStart(4)}  `
      + `${String(gain).padStart(8)}  ${String(lives).padStart(4)}`);
  }
  const b = await ev(probe);
  const dt = (b.t - a.t) / 1000;
  const f = (x) => (x / dt).toFixed(1);
  console.log(`\n=== 速率（${dt.toFixed(1)}s 窗口）===`);
  console.log(`  帧率          ${f(b.f - a.f)} fps`);
  console.log(`  物理          ${f(b.ps - a.ps)} 步/秒   ← 评测台是 50（1:1 实时）`);
  console.log(`  脑 tick       ${f(b.bt - a.bt)} tick/秒`);
  console.log(`  脑请求成功    ${f(b.ok - a.ok)} req/s`);
  console.log(`  脑忙跳过      ${f(b.busy - a.busy)} 次/s`);
  console.log(`  pend 峰值     ${b.pendMax} 步（= 一个决策被复用给几步物理）`);
  console.log(`\n=== 得分 ===`);
  console.log(`  本段过管 ${totalPass}   重开局 ${lives} 次`);
  console.log(`  score=${b.sc}  best=${b.best}  y=${b.y}  dead=${b.dead}  cause=${b.cause}`);
  console.log(`  → 平均每局 ${(totalPass / Math.max(lives, 1)).toFixed(2)} 管`);
  if (b.err) console.log(`  页面错误: ${b.err}`);
  if (errs.length) console.log(`  异常: ${errs.slice(0, 3).join(' | ')}`);
  ws.close(); killTree();
  process.exit(0);
})().catch(e => { console.error('失败:', e.message); killTree(); process.exit(1); });
