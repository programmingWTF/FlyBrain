// 限时观察页面状态（不提前退出）。
// 用法: node demo/_watch.js "<url>" [总秒数] [采样间隔秒]
const { spawn } = require('child_process');
const os = require('os'), path = require('path'), fs = require('fs');

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const URL = process.argv[2] || 'http://127.0.0.1:8620/';
const TOTAL = Number(process.argv[3] || 60);
const EVERY = Number(process.argv[4] || 6);
const PORT = 9400 + Math.floor(Math.random() * 100);
const prof = path.join(os.tmpdir(), 'dshw_' + Date.now());
fs.mkdirSync(prof, { recursive: true });
const child = spawn(EDGE, ['--headless=new', '--disable-gpu', '--no-first-run',
  `--user-data-dir=${prof}`, `--remote-debugging-port=${PORT}`, 'about:blank'],
  { stdio: 'ignore' });

// ⚠️ 必须用 taskkill /T 杀**整棵进程树**。
// `child.kill()` 只杀直接子进程，Edge 的 gpu-process / renderer / crashpad-handler
// 会变成孤儿留在后台 —— 我之前的诊断脚本就是这样泄漏了 37 个无头 Edge，
// 把系统 CPU 吃到 50~60%，反过来让被测的服务器和页面变慢（自污染）。
function killTree() {
  try {
    spawn('taskkill', ['/F', '/T', '/PID', String(child.pid)], { stdio: 'ignore' });
  } catch (_) {
    try { child.kill(); } catch (_) {}
  }
}
process.on('exit', killTree);
process.on('SIGINT', () => { killTree(); process.exit(1); });

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
  const ws = new WebSocket(u);
  let id = 0; const pend = new Map();
  const send = (m, p = {}) => new Promise(r => {
    const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method: m, params: p }));
  });
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); }
  };
  await new Promise(r => { ws.onopen = r; });
  await send('Runtime.enable'); await send('Page.enable');
  await send('Page.navigate', { url: URL });
  const ev = async e => {
    const r = await send('Runtime.evaluate', { expression: e, returnByValue: true });
    if (r.result?.exceptionDetails) return 'ERR:' + r.result.exceptionDetails.text;
    return r.result?.result?.value;
  };
  console.log('URL:', URL, ` 共观察 ${TOTAL}s`);
  console.log('  秒   score  bTicks sbCalls sbBusy sbOk  sbFin  dead  cause');
  let best = 0;
  for (let t = EVERY; t <= TOTAL; t += EVERY) {
    await sleep(EVERY * 1000);
    const row = await ev(`(function(){var d=window.__dbg;if(!d)return 'no-dbg';
      var g=d.G||window.G||{};
      return [g.score||0,d.brainTicks||0,d.sbCalls||0,d.sbBusy||0,
              d.sbOk||0,d.sbFin||0,(g.dead?1:0),g.cause||'-'].join('|')})()`);
    if (row === 'no-dbg') { console.log(`  ${t}   页面没有 __dbg（脚本没跑？）`); continue; }
    const c = String(row).split('|');
    best = Math.max(best, Number(c[0]) || 0);
    console.log(`  ${String(t).padStart(3)}  ${c[0].padStart(5)}  ${c[1].padStart(6)} ${c[2].padStart(6)} ${c[3].padStart(5)} ${c[4].padStart(5)} ${c[5].padStart(6)}  ${c[6]}   ${c[7]}`);
  }
  console.log(`  期间最高分 ${best}`);
  ws.close(); killTree();
  process.exit(0);
})().catch(e => { console.error('失败:', e.message); killTree(); process.exit(1); });
