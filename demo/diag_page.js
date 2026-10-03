// 用 CDP 连无头 Edge，打开页面并抓：控制台报错 / 未捕获异常 / 请求失败 / 最终页面状态
// 用法: node demo/_diag_page.js
const { spawn } = require('child_process');
const http = require('http');
const os = require('os');
const path = require('path');
const fs = require('fs');

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const URL = 'http://127.0.0.1:8620/';
const PORT = 9333;
const prof = path.join(os.tmpdir(), 'dsh_cdp_' + Date.now());
fs.mkdirSync(prof, { recursive: true });

const child = spawn(EDGE, [
  '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  `--user-data-dir=${prof}`, `--remote-debugging-port=${PORT}`, 'about:blank',
], { stdio: 'ignore' });

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function getWsUrl() {
  // ⚠️ 必须连**页面级** target：浏览器级 WebSocket 上 Runtime.evaluate 不存在
  //    （第一次就踩了这个：报 "'Runtime.evaluate' wasn't found"）。
  for (let i = 0; i < 80; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      const list = await r.json();
      const page = list.find(t => t.type === 'page' && t.webSocketDebuggerUrl);
      if (page) return page.webSocketDebuggerUrl;
    } catch (_) {}
    await sleep(300);
  }
  throw new Error('连不上 CDP 页面 target');
}

(async () => {
  const wsUrl = await getWsUrl();
  const ws = new WebSocket(wsUrl);
  let id = 0;
  const pending = new Map();
  const send = (method, params = {}, sessionId) => new Promise((res) => {
    const mid = ++id;
    pending.set(mid, res);
    ws.send(JSON.stringify({ id: mid, method, params, sessionId }));
  });

  const logs = [];
  let sessionId = null;

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); return; }
    const m = msg.method;
    if (m === 'Runtime.consoleAPICalled') {
      const txt = (msg.params.args || []).map(a => a.value ?? a.description ?? a.type).join(' ');
      logs.push(`[console.${msg.params.type}] ${txt}`);
    } else if (m === 'Runtime.exceptionThrown') {
      const d = msg.params.exceptionDetails;
      const loc = d.url ? ` @ ${d.url}:${(d.lineNumber ?? 0) + 1}:${(d.columnNumber ?? 0) + 1}` : '';
      logs.push(`[EXCEPTION] ${d.text}${loc}\n      ${(d.exception?.description || '').slice(0, 600)}`);
    } else if (m === 'Log.entryAdded') {
      const e = msg.params.entry;
      logs.push(`[log.${e.level}] ${e.text} ${e.url || ''}`.slice(0, 500));
    } else if (m === 'Network.loadingFailed') {
      logs.push(`[NET FAIL] ${msg.params.errorText} ${msg.params.type}`);
    } else if (m === 'Network.responseReceived') {
      const r = msg.params.response;
      if (r.status >= 400) logs.push(`[HTTP ${r.status}] ${r.url}`);
    }
  };

  await new Promise(r => { ws.onopen = r; });
  await send('Runtime.enable');
  await send('Log.enable');
  await send('Network.enable');
  await send('Page.enable');

  await send('Page.navigate', { url: URL });
  await sleep(25000);   // 等页面跑完（含 1.7MB coords + init3D）

  const evalJs = async (expr) => {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true });
    if (r.error) return 'CDP ERROR: ' + JSON.stringify(r.error);
    if (r.result?.exceptionDetails) return 'EVAL ERROR: ' + r.result.exceptionDetails.text;
    return r.result?.result?.value;
  };

  console.log('=== 控制台 / 异常 / 网络 ===');
  if (!logs.length) console.log('  (无)');
  logs.forEach(l => console.log('  ' + l));

  console.log('\n=== 页面状态 ===');
  console.log('  location      :', await evalJs('location.href'));
  console.log('  __dbg.frames  :', await evalJs('window.__dbg ? window.__dbg.frames : "(无)"'));
  console.log('  __dbg.err     :', await evalJs('window.__dbg ? String(window.__dbg.err) : "(无)"'));
  console.log('  stat 文本     :', await evalJs('(document.getElementById("stat")||{}).textContent'));
  console.log('  是否有 three  :', await evalJs('typeof window.__dbg'));
  console.log('  canvas game   :', await evalJs('(function(){var c=document.getElementById("game");return c?c.width+"x"+c.height:"无"})()'));
  console.log('  canvas brain  :', await evalJs('(function(){var c=document.getElementById("brain");return c?c.width+"x"+c.height:"无"})()'));

  // 脑 tick 数 vs 物理帧：错配修好后两者应当 1:1（评测台就是严格 1:1）。
  // ⚠️ 口径：这里是 **Δframes : ΔbrainTicks**，不是反过来。
  //    <1 表示脑比物理慢（页面整体变慢）；>1 表示物理跑在脑前面（鸟偏重，是之前那个 bug）。
  const t1 = await evalJs('window.__dbg && window.__dbg.brainTicks');
  const f1 = await evalJs('window.__dbg && window.__dbg.frames');
  await sleep(8000);
  const t2 = await evalJs('window.__dbg && window.__dbg.brainTicks');
  const f2 = await evalJs('window.__dbg && window.__dbg.frames');
  const dt = (t2 - t1) || 0, df = (f2 - f1) || 0;
  console.log('  brainTicks    :', t1, '->', t2, `  (Δ${dt})`);
  console.log('  frames        :', f1, '->', f2, `  (Δ${df})`);
  if (dt > 0) {
    const brainHz = dt / 8;                 // 实测脑速率（tick/秒）
    console.log(`  脑速率        : ${brainHz.toFixed(1)} tick/s  (dt=20ms 应为 50；`
      + `低了就是页面整体变慢)`);
    console.log(`  帧 : 脑tick   : ${(df / dt).toFixed(2)} : 1  `
      + `(1.00 = 物理与脑严格同步；<1 = 脑跟不上、游戏变慢)`);
  }
  console.log('  当前分数      :', await evalJs('(window.G && G.score) ?? "?"'),
              ' 死因:', await evalJs('(window.G && G.cause) || "无"'));

  // 连续观察：页面到底能不能过管子（能过就说明时序改对了）
  console.log('\n=== 连续观察分数（每 2 秒一次，共 ~24 秒）===');
  const seen = [];
  for (let i = 0; i < 12; i++) {
    const s = await evalJs('(window.G && G.score) ?? -1');
    const c = await evalJs('(window.G && G.cause) || "-"');
    const np = await evalJs('(window.G && G.pipes && G.pipes.length) ?? -1');
    seen.push(s);
    console.log(`  t+${(i + 1) * 2}s  分数=${s}  管子数=${np}  死因=${c}`);
  }
  console.log(`  最高分 ${Math.max(...seen)}  末次 ${seen[seen.length - 1]}`);

  ws.close();
  try { child.kill(); } catch (_) {}
  process.exit(0);
})().catch(e => { console.error('诊断失败:', e.message); try { child.kill(); } catch (_) {} process.exit(1); });
