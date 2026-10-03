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

  ws.close();
  try { child.kill(); } catch (_) {}
  process.exit(0);
})().catch(e => { console.error('诊断失败:', e.message); try { child.kill(); } catch (_) {} process.exit(1); });
