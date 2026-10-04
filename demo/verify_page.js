// 服务端权威架构的前端验收：页面能起、无异常、画面在动、状态来自服务端。
// 用法: node demo/verify_page.js [秒数]
//
// ⚠️ 这个脚本**必须确保不留残留进程**。之前我因为 killTree 没杀干净，
//    积累了几百个无头 Edge 进程把机器占满，导致后面所有测量都失真。
//    所以这里：① 记住所有子进程 ② 退出时 taskkill /F /T ③ 最后再扫一遍兜底。
const { spawn, spawnSync } = require('child_process');
const os = require('os'), path = require('path'), fs = require('fs');

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const SECS = Number(process.argv[2] || 35);
const URL = process.argv[3] || 'http://127.0.0.1:8620/';
const PORT = 9500 + Math.floor(Math.random() * 300);
const prof = path.join(os.tmpdir(), 'dshvp_' + Date.now());
fs.mkdirSync(prof, { recursive: true });

const children = [];
function spawnEdge(args) {
  const c = spawn(EDGE, args, { stdio: 'ignore' });
  children.push(c);
  return c;
}
function killAll() {
  for (const c of children) {
    try { spawnSync('taskkill', ['/F', '/T', '/PID', String(c.pid)], { stdio: 'ignore' }); }
    catch (_) {}
  }
  // 兜底：按 profile 路径精确杀（只杀我们这个临时 profile 的进程）
  try {
    const ps = spawnSync('powershell', ['-NoProfile', '-Command',
      `Get-CimInstance Win32_Process -Filter "Name='msedge.exe'" | ` +
      `Where-Object { $_.CommandLine -like '*${path.basename(prof)}*' } | ` +
      `ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`],
      { stdio: 'ignore' });
  } catch (_) {}
}
process.on('exit', killAll);
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => { killAll(); process.exit(1); });

const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  spawnEdge(['--headless=new', '--disable-gpu', '--no-first-run',
    `--user-data-dir=${prof}`, `--remote-debugging-port=${PORT}`, 'about:blank']);

  let wsUrl = null;
  for (let i = 0; i < 80 && !wsUrl; i++) {
    try {
      const l = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const p = l.find(t => t.type === 'page' && t.webSocketDebuggerUrl);
      if (p) wsUrl = p.webSocketDebuggerUrl;
    } catch (_) {}
    if (!wsUrl) await sleep(300);
  }
  if (!wsUrl) { console.log('  ❌ 连不上调试端口'); killAll(); process.exit(1); }

  const ws = new WebSocket(wsUrl);
  let id = 0; const pend = new Map(); const errs = []; const netErrs = [];
  const send = (m, p = {}) => new Promise(r => {
    const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method: m, params: p }));
  });
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); return; }
    if (m.method === 'Runtime.exceptionThrown') {
      const d = m.params.exceptionDetails;
      errs.push(`${d.text} @${(d.url || '').split('/').pop()}:${(d.lineNumber ?? 0) + 1}`);
    }
    if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') {
      const t = m.params.entry.text;
      if (!/runtime\.lastError|favicon/i.test(t)) errs.push('CONSOLE ' + t);
    }
    if (m.method === 'Network.loadingFailed') {
      netErrs.push(`${m.params.type} ${m.params.errorText}`);
    }
  };
  await new Promise(r => { ws.onopen = r; });
  await send('Runtime.enable'); await send('Page.enable');
  await send('Log.enable'); await send('Network.enable');
  await send('Page.navigate', { url: URL });
  const ev = async (e, aw) => {
    const r = await send('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: !!aw });
    if (r.result?.exceptionDetails) return 'EVAL-ERR ' + r.result.exceptionDetails.text;
    return r.result?.result?.value;
  };

  console.log(`验收 ${SECS}s  →  ${URL}`);
  await sleep(20000);                       // 等脑加载完成

  const snap = `JSON.stringify({
    frames: (window.__dbg&&__dbg.frames)||0,
    sc: (window.__dbg&&__dbg.G&&__dbg.G.score)||0,
    y: Math.round((window.__dbg&&__dbg.G&&__dbg.G.y)||0),
    pipes: ((window.__dbg&&__dbg.G&&__dbg.G.pipes)||[]).length,
    srvT: (window.__dbg&&__dbg.G&&__dbg.G.srvT)||0,
    rate: (window.__dbg&&__dbg.G&&__dbg.G.srvStats&&__dbg.G.srvStats.rate)||0,
    ticks: (window.__dbg&&__dbg.G&&__dbg.G.srvStats&&__dbg.G.srvStats.ticks)||0,
    cv: (document.querySelector('#game')||{}).width||0,
    hold: document.querySelector('#stat') ? document.querySelector('#stat').textContent.slice(0,26) : '',
    err: (window.__dbg&&__dbg.err)||null,
    errN: (window.__dbg&&__dbg.errN)||0,
    d3: (window.__three&&__three._n)||0,
    pts: (window.__three&&__three.geo&&__three.geo.attributes&&__three.geo.attributes.position)?__three.geo.attributes.position.count:0,
    vizN: ((window.__dbg&&__dbg.S&&__dbg.S.resp&&__dbg.S.resp.viz_spike)||[]).length,
    litN: ((window.__dbg&&__dbg.S&&__dbg.S.resp&&__dbg.S.resp.lit_spike)||[]).length,
    eff: (window.__dbg&&__dbg.S&&__dbg.S.resp&&__dbg.S.resp.eff)||0
  })`;

  let prev = null; const samples = [];
  for (let t = 4; t <= SECS; t += 4) {
    await sleep(4000);
    const raw = await ev(snap);
    if (typeof raw === 'string' && raw.startsWith('EVAL-ERR')) { console.log('  ' + raw); break; }
    const s = JSON.parse(raw);
    if (prev) samples.push({ dt: 4, dFrame: s.frames - prev.frames, dTicks: s.ticks - prev.ticks });
    console.log(`  t=${String(t).padStart(3)}s  帧${String(s.frames).padStart(5)}  `
      + `分数${String(s.sc).padStart(3)}  y=${String(s.y).padStart(3)}  管${s.pipes}  `
      + `服务端${String(s.ticks).padStart(6)}t (${s.rate}/s)  画布${s.cv}  `
      + `脑图${s.vizN}/${s.litN}  3D帧${s.d3}/点数${s.pts}`);
    prev = s;
  }

  const last = JSON.parse(await ev(snap));
  console.log(`\n  状态栏: ${last.hold}`);
  const fps = samples.length ? samples.reduce((a, b) => a + b.dFrame, 0) / (samples.length * 4) : 0;
  const srvRate = samples.length ? samples.reduce((a, b) => a + b.dTicks, 0) / (samples.length * 4) : 0;
  console.log(`  渲染帧率 ${fps.toFixed(1)} fps   （与游戏速度无关）`);
  console.log(`  服务端仿真 ${srvRate.toFixed(2)} 步/秒   ← 目标恒定 50`);
  console.log(`  ${Math.abs(srvRate - 50) < 2.5 ? '✅ 速度恒定' : '❌ 速度不达标'}`);
  console.log(`  画面在动: ${samples.some(x => x.dFrame > 0) ? '是' : '否'}   `
    + `分数在涨: ${last.sc > 0 ? '是(' + last.sc + ')' : '尚未'}`);
  // ---- ⚠️ 最关键的一条：**渲染循环本身有没有出错**。
  // `loop()` 的 try/catch 会把每帧的异常吞进 DBG.err 并让绘制全部跳过 ——
  // 症状是"游戏卡住 + 3D 面板全黑"，而 console 一个错都不报。
  // 我因为这个漏掉过一个"引用了从未定义的常量"的致命 bug，所以这里显式检查。
  if (last.err || last.errN > 0) {
    console.log(`  ❌ 渲染循环出错 ${last.errN} 次：${String(last.err).split('\n')[0]}`);
  } else {
    console.log('  ✅ 渲染循环无错（DBG.err 为空）');
  }
  if (last.d3 > 0 && last.pts > 0) {
    console.log(`  ✅ 3D 面板在画：${last.d3} 次渲染 / ${last.pts.toLocaleString()} 个点`);
  } else {
    console.log(`  ⚠️ 3D 面板没在画（draw_calls=${last.d3}，点数=${last.pts}）`);
  }

  const realErrs = errs.filter(e => !/lastError|favicon/i.test(e));
  console.log(realErrs.length ? `  ❌ 异常: ${[...new Set(realErrs)].slice(0, 5).join(' | ')}` : '  ✅ 无异常');
  if (netErrs.length) console.log(`  网络失败: ${[...new Set(netErrs)].slice(0, 4).join(' | ')}`);

  ws.close();
  killAll();
  await sleep(1500);
  killAll();
  process.exit(0);
})().catch(e => { console.error('失败:', e.message); killAll(); process.exit(1); });
