#!/usr/bin/env node
/**
 * 训练监控面板：http://127.0.0.1:8788
 *
 *   node dashboard.mjs            # 默认 8788
 *   node dashboard.mjs --port 9000
 *
 * 只读 runs/ 与系统状态，不干扰训练。页面每 5 秒自动刷新。
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number((() => {
  const i = process.argv.indexOf('--port');
  return i < 0 ? 8788 : process.argv[i + 1];
})());

// 采集脚本只用标准库 + psutil；FlyBrain 的 env 里有 psutil，所以优先用它。
const PY_CANDIDATES = [
  'D:\\Code\\FlyBrain\\env\\python.exe',
  'D:\\Code\\DQN\\env\\python.exe',
];
const PY = PY_CANDIDATES.find((p) => fs.existsSync(p)) || 'python';

let cache = { t: 0, data: null };
const TTL_MS = 3000;   // 采集结果缓存 3 秒，避免页面轮询把 CPU 拉满

function collect() {
  return new Promise((resolve) => {
    const p = spawn(PY, [path.join(ROOT, 'scripts', 'status_json.py')], {
      cwd: ROOT,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    });
    let out = '';
    let err = '';
    p.stdout.on('data', (d) => (out += d));
    p.stderr.on('data', (d) => (err += d));
    p.on('close', (code) => {
      if (code !== 0) return resolve({ error: `status_json.py 退出码 ${code}`, detail: err.slice(-600) });
      try {
        resolve(JSON.parse(out));
      } catch (e) {
        resolve({ error: `JSON 解析失败: ${e.message}`, detail: out.slice(0, 400) });
      }
    });
    p.on('error', (e) => resolve({ error: `无法启动 ${PY}: ${e.message}` }));
  });
}

const server = http.createServer(async (req, res) => {
  const url = req.url.split('?')[0];
  if (url === '/api/status') {
    const now = Date.now();
    if (!cache.data || now - cache.t > TTL_MS) {
      cache = { t: now, data: await collect() };
    }
    res.writeHead(200, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
    res.end(JSON.stringify(cache.data));
    return;
  }
  if (url === '/' || url === '/index.html') {
    const f = path.join(ROOT, 'dashboard.html');
    if (!fs.existsSync(f)) {
      res.writeHead(500).end('缺少 dashboard.html');
      return;
    }
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' });
    fs.createReadStream(f).pipe(res);
    return;
  }
  res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' }).end('404');
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.log(`端口 ${PORT} 已被占用，直接打开：http://127.0.0.1:${PORT}/`);
    console.log(`换端口：node dashboard.mjs --port 8789`);
    process.exit(0);
  }
  throw err;
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`训练监控面板：http://127.0.0.1:${PORT}/`);
  console.log(`（每 5 秒自动刷新；ctrl+c 停止）`);
});
