#!/usr/bin/env node
/**
 * 下载 FlyWire FAFB v783 的「连接组 + 元数据」（公开 GCS，免认证）。
 *
 *   fafb_783_simple_edgelist.feather   288.6 MB  神经元→神经元的连接（本项目的图 A 来源）
 *   fafb_783_meta.feather               12.9 MB  细胞类型/分类/侧别等元数据
 *
 * 用法：node scripts/download_connectome.mjs
 */
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url))); // flyflappy/
const OUT = path.join(ROOT, 'data');
const BUCKET = 'lee-lab_brain-and-nerve-cord-fly-connectome';
const PREFIX = 'compiled_data/fafb_783/';

const FILES = ['fafb_783_simple_edgelist.feather', 'fafb_783_meta.feather'];

async function head(name) {
  const url = `https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${encodeURIComponent(PREFIX + name)}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`无法获取 ${name} 元数据：HTTP ${r.status}`);
  const j = await r.json();
  return Number(j.size);
}

async function download(name) {
  const dst = path.join(OUT, name);
  const want = await head(name);
  // 断点续传：已完整就跳过
  if (fs.existsSync(dst)) {
    const got = (await fsp.stat(dst)).size;
    if (got === want) {
      console.log(`  ${name} 已存在且完整（${(want / 1e6).toFixed(1)} MB），跳过`);
      return;
    }
    console.log(`  ${name} 大小不符（${got} != ${want}），重新下载`);
  }
  const url = `https://storage.googleapis.com/${BUCKET}/${PREFIX}${name}`;
  console.log(`  下载 ${name} （${(want / 1e6).toFixed(1)} MB）...`);
  const t0 = Date.now();
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${name} HTTP ${res.status}`);
  await pipeline(Readable.fromWeb(res.body), fs.createWriteStream(dst));
  const got = (await fsp.stat(dst)).size;
  if (got !== want) throw new Error(`${name} 下载不完整：${got} != ${want}`);
  const el = (Date.now() - t0) / 1000;
  console.log(`  ✓ ${name}  ${(got / 1e6).toFixed(1)} MB  ${el.toFixed(1)}s  ${(got / 1e6 / el).toFixed(2)} MB/s`);
}

await fsp.mkdir(OUT, { recursive: true });
console.log(`目标目录：${OUT}`);
for (const f of FILES) await download(f);
console.log('完成。');
