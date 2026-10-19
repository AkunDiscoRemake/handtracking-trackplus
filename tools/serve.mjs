#!/usr/bin/env node
// Static server for development. Sends COOP/COEP so the page is
// cross-origin isolated (enables multi-threaded WASM). Camera access requires
// a secure context: http://localhost works; for a phone use HTTPS (see README).
//
//   node tools/serve.mjs [port] [--no-isolation] [--root build/site]
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { join, extname, normalize, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ri = process.argv.indexOf('--root');
const ROOT = ri > 0 ? resolve(process.argv[ri + 1]) : join(dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.argv.find((a) => /^\d+$/.test(a)) || 8080);
const isolate = !process.argv.includes('--no-isolation');
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.wasm': 'application/wasm', '.glb': 'model/gltf-binary',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.bin': 'application/octet-stream',
  '.onnx': 'application/octet-stream', '.md': 'text/markdown; charset=utf-8',
};

createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://x');
    let path = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, '');
    if (path.endsWith('/')) path += 'index.html';
    const file = join(ROOT, path);
    if (!file.startsWith(ROOT) || /node_modules|\.git/.test(file)) { res.writeHead(403).end(); return; }
    const st = await stat(file);
    if (!st.isFile()) { res.writeHead(404).end(); return; }
    const headers = { 'content-type': TYPES[extname(file)] || 'application/octet-stream', 'cache-control': 'no-cache' };
    if (isolate) {
      headers['cross-origin-opener-policy'] = 'same-origin';
      headers['cross-origin-embedder-policy'] = 'require-corp';
      headers['cross-origin-resource-policy'] = 'same-origin';
    }
    res.writeHead(200, headers);
    res.end(await readFile(file));
  } catch {
    res.writeHead(404).end('not found');
  }
}).listen(port, () => console.log(`http://localhost:${port}/  (cross-origin isolation: ${isolate ? 'on' : 'off'})`));
