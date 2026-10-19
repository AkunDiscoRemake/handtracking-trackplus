#!/usr/bin/env node
// Builds build/preview/: the same app packaged for a sandboxed static host
// that wraps the page in its own <html>/<head> and cannot fetch large binaries
// from CDNs. Differences from the normal build: WASM-only ONNX Runtime
// (14 MB binary instead of the 28 MB JSEP one) and the sample photo as the
// start source (camera access is usually refused inside sandboxed frames).
import { readFile, writeFile, mkdir, copyFile, readdir } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(ROOT, 'build', 'preview');
const html = await readFile(join(ROOT, 'index.html'), 'utf8');
const headRaw = html.match(/<head>([\s\S]*?)<\/head>/)[1]
  .replace(/<meta charset[^>]*>\s*/, '').replace(/<meta name="viewport"[^>]*>\s*/, '');
const head = headRaw.replace(/\s*<title>[^<]*<\/title>/, '');
const title = headRaw.match(/<title>[^<]*<\/title>/)[0];
const body = html.match(/<body>([\s\S]*?)<\/body>/)[1];
const config = `<script>window.AURA_CONFIG = { ortVariant: 'wasm', startSource: 'sample' };</script>\n`;
await mkdir(OUT, { recursive: true });
await writeFile(join(OUT, 'index.html'), title + '\n' + head.trim() + '\n' + config + body.trim() + '\n');

async function walk(dir) {
  const out = [];
  for (const e of await readdir(join(ROOT, dir), { withFileTypes: true })) {
    const p = `${dir}/${e.name}`;
    if (e.isDirectory()) out.push(...await walk(p)); else out.push(p);
  }
  return out;
}
const manifest = JSON.parse(await readFile(join(ROOT, 'models', 'manifest.json'), 'utf8'));
const files = [
  'css/app.css',
  ...(await walk('js')),
  'vendor/three/three.module.js', 'vendor/three/three.core.js',
  'vendor/three/addons/loaders/GLTFLoader.js', 'vendor/three/addons/utils/BufferGeometryUtils.js', 'vendor/three/addons/utils/SkeletonUtils.js',
  'vendor/ort/ort.wasm.min.mjs', 'vendor/ort/ort-wasm-simd-threaded.mjs', 'vendor/ort/ort-wasm-simd-threaded.wasm',
  'assets/hands/left_hand.glb', 'assets/hands/right_hand.glb',
  'assets/samples/vsign_pd.png', 'assets/samples/move_pd.png',
  'models/manifest.json', 'models/calibration.json',
  ...Object.values(manifest.models).flatMap((m) => m.parts.map((p) => `models/${p}`)),
];
for (const f of files) {
  await mkdir(dirname(join(OUT, f)), { recursive: true });
  await copyFile(join(ROOT, f), join(OUT, f));
}
await writeFile(join(OUT, 'files.json'), JSON.stringify(files, null, 1));
console.log(`build/preview: index.html + ${files.length} files`);
