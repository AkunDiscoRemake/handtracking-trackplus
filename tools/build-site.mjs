#!/usr/bin/env node
// Builds build/site/: the full app (all ONNX Runtime backends: WebNN, WebGPU,
// WASM) as a static site for any HTTPS host, e.g. GitHub Pages. Requires
// `npm run setup` first (vendor/ and models/).
import { readFile, writeFile, mkdir, copyFile, readdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(ROOT, 'build', 'site');
if (!existsSync(join(ROOT, 'vendor', 'ort', 'ort.all.min.mjs'))) {
  console.error('vendor/ missing: run `npm run setup` first');
  process.exit(1);
}
await rm(OUT, { recursive: true, force: true });

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
  'index.html', 'coi-sw.js', 'css/app.css', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
  ...(await walk('js')),
  ...(await walk('vendor')),
  ...(await walk('assets')),
  'models/manifest.json', 'models/calibration.json',
  ...Object.values(manifest.models).flatMap((m) => m.parts.map((p) => `models/${p}`)),
];
for (const f of files) {
  await mkdir(dirname(join(OUT, f)), { recursive: true });
  await copyFile(join(ROOT, f), join(OUT, f));
}
await writeFile(join(OUT, '.nojekyll'), '');
console.log(`build/site: ${files.length} files`);
