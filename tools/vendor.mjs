#!/usr/bin/env node
// Copies the exact runtime files the app needs from node_modules into vendor/
// (no CDN at runtime, works offline). Run after `npm install`.
import { mkdir, copyFile, readFile, writeFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const NM = join(ROOT, 'node_modules');
const files = [
  ['three/build/three.module.js', 'three/three.module.js'],
  ['three/build/three.core.js', 'three/three.core.js'],
  ['three/examples/jsm/loaders/GLTFLoader.js', 'three/addons/loaders/GLTFLoader.js'],
  ['three/examples/jsm/utils/BufferGeometryUtils.js', 'three/addons/utils/BufferGeometryUtils.js'],
  ['three/examples/jsm/utils/SkeletonUtils.js', 'three/addons/utils/SkeletonUtils.js'],
  ['three/LICENSE', 'three/LICENSE'],
  // ONNX Runtime Web: full build (WebNN + WebGPU + WebGL + WASM, JSEP binary)
  ['onnxruntime-web/dist/ort.all.min.mjs', 'ort/ort.all.min.mjs'],
  ['onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs', 'ort/ort-wasm-simd-threaded.jsep.mjs'],
  ['onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm', 'ort/ort-wasm-simd-threaded.jsep.wasm'],
  // WASM-only build (smaller binary, CPU fallback)
  ['onnxruntime-web/dist/ort.wasm.min.mjs', 'ort/ort.wasm.min.mjs'],
  ['onnxruntime-web/dist/ort-wasm-simd-threaded.mjs', 'ort/ort-wasm-simd-threaded.mjs'],
  ['onnxruntime-web/dist/ort-wasm-simd-threaded.wasm', 'ort/ort-wasm-simd-threaded.wasm'],
];
for (const [src, dst] of files) {
  const out = join(ROOT, 'vendor', dst);
  await mkdir(dirname(out), { recursive: true });
  await copyFile(join(NM, src), out);
}
// The onnxruntime-web npm package declares MIT but ships no LICENSE file.
await writeFile(join(ROOT, 'vendor', 'ort', 'LICENSE'), `MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
`);
const versions = {};
for (const p of ['three', 'onnxruntime-web']) versions[p] = JSON.parse(await readFile(join(NM, p, 'package.json'), 'utf8')).version;
await writeFile(join(ROOT, 'vendor', 'versions.json'), JSON.stringify(versions, null, 2) + '\n');
console.log('vendor/ ready', versions);
