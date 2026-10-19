#!/usr/bin/env node
// Measures real model latency with onnxruntime-web (WASM) in Node on this
// machine. This is a CPU reference only; browser WebGPU/WebNN numbers must be
// measured on the target device with the in-app benchmark (Ajustes →
// Benchmark) or the CSV recorder.
//   ORT_THREADS=4 node tools/bench-node.mjs [runs=20]
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import os from 'node:os';
import { decodePng } from './lib/png.mjs';
import { ROOT, ort, loadModel, runDetector, runPose } from './lib/pipeline-node.mjs';
import { roiFromBox } from '../js/core/decode.js';

const runs = Number(process.argv[2] || 20);
const img = decodePng(await readFile(join(ROOT, 'tests', 'fixtures', 'move_pd.png')));
const det = await loadModel('detector');
const pose = await loadModel('pose');
const b = (await runDetector(det, img)).boxes[0];
const roi = roiFromBox(b.x1, b.y1, b.x2, b.y2);
const stats = (a) => {
  const s = [...a].sort((x, y) => x - y);
  return { median: s[Math.floor(s.length / 2)], p95: s[Math.min(s.length - 1, Math.floor(s.length * 0.95))], min: s[0] };
};
async function time(fn) {
  for (let i = 0; i < 3; i++) await fn();
  const t = [];
  for (let i = 0; i < runs; i++) t.push(await fn());
  return stats(t);
}
const detT = await time(async () => (await runDetector(det, img)).ms);
const pose1 = await time(async () => (await runPose(pose, img, [roi])).ms);
const pose2 = await time(async () => (await runPose(pose, img, [roi, { ...roi, cx: roi.cx + 5 }])).ms);
const cpu = os.cpus()[0]?.model || 'unknown CPU';
const f = (s) => `median ${s.median.toFixed(1)} ms · p95 ${s.p95.toFixed(1)} ms · min ${s.min.toFixed(1)} ms`;
console.log(`onnxruntime-web ${ort.env.versions?.web || ''} WASM · threads ${ort.env.wasm.numThreads} · ${cpu} · Node ${process.version} · ${runs} runs`);
console.log(`RTMDet-nano 320×320        ${f(detT)}`);
console.log(`RTMPose-m 256×256 batch 1  ${f(pose1)}`);
console.log(`RTMPose-m 256×256 batch 2  ${f(pose2)}`);
