#!/usr/bin/env node
// Runs the real detector + pose models (onnxruntime-web, WASM, in Node) on the
// fixture photos, prints detections/keypoints/timings and writes annotated
// PNGs. Usage: node tools/validate-models.mjs [outDir]
import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { decodePng } from './lib/png.mjs';
import { ROOT, loadModel, detectAndPose, encodePng, drawPoint, drawLine } from './lib/pipeline-node.mjs';
import { EDGES } from '../js/core/handModel.js';
import { classifyGesture } from '../js/core/gestures.js';
import { liftHand, palmChirality } from '../js/core/lift.js';

const outDir = process.argv[2] || join(ROOT, 'build', 'validation');
await mkdir(outDir, { recursive: true });
const det = await loadModel('detector');
const pose = await loadModel('pose');
const dir = join(ROOT, 'tests', 'fixtures');
for (const f of (await readdir(dir)).filter((x) => x.endsWith('.png'))) {
  const img = decodePng(await readFile(join(dir, f)));
  const r = await detectAndPose(det, pose, img);
  console.log(`\n${f} ${img.width}x${img.height}  det ${r.detMs.toFixed(1)} ms  pose ${r.poseMs.toFixed(1)} ms`);
  for (const b of r.boxes.slice(0, 4)) console.log(`  box score=${b.score.toFixed(3)} [${[b.x1, b.y1, b.x2, b.y2].map((v) => v.toFixed(0)).join(', ')}]`);
  for (const h of r.hands) {
    const mean = h.scores.reduce((a, b) => a + b, 0) / h.scores.length;
    const f0 = Math.max(img.width, img.height) / (2 * Math.tan((67 / 2) * Math.PI / 180));
    const k3 = new Float32Array(63);
    const chir = palmChirality(h.kp);
    const lift = liftHand(h.kp, chir < 0 ? 'right' : 'left', { f: f0, cx: img.width / 2, cy: img.height / 2 }, null, k3);
    const g = classifyGesture(k3, h.kp);
    console.log(`  hand mean score=${mean.toFixed(3)} min=${Math.min(...h.scores).toFixed(3)} chirality=${chir} depth≈${lift.depth?.toFixed(2)} m gesture=${g.name} ext=[${g.ext.map((e) => e.toFixed(2))}]`);
    console.log('  kp:', Array.from(h.kp).map((v) => v.toFixed(0)).join(' '));
    for (const [a, b] of EDGES) drawLine(img, h.kp[2 * a], h.kp[2 * a + 1], h.kp[2 * b], h.kp[2 * b + 1], [0, 255, 120]);
    for (let k = 0; k < 21; k++) drawPoint(img, h.kp[2 * k], h.kp[2 * k + 1], 3, k === 0 ? [255, 0, 0] : [255, 220, 0]);
  }
  await writeFile(join(outDir, f.replace('.png', '_out.png')), encodePng(img.width, img.height, img.data));
}
console.log(`\nannotated images → ${outDir}`);
