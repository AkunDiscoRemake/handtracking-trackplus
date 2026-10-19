#!/usr/bin/env node
// Records reference outputs of both models on a fixed real photo with the
// WASM CPU backend. The browser worker reproduces the exact same input tensor
// (lossless PNG + the same JS warp) and rejects any accelerator whose output
// diverges from this reference.
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { decodePng } from './lib/png.mjs';
import { ROOT, loadModel, runDetector, runPose } from './lib/pipeline-node.mjs';
import { roiFromBox, cropAffine, applyAffine, POSE_SIZE } from '../js/core/decode.js';

const IMAGE = 'assets/samples/move_pd.png';
const img = decodePng(await readFile(join(ROOT, IMAGE)));
const det = await loadModel('detector');
const pose = await loadModel('pose');
const d = await runDetector(det, img);
const b = d.boxes[0];
const roi = roiFromBox(b.x1, b.y1, b.x2, b.y2);
const p = await runPose(pose, img, [roi]);
// Keypoints in crop space (what the worker compares after decodeSimcc).
const A = cropAffine(roi.cx, roi.cy, roi.size, roi.theta, POSE_SIZE);
const kp = [];
for (let k = 0; k < 21; k++) kp.push(...applyAffine(A.fwd, p.res[0].kp[2 * k], p.res[0].kp[2 * k + 1]).map((v) => +v.toFixed(3)));
const out = {
  image: IMAGE,
  recordedWith: 'onnxruntime-web WASM (CPU), Node ' + process.version,
  det: { expected: [b.x1, b.y1, b.x2, b.y2, b.score].map((v) => +v.toFixed(4)) },
  pose: { roi, kp, scores: Array.from(p.res[0].scores, (v) => +v.toFixed(4)) },
};
await writeFile(join(ROOT, 'models', 'calibration.json'), JSON.stringify(out, null, 1) + '\n');
console.log('models/calibration.json', out.det.expected);
