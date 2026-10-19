// Node-side reference pipeline: the same pre/post-processing as the browser
// worker (js/core/*) on top of onnxruntime-web's WASM backend. Used by the
// validation tool, the benchmark and the tests.
import * as ort from 'onnxruntime-web';
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { deflateSync } from 'node:zlib';
import {
  DET_SIZE, POSE_SIZE, DET_MEAN_BGR, DET_STD_BGR, POSE_MEAN_RGB, POSE_STD_RGB,
  letterbox, parseDetections, roiFromBox, cropAffine, decodeSimcc, applyAffine,
} from '../../js/core/decode.js';
import { warpToTensor } from '../../js/core/warp.js';
import { NUM_KP } from '../../js/core/handModel.js';

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

ort.env.wasm.numThreads = Number(process.env.ORT_THREADS || 1);
export { ort };
ort.env.logLevel = 'error';

export async function loadModel(key) {
  const manifest = JSON.parse(await readFile(join(ROOT, 'models', 'manifest.json'), 'utf8'));
  const m = manifest.models[key];
  const bufs = await Promise.all(m.parts.map((p) => readFile(join(ROOT, 'models', p))));
  const data = Buffer.concat(bufs);
  const h = createHash('sha256').update(data).digest('hex');
  if (h !== m.sha256) throw new Error(`${key}: sha256 mismatch`);
  return ort.InferenceSession.create(new Uint8Array(data), { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
}

export async function runDetector(sess, img) {
  const lb = letterbox(img.width, img.height, DET_SIZE);
  const inv = { a: 1 / lb.scale, b: 0, c: 0, d: 1 / lb.scale, e: 0, f: 0 };
  const t = new Float32Array(3 * DET_SIZE * DET_SIZE);
  // Pad region: sample outside the source → border 114 (mmdet pad value).
  warpToTensor(img.data, img.width, img.height, inv, DET_SIZE, DET_SIZE, DET_MEAN_BGR, DET_STD_BGR, 'bgr', t, 0, 114);
  const t0 = performance.now();
  const out = await sess.run({ input: new ort.Tensor('float32', t, [1, 3, DET_SIZE, DET_SIZE]) });
  const ms = performance.now() - t0;
  const dets = out.dets;
  const boxes = parseDetections(dets.data, dets.dims[1], lb.scale, img.width, img.height, 0.05);
  return { boxes, ms };
}

export async function runPose(sess, img, rois) {
  const n = rois.length;
  const plane = 3 * POSE_SIZE * POSE_SIZE;
  const t = new Float32Array(n * plane);
  const crops = rois.map((r, i) => {
    const A = cropAffine(r.cx, r.cy, r.size, r.theta);
    warpToTensor(img.data, img.width, img.height, A.inv, POSE_SIZE, POSE_SIZE, POSE_MEAN_RGB, POSE_STD_RGB, 'rgb', t, i * plane, 0);
    return A;
  });
  const t0 = performance.now();
  const out = await sess.run({ input: new ort.Tensor('float32', t, [n, 3, POSE_SIZE, POSE_SIZE]) });
  const ms = performance.now() - t0;
  const res = crops.map((A, b) => {
    const xy = new Float32Array(NUM_KP * 2), sc = new Float32Array(NUM_KP);
    decodeSimcc(out.simcc_x.data, out.simcc_y.data, b, xy, sc);
    const kp = new Float32Array(NUM_KP * 2);
    for (let k = 0; k < NUM_KP; k++) {
      const [x, y] = applyAffine(A.inv, xy[2 * k], xy[2 * k + 1]);
      kp[2 * k] = x; kp[2 * k + 1] = y;
    }
    return { kp, scores: sc };
  });
  return { res, ms };
}

export async function detectAndPose(det, pose, img, detThr = 0.3) {
  const d = await runDetector(det, img);
  const boxes = d.boxes.filter((b) => b.score >= detThr).slice(0, 2);
  if (!boxes.length) return { boxes: d.boxes, hands: [], detMs: d.ms, poseMs: 0 };
  const p = await runPose(pose, img, boxes.map((b) => roiFromBox(b.x1, b.y1, b.x2, b.y2)));
  return { boxes: d.boxes, hands: p.res, detMs: d.ms, poseMs: p.ms };
}

export function encodePng(w, h, rgba) {
  const raw = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 4 + 1)] = 0;
    Buffer.from(rgba.buffer, rgba.byteOffset + y * w * 4, w * 4).copy(raw, y * (w * 4 + 1) + 1);
  }
  const chunk = (type, data) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td) >>> 0);
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 6;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

let CRC;
function crc32(buf) {
  if (!CRC) {
    CRC = new Uint32Array(256);
    for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; CRC[n] = c; }
  }
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC[(c ^ buf[i]) & 255] ^ (c >>> 8);
  return c ^ 0xffffffff;
}

// Draws dots/lines into an RGBA buffer (debug images only).
export function drawPoint(img, x, y, r, col) {
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
    if (dx * dx + dy * dy > r * r) continue;
    const xx = Math.round(x + dx), yy = Math.round(y + dy);
    if (xx < 0 || yy < 0 || xx >= img.width || yy >= img.height) continue;
    const p = (yy * img.width + xx) * 4;
    img.data[p] = col[0]; img.data[p + 1] = col[1]; img.data[p + 2] = col[2]; img.data[p + 3] = 255;
  }
}
export function drawLine(img, x0, y0, x1, y1, col) {
  const n = Math.ceil(Math.hypot(x1 - x0, y1 - y0));
  for (let i = 0; i <= n; i++) drawPoint(img, x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, 1, col);
}
