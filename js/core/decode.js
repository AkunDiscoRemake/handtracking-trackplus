// Pre/post-processing math shared by the worker and the tests. Pure functions,
// no DOM, so they run unchanged in Node.

import { NUM_KP } from './handModel.js';

export const DET_SIZE = 320;
export const POSE_SIZE = 256;
export const SIMCC_SPLIT = 2.0;
export const SIMCC_BINS = POSE_SIZE * SIMCC_SPLIT; // 512

// RTMDet (mmdet) normalisation: BGR order, ImageNet mean/std in BGR.
export const DET_MEAN_BGR = [103.53, 116.28, 123.675];
export const DET_STD_BGR = [57.375, 57.12, 58.395];
// RTMPose (mmpose) normalisation: RGB order.
export const POSE_MEAN_RGB = [123.675, 116.28, 103.53];
export const POSE_STD_RGB = [58.395, 57.12, 57.375];

// ------------------------------------------------------------------ letterbox
// mmdet Resize(keep_ratio) + Pad(320): image drawn at the top-left, padded
// right/bottom with 114.
export function letterbox(srcW, srcH, size = DET_SIZE) {
  const s = Math.min(size / srcW, size / srcH);
  return { scale: s, w: Math.round(srcW * s), h: Math.round(srcH * s) };
}

// ------------------------------------------------------------- affine crops
// Square crop of side `size` (image px) centred at (cx, cy), rotated by
// `theta` so that the image direction at angle theta - π/2 maps to "up" in the
// crop. Returns the canvas transform (image → crop) and its inverse.
export function cropAffine(cx, cy, size, theta, outSize = POSE_SIZE) {
  const s = outSize / size;
  const c = Math.cos(theta), n = Math.sin(theta);
  const a = s * c, b = -s * n, cc = s * n, d = s * c;
  const e = outSize / 2 - a * cx - cc * cy;
  const f = outSize / 2 - b * cx - d * cy;
  // inverse: img = centre + R(θ) (crop - N/2) / s
  const inv = {
    a: c / s, b: n / s, c: -n / s, d: c / s,
    e: cx - (c / s) * (outSize / 2) + (n / s) * (outSize / 2),
    f: cy - (n / s) * (outSize / 2) - (c / s) * (outSize / 2),
  };
  return { fwd: { a, b, c: cc, d, e, f }, inv, cx, cy, size, theta, outSize };
}

export function applyAffine(m, x, y) {
  return [m.a * x + m.c * y + m.e, m.b * x + m.d * y + m.f];
}

// Rotation that makes the wrist→middle-MCP direction point up in the crop.
export function uprightAngle(wx, wy, mx, my) {
  return Math.atan2(my - wy, mx - wx) + Math.PI / 2;
}

// ROI (centre, side, angle) from 21 keypoints in image px. The box is taken
// in the rotated frame so it is tight for any hand orientation.
export function roiFromKeypoints(kp, { pad = 1.25, extra = 1.12, rotate = true } = {}) {
  const theta = rotate ? uprightAngle(kp[0], kp[1], kp[18], kp[19]) : 0;
  const c = Math.cos(-theta), s = Math.sin(-theta);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < NUM_KP; i++) {
    const x = kp[2 * i], y = kp[2 * i + 1];
    const rx = c * x - s * y, ry = s * x + c * y;
    if (rx < minX) minX = rx; if (rx > maxX) maxX = rx;
    if (ry < minY) minY = ry; if (ry > maxY) maxY = ry;
  }
  const rcx = (minX + maxX) / 2, rcy = (minY + maxY) / 2;
  // back to image frame
  const ci = Math.cos(theta), si = Math.sin(theta);
  const cx = ci * rcx - si * rcy, cy = si * rcx + ci * rcy;
  const size = Math.max(maxX - minX, maxY - minY) * pad * extra;
  return { cx, cy, size, theta };
}

// ROI from a detector box (x1, y1, x2, y2), mmpose TopDownGetBboxCenterScale
// with padding 1.25 and a square aspect (input is 256×256).
export function roiFromBox(x1, y1, x2, y2, pad = 1.25) {
  return { cx: (x1 + x2) / 2, cy: (y1 + y2) / 2, size: Math.max(x2 - x1, y2 - y1) * pad, theta: 0 };
}

// --------------------------------------------------------------- SimCC decode
// For batch element `b`, writes keypoints in crop pixel coordinates into
// `outXY` (length 42) and confidences into `outScore` (length 21). Score is
// min(max_x, max_y) as in mmpose get_simcc_maximum. A 3-point parabola refines
// the argmax to sub-bin precision.
export function decodeSimcc(simccX, simccY, b, outXY, outScore, bins = SIMCC_BINS, numKp = NUM_KP) {
  const base = b * numKp * bins;
  for (let k = 0; k < numKp; k++) {
    const ox = base + k * bins;
    const [ix, vx] = argmax(simccX, ox, bins);
    const [iy, vy] = argmax(simccY, ox, bins);
    const fx = ix + parabolaOffset(simccX, ox, ix, bins);
    const fy = iy + parabolaOffset(simccY, ox, iy, bins);
    outXY[2 * k] = fx / SIMCC_SPLIT;
    outXY[2 * k + 1] = fy / SIMCC_SPLIT;
    outScore[k] = Math.min(vx, vy);
  }
}

function argmax(a, off, n) {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = a[off + i];
    if (v > bv) { bv = v; bi = i; }
  }
  return [bi, bv];
}

function parabolaOffset(a, off, i, n) {
  if (i <= 0 || i >= n - 1) return 0;
  const l = a[off + i - 1], c = a[off + i], r = a[off + i + 1];
  const den = l - 2 * c + r;
  if (den >= 0) return 0;
  const d = 0.5 * (l - r) / den;
  return d > 0.5 ? 0.5 : d < -0.5 ? -0.5 : d;
}

// --------------------------------------------------------------- detections
// mmdeploy end2end RTMDet output: dets [1, N, 5] (x1, y1, x2, y2, score) in
// letterboxed input pixels (NMS already applied inside the graph).
export function parseDetections(dets, n, lbScale, srcW, srcH, scoreThr = 0.3) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const s = dets[i * 5 + 4];
    if (!(s >= scoreThr)) continue;
    const x1 = clamp(dets[i * 5] / lbScale, 0, srcW);
    const y1 = clamp(dets[i * 5 + 1] / lbScale, 0, srcH);
    const x2 = clamp(dets[i * 5 + 2] / lbScale, 0, srcW);
    const y2 = clamp(dets[i * 5 + 3] / lbScale, 0, srcH);
    if (x2 - x1 < 4 || y2 - y1 < 4) continue;
    out.push({ x1, y1, x2, y2, score: s });
  }
  out.sort((a, b) => b.score - a.score);
  return nms(out, 0.5);
}

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

// Greedy NMS over boxes sorted by descending score.
export function nms(boxes, thr) {
  const keep = [];
  for (const b of boxes) {
    if (keep.every((k) => iou(k, b) < thr)) keep.push(b);
  }
  return keep;
}

export function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
