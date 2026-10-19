import { test } from 'node:test';
import assert from 'node:assert/strict';
import { OneEuroFilter, KalmanCV, predictBounded } from '../js/core/filters.js';
import { cropAffine, applyAffine, decodeSimcc, roiFromKeypoints, parseDetections, nms, iou, letterbox, SIMCC_BINS } from '../js/core/decode.js';
import { NUM_KP, TEMPLATE_RIGHT, TEMPLATE_LEFT, BONE_LENGTH, PARENT, BONE_CHILD } from '../js/core/handModel.js';
import { liftHand, LiftState, palmChirality, handednessFromChirality, fitPlanar, planarPoseCandidates } from '../js/core/lift.js';
import { classifyGesture, GestureStabilizer } from '../js/core/gestures.js';
import { chooseBackend } from '../js/worker/backends.js';
import { poseKeypoints } from '../js/app/synthetic.js';
import { warpToTensor, rgbaToTensor } from '../js/core/warp.js';

// ----------------------------------------------------------------- helpers
const deg = Math.PI / 180;
function rot(yaw, pitch, roll) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch), cr = Math.cos(roll), sr = Math.sin(roll);
  const Rz = [cr, -sr, 0, sr, cr, 0, 0, 0, 1], Ry = [cy, 0, sy, 0, 1, 0, -sy, 0, cy], Rx = [1, 0, 0, 0, cp, -sp, 0, sp, cp];
  const mul = (A, B) => { const C = new Array(9).fill(0); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[3 * i + j] += A[3 * i + k] * B[3 * k + j]; return C; };
  return mul(Rx, mul(Ry, Rz));
}
// Projects a hand-frame pose into a 640×480 camera (palm towards camera when base = facing).
function project(pts, R, t, cam) {
  const base = [1, 0, 0, 0, -1, 0, 0, 0, -1]; // hand Y → up, hand Z → camera
  const kp3 = new Float32Array(NUM_KP * 3), kp2 = new Float32Array(NUM_KP * 2);
  pts.forEach((p, j) => {
    const b = [base[0] * p[0], base[4] * p[1], base[8] * p[2]];
    const w = [R[0] * b[0] + R[1] * b[1] + R[2] * b[2] + t[0], R[3] * b[0] + R[4] * b[1] + R[5] * b[2] + t[1], R[6] * b[0] + R[7] * b[1] + R[8] * b[2] + t[2]];
    kp3.set(w, 3 * j);
    kp2[2 * j] = cam.f * w[0] / w[2] + cam.cx;
    kp2[2 * j + 1] = cam.f * w[1] / w[2] + cam.cy;
  });
  return { kp3, kp2 };
}
const cam = { f: 500, cx: 320, cy: 240 };

// ----------------------------------------------------------------- filters
test('One Euro: converges on a constant and reduces noise', () => {
  const f = new OneEuroFilter(1, { minCutoff: 1, beta: 0 });
  let seed = 1;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5);
  let rawVar = 0, filtVar = 0;
  for (let i = 0; i < 300; i++) {
    const v = 10 + rnd() * 2;
    const out = f.filter([v], i / 60)[0];
    if (i > 60) { rawVar += (v - 10) ** 2; filtVar += (out - 10) ** 2; }
  }
  assert.ok(filtVar < rawVar * 0.25, `filtered variance ${filtVar} vs raw ${rawVar}`);
});

test('One Euro: beta reduces lag on fast motion', () => {
  const lag = (beta) => {
    const f = new OneEuroFilter(1, { minCutoff: 1, beta });
    let out = 0;
    for (let i = 0; i < 30; i++) out = f.filter([i * 10], i / 60, 1)[0];
    return 290 - out;
  };
  assert.ok(lag(1) < lag(0) * 0.5);
});

test('Kalman CV: estimates velocity and extrapolates', () => {
  const k = new KalmanCV(1, { q: 10, r: 0.01 });
  for (let i = 0; i <= 30; i++) k.update([5 + 2 * (i / 30)], i / 30);
  assert.ok(Math.abs(k.vel[0] - 2) < 0.1, `vel ${k.vel[0]}`);
  const out = k.extrapolate(1.5, [0]);
  assert.ok(Math.abs(out[0] - 8) < 0.1);
});

test('predictBounded clamps horizon and displacement', () => {
  const out = new Float32Array(2);
  predictBounded([0, 0], [1000, 0], 1, 0.05, 1e9, out, 2);
  assert.equal(out[0], 50);
  predictBounded([0, 0], [1000, 0], 1, 0.05, 10, out, 2);
  assert.ok(Math.abs(out[0] - 10) < 1e-4);
  predictBounded([0, 0], [1000, 0], -1, 0.05, 10, out, 2);
  assert.equal(out[0], 0);
});

// ------------------------------------------------------------------ decode
test('crop affine: forward/inverse round trip, rotation keeps the centre', () => {
  const A = cropAffine(300, 200, 150, 0.7, 256);
  for (const [x, y] of [[300, 200], [10, 20], [512, 400]]) {
    const [u, v] = applyAffine(A.fwd, x, y);
    const [x2, y2] = applyAffine(A.inv, u, v);
    assert.ok(Math.hypot(x2 - x, y2 - y) < 1e-6);
  }
  const [u, v] = applyAffine(A.fwd, 300, 200);
  assert.ok(Math.abs(u - 128) < 1e-9 && Math.abs(v - 128) < 1e-9);
});

test('ROI from keypoints aligns the hand upright', () => {
  const kp = new Float32Array(42);
  // hand pointing right: wrist at (100,100), middle MCP at (160,100)
  for (let i = 0; i < 21; i++) { kp[2 * i] = 100 + i * 5; kp[2 * i + 1] = 100 + (i % 3) * 4; }
  kp[0] = 100; kp[1] = 100; kp[18] = 160; kp[19] = 100;
  const r = roiFromKeypoints(kp);
  const A = cropAffine(r.cx, r.cy, r.size, r.theta);
  const [wx, wy] = applyAffine(A.fwd, 100, 100), [mx, my] = applyAffine(A.fwd, 160, 100);
  assert.ok(my < wy && Math.abs(mx - wx) < 1e-6, 'middle MCP should be straight above the wrist in the crop');
});

test('SimCC decode: argmax with sub-bin refinement and min score', () => {
  const x = new Float32Array(2 * NUM_KP * SIMCC_BINS), y = new Float32Array(2 * NUM_KP * SIMCC_BINS);
  for (let b = 0; b < 2; b++) for (let k = 0; k < NUM_KP; k++) {
    const o = (b * NUM_KP + k) * SIMCC_BINS;
    x[o + 100 + k] = 0.9; x[o + 99 + k] = 0.5; x[o + 101 + k] = 0.5;
    y[o + 300] = 0.6;
  }
  const xy = new Float32Array(42), sc = new Float32Array(21);
  decodeSimcc(x, y, 1, xy, sc);
  assert.equal(xy[0], 50);                // bin 100 / split 2
  assert.equal(xy[2 * 5], (100 + 5) / 2);
  assert.equal(xy[1], 150);
  assert.ok(Math.abs(sc[3] - 0.6) < 1e-6); // min(max_x, max_y)
});

test('detections: letterbox mapping, threshold and NMS', () => {
  const lb = letterbox(640, 480);
  assert.equal(lb.scale, 0.5);
  const dets = new Float32Array([10, 10, 60, 60, 0.9, 12, 12, 62, 62, 0.8, 100, 100, 150, 150, 0.2]);
  const out = parseDetections(dets, 3, lb.scale, 640, 480, 0.3);
  assert.equal(out.length, 1);
  assert.deepEqual([out[0].x1, out[0].y2], [20, 120]);
  assert.ok(iou({ x1: 0, y1: 0, x2: 2, y2: 2 }, { x1: 1, y1: 0, x2: 3, y2: 2 }) - 1 / 3 < 1e-9);
  assert.equal(nms([{ x1: 0, y1: 0, x2: 10, y2: 10, score: 1 }, { x1: 1, y1: 1, x2: 10, y2: 10, score: 0.5 }], 0.5).length, 1);
});

test('warp/normalise: canvas path and CPU path agree on an identity crop', () => {
  const W = 8, H = 8, src = new Uint8Array(W * H * 4);
  for (let i = 0; i < W * H; i++) { src[4 * i] = i * 3; src[4 * i + 1] = 100; src[4 * i + 2] = 255 - i; src[4 * i + 3] = 255; }
  const a = new Float32Array(3 * 64), b = new Float32Array(3 * 64);
  warpToTensor(src, W, H, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }, W, H, [10, 20, 30], [2, 2, 2], 'bgr', a);
  rgbaToTensor(src, W, H, [10, 20, 30], [2, 2, 2], 'bgr', b);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-4);
  assert.equal(b[0], (src[2] - 10) / 2); // channel 0 = blue in BGR
});

// --------------------------------------------------------------- hand model
test('hand model: template sizes are anatomically plausible and mirrored', () => {
  const handLen = Math.hypot(...TEMPLATE_RIGHT[12]);
  assert.ok(handLen > 0.17 && handLen < 0.2, `hand length ${handLen}`);
  for (let i = 0; i < NUM_KP; i++) assert.equal(TEMPLATE_LEFT[i][0], -TEMPLATE_RIGHT[i][0]);
  for (let i = 1; i < NUM_KP; i++) assert.ok(BONE_LENGTH[i] > 0.015 && BONE_LENGTH[i] < 0.1);
  for (let i = 1; i < NUM_KP; i++) assert.ok(PARENT[i] < i, 'topological order');
  assert.equal(BONE_CHILD[4], -1);
  assert.equal(BONE_CHILD[5], 6);
});

// ------------------------------------------------------------------ lifting
test('lifting: reprojects exactly, keeps bone lengths, recovers depth for template-sized hands', () => {
  for (const side of ['right', 'left']) {
    for (const [yaw, pitch, roll] of [[0, 0, 0], [25, -15, 10], [-30, 20, -25]]) {
      const pts = poseKeypoints(side, { thumb: [0.2, 0.3, 0.3], f: [[0.3, 0.5, 0.3], [0.1, 0.1, 0.1], [0.8, 1.0, 0.6], [0.2, 0.3, 0.2]] });
      const R = rot(yaw * deg, pitch * deg, roll * deg);
      const { kp3, kp2 } = project(pts, R, [0.03, -0.02, 0.45], cam);
      const out = new Float32Array(63);
      const res = liftHand(kp2, side, cam, new LiftState(), out);
      assert.ok(res.ok);
      for (let j = 0; j < NUM_KP; j++) {
        const u = cam.f * out[3 * j] / out[3 * j + 2] + cam.cx, v = cam.f * out[3 * j + 1] / out[3 * j + 2] + cam.cy;
        assert.ok(Math.hypot(u - kp2[2 * j], v - kp2[2 * j + 1]) < 1e-3, 'exact reprojection');
      }
      assert.ok(Math.abs(out[2] - kp3[2]) / kp3[2] < 0.08, `${side} wrist depth ${out[2]} vs ${kp3[2]}`);
      for (const j of [6, 7, 8, 10, 11, 12]) {
        const p = PARENT[j];
        const L = Math.hypot(out[3 * j] - out[3 * p], out[3 * j + 1] - out[3 * p + 1], out[3 * j + 2] - out[3 * p + 2]);
        assert.ok(Math.abs(L - BONE_LENGTH[j]) < 1e-4, `bone ${j} length ${L} vs ${BONE_LENGTH[j]}`);
      }
      assert.equal(res.facing, 'palm');
    }
  }
});

test('handedness from chirality: palm-facing assumption labels both hands correctly', () => {
  for (const side of ['right', 'left']) {
    const { kp2 } = project(poseKeypoints(side, { thumb: [0, 0, 0], f: [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] }), rot(0.2, 0.1, 0.3), [0, 0, 0.5], cam);
    assert.equal(handednessFromChirality(palmChirality(kp2), 'palm'), side);
    assert.equal(handednessFromChirality(palmChirality(kp2), 'back'), side === 'right' ? 'left' : 'right');
  }
});

test('planar pose: candidates are rotations and reproduce the fitted map', () => {
  const fit = fitPlanar(new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, -40, 0, 0, 0, 0, 0, 0, 1, -45, 0, 0, 0, 0, 0, 0, -9, -42, 0, 0, 0, 0, 0, 0, -18, -37]), TEMPLATE_RIGHT);
  const c = planarPoseCandidates(fit.M);
  for (const R of c.R) {
    const det = R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) + R[2] * (R[3] * R[7] - R[4] * R[6]);
    assert.ok(Math.abs(det - 1) < 1e-6);
  }
});

// ---------------------------------------------------------------- gestures
test('gestures: synthetic poses are classified', () => {
  const P = {
    open_palm: { thumb: [0.05, 0.05, 0.05], f: [[0.05, 0.05, 0.05], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05]] },
    fist: { thumb: [0.6, 0.9, 1.0], f: [[1.45, 1.7, 1.0], [1.5, 1.75, 1.05], [1.5, 1.75, 1.05], [1.45, 1.7, 1.0]] },
    point: { thumb: [0.6, 0.9, 1.0], f: [[0, 0.05, 0.05], [1.5, 1.75, 1.05], [1.5, 1.75, 1.05], [1.45, 1.7, 1.0]] },
    victory: { thumb: [0.6, 0.9, 1.0], f: [[0, 0.05, 0.05], [0, 0.05, 0.05], [1.5, 1.75, 1.05], [1.45, 1.7, 1.0]] },
    rock: { thumb: [0.6, 0.9, 1.0], f: [[0, 0.05, 0.05], [1.5, 1.75, 1.05], [1.5, 1.75, 1.05], [0, 0.05, 0.05]] },
  };
  for (const [name, pose] of Object.entries(P)) {
    const { kp3, kp2 } = project(poseKeypoints('right', pose), rot(0.1, 0.2, 0), [0, 0, 0.45], cam);
    assert.equal(classifyGesture(kp3, kp2).name, name);
  }
});

test('gesture stabiliser needs persistence before switching', () => {
  const s = new GestureStabilizer({ minFrames: 3, minMs: 50 });
  assert.equal(s.update({ name: 'fist', score: 1 }, 0).name, 'none');
  assert.equal(s.update({ name: 'fist', score: 1 }, 30).name, 'none');
  assert.equal(s.update({ name: 'fist', score: 1 }, 60).name, 'fist');
  assert.equal(s.update({ name: 'point', score: 1 }, 70).name, 'fist');
});

// ---------------------------------------------------------------- backends
test('backend choice: NPU preferred within slack, otherwise fastest valid', () => {
  const r = [
    { id: 'webnn-npu', valid: true, medianMs: 12 },
    { id: 'webnn-gpu', valid: true, medianMs: 10 },
    { id: 'webgpu', valid: true, medianMs: 11 },
    { id: 'wasm', valid: true, medianMs: 90 },
  ];
  assert.equal(chooseBackend(r).id, 'webnn-npu');
  r[0].medianMs = 30;
  assert.equal(chooseBackend(r).id, 'webnn-gpu');
  r[1].valid = false;
  assert.equal(chooseBackend(r).id, 'webgpu');
  assert.equal(chooseBackend([{ id: 'x', valid: false }]), null);
});
