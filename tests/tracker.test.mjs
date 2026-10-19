// Tracker behaviour on a scripted two-hand sequence: births, handedness,
// occlusion coasting with identity preserved, recovery and removal.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { HandTracker } from '../js/core/tracker.js';
import { poseKeypoints } from '../js/app/synthetic.js';
import { NUM_KP } from '../js/core/handModel.js';

const cam = { f: 500, cx: 320, cy: 240 };
const open = { thumb: [0.05, 0.05, 0.05], f: [[0.05, 0.05, 0.05], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05]] };

function hand2d(side, x, z = 0.5) {
  const pts = poseKeypoints(side, open);
  const kp = new Float32Array(NUM_KP * 2);
  pts.forEach((p, j) => {
    const w = [p[0] + x, -p[1] + 0.05, -p[2] + z]; // palm towards camera
    kp[2 * j] = cam.f * w[0] / w[2] + cam.cx;
    kp[2 * j + 1] = cam.f * w[1] / w[2] + cam.cy;
  });
  return kp;
}
const scores = (s) => new Float32Array(NUM_KP).fill(s);

test('two hands: birth with correct handedness, occlusion coasting, recovery with same id', () => {
  const tr = new HandTracker({ facing: 'palm', coastMs: 400 });
  // Frame 0: detector proposes two new hands (person's right hand appears on image left).
  let p = tr.plan(0);
  assert.equal(p.runDetector, true);
  let out = tr.update(0, [
    { trackId: null, kp: hand2d('right', -0.12), scores: scores(0.6) },
    { trackId: null, kp: hand2d('left', 0.12), scores: scores(0.6) },
  ], cam, 640, 480);
  assert.equal(out.length, 2);
  const right = out.find((h) => h.handedness === 'right'), left = out.find((h) => h.handedness === 'left');
  assert.ok(right && left);
  assert.ok(right.kp2d[0] < left.kp2d[0]);

  // Tracking frames: ROIs come from the tracks; detector not needed every frame.
  let detRuns = 0;
  for (let i = 1; i <= 10; i++) {
    const t = i / 30;
    p = tr.plan(t);
    if (p.runDetector) detRuns++;
    assert.equal(p.rois.length, 2);
    out = tr.update(t, p.rois.map((r) => {
      const isRight = r.trackId === right.id;
      return { trackId: r.trackId, kp: hand2d(isRight ? 'right' : 'left', (isRight ? -0.12 : 0.12) + i * 0.001), scores: scores(0.6) };
    }), cam, 640, 480);
  }
  assert.ok(detRuns <= 1, `detector ran ${detRuns}× while both hands were tracked`);

  // Occlusion of the right hand for 200 ms: low pose score → coasting, same id.
  for (let i = 11; i <= 16; i++) {
    const t = i / 30;
    p = tr.plan(t);
    out = tr.update(t, p.rois.map((r) => ({
      trackId: r.trackId,
      kp: hand2d(r.trackId === right.id ? 'right' : 'left', r.trackId === right.id ? -0.11 : 0.13),
      scores: scores(r.trackId === right.id ? 0.1 : 0.6),
    })), cam, 640, 480);
    const rh = out.find((h) => h.id === right.id);
    assert.ok(rh, 'occluded hand kept');
    assert.equal(rh.state, 'coasting');
    if (i > 11) assert.ok(p.runDetector, 'detector searches while a hand is coasting');
  }
  // Recovery: right hand visible again → same identity, tracking state.
  const t = 17 / 30;
  p = tr.plan(t);
  out = tr.update(t, p.rois.map((r) => ({ trackId: r.trackId, kp: hand2d(r.trackId === right.id ? 'right' : 'left', r.trackId === right.id ? -0.11 : 0.13), scores: scores(0.6) })), cam, 640, 480);
  const rh = out.find((h) => h.id === right.id);
  assert.equal(rh.state, 'tracking');
  assert.equal(rh.handedness, 'right');
  assert.equal(tr.stats.recoveries, 1);

  // Long absence → removed.
  let tt = t;
  for (let i = 0; i < 20; i++) {
    tt += 1 / 30;
    p = tr.plan(tt);
    out = tr.update(tt, p.rois.map((r) => ({ trackId: r.trackId, kp: hand2d('left', 0.13), scores: scores(r.trackId === right.id ? 0.05 : 0.6) })), cam, 640, 480);
  }
  assert.ok(!out.some((h) => h.id === right.id), 'lost hand removed after coastMs');
  assert.ok(out.some((h) => h.id === left.id));
});

test('idle mode throttles the detector when no hands are present', () => {
  const tr = new HandTracker({ idleAfterMs: 500, idleDetHz: 5 });
  let runs = 0;
  for (let i = 0; i < 90; i++) {
    const t = 1 + i / 30;
    tr.lastHandT = 0;
    const p = tr.plan(t);
    if (!p.skip && p.runDetector) { runs++; tr.lastDetT = t; }
  }
  assert.ok(runs <= 17 && runs >= 13, `detector runs in 3 s idle: ${runs}`);
});

test('duplicate ROIs converging on one hand collapse to one track', () => {
  const tr = new HandTracker();
  tr.update(0, [{ trackId: null, kp: hand2d('right', 0), scores: scores(0.6) }], cam, 640, 480);
  const out = tr.update(0.033, [{ trackId: null, kp: hand2d('right', 0.002), scores: scores(0.7) }], cam, 640, 480);
  assert.equal(out.length, 1);
});
