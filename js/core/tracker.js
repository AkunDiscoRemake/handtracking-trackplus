// Two-hand tracker: decides when to run the detector, which ROIs to feed the
// pose model, associates results with persistent tracks, filters keypoints,
// lifts them to 3D and survives short occlusions.
//
// Coordinates are image pixels of the processed frame; times are seconds on a
// monotonic clock.

import { NUM_KP } from './handModel.js';
import { OneEuroFilter, KalmanCV } from './filters.js';
import { roiFromKeypoints, roiFromBox, iou } from './decode.js';
import { liftHand, LiftState, palmChirality, handednessFromChirality, palmSizePx } from './lift.js';
import { classifyGesture, GestureStabilizer } from './gestures.js';

export const DEFAULTS = Object.freeze({
  maxHands: 2,
  minHandScore: 0.32,     // mean SimCC confidence to keep a hand (hands ≈0.5, background ≈0.2)
  minNewHandScore: 0.38,  // stricter threshold to start a new track
  detScore: 0.3,
  coastMs: 450,           // how long a lost hand is predicted before removal
  idleAfterMs: 1500,      // no hands for this long → idle detector rate
  idleDetHz: 8,
  searchEveryN: 3,        // detector cadence while fewer than maxHands tracked
  confirmEveryN: 45,      // detector sanity check while all hands are tracked
  facing: 'palm',         // handedness assumption at track birth: 'palm' | 'back'
  oneEuro: { minCutoff: 1.2, beta: 0.35, dCutoff: 1.0 },
  rotateRoi: true,
});

let nextId = 1;

class Track {
  constructor(t, kp, scores, handedness, opts) {
    this.id = nextId++;
    this.handedness = handedness;
    this.born = t;
    this.lastSeen = t;
    this.state = 'tracking';
    this.hits = 1;
    this.score = mean(scores);
    this.kpRaw = new Float32Array(kp);
    this.kpScores = new Float32Array(scores);
    this.euro = new OneEuroFilter(NUM_KP * 2, opts.oneEuro);
    this.kf = new KalmanCV(NUM_KP * 2, { q: 4e5, r: 9, p0: 1e4 });
    this.roiKf = new KalmanCV(3, { q: 2e5, r: 16, p0: 1e4 });
    this.lift = new LiftState();
    this.gesture = new GestureStabilizer();
    this.kp3d = new Float32Array(NUM_KP * 3);
    this.prev3d = new Float32Array(NUM_KP * 3);
    this.vel3d = new Float32Array(NUM_KP * 3);
    this.t3d = -1;
    this.roi = null;
    this.gest = { name: 'none', score: 0 };
    this.facing = 'palm';
    this.R = null;
    this.depth = 0;
    this.noise = 0;
    this.measure(t, kp, scores, opts);
  }

  measure(t, kp, scores, opts) {
    const size = palmSizePx(kp);
    // Large jumps (re-acquisition after occlusion) restart the filters
    // instead of smearing the old position into the new one.
    if (this.euro.t >= 0) {
      let jump = 0;
      for (let i = 0; i < NUM_KP * 2; i++) jump = Math.max(jump, Math.abs(kp[i] - this.euro.x[i]));
      if (jump > 1.5 * size) { this.euro.reset(); this.kf.reset(); this.lift.reset(); }
    }
    this.kpRaw.set(kp);
    this.kpScores.set(scores);
    this.euro.filter(kp, t, size);
    // High-frequency residual (px RMS) between raw and filtered keypoints:
    // a direct measure of landmark jitter.
    let r2 = 0;
    for (let i = 0; i < NUM_KP * 2; i++) { const d = kp[i] - this.euro.x[i]; r2 += d * d; }
    this.noise = 0.9 * this.noise + 0.1 * Math.sqrt(r2 / NUM_KP);
    // Measurement noise grows when the model is unsure.
    this.kf.update(kp, t, 1 / Math.max(0.15, mean(scores)));
    const roi = roiFromKeypoints(this.euro.x, { rotate: opts.rotateRoi });
    this.roiKf.update([roi.cx, roi.cy, roi.size], t);
    this.roi = roi;
    this.score = 0.6 * this.score + 0.4 * mean(scores);
    this.lastSeen = t;
    this.state = 'tracking';
    this.hits++;
  }

  // ROI for the next frame at time t, from the ROI Kalman filter.
  predictRoi(t) {
    const p = this.roiKf.extrapolate(Math.min(t, this.roiKf.t + 0.1), [0, 0, 0]);
    const grow = this.state === 'coasting' ? 1.25 : 1.0;
    return { cx: p[0], cy: p[1], size: Math.max(24, p[2] * grow), theta: this.roi ? this.roi.theta : 0 };
  }

  box() {
    const k = this.euro.x;
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    for (let i = 0; i < NUM_KP; i++) {
      x1 = Math.min(x1, k[2 * i]); x2 = Math.max(x2, k[2 * i]);
      y1 = Math.min(y1, k[2 * i + 1]); y2 = Math.max(y2, k[2 * i + 1]);
    }
    return { x1, y1, x2, y2 };
  }
}

export class HandTracker {
  constructor(opts = {}) {
    this.opts = { ...DEFAULTS, ...opts, oneEuro: { ...DEFAULTS.oneEuro, ...(opts.oneEuro || {}) } };
    this.tracks = [];
    this.frame = 0;
    this.lastHandT = -Infinity;
    this.lastDetT = -Infinity;
    this.stats = { detRuns: 0, poseRuns: 0, recoveries: 0, births: 0, deaths: 0 };
  }

  setOptions(o) {
    Object.assign(this.opts, o);
    if (o.oneEuro) for (const tr of this.tracks) Object.assign(tr.euro, o.oneEuro);
  }

  reset() { this.tracks = []; this.lastHandT = -Infinity; }

  // Decides the work for a frame. Returns { skip, runDetector, rois }.
  plan(t) {
    this.frame++;
    const o = this.opts;
    const tracking = this.tracks.filter((tr) => tr.state === 'tracking').length;
    const coasting = this.tracks.some((tr) => tr.state === 'coasting');
    const idle = this.tracks.length === 0 && (t - this.lastHandT) * 1000 > o.idleAfterMs;
    let runDetector = false;
    if (this.tracks.length === 0) {
      if (idle && (t - this.lastDetT) < 1 / o.idleDetHz) return { skip: true, runDetector: false, rois: [] };
      runDetector = true;
    } else if (coasting) runDetector = true;
    else if (tracking < o.maxHands) runDetector = this.frame % o.searchEveryN === 0;
    else runDetector = this.frame % o.confirmEveryN === 0;
    const rois = this.tracks.map((tr) => ({ trackId: tr.id, roi: tr.predictRoi(t) }));
    return { skip: false, runDetector, rois, idle };
  }

  // New ROIs from detector boxes that do not overlap an existing track.
  roisFromDetections(dets) {
    const o = this.opts;
    const out = [];
    const free = o.maxHands - this.tracks.length;
    for (const d of dets) {
      if (d.score < o.detScore) continue;
      const overlaps = this.tracks.some((tr) => {
        const b = tr.box();
        const r = tr.roi ? tr.roi.size / 2 : 0;
        const ext = { x1: b.x1 - r * 0.2, y1: b.y1 - r * 0.2, x2: b.x2 + r * 0.2, y2: b.y2 + r * 0.2 };
        return iou(ext, d) > 0.25 || (pointInBox(centre(d), ext) && tr.state === 'tracking');
      });
      // A detection near a coasting track is a re-acquisition candidate.
      const coast = this.tracks.find((tr) => tr.state === 'coasting' && nearTrack(tr, d));
      if (coast) { out.push({ trackId: coast.id, roi: roiFromBox(d.x1, d.y1, d.x2, d.y2), fromDet: true }); continue; }
      if (overlaps) continue;
      if (out.filter((r) => r.trackId == null).length >= Math.max(0, free)) continue;
      out.push({ trackId: null, roi: roiFromBox(d.x1, d.y1, d.x2, d.y2), fromDet: true, detScore: d.score });
    }
    return out;
  }

  // results: [{ trackId|null, kp: Float32Array(42) image px, scores: Float32Array(21) }]
  // cam: { f, cx, cy }. Returns the list of output hands.
  update(t, results, cam, frameW, frameH) {
    const o = this.opts;
    const seen = new Set();
    // 1. Updates for existing tracks (best result per track).
    const byTrack = new Map();
    for (const r of results) {
      if (r.trackId == null) continue;
      const s = mean(r.scores);
      const prev = byTrack.get(r.trackId);
      if (!prev || s > prev.s) byTrack.set(r.trackId, { r, s });
    }
    for (const tr of this.tracks) {
      const m = byTrack.get(tr.id);
      if (m && m.s >= o.minHandScore) {
        if (tr.state === 'coasting') this.stats.recoveries++;
        tr.measure(t, m.r.kp, m.r.scores, o);
        seen.add(tr.id);
      }
    }
    // 2. New tracks.
    for (const r of results) {
      if (r.trackId != null) continue;
      const s = mean(r.scores);
      if (s < o.minNewHandScore || this.tracks.length >= o.maxHands) continue;
      // Reject if it lands on an existing hand (two ROIs converging).
      const box = kpBox(r.kp);
      if (this.tracks.some((tr) => iou(tr.box(), box) > 0.35)) continue;
      const chir = palmChirality(r.kp);
      let hand = handednessFromChirality(chir, o.facing);
      const other = this.tracks.find((tr) => tr.handedness === hand);
      if (other) hand = hand === 'right' ? 'left' : 'right';
      const tr = new Track(t, r.kp, r.scores, hand, o);
      this.tracks.push(tr);
      seen.add(tr.id);
      this.stats.births++;
    }
    // 3. Coasting / removal.
    for (const tr of this.tracks) {
      if (seen.has(tr.id)) continue;
      tr.state = 'coasting';
      tr.score *= 0.8;
    }
    const before = this.tracks.length;
    this.tracks = this.tracks.filter((tr) => (t - tr.lastSeen) * 1000 <= o.coastMs);
    this.stats.deaths += before - this.tracks.length;
    // 4. Duplicate suppression.
    for (let i = 0; i < this.tracks.length; i++) {
      for (let j = i + 1; j < this.tracks.length; j++) {
        const a = this.tracks[i], b = this.tracks[j];
        if (iou(a.box(), b.box()) > 0.5) {
          const drop = a.score >= b.score ? j : i;
          this.tracks.splice(drop, 1);
          j = this.tracks.length; i = -1;
          break;
        }
      }
    }
    if (this.tracks.some((tr) => tr.state === 'tracking')) this.lastHandT = t;
    return this.outputs(t, cam, frameW, frameH);
  }

  outputs(t, cam, frameW, frameH) {
    const out = [];
    for (const tr of this.tracks) {
      // Coasting: extrapolate with decaying velocity.
      let kp2d = tr.euro.x;
      let vel = tr.kf.vel;
      if (tr.state === 'coasting') {
        const dt = Math.min(t - tr.lastSeen, 0.25);
        const k = new Float32Array(NUM_KP * 2);
        const damp = Math.exp(-dt / 0.15);
        for (let i = 0; i < NUM_KP * 2; i++) k[i] = tr.euro.x[i] + tr.kf.vel[i] * dt * damp * 0.5;
        kp2d = k;
        vel = new Float64Array(NUM_KP * 2).map((_, i) => tr.kf.vel[i] * damp);
      }
      const lift = liftHand(kp2d, tr.handedness, cam, tr.lift, tr.kp3d);
      if (lift.ok) {
        tr.R = lift.R; tr.depth = lift.depth; tr.facing = lift.facing;
        if (tr.t3d >= 0 && t > tr.t3d) {
          const dt = t - tr.t3d;
          for (let i = 0; i < NUM_KP * 3; i++) {
            const v = (tr.kp3d[i] - tr.prev3d[i]) / dt;
            tr.vel3d[i] = 0.5 * tr.vel3d[i] + 0.5 * v;
          }
        }
        tr.prev3d.set(tr.kp3d);
        tr.t3d = t;
        if (tr.state === 'tracking') tr.gest = tr.gesture.update(classifyGesture(tr.kp3d, kp2d), t * 1000);
      }
      const age = t - tr.lastSeen;
      const conf = tr.state === 'tracking' ? tr.score : tr.score * Math.max(0, 1 - age * 1000 / this.opts.coastMs);
      out.push({
        id: tr.id,
        handedness: tr.handedness,
        facing: tr.facing,
        state: tr.state,
        score: conf,
        kp2d: Float32Array.from(kp2d),
        vel2d: Float32Array.from(vel),
        kpRaw: Float32Array.from(tr.kpRaw),
        kpScores: Float32Array.from(tr.kpScores),
        kp3d: Float32Array.from(tr.kp3d),
        vel3d: Float32Array.from(tr.vel3d),
        R: tr.R ? Float32Array.from(tr.R) : null,
        depth: tr.depth,
        gesture: tr.gest.name,
        gestureScore: tr.gest.score,
        roi: tr.roi,
        noisePx: tr.noise,
        frameW, frameH,
      });
    }
    return out;
  }
}

function mean(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
function kpBox(kp) {
  let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
  for (let i = 0; i < NUM_KP; i++) {
    x1 = Math.min(x1, kp[2 * i]); x2 = Math.max(x2, kp[2 * i]);
    y1 = Math.min(y1, kp[2 * i + 1]); y2 = Math.max(y2, kp[2 * i + 1]);
  }
  return { x1, y1, x2, y2 };
}
function centre(b) { return [(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2]; }
function pointInBox(p, b) { return p[0] >= b.x1 && p[0] <= b.x2 && p[1] >= b.y1 && p[1] <= b.y2; }
function nearTrack(tr, d) {
  const r = tr.roi;
  if (!r) return false;
  const [x, y] = centre(d);
  return Math.hypot(x - r.cx, y - r.cy) < r.size * 0.9;
}
