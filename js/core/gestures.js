// Rule-based gesture classification on the lifted 3D keypoints, plus a
// temporal stabiliser (hysteresis) so labels do not flicker.

import { dist, sub, norm, dot } from './handModel.js';

// Point i of a flat keypoint array with 2 or 3 components per point.
const P = (k, i, dim = 3) => (dim === 3 ? [k[3 * i], k[3 * i + 1], k[3 * i + 2]] : [k[2 * i], k[2 * i + 1], 0]);

function angleBetween(a, b) {
  return Math.acos(Math.max(-1, Math.min(1, dot(norm(a), norm(b)))));
}

// Finger extension in [0, 1] from total flexion along the chain and the
// tip-to-wrist reach relative to the PIP-to-wrist reach.
export function fingerExtension(k, mcp, dim = 3) {
  const w = P(k, 0, dim), m = P(k, mcp, dim), p = P(k, mcp + 1, dim), d = P(k, mcp + 2, dim), t = P(k, mcp + 3, dim);
  const bend = angleBetween(sub(p, m), sub(d, p)) + angleBetween(sub(d, p), sub(t, d))
    + 0.5 * angleBetween(sub(m, w), sub(p, m));
  const reach = dist(t, w) / Math.max(dist(p, w), 1e-6);
  const fromBend = 1 - smooth(0.6, 1.9, bend);
  const fromReach = smooth(1.05, 1.45, reach);
  return 0.5 * fromBend + 0.5 * fromReach;
}

export function thumbExtension(k, dim = 3) {
  const cmc = P(k, 1, dim), mcp = P(k, 2, dim), ip = P(k, 3, dim), tip = P(k, 4, dim), idx = P(k, 5, dim), pky = P(k, 17, dim);
  const bend = angleBetween(sub(mcp, cmc), sub(ip, mcp)) + angleBetween(sub(ip, mcp), sub(tip, ip));
  const palmW = Math.max(dist(idx, pky), 1e-6);
  const away = dist(tip, idx) / palmW;
  return 0.45 * (1 - smooth(0.5, 1.4, bend)) + 0.55 * smooth(0.55, 1.0, away);
}

function smooth(e0, e1, x) {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

export const GESTURES = ['none', 'open_palm', 'fist', 'point', 'victory', 'rock', 'call', 'thumbs_up', 'thumbs_down', 'pinch', 'ok'];

// Returns { name, score, ext: [thumb, index, middle, ring, pinky] }.
// The lifted 3D pose relies on an anatomical prior that can misread a
// foreshortened straight finger as bent, so when the 2D keypoints are given
// the 2D evidence (straightness and reach in the image) is blended in.
export function classifyGesture(k3d, k2d = null) {
  const ext3 = [thumbExtension(k3d), fingerExtension(k3d, 5), fingerExtension(k3d, 9),
    fingerExtension(k3d, 13), fingerExtension(k3d, 17)];
  let ext = ext3;
  if (k2d) {
    const ext2 = [thumbExtension(k2d, 2), fingerExtension(k2d, 5, 2), fingerExtension(k2d, 9, 2),
      fingerExtension(k2d, 13, 2), fingerExtension(k2d, 17, 2)];
    ext = ext3.map((e, i) => 0.5 * e + 0.5 * ext2[i]);
  }
  const on = ext.map((e) => e > 0.5);
  const palmLen = Math.max(dist(P(k3d, 0), P(k3d, 9)), 1e-6);
  const pinchD = dist(P(k3d, 4), P(k3d, 8)) / palmLen;
  const margin = (vals, want) => Math.min(...vals.map((e, i) => (want[i] ? e - 0.5 : 0.5 - e)));
  const cands = [];
  const add = (name, want, ignoreThumb = false) => {
    const idx = ignoreThumb ? [1, 2, 3, 4] : [0, 1, 2, 3, 4];
    const m = margin(idx.map((i) => ext[i]), idx.map((i) => want[i]));
    if (m > 0) cands.push({ name, score: Math.min(1, 0.5 + m) });
  };
  if (pinchD < 0.32) {
    const others = [2, 3, 4].filter((i) => on[i]).length;
    cands.push({ name: others >= 2 ? 'ok' : 'pinch', score: Math.min(1, 0.6 + (0.32 - pinchD)) });
  }
  add('open_palm', [1, 1, 1, 1, 1]);
  add('fist', [0, 0, 0, 0, 0], true);
  add('point', [0, 1, 0, 0, 0], true);
  add('victory', [0, 1, 1, 0, 0], true);
  add('rock', [0, 1, 0, 0, 1], true);
  add('call', [1, 0, 0, 0, 1]);
  if (ext[0] > 0.62 && !on[1] && !on[2] && !on[3] && !on[4]) {
    // Image y grows downwards: a thumb tip above the thumb MCP is "up".
    const up = k3d[3 * 2 + 1] - k3d[3 * 4 + 1];
    const name = up > 0 ? 'thumbs_up' : 'thumbs_down';
    const m = margin(ext, [1, 0, 0, 0, 0]);
    if (m > 0) cands.push({ name, score: Math.min(1, 0.55 + m) });
  }
  // A fist with an extended thumb is a thumbs gesture, not a fist.
  if (on[0]) {
    const i = cands.findIndex((c) => c.name === 'fist');
    if (i >= 0 && cands.some((c) => c.name.startsWith('thumbs'))) cands.splice(i, 1);
  }
  cands.sort((a, b) => b.score - a.score);
  const best = cands[0] || { name: 'none', score: 0 };
  return { name: best.name, score: best.score, ext };
}

// Hysteresis: a new label must win for `minFrames` consecutive frames and
// at least `minMs` before it replaces the current one.
export class GestureStabilizer {
  constructor({ minFrames = 3, minMs = 90 } = {}) {
    this.minFrames = minFrames;
    this.minMs = minMs;
    this.current = 'none';
    this.score = 0;
    this.pending = null;
    this.pendingCount = 0;
    this.pendingSince = 0;
  }

  update(g, tMs) {
    if (g.name === this.current) {
      this.score = 0.7 * this.score + 0.3 * g.score;
      this.pending = null;
      return { name: this.current, score: this.score };
    }
    if (g.name !== this.pending) {
      this.pending = g.name;
      this.pendingCount = 1;
      this.pendingSince = tMs;
    } else {
      this.pendingCount++;
    }
    if (this.pendingCount >= this.minFrames && tMs - this.pendingSince >= this.minMs) {
      this.current = g.name;
      this.score = g.score;
      this.pending = null;
    }
    return { name: this.current, score: this.score };
  }
}
