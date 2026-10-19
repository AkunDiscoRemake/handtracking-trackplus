// Monocular 2D → 3D hand lifting with an anatomical prior.
//
// RTMPose outputs 2D keypoints only. A single camera cannot recover metric
// depth, so this module produces a *plausible* 3D pose that reprojects exactly
// onto the detected 2D keypoints:
//
//  1. Palm pose: the 5 palm keypoints are fitted to the (planar) palm of the
//     anatomical template with a weak-perspective model. The closed-form
//     planar solution gives scale s (px per metre at the palm) and a rotation
//     up to the classic two-fold tilt ambiguity, resolved by temporal
//     continuity (and by full-perspective reprojection error on the first
//     frame of a track).
//  2. Depth: Z_palm = f / s, i.e. it assumes the user's hand has template
//     size. Error is proportional to the ratio between real and template hand
//     size (typically ±10–15 % for adults).
//  3. Fingers: each bone keeps its template length: the child keypoint is
//     placed where its camera ray meets the sphere of radius L around the
//     parent (exact perspective, two roots = two depth hypotheses). The root
//     is chosen to continue the parent bone, bend towards the palm and stay
//     close to the previous frame.
//
// Camera frame: x right, y down, z forward (OpenCV convention), metres.

import {
  NUM_KP, PARENT, PALM_KP, BONE_LENGTH, KP, template,
  sub, add, scale, dot, cross, norm, len,
} from './handModel.js';

const FINGER_BASES = [KP.THUMB_CMC, KP.INDEX_MCP, KP.MIDDLE_MCP, KP.RING_MCP, KP.PINKY_MCP];

// Signed chirality of the palm as seen in the image, measured against the
// right-hand template: < 0 means "right palm towards camera" or equivalently
// "left hand, back towards camera".
export function palmChirality(kp2d) {
  const fit = fitPlanar(kp2d, template('right'));
  return fit ? Math.sign(fit.det) || 1 : 0;
}

// Handedness of a newly born track from the image chirality and the facing
// assumption ('palm': palms towards the camera, 'back': backs of the hands
// towards the camera, e.g. first-person view).
export function handednessFromChirality(chir, assume = 'palm') {
  const palmFacingIfRight = chir < 0;
  if (assume === 'palm') return palmFacingIfRight ? 'right' : 'left';
  return palmFacingIfRight ? 'left' : 'right';
}

// Least-squares 2×2 linear map from template palm (x, y) to centred image
// points. Returns { M: [m11, m12, m21, m22], det, centroid2d, centroid3d }.
export function fitPlanar(kp2d, tpl) {
  let cx = 0, cy = 0, tx = 0, ty = 0, tz = 0;
  const n = PALM_KP.length;
  for (const i of PALM_KP) {
    cx += kp2d[2 * i]; cy += kp2d[2 * i + 1];
    tx += tpl[i][0]; ty += tpl[i][1]; tz += tpl[i][2];
  }
  cx /= n; cy /= n; tx /= n; ty /= n; tz /= n;
  // A Aᵀ (2×2) and B Aᵀ (2×2)
  let aa00 = 0, aa01 = 0, aa11 = 0, ba00 = 0, ba01 = 0, ba10 = 0, ba11 = 0;
  for (const i of PALM_KP) {
    const ax = tpl[i][0] - tx, ay = tpl[i][1] - ty;
    const bx = kp2d[2 * i] - cx, by = kp2d[2 * i + 1] - cy;
    aa00 += ax * ax; aa01 += ax * ay; aa11 += ay * ay;
    ba00 += bx * ax; ba01 += bx * ay; ba10 += by * ax; ba11 += by * ay;
  }
  const detA = aa00 * aa11 - aa01 * aa01;
  if (Math.abs(detA) < 1e-12) return null;
  const i00 = aa11 / detA, i01 = -aa01 / detA, i11 = aa00 / detA;
  const M = [
    ba00 * i00 + ba01 * i01, ba00 * i01 + ba01 * i11,
    ba10 * i00 + ba11 * i01, ba10 * i01 + ba11 * i11,
  ];
  return { M, det: M[0] * M[3] - M[1] * M[2], c2: [cx, cy], c3: [tx, ty, tz] };
}

// Both rotation hypotheses (row-major 3×3) and the scale from a planar fit.
export function planarPoseCandidates(M) {
  const [m11, m12, m21, m22] = M;
  const k = m11 * m11 + m12 * m12 + m21 * m21 + m22 * m22;
  const det = m11 * m22 - m12 * m21;
  const disc = Math.max(0, k * k - 4 * det * det);
  const s = Math.sqrt((k + Math.sqrt(disc)) / 2);
  if (!(s > 1e-9)) return null;
  const a = [m11 / s, m12 / s], b = [m21 / s, m22 / s];
  const r13 = Math.sqrt(Math.max(0, 1 - a[0] * a[0] - a[1] * a[1]));
  const ab = a[0] * b[0] + a[1] * b[1];
  const r23mag = Math.sqrt(Math.max(0, 1 - b[0] * b[0] - b[1] * b[1]));
  const r23 = ab > 0 ? -r23mag : r23mag;
  const make = (sg) => {
    let r1 = [a[0], a[1], sg * r13];
    let r2 = [b[0], b[1], sg * r23];
    r1 = norm(r1);
    r2 = norm(sub(r2, scale(r1, dot(r1, r2))));
    const r3 = cross(r1, r2);
    return [r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], r3[0], r3[1], r3[2]];
  };
  return { s, R: [make(1), make(-1)] };
}

export function mulRv(R, v) {
  return [
    R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
    R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
    R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
  ];
}

function rotSimilarity(A, B) {
  let t = 0;
  for (let i = 0; i < 9; i++) t += A[i] * B[i];
  return t; // trace(Aᵀ B)
}

// Persistent per-track lifting state (previous rotation and bone directions).
export class LiftState {
  constructor() {
    this.R = null;
    this.dirs = new Float32Array(NUM_KP * 3);
    this.hasDirs = false;
  }
  reset() { this.R = null; this.hasDirs = false; }
}

// Lifts one hand.
//  kp2d: Float32Array(42) image px; cam: { f, cx, cy } in px.
//  Writes camera-space metres into out (Float32Array(63)).
//  Returns { R, s, depth, facing, ok }.
export function liftHand(kp2d, handedness, cam, state, out) {
  const tpl = template(handedness);
  const fit = fitPlanar(kp2d, tpl);
  if (!fit) return { ok: false };
  const cand = planarPoseCandidates(fit.M);
  if (!cand) return { ok: false };
  let R;
  if (state && state.R) {
    R = rotSimilarity(cand.R[0], state.R) >= rotSimilarity(cand.R[1], state.R) ? cand.R[0] : cand.R[1];
  } else {
    // First frame: the weak-perspective fit cannot tell the two tilts apart,
    // but true perspective can (the nearer side of the palm projects larger).
    // Pick the candidate with the lower full-perspective reprojection error.
    const e0 = palmReprojError(kp2d, tpl, fit, cand.R[0], cam.f / cand.s, cam);
    const e1 = palmReprojError(kp2d, tpl, fit, cand.R[1], cam.f / cand.s, cam);
    R = e0 <= e1 ? cand.R[0] : cand.R[1];
  }
  const f = cam.f;
  const zc = f / cand.s; // depth of the palm centroid
  const unproject = (i, z) => {
    out[3 * i] = (kp2d[2 * i] - cam.cx) * z / f;
    out[3 * i + 1] = (kp2d[2 * i + 1] - cam.cy) * z / f;
    out[3 * i + 2] = z;
  };
  // Palm keypoints + thumb CMC: depth from the rigid palm pose.
  for (const i of [...PALM_KP, KP.THUMB_CMC]) {
    const rel = mulRv(R, sub(tpl[i], fit.c3));
    unproject(i, zc + rel[2]);
  }
  const palmN = [R[2], R[5], R[8]]; // hand +Z (out of the palm) in camera frame
  const mirror = handedness === 'left' ? -1 : 1;
  const thumbFlex = norm(mulRv(R, [-0.7 * mirror, 0.2, 1]));

  // Finger chains.
  for (let fi = 0; fi < 5; fi++) {
    const baseKp = FINGER_BASES[fi];
    const flex = fi === 0 ? thumbFlex : palmN;
    // Direction of the bone that ends at the finger base, in camera frame.
    let prevDir = norm(mulRv(R, sub(tpl[baseKp], tpl[PARENT[baseKp]])));
    // Template rest direction of the first phalanx.
    const restDir = norm(mulRv(R, sub(tpl[baseKp + 1], tpl[baseKp])));
    prevDir = norm(add(scale(prevDir, 0.3), restDir));
    for (let j = baseKp + 1; j <= baseKp + 3; j++) {
      const p = PARENT[j];
      const L = BONE_LENGTH[j];
      const P = [out[3 * p], out[3 * p + 1], out[3 * p + 2]];
      // Exact perspective: the child lies on its camera ray Z·r; intersect
      // the ray with the sphere |X − P| = L. The two roots are the two
      // depth hypotheses; if the ray misses (2D bone longer than L), take the
      // closest point on the ray.
      const r = [(kp2d[2 * j] - cam.cx) / f, (kp2d[2 * j + 1] - cam.cy) / f, 1];
      const a = dot(r, r), b = dot(r, P), c = dot(P, P) - L * L;
      const disc = b * b - a * c;
      const roots = disc > 0 ? [(b + Math.sqrt(disc)) / a, (b - Math.sqrt(disc)) / a] : [b / a];
      let bestZ = roots[0], bestScore = -Infinity;
      for (const z of roots) {
        const d = norm(sub(scale(r, z), P));
        let sc = dot(d, prevDir) + 0.6 * dot(d, flex);
        if (state && state.hasDirs) {
          sc += 0.8 * dot(d, [state.dirs[3 * j], state.dirs[3 * j + 1], state.dirs[3 * j + 2]]);
        }
        if (sc > bestScore) { bestScore = sc; bestZ = z; }
      }
      unproject(j, bestZ);
      const d = norm(sub([out[3 * j], out[3 * j + 1], out[3 * j + 2]], P));
      if (state) { state.dirs[3 * j] = d[0]; state.dirs[3 * j + 1] = d[1]; state.dirs[3 * j + 2] = d[2]; }
      prevDir = d;
    }
  }
  if (state) { state.R = R; state.hasDirs = true; }
  return {
    ok: true,
    R,
    s: cand.s,
    depth: zc,
    facing: palmN[2] < 0 ? 'palm' : 'back',
  };
}

// Perspective reprojection error of the rigid template palm placed with
// rotation R so that its centroid lies on the ray of the 2D centroid at depth zc.
function palmReprojError(kp2d, tpl, fit, R, zc, cam) {
  const C = [(fit.c2[0] - cam.cx) * zc / cam.f, (fit.c2[1] - cam.cy) * zc / cam.f, zc];
  let err = 0;
  for (const i of PALM_KP) {
    const X = add(C, mulRv(R, sub(tpl[i], fit.c3)));
    const u = cam.f * X[0] / X[2] + cam.cx, v = cam.f * X[1] / X[2] + cam.cy;
    err += (u - kp2d[2 * i]) ** 2 + (v - kp2d[2 * i + 1]) ** 2;
  }
  return err;
}

// Palm size in px (wrist→middle MCP and index→pinky MCP), used to normalise
// thresholds by apparent hand size.
export function palmSizePx(kp2d) {
  const a = Math.hypot(kp2d[18] - kp2d[0], kp2d[19] - kp2d[1]);
  const b = Math.hypot(kp2d[10] - kp2d[34], kp2d[11] - kp2d[35]);
  return Math.max(a, b * 1.1, 1);
}

export { len };
