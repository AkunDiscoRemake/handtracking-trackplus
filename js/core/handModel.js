// Hand topology + anatomical template shared by the 2D→3D lifting, the GLB
// generator (tools/gen-hand-glb.mjs) and the runtime rig solver.
//
// Keypoint order (21) is the COCO-WholeBody / OneHand10K order produced by
// RTMPose-hand: 0 wrist, 1-4 thumb (CMC, MCP, IP, TIP), 5-8 index
// (MCP, PIP, DIP, TIP), 9-12 middle, 13-16 ring, 17-20 pinky.
//
// Hand frame (right hand, metres): origin at the wrist, +Y towards the middle
// finger MCP, +Z out of the palm surface (palm normal), +X = Y × Z which points
// to the thumb side. The left hand is the mirror image (X negated).

export const NUM_KP = 21;

export const KP = Object.freeze({
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
});

export const FINGERS = Object.freeze([
  { name: 'thumb', chain: [0, 1, 2, 3, 4] },
  { name: 'index', chain: [0, 5, 6, 7, 8] },
  { name: 'middle', chain: [0, 9, 10, 11, 12] },
  { name: 'ring', chain: [0, 13, 14, 15, 16] },
  { name: 'pinky', chain: [0, 17, 18, 19, 20] },
]);

// Parent of each keypoint in the kinematic tree (-1 for the wrist).
export const PARENT = Object.freeze([
  -1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,
]);

// Skeleton line pairs for 2D/3D debug drawing (includes the palm outline).
export const EDGES = Object.freeze([
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
]);

// Palm keypoints treated as (approximately) rigid and coplanar for pose fitting.
export const PALM_KP = Object.freeze([0, 5, 9, 13, 17]);

const DEG = Math.PI / 180;

// Adult hand proportions (~50th percentile, hand length ≈ 18.5 cm, palm
// breadth ≈ 8.4 cm). Values are approximations from published anthropometric
// tables (phalanx lengths after Buryanov & Kotiuk 2010, rounded); they are a
// prior, not a measurement of the user.
const MCP = {
  index: [0.0235, 0.0860],
  middle: [0.0020, 0.0900],
  ring: [-0.0185, 0.0850],
  pinky: [-0.0360, 0.0760],
};
const PHALANX = {
  index: [0.0390, 0.0225, 0.0190],
  middle: [0.0440, 0.0270, 0.0205],
  ring: [0.0410, 0.0260, 0.0200],
  pinky: [0.0320, 0.0190, 0.0180],
};
// Rest splay of each finger (radians, positive towards the thumb / +X).
const SPLAY = { index: 6 * DEG, middle: 0, ring: -6 * DEG, pinky: -13 * DEG };

// Radii (metres) used by the mesh generator for each joint of each finger
// [MCP, PIP, DIP, TIP]; thumb [CMC, MCP, IP, TIP].
export const RADII = Object.freeze({
  thumb: [0.0125, 0.0105, 0.0095, 0.0082],
  index: [0.0100, 0.0088, 0.0078, 0.0068],
  middle: [0.0102, 0.0091, 0.0080, 0.0070],
  ring: [0.0097, 0.0086, 0.0076, 0.0066],
  pinky: [0.0088, 0.0076, 0.0068, 0.0060],
});

function buildTemplate() {
  const p = new Array(NUM_KP);
  p[0] = [0, 0, 0];
  // Thumb: CMC sits on the radial side of the carpus, the chain is abducted
  // ~40° from the index and pronated slightly out of the palm plane (+Z).
  p[1] = [0.0215, 0.0250, 0.0060];
  const tDir = norm([Math.sin(38 * DEG), Math.cos(38 * DEG), 0.30]);
  const tLen = [0.0420, 0.0320, 0.0270];
  let cur = p[1];
  for (let i = 0; i < 3; i++) {
    cur = add(cur, scale(tDir, tLen[i]));
    p[2 + i] = cur;
  }
  const order = ['index', 'middle', 'ring', 'pinky'];
  order.forEach((f, fi) => {
    const base = 5 + fi * 4;
    const [x, y] = MCP[f];
    p[base] = [x, y, 0];
    const d = [Math.sin(SPLAY[f]), Math.cos(SPLAY[f]), 0];
    let c = p[base];
    for (let k = 0; k < 3; k++) {
      c = add(c, scale(d, PHALANX[f][k]));
      p[base + 1 + k] = c;
    }
  });
  return p;
}

export const TEMPLATE_RIGHT = Object.freeze(buildTemplate().map((v) => Object.freeze(v)));
export const TEMPLATE_LEFT = Object.freeze(TEMPLATE_RIGHT.map((v) => Object.freeze([-v[0], v[1], v[2]])));

export function template(handedness) {
  return handedness === 'left' ? TEMPLATE_LEFT : TEMPLATE_RIGHT;
}

// Bone length of keypoint i to its parent (metres), identical for both hands.
export const BONE_LENGTH = Object.freeze(
  TEMPLATE_RIGHT.map((v, i) => (PARENT[i] < 0 ? 0 : dist(v, TEMPLATE_RIGHT[PARENT[i]]))),
);

// Rig bone names, one per keypoint. The bone named for keypoint i has its
// head at keypoint i; the wrist bone is the skeleton root.
export const BONE_NAMES = Object.freeze([
  'wrist',
  'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
  'index_mcp', 'index_pip', 'index_dip', 'index_tip',
  'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
  'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
  'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]);

// Child keypoint that defines the pointing direction of each bone (-1 for leaves).
export const BONE_CHILD = Object.freeze(
  PARENT.map((_, i) => {
    if (i === 0) return KP.MIDDLE_MCP;
    const c = PARENT.indexOf(i);
    return c;
  }),
);

// ---------------------------------------------------------------- vec helpers
export function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
export function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
export function scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
export function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
export function len(a) { return Math.hypot(a[0], a[1], a[2]); }
export function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
export function norm(a) {
  const l = len(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}
