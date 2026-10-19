#!/usr/bin/env node
// Generates assets/hands/left_hand.glb and right_hand.glb: a skinned hand mesh
// with an anatomical 21-joint skeleton (wrist + 4 joints per finger, same
// layout as the RTMPose keypoints), plus a "curl" animation clip proving the
// fingers articulate.
//
// Geometry: signed distance field (round-cone phalanges, extruded palm,
// thenar eminence, forearm stub) blended with smooth-min, polygonised with
// Naive Surface Nets and smoothed; normals from the SDF gradient.
// Skinning: per-vertex weights from distance to each bone segment
// (soft-min, up to 4 influences).
//
//   node tools/gen-hand-glb.mjs [--voxel 0.0015]

import { writeFile, mkdir } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  NUM_KP, PARENT, RADII, FINGERS, BONE_NAMES, TEMPLATE_RIGHT, KP,
  sub, add, scale, dot, cross, norm, len,
} from '../js/core/handModel.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const argVoxel = process.argv.indexOf('--voxel');
const VOXEL = argVoxel > 0 ? Number(process.argv[argVoxel + 1]) : 0.002;

// ------------------------------------------------------------------ SDF parts
function jointRadius(j) {
  if (j === 0) return 0.016;
  for (const f of FINGERS) {
    const k = f.chain.indexOf(j);
    if (k > 0) return RADII[f.name][k - 1];
  }
  return 0.01;
}

// Bone segments: segment PARENT[j] → j is driven by bone PARENT[j].
const T = TEMPLATE_RIGHT;
const SEGMENTS = [];
for (let j = 1; j < NUM_KP; j++) {
  const p = PARENT[j];
  if (p === 0 && j !== KP.THUMB_CMC) continue; // metacarpals are in the palm block
  SEGMENTS.push({ a: T[p], b: T[j], ra: jointRadius(p === 0 ? j : p), rb: jointRadius(j), bone: p, finger: true });
}
// Metacarpal segments (inside the palm, used for skin weights only).
const META = [KP.INDEX_MCP, KP.MIDDLE_MCP, KP.RING_MCP, KP.PINKY_MCP].map((j) => ({
  a: [T[j][0] * 0.55, 0.012, 0], b: T[j], ra: 0.012, rb: jointRadius(j), bone: 0,
}));

// Round cone (capsule with linearly varying radius).
function sdRoundCone(p, a, b, ra, rb) {
  const ba = sub(b, a), pa = sub(p, a);
  const l2 = dot(ba, ba);
  const t = Math.max(0, Math.min(1, dot(pa, ba) / l2));
  const q = sub(pa, scale(ba, t));
  return len(q) - (ra + (rb - ra) * t);
}

// Palm outline (XY), counter-clockwise, slightly inset from the knuckles.
const PALM_POLY = [
  [0.020, 0.004], [0.026, 0.040], [0.025, 0.080], [0.012, 0.086],
  [-0.004, 0.088], [-0.020, 0.082], [-0.034, 0.072], [-0.036, 0.030], [-0.024, 0.002],
];

function sdPolygon(px, py, poly) {
  let d = Infinity, s = 1;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [ax, ay] = poly[i], [bx, by] = poly[j];
    const ex = bx - ax, ey = by - ay, wx = px - ax, wy = py - ay;
    const t = Math.max(0, Math.min(1, (wx * ex + wy * ey) / (ex * ex + ey * ey)));
    const dx = wx - ex * t, dy = wy - ey * t;
    d = Math.min(d, dx * dx + dy * dy);
    const c1 = py >= ay, c2 = py < by, c3 = ex * wy > ey * wx;
    if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) s = -s;
  }
  return s * Math.sqrt(d);
}

function sdPalm(p) {
  const round = 0.0085;
  const d2 = sdPolygon(p[0], p[1], PALM_POLY);
  // Palm is thicker at the base (thenar/hypothenar) than at the knuckles.
  const half = 0.0105 - 0.0035 * Math.max(0, Math.min(1, p[1] / 0.085));
  const dz = Math.abs(p[2] - 0.0015) - (half - round);
  const ox = Math.max(d2 + round, 0), oz = Math.max(dz, 0);
  return Math.hypot(ox, oz) + Math.min(Math.max(d2 + round, dz), 0) - round;
}

function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

function flatten(p, zs) { return [p[0], p[1], p[2] / zs]; }

function sdHand(p) {
  let d = sdPalm(p);
  // Forearm stub, slightly flattened.
  d = smin(d, sdRoundCone(flatten(p, 0.72), [0, -0.055, 0], [0, 0.006, 0], 0.025, 0.021) * 0.8, 0.012);
  // Thenar eminence (muscle at the thumb base) and hypothenar.
  d = smin(d, sdRoundCone(flatten(p, 0.8), [0.010, 0.012, 0.004], [0.034, 0.046, 0.006], 0.013, 0.010), 0.010);
  d = smin(d, sdRoundCone(flatten(p, 0.8), [-0.022, 0.012, 0.003], [-0.030, 0.060, 0.002], 0.011, 0.009), 0.010);
  for (const s of SEGMENTS) {
    const k = s.bone === 0 || s.bone === KP.THUMB_CMC ? 0.010 : 0.005;
    d = smin(d, sdRoundCone(p, s.a, s.b, s.ra, s.rb), k);
  }
  return d;
}

// --------------------------------------------------------------- surface nets
function surfaceNets(f, min, max, h) {
  const nx = Math.ceil((max[0] - min[0]) / h) + 1;
  const ny = Math.ceil((max[1] - min[1]) / h) + 1;
  const nz = Math.ceil((max[2] - min[2]) / h) + 1;
  const field = new Float32Array(nx * ny * nz);
  const id = (x, y, z) => x + nx * (y + ny * z);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    field[id(x, y, z)] = f([min[0] + x * h, min[1] + y * h, min[2] + z * h]);
  }
  const cellVert = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const cid = (x, y, z) => x + (nx - 1) * (y + (ny - 1) * z);
  const verts = [];
  const corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  for (let z = 0; z < nz - 1; z++) for (let y = 0; y < ny - 1; y++) for (let x = 0; x < nx - 1; x++) {
    const v = corners.map(([a, b, c]) => field[id(x + a, y + b, z + c)]);
    let neg = 0;
    for (const s of v) if (s < 0) neg++;
    if (neg === 0 || neg === 8) continue;
    let sx = 0, sy = 0, sz = 0, n = 0;
    for (const [i, j] of edges) {
      if ((v[i] < 0) === (v[j] < 0)) continue;
      const t = v[i] / (v[i] - v[j]);
      sx += corners[i][0] + t * (corners[j][0] - corners[i][0]);
      sy += corners[i][1] + t * (corners[j][1] - corners[i][1]);
      sz += corners[i][2] + t * (corners[j][2] - corners[i][2]);
      n++;
    }
    cellVert[cid(x, y, z)] = verts.length;
    verts.push([min[0] + (x + sx / n) * h, min[1] + (y + sy / n) * h, min[2] + (z + sz / n) * h]);
  }
  const tris = [];
  const quad = (a, b, c, d, flip) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (flip) tris.push(a, c, b, a, d, c); else tris.push(a, b, c, a, c, d);
  };
  for (let z = 1; z < nz - 1; z++) for (let y = 1; y < ny - 1; y++) for (let x = 1; x < nx - 1; x++) {
    const s0 = field[id(x, y, z)] < 0;
    // edge along +x
    if (x < nx - 1 && s0 !== (field[id(x + 1, y, z)] < 0)) {
      quad(cellVert[cid(x, y - 1, z - 1)], cellVert[cid(x, y, z - 1)], cellVert[cid(x, y, z)], cellVert[cid(x, y - 1, z)], !s0);
    }
    if (y < ny - 1 && s0 !== (field[id(x, y + 1, z)] < 0)) {
      quad(cellVert[cid(x - 1, y, z - 1)], cellVert[cid(x - 1, y, z)], cellVert[cid(x, y, z)], cellVert[cid(x, y, z - 1)], !s0);
    }
    if (z < nz - 1 && s0 !== (field[id(x, y, z + 1)] < 0)) {
      quad(cellVert[cid(x - 1, y - 1, z)], cellVert[cid(x, y - 1, z)], cellVert[cid(x, y, z)], cellVert[cid(x - 1, y, z)], !s0);
    }
  }
  return { verts, tris };
}

function gradient(f, p, e = 2e-4) {
  return norm([
    f([p[0] + e, p[1], p[2]]) - f([p[0] - e, p[1], p[2]]),
    f([p[0], p[1] + e, p[2]]) - f([p[0], p[1] - e, p[2]]),
    f([p[0], p[1], p[2] + e]) - f([p[0], p[1], p[2] - e]),
  ]);
}

// Taubin-style smoothing (λ/μ) keeps volume while removing voxel steps, then
// re-projects each vertex onto the zero level set along the gradient.
function smooth(verts, tris, f, iters = 3) {
  const nb = verts.map(() => new Set());
  for (let i = 0; i < tris.length; i += 3) {
    const [a, b, c] = [tris[i], tris[i + 1], tris[i + 2]];
    nb[a].add(b); nb[a].add(c); nb[b].add(a); nb[b].add(c); nb[c].add(a); nb[c].add(b);
  }
  const step = (k) => {
    const out = verts.map((v, i) => {
      if (!nb[i].size) return v;
      let s = [0, 0, 0];
      for (const j of nb[i]) s = add(s, verts[j]);
      s = scale(s, 1 / nb[i].size);
      return add(v, scale(sub(s, v), k));
    });
    for (let i = 0; i < verts.length; i++) verts[i] = out[i];
  };
  for (let it = 0; it < iters; it++) { step(0.5); step(-0.53); }
  for (let i = 0; i < verts.length; i++) {
    for (let k = 0; k < 2; k++) {
      const d = f(verts[i]);
      verts[i] = sub(verts[i], scale(gradient(f, verts[i]), d));
    }
  }
}

// ------------------------------------------------------------------ skinning
function distToSegment(p, a, b) {
  const ba = sub(b, a), pa = sub(p, a);
  const t = Math.max(0, Math.min(1, dot(pa, ba) / dot(ba, ba)));
  return { d: len(sub(pa, scale(ba, t))), t };
}

function skinWeights(p) {
  const sigma = 0.0035;
  const acc = new Map();
  const cands = [];
  for (const s of [...SEGMENTS, ...META]) {
    const { d, t } = distToSegment(p, s.a, s.b);
    const surf = d - (s.ra + (s.rb - s.ra) * t);
    cands.push({ bone: s.bone, surf });
  }
  // Points behind the wrist / forearm belong to the wrist bone.
  if (p[1] < 0.012) cands.push({ bone: 0, surf: -0.01 });
  // Thenar region follows the thumb metacarpal partly.
  const th = distToSegment(p, [0.010, 0.012, 0.004], [0.034, 0.046, 0.006]);
  cands.push({ bone: 0, surf: th.d - 0.013 + 0.002 });
  const m = Math.min(...cands.map((c) => c.surf));
  for (const c of cands) {
    const w = Math.exp(-(c.surf - m) / sigma);
    acc.set(c.bone, (acc.get(c.bone) || 0) + w);
  }
  const top = [...acc.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4);
  const sum = top.reduce((s, e) => s + e[1], 0);
  const joints = [0, 0, 0, 0], weights = [0, 0, 0, 0];
  top.forEach(([b, w], i) => { joints[i] = b; weights[i] = w / sum; });
  // Drop negligible influences and renormalise (keeps the validator happy).
  let s2 = 0;
  for (let i = 0; i < 4; i++) { if (weights[i] < 0.01) { weights[i] = 0; joints[i] = 0; } s2 += weights[i]; }
  for (let i = 0; i < 4; i++) weights[i] /= s2;
  return { joints, weights };
}

// ---------------------------------------------------------------- quaternions
function quatAxisAngle(axis, ang) {
  const a = norm(axis), s = Math.sin(ang / 2);
  return [a[0] * s, a[1] * s, a[2] * s, Math.cos(ang / 2)];
}

// ---------------------------------------------------------------------- glTF
function buildGlb(handedness, mesh) {
  const mx = handedness === 'left' ? -1 : 1;
  const M = (v) => [v[0] * mx, v[1], v[2]];
  const tpl = TEMPLATE_RIGHT.map(M);
  const nV = mesh.verts.length;
  const pos = new Float32Array(nV * 3), nrm = new Float32Array(nV * 3);
  const jnt = new Uint8Array(nV * 4), wgt = new Float32Array(nV * 4);
  const pmin = [Infinity, Infinity, Infinity], pmax = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nV; i++) {
    const p = M(mesh.verts[i]), n = M(mesh.normals[i]);
    pos.set(p, 3 * i); nrm.set(n, 3 * i);
    for (let k = 0; k < 3; k++) { pmin[k] = Math.min(pmin[k], p[k]); pmax[k] = Math.max(pmax[k], p[k]); }
    jnt.set(mesh.skin[i].joints, 4 * i); wgt.set(mesh.skin[i].weights, 4 * i);
  }
  const idx32 = nV > 65535;
  const idx = idx32 ? new Uint32Array(mesh.tris.length) : new Uint16Array(mesh.tris.length);
  for (let i = 0; i < mesh.tris.length; i += 3) {
    // Mirroring flips winding: swap two indices for the left hand.
    idx[i] = mesh.tris[i];
    idx[i + 1] = mx < 0 ? mesh.tris[i + 2] : mesh.tris[i + 1];
    idx[i + 2] = mx < 0 ? mesh.tris[i + 1] : mesh.tris[i + 2];
  }
  // Inverse bind matrices: bones have identity rest rotation, so IBM = T(-p).
  const ibm = new Float32Array(NUM_KP * 16);
  for (let j = 0; j < NUM_KP; j++) {
    const m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -tpl[j][0], -tpl[j][1], -tpl[j][2], 1];
    ibm.set(m, 16 * j);
  }
  // Animation "curl": rest → fist → rest. Flex axis = d × Z (palm normal).
  const times = new Float32Array([0, 1, 2]);
  const animBones = [];
  const FLEX = { thumb: [0.35, 0.55, 0.75], other: [1.25, 1.65, 1.05] };
  for (const f of FINGERS) {
    const flex = f.name === 'thumb' ? FLEX.thumb : FLEX.other;
    for (let k = 1; k <= 3; k++) {
      const j = f.chain[k];
      const d = norm(sub(tpl[f.chain[k + 1]], tpl[j]));
      let axis = cross(d, [0, 0, 1]);
      if (f.name === 'thumb') axis = cross(d, norm([-0.6 * mx, 0, 1]));
      const q = quatAxisAngle(axis, flex[k - 1]);
      animBones.push({ node: j, out: new Float32Array([0, 0, 0, 1, ...q, 0, 0, 0, 1]) });
    }
  }

  // Binary buffer layout.
  const parts = [];
  let byteLen = 0;
  const addView = (arr, target) => {
    const pad = (4 - (byteLen % 4)) % 4;
    if (pad) { parts.push(new Uint8Array(pad)); byteLen += pad; }
    const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
    parts.push(bytes);
    const view = { buffer: 0, byteOffset: byteLen, byteLength: bytes.byteLength };
    if (target) view.target = target;
    byteLen += bytes.byteLength;
    return view;
  };
  const bufferViews = [], accessors = [];
  const acc = (arr, type, componentType, count, target, extra = {}) => {
    bufferViews.push(addView(arr, target));
    accessors.push({ bufferView: bufferViews.length - 1, componentType, count, type, ...extra });
    return accessors.length - 1;
  };
  const aPos = acc(pos, 'VEC3', 5126, nV, 34962, { min: pmin, max: pmax });
  const aNrm = acc(nrm, 'VEC3', 5126, nV, 34962);
  const aJnt = acc(jnt, 'VEC4', 5121, nV, 34962);
  const aWgt = acc(wgt, 'VEC4', 5126, nV, 34962);
  const aIdx = acc(idx, 'SCALAR', idx32 ? 5125 : 5123, idx.length, 34963);
  const aIbm = acc(ibm, 'MAT4', 5126, NUM_KP);
  const aTime = acc(times, 'SCALAR', 5126, 3, undefined, { min: [0], max: [2] });
  const samplers = [], channels = [];
  for (const b of animBones) {
    const aOut = acc(b.out, 'VEC4', 5126, 3);
    samplers.push({ input: aTime, output: aOut, interpolation: 'LINEAR' });
    channels.push({ sampler: samplers.length - 1, target: { node: b.node, path: 'rotation' } });
  }

  const side = handedness === 'left' ? 'L' : 'R';
  const nodes = [];
  for (let j = 0; j < NUM_KP; j++) {
    const p = PARENT[j];
    const t = p < 0 ? tpl[j] : sub(tpl[j], tpl[p]);
    const children = PARENT.map((pp, c) => (pp === j ? c : -1)).filter((c) => c >= 0);
    const node = { name: `${BONE_NAMES[j]}_${side}`, translation: t.map((v) => +v.toFixed(6)), extras: { landmark: j } };
    if (children.length) node.children = children;
    nodes.push(node);
  }
  nodes.push({ name: `HandMesh_${side}`, mesh: 0, skin: 0 });
  const json = {
    asset: { version: '2.0', generator: 'aura-hands tools/gen-hand-glb.mjs', copyright: 'Apache-2.0' },
    extras: {
      handedness,
      units: 'metres',
      frame: 'origin at wrist; +Y towards middle MCP; +Z palm normal; thumb towards ' + (mx > 0 ? '+X' : '-X'),
      landmarks: 'RTMPose / COCO-WholeBody hand order (21 keypoints)',
    },
    scene: 0,
    scenes: [{ name: `${handedness}_hand`, nodes: [0, NUM_KP] }],
    nodes,
    meshes: [{
      name: `hand_${handedness}`,
      primitives: [{
        attributes: { POSITION: aPos, NORMAL: aNrm, JOINTS_0: aJnt, WEIGHTS_0: aWgt },
        indices: aIdx, material: 0, mode: 4,
      }],
    }],
    materials: [{
      name: 'skin',
      pbrMetallicRoughness: { baseColorFactor: [0.80, 0.60, 0.50, 1], metallicFactor: 0, roughnessFactor: 0.55 },
    }],
    skins: [{ name: `armature_${side}`, inverseBindMatrices: aIbm, joints: [...Array(NUM_KP).keys()], skeleton: 0 }],
    animations: [{ name: 'curl', samplers, channels }],
    accessors,
    bufferViews,
    buffers: [{ byteLength: 0 }],
  };
  const bin = new Uint8Array(Math.ceil(byteLen / 4) * 4);
  let o = 0;
  for (const p of parts) { bin.set(p, o); o += p.length; }
  json.buffers[0].byteLength = bin.length;
  let jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jpad = (4 - (jsonBytes.length % 4)) % 4;
  if (jpad) { const t = new Uint8Array(jsonBytes.length + jpad).fill(0x20); t.set(jsonBytes); jsonBytes = t; }
  const total = 12 + 8 + jsonBytes.length + 8 + bin.length;
  const glb = Buffer.alloc(total);
  glb.writeUInt32LE(0x46546c67, 0); glb.writeUInt32LE(2, 4); glb.writeUInt32LE(total, 8);
  glb.writeUInt32LE(jsonBytes.length, 12); glb.writeUInt32LE(0x4e4f534a, 16);
  Buffer.from(jsonBytes).copy(glb, 20);
  const bo = 20 + jsonBytes.length;
  glb.writeUInt32LE(bin.length, bo); glb.writeUInt32LE(0x004e4942, bo + 4);
  Buffer.from(bin).copy(glb, bo + 8);
  return glb;
}

export function generateMesh(voxel = VOXEL) {
  // Bounds include the abducted, pronated thumb (tip reaches z ≈ +0.045).
  const min = [-0.066, -0.078, -0.032], max = [0.104, 0.208, 0.07];
  const { verts, tris } = surfaceNets(sdHand, min, max, voxel);
  smooth(verts, tris, sdHand, 3);
  const normals = verts.map((v) => gradient(sdHand, v));
  const skin = verts.map((v) => skinWeights(v));
  return { verts, tris, normals, skin };
}

async function main() {
  const t0 = Date.now();
  const mesh = generateMesh();
  const dir = join(ROOT, 'assets', 'hands');
  await mkdir(dir, { recursive: true });
  for (const h of ['left', 'right']) {
    const glb = buildGlb(h, mesh);
    await writeFile(join(dir, `${h}_hand.glb`), glb);
    console.log(`${h}_hand.glb  ${mesh.verts.length} vertices, ${mesh.tris.length / 3} triangles, ${(glb.length / 1024).toFixed(0)} KiB`);
  }
  console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)} s (voxel ${VOXEL * 1000} mm)`);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
