// Inference worker: owns ONNX Runtime, the models, preprocessing buffers and
// the hand tracker. The main thread sends camera frames as ImageBitmaps (one
// in flight at a time) and receives tracked hands + timings.

import { HandTracker } from '../core/tracker.js';
import {
  DET_SIZE, POSE_SIZE, DET_MEAN_BGR, DET_STD_BGR, POSE_MEAN_RGB, POSE_STD_RGB,
  letterbox, cropAffine, decodeSimcc, applyAffine, parseDetections, roiFromBox,
} from '../core/decode.js';
import { rgbaToTensor, warpToTensor } from '../core/warp.js';
import { NUM_KP } from '../core/handModel.js';
import { loadModelBytes } from './modelStore.js';
import { CANDIDATES, probeEnvironment, isAvailable, chooseBackend, median, socHint } from './backends.js';

const POSE_PLANE = 3 * POSE_SIZE * POSE_SIZE;
const MAX_BATCH = 4;

let ort = null;
let ortVariant = '';
const sessions = {};      // key → { session, backend, fixedBatch }
const backendReport = {}; // key → [{ id, valid, medianMs, error }]
let tracker = null;
let env = null;
let calib = null;
let frameCount = 0;

// Reusable buffers (no per-frame allocation of large arrays).
const detBuf = new Float32Array(3 * DET_SIZE * DET_SIZE);
const poseBuf = new Float32Array(MAX_BATCH * POSE_PLANE);
const xyBuf = new Float32Array(NUM_KP * 2);
const scBuf = new Float32Array(NUM_KP);
let detCanvas, detCtx, poseCanvas, poseCtx;

function ensureCanvases() {
  if (detCtx) return;
  detCanvas = new OffscreenCanvas(DET_SIZE, DET_SIZE);
  detCtx = detCanvas.getContext('2d', { willReadFrequently: true, alpha: false });
  poseCanvas = new OffscreenCanvas(POSE_SIZE, POSE_SIZE);
  poseCtx = poseCanvas.getContext('2d', { willReadFrequently: true, alpha: false });
  poseCtx.imageSmoothingQuality = 'medium';
}

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const status = (text, extra = {}) => post({ type: 'status', text, ...extra });

// ------------------------------------------------------------------ ORT load
async function loadOrt(cfg) {
  const local = new URL('../../vendor/ort/', import.meta.url).href;
  const variants = [];
  if (cfg.ortVariant !== 'wasm') {
    variants.push({ name: 'all', mod: `${local}ort.all.min.mjs`, wasm: cfg.wasmBase || local });
    if (cfg.wasmFallbackBase) variants.push({ name: 'all', mod: `${local}ort.all.min.mjs`, wasm: cfg.wasmFallbackBase });
  }
  variants.push({ name: 'wasm', mod: `${local}ort.wasm.min.mjs`, wasm: local });
  return variants;
}

async function initOrtVariant(v) {
  const m = await import(v.mod);
  m.env.wasm.wasmPaths = v.wasm;
  m.env.wasm.numThreads = env.threads;
  m.env.wasm.proxy = false;
  m.env.logLevel = 'error';
  return m;
}

async function createSession(bytes, cand, key) {
  const ep = cand.ep();
  const opts = { executionProviders: [ep], graphOptimizationLevel: 'all' };
  let fixedBatch = 0;
  // WebNN needs static shapes: pin the pose batch to 1.
  if (cand.id.startsWith('webnn') && key === 'pose') { opts.freeDimensionOverrides = { batch: 1 }; fixedBatch = 1; }
  const session = await ort.InferenceSession.create(bytes, opts);
  return { session, backend: cand.id, label: cand.label, fixedBatch };
}

// ------------------------------------------------------------- calibration
async function loadCalibration(base) {
  const res = await fetch(new URL('calibration.json', base));
  if (!res.ok) return null;
  const c = await res.json();
  const blob = await (await fetch(new URL(c.image, new URL('../', base)))).blob();
  const bmp = await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  const cv = new OffscreenCanvas(bmp.width, bmp.height);
  const cx = cv.getContext('2d', { willReadFrequently: true });
  cx.drawImage(bmp, 0, 0);
  const rgba = cx.getImageData(0, 0, bmp.width, bmp.height).data;
  const W = bmp.width, H = bmp.height;
  bmp.close();
  const lb = letterbox(W, H);
  const det = new Float32Array(3 * DET_SIZE * DET_SIZE);
  warpToTensor(rgba, W, H, { a: 1 / lb.scale, b: 0, c: 0, d: 1 / lb.scale, e: 0, f: 0 }, DET_SIZE, DET_SIZE, DET_MEAN_BGR, DET_STD_BGR, 'bgr', det, 0, 114);
  const A = cropAffine(c.pose.roi.cx, c.pose.roi.cy, c.pose.roi.size, c.pose.roi.theta);
  const pose = new Float32Array(POSE_PLANE);
  warpToTensor(rgba, W, H, A.inv, POSE_SIZE, POSE_SIZE, POSE_MEAN_RGB, POSE_STD_RGB, 'rgb', pose, 0, 0);
  return { ...c, W, H, lb, detTensor: det, poseTensor: pose };
}

async function runCalib(key, s) {
  if (key === 'detector') {
    const out = await s.session.run({ input: new ort.Tensor('float32', calib.detTensor, [1, 3, DET_SIZE, DET_SIZE]) });
    const boxes = parseDetections(out.dets.data, out.dets.dims[1], calib.lb.scale, calib.W, calib.H, 0.05);
    disposeOutputs(out);
    return boxes;
  }
  const out = await s.session.run({ input: new ort.Tensor('float32', calib.poseTensor, [1, 3, POSE_SIZE, POSE_SIZE]) });
  const xy = new Float32Array(NUM_KP * 2), sc = new Float32Array(NUM_KP);
  decodeSimcc(out.simcc_x.data, out.simcc_y.data, 0, xy, sc);
  disposeOutputs(out);
  return { xy, sc };
}

function checkCalib(key, r) {
  if (key === 'detector') {
    const e = calib.det.expected;
    const b = r[0];
    if (!b) return 'nenhuma detecção na calibração';
    const err = Math.max(Math.abs(b.x1 - e[0]), Math.abs(b.y1 - e[1]), Math.abs(b.x2 - e[2]), Math.abs(b.y2 - e[3]));
    if (err > 6 || Math.abs(b.score - e[4]) > 0.06) return `diverge da referência (Δbox ${err.toFixed(1)} px, Δscore ${(b.score - e[4]).toFixed(3)})`;
    return null;
  }
  let maxErr = 0;
  for (let k = 0; k < NUM_KP; k++) {
    if (calib.pose.scores[k] < 0.3) continue;
    maxErr = Math.max(maxErr, Math.hypot(r.xy[2 * k] - calib.pose.kp[2 * k], r.xy[2 * k + 1] - calib.pose.kp[2 * k + 1]));
  }
  const ds = Math.abs(mean(r.sc) - mean(calib.pose.scores));
  if (maxErr > 4 || ds > 0.06) return `diverge da referência (erro ${maxErr.toFixed(1)} px, Δscore ${ds.toFixed(3)})`;
  return null;
}

async function benchSession(key, s, runs = 5) {
  const times = [];
  for (let i = 0; i < runs + 2; i++) {
    const t0 = performance.now();
    await runCalib(key, s);
    if (i >= 2) times.push(performance.now() - t0);
  }
  return median(times);
}

// Tries every available backend for one model, validates and benchmarks.
async function selectBackend(key, bytes, cfg) {
  const report = [];
  const order = cfg.forceBackend && cfg.forceBackend !== 'auto'
    ? CANDIDATES.filter((c) => c.id === cfg.forceBackend || c.id === 'wasm')
    : CANDIDATES;
  let chosen = null;
  const built = [];
  for (const cand of order) {
    if (!isAvailable(cand, env)) { report.push({ id: cand.id, valid: false, error: 'indisponível neste navegador' }); continue; }
    status(`${key}: testando ${cand.label}…`);
    try {
      const s = await createSession(bytes, cand, key);
      let err = null;
      if (calib) err = checkCalib(key, await runCalib(key, s));
      if (err) { report.push({ id: cand.id, valid: false, error: err }); await s.session.release?.(); continue; }
      const ms = await benchSession(key, s, cfg.quickSelect ? 2 : 5);
      report.push({ id: cand.id, valid: true, medianMs: ms });
      built.push(s);
      // Early exit: an accelerator that already beats a typical CPU budget
      // is good enough; skip building the remaining (slower to init) ones.
      if (cfg.quickSelect && cand.id !== 'wasm') break;
    } catch (e) {
      report.push({ id: cand.id, valid: false, error: String(e.message || e).slice(0, 160) });
    }
  }
  const pick = chooseBackend(report, cfg.npuSlack || 1.3);
  if (pick) chosen = built.find((b) => b.backend === pick.id);
  for (const b of built) if (b !== chosen) await b.session.release?.();
  backendReport[key] = report;
  if (!chosen) throw new Error(`${key}: nenhum backend válido`);
  return chosen;
}

// --------------------------------------------------------------------- init
async function init(cfg) {
  env = await probeEnvironment();
  env.soc = socHint(env, cfg.uaModel || '');
  status('carregando runtime…', { env });
  const modelBase = new URL(cfg.modelBase || '../../models/', import.meta.url);
  const manifest = await (await fetch(new URL('manifest.json', modelBase))).json();
  const bytes = {};
  for (const key of ['detector', 'pose']) {
    const entry = manifest.models[key];
    bytes[key] = await loadModelBytes(entry, modelBase, (l, t, cached) => status(
      cached ? `${entry.name}: cache local` : `baixando ${entry.name}: ${(l / 1e6).toFixed(1)}/${(t / 1e6).toFixed(1)} MB`,
      { progress: { key, loaded: l, total: t } }));
  }
  try { calib = await loadCalibration(modelBase); } catch { calib = null; }

  let lastErr = null;
  for (const v of await loadOrt(cfg)) {
    try {
      ort = await initOrtVariant(v);
      ortVariant = `${v.name} (${v.wasm.startsWith('http') && !v.wasm.startsWith(self.location.origin) ? 'CDN' : 'local'})`;
      // Fail fast if the wasm binary itself cannot be fetched/compiled.
      sessions.detector = await selectBackend('detector', bytes.detector, cfg);
      break;
    } catch (e) {
      lastErr = e;
      ort = null;
    }
  }
  if (!ort) throw lastErr || new Error('ONNX Runtime indisponível');
  sessions.pose = await selectBackend('pose', bytes.pose, cfg);
  tracker = new HandTracker(cfg.tracker || {});
  ensureCanvases();
  post({
    type: 'ready',
    env,
    ortVariant,
    ortVersion: ort.env.versions?.web || '',
    backends: { detector: sessions.detector.backend, pose: sessions.pose.backend },
    labels: { detector: sessions.detector.label, pose: sessions.pose.label },
    report: backendReport,
    calibrated: !!calib,
  });
}

// ------------------------------------------------------------ preprocessing
function mirrorCompose(m, W, mirror) {
  if (!mirror) return [m.a, m.b, m.c, m.d, m.e, m.f];
  return [-m.a, -m.b, m.c, m.d, m.a * W + m.e, m.b * W + m.f];
}

function prepDetector(bitmap, mirror) {
  const W = bitmap.width, H = bitmap.height;
  const lb = letterbox(W, H);
  detCtx.setTransform(1, 0, 0, 1, 0, 0);
  detCtx.fillStyle = 'rgb(114,114,114)';
  detCtx.fillRect(0, 0, DET_SIZE, DET_SIZE);
  detCtx.setTransform(...mirrorCompose({ a: lb.scale, b: 0, c: 0, d: lb.scale, e: 0, f: 0 }, W, mirror));
  detCtx.drawImage(bitmap, 0, 0);
  const px = detCtx.getImageData(0, 0, DET_SIZE, DET_SIZE).data;
  rgbaToTensor(px, DET_SIZE, DET_SIZE, DET_MEAN_BGR, DET_STD_BGR, 'bgr', detBuf, 0);
  return lb;
}

function prepPose(bitmap, roi, slot, mirror) {
  const A = cropAffine(roi.cx, roi.cy, roi.size, roi.theta);
  poseCtx.setTransform(1, 0, 0, 1, 0, 0);
  poseCtx.fillStyle = '#000';
  poseCtx.fillRect(0, 0, POSE_SIZE, POSE_SIZE);
  poseCtx.setTransform(...mirrorCompose(A.fwd, bitmap.width, mirror));
  poseCtx.drawImage(bitmap, 0, 0);
  const px = poseCtx.getImageData(0, 0, POSE_SIZE, POSE_SIZE).data;
  rgbaToTensor(px, POSE_SIZE, POSE_SIZE, POSE_MEAN_RGB, POSE_STD_RGB, 'rgb', poseBuf, slot * POSE_PLANE);
  return A;
}

function disposeOutputs(out) {
  for (const k in out) out[k].dispose?.();
}

async function runPose(rois, crops) {
  const s = sessions.pose;
  const results = [];
  const step = s.fixedBatch || MAX_BATCH;
  for (let start = 0; start < rois.length; start += step) {
    const n = Math.min(step, rois.length - start);
    const view = poseBuf.subarray(start * POSE_PLANE, (start + n) * POSE_PLANE);
    const out = await s.session.run({ input: new ort.Tensor('float32', view, [n, 3, POSE_SIZE, POSE_SIZE]) });
    const sx = out.simcc_x.data, sy = out.simcc_y.data;
    for (let b = 0; b < n; b++) {
      decodeSimcc(sx, sy, b, xyBuf, scBuf);
      const A = crops[start + b];
      const kp = new Float32Array(NUM_KP * 2);
      for (let k = 0; k < NUM_KP; k++) {
        const [x, y] = applyAffine(A.inv, xyBuf[2 * k], xyBuf[2 * k + 1]);
        kp[2 * k] = x; kp[2 * k + 1] = y;
      }
      results.push({ trackId: rois[start + b].trackId, kp, scores: Float32Array.from(scBuf) });
    }
    disposeOutputs(out);
  }
  return results;
}

// -------------------------------------------------------------------- frame
async function processFrame(msg) {
  const { bitmap, t, mirror, hfovDeg, frameId } = msg;
  const W = bitmap.width, H = bitmap.height;
  const ts = t / 1000;
  const T = { pre: 0, det: 0, pose: 0, post: 0 };
  const t0 = performance.now();
  frameCount++;
  const plan = tracker.plan(ts);
  const cam = { f: (Math.max(W, H) / 2) / Math.tan((hfovDeg * Math.PI) / 360), cx: W / 2, cy: H / 2 };
  if (plan.skip) {
    bitmap.close();
    post({ type: 'result', frameId, t, W, H, hands: [], dets: [], timings: { ...T, total: performance.now() - t0 }, plan: { skip: true, idle: true }, stats: tracker.stats });
    return;
  }
  const rois = plan.rois.map((r) => ({ ...r }));
  let dets = [];
  if (plan.runDetector) {
    let a = performance.now();
    const lb = prepDetector(bitmap, mirror);
    T.pre += performance.now() - a;
    a = performance.now();
    const out = await sessions.detector.session.run({ input: new ort.Tensor('float32', detBuf, [1, 3, DET_SIZE, DET_SIZE]) });
    T.det = performance.now() - a;
    dets = parseDetections(out.dets.data, out.dets.dims[1], lb.scale, W, H, 0.2);
    disposeOutputs(out);
    tracker.lastDetT = ts;
    tracker.stats.detRuns++;
    rois.push(...tracker.roisFromDetections(dets));
  }
  let results = [];
  if (rois.length) {
    let a = performance.now();
    const crops = rois.slice(0, MAX_BATCH).map((r, i) => prepPose(bitmap, r.roi, i, mirror));
    T.pre += performance.now() - a;
    a = performance.now();
    results = await runPose(rois.slice(0, MAX_BATCH), crops);
    T.pose = performance.now() - a;
    tracker.stats.poseRuns++;
  }
  bitmap.close();
  const a = performance.now();
  const hands = tracker.update(ts, results, cam, W, H);
  T.post = performance.now() - a;
  T.total = performance.now() - t0;
  post({
    type: 'result', frameId, t, W, H, hands, cam,
    dets: dets.map((d) => [d.x1, d.y1, d.x2, d.y2, d.score]),
    rois: rois.map((r) => ({ ...r.roi, trackId: r.trackId, fromDet: !!r.fromDet })),
    timings: T,
    plan: { runDetector: plan.runDetector, nRois: rois.length, idle: !!plan.idle },
    stats: tracker.stats,
  });
}

// ---------------------------------------------------------------- benchmark
async function benchmark(runs) {
  const out = {};
  for (const key of ['detector', 'pose']) out[key] = { backend: sessions[key].backend, medianMs: await benchSession(key, sessions[key], runs) };
  post({ type: 'bench', result: out });
}

function mean(a) { let s = 0; for (const v of a) s += v; return s / a.length; }

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') await init(m.config);
    else if (m.type === 'frame') {
      if (!tracker) { m.bitmap.close(); return; }
      await processFrame(m);
    } else if (m.type === 'config') tracker?.setOptions(m.tracker);
    else if (m.type === 'reset') tracker?.reset();
    else if (m.type === 'bench') await benchmark(m.runs || 20);
  } catch (err) {
    if (m.type === 'frame') { try { m.bitmap.close(); } catch { /* closed */ } }
    post({ type: 'error', phase: m.type, message: String(err?.message || err), stack: String(err?.stack || '') });
  }
};
