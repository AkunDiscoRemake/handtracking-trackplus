// Main thread: media sources, frame pacing towards the inference worker,
// motion prediction to the displayed frame, rendering and the HUD.

import { HandScene } from './render/scene.js';
import { Overlay2D, gestureLabel } from './render/overlay2d.js';
import { Metrics } from './app/metrics.js';
import { SyntheticSource } from './app/synthetic.js';
import { predictBounded } from './core/filters.js';
import { palmSizePx } from './core/lift.js';
import { NUM_KP } from './core/handModel.js';
import { detectEnvironment, cameraPermission, openCamera, helpFor, chromeIntentUrl } from './app/cameraAccess.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// Optional build-time overrides (the hosted preview sets these inline because
// its frame passes no query string and cannot fetch the 28 MB JSEP binary).
const BOOT = window.AURA_CONFIG || {};

// ------------------------------------------------------------------ settings
const DEFAULTS = {
  mode: 'hand_aura', facing: 'palm', hfov: 67, procRes: 640, maxInfFps: 30,
  minCutoff: 1.2, beta: 0.35, predict: true, maxPredMs: 80,
  auraColor: '#000000', auraOuter: 0.95, auraInner: 0.9, auraShells: 1.0, auraFlow: 0.035,
  labels: true, backend: 'auto', adaptive: true,
};
const STORE_KEY = 'aura-hands-settings-v1';
const settings = { ...DEFAULTS, ...loadStored() };
if (params.get('backend')) settings.backend = params.get('backend');
if (params.get('mode')) settings.mode = params.get('mode');

function loadStored() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY) || '{}'); } catch { return {}; }
}
function saveSettings() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(settings)); } catch { /* private mode */ }
}

// --------------------------------------------------------------------- state
const video = $('video'), still = $('still'), stage = $('stage');
const scene = new HandScene($('gl'));
const overlay = new Overlay2D($('overlay'));
const metrics = new Metrics();
const synth = new SyntheticSource();
const S = {
  source: null,          // 'camera-env' | 'camera-user' | 'video' | 'image' | 'sample' | 'synthetic'
  stream: null,
  mirror: false,
  fit: 'cover',
  worker: null,
  ready: false,
  busy: false,
  lastSend: 0,
  frameId: 0,
  result: null,
  presented: null,       // latest presented video frame { t, now }
  timeBasis: '',
  info: null,
  lowPerfSince: 0,
  highPerfSince: 0,
  hudAt: 0,
  battery: null,
};

// ------------------------------------------------------------------- worker
function startWorker() {
  S.ready = false;
  S.busy = false;
  S.worker?.terminate();
  const w = new Worker(new URL('./worker/inference.worker.js', import.meta.url), { type: 'module' });
  S.worker = w;
  w.onmessage = (e) => onWorkerMessage(e.data);
  w.onerror = (e) => setLoad(`Erro no worker: ${e.message || e}`, null, true);
  const uaModel = navigator.userAgentData?.platform || '';
  w.postMessage({
    type: 'init',
    config: {
      forceBackend: settings.backend,
      wasmBase: params.get('wasmBase') || undefined,
      wasmFallbackBase: BOOT.ortVariant === 'wasm' ? undefined : 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
      ortVariant: params.get('ort') || BOOT.ortVariant || 'all',
      uaModel,
      tracker: trackerOptions(),
    },
  });
  setLoad('Carregando modelos…', 0);
}

function trackerOptions() {
  return { facing: settings.facing, oneEuro: { minCutoff: +settings.minCutoff, beta: +settings.beta } };
}

function onWorkerMessage(m) {
  if (m.type === 'status') {
    const p = m.progress ? m.progress.loaded / m.progress.total : null;
    setLoad(m.text, p);
    if (m.env) S.env = m.env;
  } else if (m.type === 'ready') {
    S.ready = true;
    S.info = m;
    setLoad(`Pronto · detector: ${m.labels.detector} · pose: ${m.labels.pose}`, 1);
  } else if (m.type === 'result') {
    S.busy = false;
    const now = performance.now();
    S.result = { ...m, recvT: now };
    if (!m.plan?.skip) {
      metrics.inference.tick(now);
      metrics.infMs.push(m.timings.total);
      if (m.timings.det) metrics.detMs.push(m.timings.det);
      if (m.timings.pose) metrics.poseMs.push(m.timings.pose);
      metrics.preMs.push(m.timings.pre);
      if (S.timeBasis === 'capture') metrics.captureToResult.push(now - m.t);
      for (const h of m.hands) if (h.state === 'tracking') metrics.noisePx.push(h.noisePx);
    }
  } else if (m.type === 'bench') {
    $('benchOut').textContent = Object.entries(m.result)
      .map(([k, v]) => `${k}: ${v.backend} · mediana ${v.medianMs.toFixed(1)} ms`).join('\n');
  } else if (m.type === 'error') {
    S.busy = false;
    console.error(m.message, m.stack);
    if (m.phase === 'init') setLoad(`Falha ao iniciar a inferência: ${m.message}`, null, true);
    else $('startMsg').textContent = m.message;
  }
}

function setLoad(text, frac, isError = false) {
  S.loadText = text;
  $('loadText').textContent = text;
  if (frac != null) $('loadBar').style.width = `${Math.round(frac * 100)}%`;
  $('loadState').classList.toggle('error', isError);
  if (isError) $('startMsg').textContent = text;
}

// ------------------------------------------------------------------ sources
async function useSource(src) {
  stopMedia();
  $('startMsg').textContent = '';
  S.result = null;
  S.worker?.postMessage({ type: 'reset' });
  metrics.camera.reset();
  metrics.inference.reset();
  stage.classList.toggle('light', src === 'synthetic');
  $('demoBadge').hidden = src !== 'synthetic';
  if (src === 'synthetic') { video.hidden = true; still.hidden = true; }
  try {
    if (src === 'camera-env' || src === 'camera-user') await startCamera(src === 'camera-env' ? 'environment' : 'user');
    else if (src === 'sample') await showImage(SAMPLES[(S.sampleIdx = ((S.sampleIdx ?? -1) + 1) % SAMPLES.length)]);
    else if (src === 'file') { $('fileInput').click(); return; }
    else if (src === 'synthetic') { S.source = 'synthetic'; S.fit = 'cover'; }
    hideCamHelp();
    $('start').classList.add('hidden');
  } catch (e) {
    $('start').classList.remove('hidden');
    if (e.kind) showCamHelp(e.kind, src);
    else $('startMsg').textContent = `Não foi possível abrir a fonte: ${e.message || e}`;
  }
}

const SAMPLES = ['assets/samples/vsign_pd.png', 'assets/samples/move_pd.png'];

async function startCamera(facingMode) {
  const btn = $('btnCamEnv');
  btn.disabled = true;
  btn.textContent = 'Aguardando permissão…';
  let stream;
  try {
    stream = await openCamera(facingMode);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Permitir câmera e começar';
  }
  S.stream = stream;
  // The real lens may differ from the request (e.g. a laptop with one camera).
  const actual = stream.getVideoTracks()[0]?.getSettings?.().facingMode;
  S.source = facingMode === 'environment' ? 'camera-env' : 'camera-user';
  S.mirror = (actual || facingMode) === 'user';
  S.fit = 'cover';
  video.style.objectFit = 'cover';
  stage.classList.toggle('mirror', S.mirror);
  video.srcObject = stream;
  still.hidden = true;
  video.hidden = false;
  await video.play();
  startVideoFrameLoop();
  refreshCamStatus();
}

// ------------------------------------------------------------ camera access UI
const CAM_ENV = detectEnvironment();
let camRetrySrc = 'camera-env';

function setCamStatus(state, text) {
  $('camStatus').dataset.state = state;
  $('camStatusText').textContent = text;
}

async function refreshCamStatus() {
  if (!CAM_ENV.secure) return setCamStatus('insecure', 'Página sem HTTPS: a câmera não pode ser usada aqui.');
  if (CAM_ENV.inApp) return setCamStatus('inapp', `Aberto dentro de ${CAM_ENV.inAppName}: abra no ${CAM_ENV.ios ? 'Safari' : 'Chrome'} para usar a câmera.`);
  if (!CAM_ENV.hasApi) return setCamStatus('denied', 'Este navegador não oferece acesso à câmera.');
  const st = await cameraPermission();
  const apply = (state) => {
    if (state === 'granted') setCamStatus('granted', 'Câmera liberada.');
    else if (state === 'denied') setCamStatus('denied', 'Câmera bloqueada neste site. Veja como liberar abaixo.');
    else setCamStatus('prompt', 'O navegador vai pedir permissão quando você tocar no botão.');
  };
  if (!st) return setCamStatus('prompt', 'O navegador vai pedir permissão quando você tocar no botão.');
  apply(st.state);
  if (st.state === 'denied') showCamHelp('denied', camRetrySrc);
  // Unblocked from the site settings while the page is open → start right away.
  st.onchange = () => {
    apply(st.state);
    if (st.state === 'granted' && !$('camHelp').hidden) useSource(camRetrySrc);
  };
}

function showCamHelp(kind, src) {
  camRetrySrc = src && src.startsWith('camera') ? src : 'camera-env';
  const h = helpFor(kind, CAM_ENV);
  $('camHelpTitle').textContent = h.title;
  $('camHelpSteps').replaceChildren(...h.steps.map((t) => { const li = document.createElement('li'); li.textContent = t; return li; }));
  $('actRetry').hidden = !h.actions.includes('retry');
  $('actReload').hidden = !h.actions.includes('reload');
  $('actCopy').hidden = !h.actions.includes('copy') && kind !== 'inapp';
  $('actChrome').hidden = !h.actions.includes('openChrome');
  $('actChrome').href = chromeIntentUrl();
  $('camLink').value = location.href.split('#')[0];
  $('camHelp').hidden = false;
  $('camHelp').scrollIntoView({ block: 'nearest' });
}

function hideCamHelp() { $('camHelp').hidden = true; }

async function showImage(url) {
  S.source = 'image';
  S.mirror = false;
  S.fit = 'contain';
  stage.classList.remove('mirror');
  video.hidden = true;
  still.hidden = false;
  still.src = url;
  await still.decode();
  S.timeBasis = 'static';
}

function showFile(file) {
  stopMedia();
  const url = URL.createObjectURL(file);
  if (file.type.startsWith('image/')) {
    showImage(url).then(() => $('start').classList.add('hidden'));
    return;
  }
  S.source = 'video';
  S.mirror = false;
  S.fit = 'contain';
  video.style.objectFit = 'contain';
  stage.classList.remove('mirror', 'light');
  still.hidden = true;
  video.hidden = false;
  video.srcObject = null;
  video.src = url;
  video.loop = true;
  video.play().then(() => { $('start').classList.add('hidden'); startVideoFrameLoop(); })
    .catch((e) => { $('startMsg').textContent = `Vídeo não reproduzível: ${e.message}`; });
}

function stopMedia() {
  if (S.stream) { for (const t of S.stream.getTracks()) t.stop(); S.stream = null; }
  if (video.src) { URL.revokeObjectURL(video.src); video.removeAttribute('src'); }
  video.srcObject = null;
  S.presented = null;
  S.source = null;
}

// Frame pacing: every presented video frame is timestamped; a frame is sent
// to the worker only when it is idle (one in flight → no queueing latency)
// and the inference FPS cap allows it. Others are counted as dropped.
function startVideoFrameLoop() {
  const token = (S.loopToken = {});
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    const cb = (now, meta) => {
      if (S.loopToken !== token) return;
      metrics.camera.tick(now);
      // captureTime exists for camera streams (same clock as performance.now).
      const t = meta.captureTime ?? meta.expectedDisplayTime ?? now;
      S.timeBasis = meta.captureTime != null ? 'capture' : 'presentation';
      S.presented = { t, now };
      trySend(video, t);
      video.requestVideoFrameCallback(cb);
    };
    video.requestVideoFrameCallback(cb);
  } else {
    // Fallback: poll currentTime changes from rAF.
    let lastTime = -1;
    const poll = () => {
      if (S.loopToken !== token) return;
      if (video.currentTime !== lastTime) {
        lastTime = video.currentTime;
        const now = performance.now();
        metrics.camera.tick(now);
        S.timeBasis = 'presentation';
        S.presented = { t: now, now };
        trySend(video, now);
      }
      requestAnimationFrame(poll);
    };
    requestAnimationFrame(poll);
  }
}

function trySend(media, t) {
  if (!S.ready || document.hidden) return;
  if (S.busy) { metrics.dropped++; return; }
  const now = performance.now();
  if (now - S.lastSend < 1000 / Number(settings.maxInfFps) - 2) return;
  const w = media.videoWidth || media.naturalWidth, h = media.videoHeight || media.naturalHeight;
  if (!w || !h) return;
  const k = Math.min(1, Number(settings.procRes) / Math.max(w, h));
  const rw = Math.round(w * k), rh = Math.round(h * k);
  S.busy = true;
  S.lastSend = now;
  const id = ++S.frameId;
  createImageBitmap(media, { resizeWidth: rw, resizeHeight: rh, resizeQuality: 'medium' }).then((bitmap) => {
    metrics.sent++;
    S.worker.postMessage({ type: 'frame', bitmap, t, frameId: id, mirror: S.mirror, hfovDeg: Number(settings.hfov) }, [bitmap]);
  }).catch(() => { S.busy = false; });
}

// ------------------------------------------------------------------- render
const predBuf = { left: [new Float32Array(NUM_KP * 2), new Float32Array(NUM_KP * 3)], right: [new Float32Array(NUM_KP * 2), new Float32Array(NUM_KP * 3)] };

function frame(now) {
  requestAnimationFrame(frame);
  metrics.render.tick(now);
  if (S.source === 'image') trySend(still, now);

  let res = null, targetT = now;
  if (S.source === 'synthetic') {
    const s = synth.frame(now, Number(settings.hfov));
    res = { ...s, t: now, recvT: now, dets: [], rois: [] };
  } else if (S.result) {
    res = S.result;
    // Target = timestamp of the video frame currently on screen; for a still
    // image there is nothing to predict.
    targetT = S.source === 'image' ? res.t : (S.presented ? S.presented.t : now);
  }

  const hands = [];
  if (res) {
    const horizonMs = Math.max(0, targetT - res.t);
    const maxPred = settings.predict ? Number(settings.maxPredMs) : 0;
    if (res.hands.length && S.source !== 'synthetic' && S.source !== 'image') {
      metrics.residual.push(Math.max(0, horizonMs - maxPred));
      metrics.captureToDisplay.push(now + (metrics.render.iv.mean() || 16.7) - res.t);
    }
    for (const h of res.hands) {
      const buf = predBuf[h.handedness] || predBuf.right;
      const size = palmSizePx(h.kp2d);
      const k2 = predictBounded(h.kp2d, h.vel2d, horizonMs / 1000, maxPred / 1000, 0.35 * size, buf[0], 2);
      const k3 = predictBounded(h.kp3d, h.vel3d, horizonMs / 1000, maxPred / 1000, 0.05, buf[1], 3);
      const opacity = h.state === 'tracking' ? smoothstep(0.26, 0.42, h.score) : Math.min(1, h.score / 0.4);
      hands.push({ ...h, kp2dPred: k2, kp3d: k3, opacity });
    }
    const f = res.cam?.f ?? (Math.max(res.W, res.H) / 2) / Math.tan((Number(settings.hfov) * Math.PI) / 360);
    scene.setView(res.W, res.H, f, S.fit);
  }
  scene.mode = settings.mode;
  scene.update(hands, now / 1000);
  scene.render();
  overlay.draw(scene.frameToScreen(), res, hands, { debug: settings.mode === 'debug', labels: settings.labels });
  adaptResolution(now);
  if (now - S.hudAt > 250) { S.hudAt = now; updateHud(hands); }
  metrics.record({
    t: now.toFixed(1), renderFps: metrics.render.fps().toFixed(1), infMs: res?.timings?.total?.toFixed(2) ?? '',
    c2r: res && S.timeBasis === 'capture' ? (res.recvT - res.t).toFixed(1) : '', hands: hands.length,
  });
}

function smoothstep(a, b, x) { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }

// Lower the WebGL pixel ratio when the display rate cannot be held, and
// restore it when there is headroom (saves GPU time and battery).
function adaptResolution(now) {
  if (!settings.adaptive) { scene.setPixelScale(1); return; }
  const fps = metrics.render.fps();
  if (!fps) return;
  if (fps < 45) {
    S.highPerfSince = 0;
    S.lowPerfSince ||= now;
    if (now - S.lowPerfSince > 2500 && scene.pixelScale > 0.5) { scene.setPixelScale(+(scene.pixelScale - 0.1).toFixed(2)); S.lowPerfSince = now; }
  } else if (fps > 57) {
    S.lowPerfSince = 0;
    S.highPerfSince ||= now;
    if (now - S.highPerfSince > 6000 && scene.pixelScale < 1) { scene.setPixelScale(+(scene.pixelScale + 0.1).toFixed(2)); S.highPerfSince = now; }
  }
}

// ---------------------------------------------------------------------- HUD
function updateHud(hands) {
  const m = metrics.snapshot();
  const info = S.info;
  if (S.source === 'synthetic') $('hBackend').textContent = 'demo sintética (sem inferência)';
  else if (info) $('hBackend').textContent = `det ${info.labels.detector} · pose ${info.labels.pose}`;
  else $('hBackend').textContent = S.ready ? '–' : (S.loadText || 'carregando…');
  $('hFps').textContent = m.renderFps;
  $('hCamFps').textContent = S.source === 'synthetic' || S.source === 'image' ? '–' : m.cameraFps;
  $('hInfFps').textContent = S.source === 'synthetic' ? '–' : m.infFps;
  $('hInf').textContent = S.source === 'synthetic' ? '–' : m.infMs;
  $('hC2R').textContent = S.timeBasis === 'capture' ? m.c2r : '–';
  $('hJit').textContent = m.renderJitter;
  $('hNoise').textContent = S.source === 'synthetic' ? '–' : m.noisePx;
  const conf = $('hConf');
  if (!hands.length) conf.textContent = 'nenhuma mão';
  else {
    conf.replaceChildren(...hands.map((h) => {
      const el = document.createElement('span');
      el.className = 'hand';
      const label = document.createElement('span');
      label.textContent = `${h.handedness === 'left' ? 'E' : 'D'} ${(h.score * 100).toFixed(0)}%`;
      const meter = document.createElement('span');
      meter.className = 'meter';
      const bar = document.createElement('i');
      bar.style.width = `${Math.round(Math.min(1, h.score / 0.7) * 100)}%`;
      bar.style.background = h.state === 'coasting' ? 'var(--warn)' : h.score > 0.42 ? 'var(--good)' : 'var(--warn)';
      meter.append(bar);
      const g = document.createElement('span');
      g.textContent = gestureLabel(h.gesture);
      el.append(label, meter, g);
      return el;
    }));
  }
  const dbg = $('hudDebug');
  dbg.hidden = settings.mode !== 'debug';
  if (!dbg.hidden) {
    const env = info?.env || S.env || {};
    const st = S.result?.stats;
    const rep = info ? Object.entries(info.report).map(([k, rs]) => `${k}: ` + rs.map((r) => `${r.id}${r.valid ? ` ${r.medianMs.toFixed(1)}ms` : ' ✗'}`).join(' | ')).join('\n') : '';
    dbg.textContent = [
      `ORT ${info?.ortVersion || ''} ${info?.ortVariant || ''} · threads ${env.threads ?? '–'} · COI ${env.crossOriginIsolated ? 'sim' : 'não'}`,
      `GPU ${env.gpu ? `${env.gpu.vendor} ${env.gpu.architecture}`.trim() || 'WebGPU' : 'sem WebGPU'} · WebNN ${env.webnn ? Object.entries(env.webnnContexts || {}).filter(([, v]) => v).map(([k]) => k).join('/') || 'sim' : 'não'} · SoC ${env.soc || '–'}`,
      rep,
      `inferência média ${m.infMs} ms (p95 ${m.infP95}) · det ${m.detMs} · pose ${m.poseMs} · pré ${m.preMs}`,
      `captura→resultado ${S.timeBasis === 'capture' ? `${m.c2r} ms (p95 ${m.c2rP95})` : `n/d (${S.timeBasis || '–'})`} · captura→tela≈ ${S.timeBasis === 'capture' ? m.c2d : '–'} ms`,
      `atraso residual após predição ${m.residual} ms · jitter câmera ${m.cameraJitter} ms · jitter inferência ${m.infJitter} ms`,
      `quadros enviados ${m.sent} · descartados (worker ocupado) ${m.dropped} · escala render ${scene.pixelScale.toFixed(1)}`,
      st ? `detector ${st.detRuns}× · pose ${st.poseRuns}× · trilhas novas ${st.births} · recuperações após oclusão ${st.recoveries}` : '',
      S.battery ? `bateria ${(S.battery.level * 100).toFixed(0)}%${S.battery.charging ? ' (carregando)' : ''}` : '',
    ].filter(Boolean).join('\n');
  }
}

// ----------------------------------------------------------------------- UI
function applySettings() {
  for (const b of document.querySelectorAll('.modes button')) b.setAttribute('aria-selected', String(b.dataset.mode === settings.mode));
  scene.aura.setColor(settings.auraColor);
  Object.assign(scene.aura.params, { outer: +settings.auraOuter, inner: +settings.auraInner, shells: +settings.auraShells, flow: +settings.auraFlow });
  for (const el of document.querySelectorAll('[data-set]')) {
    const k = el.dataset.set;
    if (el.type === 'checkbox') el.checked = !!settings[k]; else el.value = String(settings[k]);
  }
  for (const o of document.querySelectorAll('[data-out]')) o.textContent = settings[o.dataset.out];
}

function bindUi() {
  for (const b of document.querySelectorAll('.modes button')) {
    b.addEventListener('click', () => { settings.mode = b.dataset.mode; saveSettings(); applySettings(); });
  }
  for (const el of document.querySelectorAll('[data-set]')) {
    el.addEventListener('input', () => {
      const k = el.dataset.set;
      const prev = settings[k];
      settings[k] = el.type === 'checkbox' ? el.checked : el.type === 'range' ? Number(el.value) : el.value;
      saveSettings();
      applySettings();
      if (['facing', 'minCutoff', 'beta'].includes(k)) S.worker?.postMessage({ type: 'config', tracker: trackerOptions() });
      if (k === 'backend' && prev !== settings.backend) startWorker();
    });
  }
  for (const b of document.querySelectorAll('#start [data-src]')) b.addEventListener('click', () => useSource(b.dataset.src));
  $('btnCamEnv').addEventListener('click', () => useSource('camera-env'));
  $('btnCamUser').addEventListener('click', () => useSource('camera-user'));
  $('actRetry').addEventListener('click', () => useSource(camRetrySrc));
  $('actReload').addEventListener('click', () => location.reload());
  $('actCopy').addEventListener('click', () => {
    const url = $('camLink').value;
    navigator.clipboard?.writeText(url).then(() => { $('actCopy').textContent = 'Link copiado'; })
      .catch(() => { $('camLink').select(); });
  });
  $('btnSettings').addEventListener('click', () => $('settings').showModal());
  $('btnSource').addEventListener('click', () => $('sourceDlg').showModal());
  $('sourceDlg').addEventListener('close', () => { const v = $('sourceDlg').returnValue; if (v && v !== 'close') useSource(v); });
  $('fileInput').addEventListener('change', (e) => { const f = e.target.files?.[0]; if (f) showFile(f); e.target.value = ''; });
  $('btnBench').addEventListener('click', () => {
    if (!S.ready) { $('benchOut').textContent = 'modelos ainda carregando…'; return; }
    $('benchOut').textContent = 'medindo…';
    S.worker.postMessage({ type: 'bench', runs: 20 });
  });
  $('btnLog').addEventListener('click', () => {
    metrics.logging = !metrics.logging;
    if (metrics.logging) metrics.log = [];
    $('btnLog').textContent = metrics.logging ? 'Parar gravação' : 'Gravar métricas';
  });
  $('btnCsv').addEventListener('click', () => {
    const csv = metrics.csv();
    if (!csv) { $('benchOut').textContent = 'nada gravado ainda'; return; }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `aura-hands-metrics-${Date.now()}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  });
  window.addEventListener('resize', () => { scene.resize(); overlay.resize(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) { metrics.render.reset(); metrics.camera.reset(); } });
}

async function main() {
  applySettings();
  bindUi();
  overlay.resize();
  navigator.getBattery?.().then((b) => { S.battery = b; }).catch(() => {});
  try {
    await scene.load(new URL('../assets/hands/', import.meta.url));
  } catch (e) {
    setLoad(`Falha ao carregar os modelos 3D: ${e.message}`, null, true);
  }
  startWorker();
  refreshCamStatus();
  requestAnimationFrame(frame);
  const src = params.get('src') || BOOT.startSource;
  if (src) useSource(src);
}

main();

// Exposed for automated end-to-end checks (tools/e2e.mjs).
window.__aura = { S, metrics, scene, settings, useSource };
