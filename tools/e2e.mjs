#!/usr/bin/env node
// End-to-end check in headless Chromium (Playwright):
//  1. sample photo → real models detect and track a hand
//  2. fake camera fed with a real-hand Y4M clip (with an occlusion) → camera
//     path, capture timestamps, tracking + recovery, prediction
//  3. synthetic demo in the three render modes
// Writes screenshots and a JSON report to build/e2e/.
//   node tools/e2e.mjs [--isolation] [--webgpu]
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(ROOT, 'build', 'e2e');
await mkdir(OUT, { recursive: true });
const isolation = process.argv.includes('--isolation');
const webgpu = process.argv.includes('--webgpu');
const PORT = 8000 + Math.floor(Math.random() * 900);

let chromium;
try { ({ chromium } = await import('playwright')); } catch {
  console.error('playwright not installed: npm i -D playwright (and `npx playwright install chromium` if no browser is present)');
  process.exit(2);
}

const y4m = join(OUT, 'hand.y4m');
if (!existsSync(y4m)) {
  await run('node', [join(ROOT, 'tools', 'make-y4m.mjs'), join(ROOT, 'assets', 'samples', 'move_pd.png'), y4m, '120']);
}
const server = spawn('node', [join(ROOT, 'tools', 'serve.mjs'), String(PORT), ...(isolation ? [] : ['--no-isolation'])], { stdio: 'pipe' });
await new Promise((r) => server.stdout.once('data', r));

const args = [
  '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${y4m}`,
  '--autoplay-policy=no-user-gesture-required', '--ignore-gpu-blocklist', '--enable-gpu',
];
if (webgpu) args.push('--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader');

const browser = await chromium.launch({ args });
const page = await browser.newPage({ viewport: { width: 900, height: 640 }, deviceScaleFactor: 1 });
const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
const report = { isolation, webgpu, steps: {} };
const base = `http://localhost:${PORT}/`;

try {
  await page.goto(`${base}?mode=debug`);
  await page.waitForFunction(() => window.__aura?.S.ready || /Falha/.test(document.getElementById('loadText').textContent), null, { timeout: 240000 });
  const info = await page.evaluate(() => ({ ready: window.__aura.S.ready, text: document.getElementById('loadText').textContent, info: window.__aura.S.info && { labels: window.__aura.S.info.labels, report: window.__aura.S.info.report, env: window.__aura.S.info.env, ortVariant: window.__aura.S.info.ortVariant } }));
  report.steps.init = info;
  console.log('init:', JSON.stringify(info.info?.labels), info.text);
  if (!info.ready) throw new Error('worker failed: ' + info.text);

  // 1. sample photo
  await page.evaluate(() => window.__aura.useSource('sample'));
  await page.waitForFunction(() => window.__aura.S.result?.hands?.length > 0, null, { timeout: 60000 });
  await page.waitForTimeout(2500);
  report.steps.sample = await page.evaluate(() => {
    const r = window.__aura.S.result;
    return { hands: r.hands.map((h) => ({ hand: h.handedness, score: +h.score.toFixed(3), gesture: h.gesture, state: h.state, depth: +h.depth.toFixed(3) })), timings: r.timings, metrics: window.__aura.metrics.snapshot() };
  });
  console.log('sample:', JSON.stringify(report.steps.sample.hands), 'inf ms', report.steps.sample.metrics.infMs);
  await page.screenshot({ path: join(OUT, 'sample_debug.png') });
  await page.evaluate(() => { window.__aura.settings.mode = 'hand_aura'; });
  await page.waitForTimeout(400);
  await page.screenshot({ path: join(OUT, 'sample_hand_aura.png') });

  // 2. fake camera with a real hand clip
  await page.evaluate(() => { window.__aura.settings.mode = 'debug'; return window.__aura.useSource('camera-env'); });
  await page.waitForFunction(() => window.__aura.S.result?.hands?.length > 0, null, { timeout: 60000 });
  const seen = { tracking: 0, coasting: 0, none: 0, ids: new Set() };
  for (let i = 0; i < 60; i++) {
    const s = await page.evaluate(() => (window.__aura.S.result?.hands || []).map((h) => [h.id, h.state]));
    if (!s.length) seen.none++;
    for (const [id, st] of s) { seen[st]++; seen.ids.add(id); }
    await page.waitForTimeout(100);
  }
  report.steps.camera = await page.evaluate(() => ({ metrics: window.__aura.metrics.snapshot(), basis: window.__aura.S.timeBasis, stats: window.__aura.S.result?.stats }));
  report.steps.camera.samples = { ...seen, ids: [...seen.ids] };
  console.log('camera:', JSON.stringify(report.steps.camera.samples), JSON.stringify(report.steps.camera.stats), 'basis', report.steps.camera.basis);
  console.log('camera metrics:', JSON.stringify(report.steps.camera.metrics));
  await page.screenshot({ path: join(OUT, 'camera_debug.png') });
  await page.evaluate(() => { window.__aura.settings.mode = 'aura'; });
  await page.waitForTimeout(400);
  await page.screenshot({ path: join(OUT, 'camera_aura.png') });

  // 3. synthetic demo, all modes
  await page.evaluate(() => window.__aura.useSource('synthetic'));
  for (const mode of ['aura', 'hand_aura', 'debug']) {
    await page.evaluate((m) => { window.__aura.settings.mode = m; }, mode);
    await page.waitForTimeout(1200);
    await page.screenshot({ path: join(OUT, `synthetic_${mode}.png`) });
  }
  report.steps.synthetic = await page.evaluate(() => window.__aura.metrics.snapshot());
} catch (e) {
  report.error = String(e.message || e);
  console.error('E2E error:', report.error);
  await page.screenshot({ path: join(OUT, 'error.png') }).catch(() => {});
} finally {
  report.logs = logs.slice(-80);
  await writeFile(join(OUT, 'report.json'), JSON.stringify(report, null, 2));
  await browser.close();
  server.kill();
}
const errors = logs.filter((l) => l.startsWith('[pageerror]') || l.startsWith('[error]'));
if (errors.length) console.log('console errors:\n' + errors.join('\n'));
process.exit(report.error ? 1 : 0);

function run(cmd, a) {
  return new Promise((res, rej) => { const p = spawn(cmd, a, { stdio: 'inherit' }); p.on('exit', (c) => (c ? rej(new Error(`${cmd} ${c}`)) : res())); });
}
