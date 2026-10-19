#!/usr/bin/env node
// Screenshots a page of this project in headless Chromium.
//   node tools/shoot.mjs <path?query> <out.png> [width] [height] [waitMs]
import { spawn } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const [, , path, out, w = '900', h = '600', wait = '800'] = process.argv;
const { chromium } = await import('playwright');
const PORT = 9000 + Math.floor(Math.random() * 900);
const server = spawn('node', [join(ROOT, 'tools', 'serve.mjs'), String(PORT), '--no-isolation'], { stdio: 'pipe' });
await new Promise((r) => server.stdout.once('data', r));
const browser = await chromium.launch({ args: ['--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: +w, height: +h } });
page.on('pageerror', (e) => console.log('pageerror', e.message));
await page.goto(`http://localhost:${PORT}/${path}`);
await page.waitForFunction(() => window.__ready || window.__aura, null, { timeout: 60000 });
await page.waitForTimeout(+wait);
await page.screenshot({ path: out });
await browser.close();
server.kill();
console.log(out);
