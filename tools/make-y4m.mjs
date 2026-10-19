#!/usr/bin/env node
// Builds a Y4M clip (for Chromium's --use-file-for-fake-video-capture) from a
// real photo: the hand drifts and briefly leaves the frame (occlusion test).
//   node tools/make-y4m.mjs <in.png> <out.y4m> [frames=90] [w=640] [h=480]
import { readFile, writeFile } from 'node:fs/promises';
import { decodePng } from './lib/png.mjs';

const [, , inPath, outPath, nArg, wArg, hArg] = process.argv;
const N = Number(nArg || 90), W = Number(wArg || 640), H = Number(hArg || 480);
const img = decodePng(await readFile(inPath));
const header = Buffer.from(`YUV4MPEG2 W${W} H${H} F30:1 Ip A1:1 C420jpeg\n`, 'ascii');
const frames = [header];
const scale = Math.min(W / img.width, H / img.height) * 1.1;
for (let f = 0; f < N; f++) {
  const t = f / 30;
  // Smooth drift; frames 55–64 slide a grey panel over the hand (occlusion).
  const dx = Math.sin(t * 1.4) * 40, dy = Math.cos(t * 1.1) * 18;
  const occl = f >= 55 && f < 65;
  const yP = Buffer.alloc(W * H), uP = Buffer.alloc((W / 2) * (H / 2)), vP = Buffer.alloc((W / 2) * (H / 2));
  const rgbAt = (x, y) => {
    const sx = (x - W / 2 - dx) / scale + img.width / 2, sy = (y - H / 2 - dy) / scale + img.height / 2;
    const ix = Math.round(sx), iy = Math.round(sy);
    if (ix < 0 || iy < 0 || ix >= img.width || iy >= img.height) return [235, 235, 235];
    if (occl && x > W * 0.15 && x < W * 0.85) return [120, 120, 125];
    const p = (iy * img.width + ix) * 4;
    return [img.data[p], img.data[p + 1], img.data[p + 2]];
  };
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const [r, g, b] = rgbAt(x, y);
    yP[y * W + x] = Math.max(0, Math.min(255, Math.round(0.299 * r + 0.587 * g + 0.114 * b)));
    if ((x & 1) === 0 && (y & 1) === 0) {
      const i = (y / 2) * (W / 2) + x / 2;
      uP[i] = Math.max(0, Math.min(255, Math.round(-0.168736 * r - 0.331264 * g + 0.5 * b + 128)));
      vP[i] = Math.max(0, Math.min(255, Math.round(0.5 * r - 0.418688 * g - 0.081312 * b + 128)));
    }
  }
  frames.push(Buffer.from('FRAME\n', 'ascii'), yP, uP, vP);
}
await writeFile(outPath, Buffer.concat(frames));
console.log(`${outPath}: ${N} frames ${W}x${H}`);
