#!/usr/bin/env node
// Downloads the official OpenMMLab ONNX exports (Apache-2.0), extracts the
// graph from the zip, verifies SHA-256, and writes models/ + a manifest.
// Large models are split into <14 MB parts so they can be hosted on static
// hosts with per-file limits; the app reassembles and re-verifies them.
//
//   node tools/prepare-models.mjs [--from-dir <dir with the .zip files>]

import { createHash } from 'node:crypto';
import { inflateRawSync } from 'node:zlib';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(ROOT, 'models');
const PART_SIZE = 14 * 1024 * 1024;
const BASE = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/';

export const MODELS = [
  {
    key: 'detector',
    name: 'RTMDet-nano (hand)',
    zip: 'rtmdet_nano_8xb32-300e_hand-267f9c8f.zip',
    zipSha256: '9c0370a43c02b2fe42b4382aba7383d97cfa3ed35623b655cac4f0c25cfde402',
    member: /end2end\.onnx$/,
    file: 'rtmdet_nano_hand.onnx',
    sha256: '568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2',
    input: [1, 3, 320, 320],
  },
  {
    key: 'pose',
    name: 'RTMPose-m SimCC (hand, 21 kpt)',
    zip: 'rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip',
    zipSha256: '45c9e1aa21b8fe33859973c49bce4647645d312e7a226e398a42f9137d44a56d',
    member: /end2end\.onnx$/,
    file: 'rtmpose_m_hand.onnx',
    sha256: '39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29',
    input: ['batch', 3, 256, 256],
  },
];

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

// Minimal ZIP reader: central directory → local header → stored/deflate data.
export function unzipMember(zip, test) {
  let eocd = -1;
  for (let i = zip.length - 22; i >= Math.max(0, zip.length - 65557); i--) {
    if (zip.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error('zip: no end of central directory');
  const count = zip.readUInt16LE(eocd + 10);
  let p = zip.readUInt32LE(eocd + 16);
  for (let k = 0; k < count; k++) {
    if (zip.readUInt32LE(p) !== 0x02014b50) throw new Error('zip: bad central header');
    const method = zip.readUInt16LE(p + 10);
    const csize = zip.readUInt32LE(p + 20);
    const nameLen = zip.readUInt16LE(p + 28), extraLen = zip.readUInt16LE(p + 30), cmtLen = zip.readUInt16LE(p + 32);
    const local = zip.readUInt32LE(p + 42);
    const name = zip.toString('utf8', p + 46, p + 46 + nameLen);
    p += 46 + nameLen + extraLen + cmtLen;
    if (!test.test(name)) continue;
    const ln = zip.readUInt16LE(local + 26), le = zip.readUInt16LE(local + 28);
    const data = zip.subarray(local + 30 + ln + le, local + 30 + ln + le + csize);
    if (method === 0) return { name, data: Buffer.from(data) };
    if (method === 8) return { name, data: inflateRawSync(data) };
    throw new Error(`zip: unsupported method ${method}`);
  }
  throw new Error('zip: member not found');
}

async function fetchZip(m, fromDir) {
  if (fromDir && existsSync(join(fromDir, m.zip))) return readFile(join(fromDir, m.zip));
  const url = BASE + m.zip;
  process.stdout.write(`downloading ${url}\n`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

async function main() {
  const fi = process.argv.indexOf('--from-dir');
  const fromDir = fi > 0 ? process.argv[fi + 1] : null;
  await mkdir(OUT, { recursive: true });
  const manifest = { version: 1, source: BASE, license: 'Apache-2.0 (OpenMMLab mmpose)', models: {} };
  for (const m of MODELS) {
    const zip = await fetchZip(m, fromDir);
    if (sha256(zip) !== m.zipSha256) throw new Error(`${m.zip}: SHA-256 mismatch`);
    const { data } = unzipMember(zip, m.member);
    const h = sha256(data);
    if (h !== m.sha256) throw new Error(`${m.file}: SHA-256 mismatch (${h})`);
    await writeFile(join(OUT, m.file), data);
    const parts = [];
    if (data.length > PART_SIZE) {
      for (let o = 0, i = 0; o < data.length; o += PART_SIZE, i++) {
        const name = `${m.file}.part${i}.bin`;
        await writeFile(join(OUT, name), data.subarray(o, o + PART_SIZE));
        parts.push(name);
      }
    } else {
      // Single-part copy with a neutral extension for strict static hosts.
      const name = `${m.file}.part0.bin`;
      await writeFile(join(OUT, name), data);
      parts.push(name);
    }
    manifest.models[m.key] = { name: m.name, file: m.file, parts, size: data.length, sha256: h, input: m.input };
    process.stdout.write(`ok  ${m.file}  ${(data.length / 1e6).toFixed(1)} MB  ${parts.length} part(s)\n`);
  }
  await writeFile(join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  process.stdout.write('models/manifest.json written\n');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => { console.error(e.message); process.exit(1); });
}
