#!/usr/bin/env node
// Packs the source tree (no node_modules, vendor/, build/ or downloaded model
// binaries) into build/aura-hands-source.zip. Pure JS (deflate via zlib).
//   node tools/make-zip.mjs [out.zip]
import { readFile, writeFile, readdir, stat, mkdir } from 'node:fs/promises';
import { deflateRawSync, crc32 } from 'node:zlib';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = process.argv[2] || join(ROOT, 'build', 'aura-hands-source.zip');
const EXCLUDE = [/^node_modules\//, /^vendor\//, /^build\//, /^\.git\//, /^models\/.*\.(onnx|bin)$/, /\.y4m$/];

async function walk(dir) {
  const files = [];
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    const rel = relative(ROOT, p).split('\\').join('/');
    if (EXCLUDE.some((r) => r.test(rel + (e.isDirectory() ? '/' : '')))) continue;
    if (e.isDirectory()) files.push(...await walk(p)); else files.push(rel);
  }
  return files;
}

const files = (await walk(ROOT)).sort();
const local = [], central = [];
let offset = 0;
for (const f of files) {
  const data = await readFile(join(ROOT, f));
  const st = await stat(join(ROOT, f));
  const comp = deflateRawSync(data, { level: 9 });
  const name = Buffer.from(`aura-hands/${f}`, 'utf8');
  const crc = crc32(data) >>> 0;
  const d = st.mtime;
  const dosTime = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
  const dosDate = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
  const h = Buffer.alloc(30);
  h.writeUInt32LE(0x04034b50, 0); h.writeUInt16LE(20, 4); h.writeUInt16LE(0x0800, 6); h.writeUInt16LE(8, 8);
  h.writeUInt16LE(dosTime, 10); h.writeUInt16LE(dosDate, 12); h.writeUInt32LE(crc, 14);
  h.writeUInt32LE(comp.length, 18); h.writeUInt32LE(data.length, 22); h.writeUInt16LE(name.length, 26);
  local.push(h, name, comp);
  const c = Buffer.alloc(46);
  c.writeUInt32LE(0x02014b50, 0); c.writeUInt16LE(20, 4); c.writeUInt16LE(20, 6); c.writeUInt16LE(0x0800, 8); c.writeUInt16LE(8, 10);
  c.writeUInt16LE(dosTime, 12); c.writeUInt16LE(dosDate, 14); c.writeUInt32LE(crc, 16);
  c.writeUInt32LE(comp.length, 20); c.writeUInt32LE(data.length, 24); c.writeUInt16LE(name.length, 28);
  c.writeUInt32LE(offset, 42);
  central.push(c, name);
  offset += 30 + name.length + comp.length;
}
const cdir = Buffer.concat(central);
const end = Buffer.alloc(22);
end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(files.length, 8); end.writeUInt16LE(files.length, 10);
end.writeUInt32LE(cdir.length, 12); end.writeUInt32LE(offset, 16);
await mkdir(dirname(out), { recursive: true });
const zip = Buffer.concat([...local, cdir, end]);
await writeFile(out, zip);
console.log(`${out}: ${files.length} files, ${(zip.length / 1024).toFixed(0)} KiB`);
