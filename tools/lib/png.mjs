// Minimal PNG decoder (8-bit grey / RGB / RGBA, non-interlaced) for tests and
// tools, so the Node side needs no native image dependency.
import { inflateSync } from 'node:zlib';

export function decodePng(buf) {
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < 8; i++) if (buf[i] !== sig[i]) throw new Error('not a PNG');
  let off = 8, w = 0, h = 0, depth = 0, ctype = 0, interlace = 0;
  const idat = [];
  while (off < buf.length) {
    const n = buf.readUInt32BE(off);
    const type = buf.toString('ascii', off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + n);
    if (type === 'IHDR') {
      w = data.readUInt32BE(0); h = data.readUInt32BE(4);
      depth = data[8]; ctype = data[9]; interlace = data[12];
    } else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    off += 12 + n;
  }
  if (depth !== 8 || interlace) throw new Error(`unsupported PNG (depth ${depth}, interlace ${interlace})`);
  const bpp = { 0: 1, 2: 3, 6: 4, 4: 2 }[ctype];
  if (!bpp) throw new Error(`unsupported colour type ${ctype}`);
  const raw = inflateSync(Buffer.concat(idat));
  const stride = w * bpp;
  const px = new Uint8Array(stride * h);
  let prev = new Uint8Array(stride);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const cur = px.subarray(y * stride, (y + 1) * stride);
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? cur[x - bpp] : 0, b = prev[x], c = x >= bpp ? prev[x - bpp] : 0;
      let v = line[x];
      if (f === 1) v += a;
      else if (f === 2) v += b;
      else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const p = a + b - c; const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      cur[x] = v & 255;
    }
    prev = cur;
  }
  const rgba = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    if (bpp === 1) { rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = px[i]; rgba[4 * i + 3] = 255; }
    else if (bpp === 2) { rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = px[2 * i]; rgba[4 * i + 3] = px[2 * i + 1]; }
    else if (bpp === 3) { rgba[4 * i] = px[3 * i]; rgba[4 * i + 1] = px[3 * i + 1]; rgba[4 * i + 2] = px[3 * i + 2]; rgba[4 * i + 3] = 255; }
    else rgba.set(px.subarray(4 * i, 4 * i + 4), 4 * i);
  }
  return { width: w, height: h, data: rgba };
}
