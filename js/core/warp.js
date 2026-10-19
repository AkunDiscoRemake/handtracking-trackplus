// CPU affine warp + normalisation straight into an NCHW float tensor.
// Used by the Node validation tools/tests, and by the worker when
// OffscreenCanvas 2D is unavailable. Bilinear sampling, constant border.

// src: Uint8Array/Uint8ClampedArray RGBA, srcW × srcH.
// inv: affine crop→image {a,b,c,d,e,f} (x = a·u + c·v + e, y = b·u + d·v + f).
// order: 'rgb' or 'bgr'; mean/std in that same channel order.
// out: Float32Array, written at `offset` as [3, outH, outW].
export function warpToTensor(src, srcW, srcH, inv, outW, outH, mean, std, order, out, offset = 0, border = 0) {
  const plane = outW * outH;
  const ch = order === 'bgr' ? [2, 1, 0] : [0, 1, 2];
  const s0 = 1 / std[0], s1 = 1 / std[1], s2 = 1 / std[2];
  const b0 = (border - mean[0]) * s0, b1 = (border - mean[1]) * s1, b2 = (border - mean[2]) * s2;
  const o0 = offset, o1 = offset + plane, o2 = offset + 2 * plane;
  for (let v = 0; v < outH; v++) {
    // sample at pixel centres
    const vc = v + 0.5;
    for (let u = 0; u < outW; u++) {
      const uc = u + 0.5;
      const x = inv.a * uc + inv.c * vc + inv.e - 0.5;
      const y = inv.b * uc + inv.d * vc + inv.f - 0.5;
      const idx = v * outW + u;
      const x0 = Math.floor(x), y0 = Math.floor(y);
      if (x0 < -1 || y0 < -1 || x0 >= srcW || y0 >= srcH) {
        out[o0 + idx] = b0; out[o1 + idx] = b1; out[o2 + idx] = b2;
        continue;
      }
      const fx = x - x0, fy = y - y0;
      let r0 = 0, r1 = 0, r2 = 0;
      for (let k = 0; k < 4; k++) {
        const xx = x0 + (k & 1), yy = y0 + (k >> 1);
        const w = ((k & 1) ? fx : 1 - fx) * ((k >> 1) ? fy : 1 - fy);
        if (w === 0) continue;
        let c0, c1, c2;
        if (xx < 0 || yy < 0 || xx >= srcW || yy >= srcH) { c0 = c1 = c2 = border; } else {
          const p = (yy * srcW + xx) * 4;
          c0 = src[p + ch[0]]; c1 = src[p + ch[1]]; c2 = src[p + ch[2]];
        }
        r0 += w * c0; r1 += w * c1; r2 += w * c2;
      }
      out[o0 + idx] = (r0 - mean[0]) * s0;
      out[o1 + idx] = (r1 - mean[1]) * s1;
      out[o2 + idx] = (r2 - mean[2]) * s2;
    }
  }
  return out;
}

// RGBA (already letterboxed/warped by a canvas) → normalised NCHW tensor.
export function rgbaToTensor(rgba, w, h, mean, std, order, out, offset = 0) {
  const plane = w * h;
  const c0 = order === 'bgr' ? 2 : 0, c2 = order === 'bgr' ? 0 : 2;
  const s0 = 1 / std[0], s1 = 1 / std[1], s2 = 1 / std[2];
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    out[offset + i] = (rgba[p + c0] - mean[0]) * s0;
    out[offset + plane + i] = (rgba[p + 1] - mean[1]) * s1;
    out[offset + 2 * plane + i] = (rgba[p + c2] - mean[2]) * s2;
  }
  return out;
}
