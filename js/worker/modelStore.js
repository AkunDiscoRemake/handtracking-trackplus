// Fetches model files (possibly split into parts), verifies SHA-256 and keeps
// them in the Cache Storage API so later launches start offline and fast.

const CACHE = 'aura-hands-models-v1';

async function sha256Hex(buf) {
  const h = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

async function openCache() {
  try { return await caches.open(CACHE); } catch { return null; }
}

// Returns the model bytes as a Uint8Array. onProgress(loaded, total).
export async function loadModelBytes(entry, baseUrl, onProgress) {
  const cache = await openCache();
  const key = new URL(`${entry.file}?sha=${entry.sha256}`, baseUrl).href;
  if (cache) {
    const hit = await cache.match(key);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      onProgress?.(buf.length, buf.length, true);
      return buf;
    }
  }
  const out = new Uint8Array(entry.size);
  let off = 0;
  for (const part of entry.parts) {
    const res = await fetch(new URL(part, baseUrl));
    if (!res.ok) throw new Error(`${part}: HTTP ${res.status}`);
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (off + value.length > out.length) throw new Error(`${entry.file}: larger than manifest size`);
      out.set(value, off);
      off += value.length;
      onProgress?.(off, entry.size, false);
    }
  }
  if (off !== entry.size) throw new Error(`${entry.file}: expected ${entry.size} bytes, got ${off}`);
  const h = await sha256Hex(out);
  if (h !== entry.sha256) throw new Error(`${entry.file}: SHA-256 mismatch`);
  if (cache) {
    try { await cache.put(key, new Response(out, { headers: { 'content-type': 'application/octet-stream' } })); } catch { /* quota */ }
  }
  return out;
}
