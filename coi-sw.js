// Service worker that makes the page cross-origin isolated on static hosts
// that cannot send COOP/COEP headers (e.g. GitHub Pages). It re-serves
// same-origin responses with the isolation headers, which enables
// SharedArrayBuffer and therefore multi-threaded WASM inference.
// Registered from index.html only when the page is not already isolated.

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  if (new URL(req.url).origin !== self.location.origin) return;
  e.respondWith((async () => {
    const res = await fetch(req);
    if (res.status === 0 || res.type === 'opaque') return res;
    const headers = new Headers(res.headers);
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    headers.set('Cross-Origin-Resource-Policy', 'same-origin');
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
  })());
});
