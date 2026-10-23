// Offline cache: the model and wasm are large and never change for a version.
const C = 'chehra-v1';
self.addEventListener('install', e => { self.skipWaiting(); });
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(k => Promise.all(k.filter(x => x !== C).map(x => caches.delete(x)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== location.origin) return;
  const vendor = u.pathname.includes('/vendor/');
  e.respondWith(vendor
    ? caches.match(e.request).then(r => r || fetch(e.request).then(n => { const c = n.clone(); caches.open(C).then(x => x.put(e.request, c)); return n; }))
    : fetch(e.request).then(n => { const c = n.clone(); caches.open(C).then(x => x.put(e.request, c)); return n; }).catch(() => caches.match(e.request)));
});
