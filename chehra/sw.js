// Offline: the app shell is precached; the model and wasm are cached on first use (large, versioned with the app).
const C = 'chehra-v6';
const SHELL = ['./', 'index.html', 'app.js', 'track.js', 'audio.js', 'manifest.webmanifest', 'icon.svg', 'icon-192.png'];
self.addEventListener('install', e => { self.skipWaiting(); e.waitUntil(caches.open(C).then(c => c.addAll(SHELL)).catch(() => {})); });
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(k => Promise.all(k.filter(x => x !== C).map(x => caches.delete(x)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== location.origin) return;
  const put = n => { if (n.ok) { const c = n.clone(); caches.open(C).then(x => x.put(e.request, c)); } return n; };
  e.respondWith(u.pathname.includes('/vendor/')
    ? caches.match(e.request).then(r => r || fetch(e.request).then(put))
    : fetch(e.request).then(put).catch(() => caches.match(e.request, { ignoreSearch: true })));
});
