const CACHE_NAME = 'altru-ocr-v1';
const MODEL_CACHE = 'altru-ocr-models-v1';

const APP_SHELL = [
  './',
  './index.html',
  './app.js',
  './app.css',
  './manifest.json',
  './icon-192.png',
  './icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k !== CACHE_NAME && k !== MODEL_CACHE)
          .map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // ONNX models + WASM + dictionaries: cache-first, long-lived
  if (
    url.pathname.endsWith('.onnx') ||
    url.pathname.endsWith('.wasm') ||
    (url.pathname.endsWith('.txt') && url.pathname.includes('key'))
  ) {
    event.respondWith(
      caches.open(MODEL_CACHE).then((cache) =>
        cache.match(event.request).then(
          (cached) =>
            cached ||
            fetch(event.request).then((response) => {
              if (response.ok) cache.put(event.request, response.clone());
              return response;
            })
        )
      )
    );
    return;
  }

  // App shell: cache-first with network fallback
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
