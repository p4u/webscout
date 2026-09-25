import { defineConfig } from 'vite';

// The API is same-origin in production (nginx proxies /api/ to the api service).
// In development we proxy to a locally running `webscout --api --api-port 8080`.
const target = process.env.WEBSCOUT_API_ORIGIN || 'http://127.0.0.1:8080';

export default defineConfig({
  // Everything is bundled: no CDN references at runtime, so the container works offline.
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2020',
    assetsDir: 'assets',
  },
  server: {
    port: 5173,
    proxy: {
      '/mcp': { target, changeOrigin: true },
      '/api': {
        target,
        changeOrigin: true,
        // NDJSON is streamed; never let the dev proxy buffer it.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['cache-control'] = 'no-transform';
          });
        },
      },
    },
  },
  preview: {
    port: 4173,
    proxy: { '/api': { target, changeOrigin: true }, '/mcp': { target, changeOrigin: true } },
  },
});
