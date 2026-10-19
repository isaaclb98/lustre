import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

// Dev-server proxy target. In Isaac's local dev setup this points at
// the FastAPI backend on localhost:8765; in the k8s cluster the
// Deployment sets API_PROXY_TARGET=http://lustre:8000 (the
// in-cluster Service), so vite proxies API + photo-raw + zip routes
// directly to the backend pod instead of relying on Caddy's path-
// matching routing. Removing the Caddy-side path split makes Caddy a
// pure passthrough and prevents the kubelet→projected-volume→caddy
// reload race that can leave Caddy serving the OLD Caddyfile while
// the ConfigMap update sits in transit.
const API_PROXY_TARGET = process.env.API_PROXY_TARGET || 'http://localhost:8765';

export default defineConfig({
  plugins: [sveltekit()],
  // Svelte 5 ships separate server and client runtimes. jsdom is a
  // browser environment, so force the client bundle here. Without it,
  // @testing-library/svelte mounts the server bundle and explodes with
  // "lifecycle_function_unavailable".
  define: {
    'import.meta.env.SSR': 'false'
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      // Proxy API + photo + zip endpoints to FastAPI in dev. In prod
      // (k8s cluster) the Deployment sets API_PROXY_TARGET to the
      // in-cluster Service; in local dev it stays on localhost:8765.
      //
      // SvelteKit routes match first when listed in src/routes/. The
      // proxy is only invoked for paths no SvelteKit route handles.
      // We use exact-prefix matches (regex) so /albums/{id} and
      // /photo/{id} (the SvelteKit pages) don't get forwarded to
      // FastAPI, but /albums/{id}/download.zip and /photo/{id}/raw do.
      '/api': {
        target: API_PROXY_TARGET,
        changeOrigin: true
      },
      '/healthz': {
        target: API_PROXY_TARGET,
        changeOrigin: true
      },
      // Backend photo RAW endpoint must be proxied; SvelteKit pages
      // for /photo/{id} match first because the file route is more
      // specific, but the dev proxy still must serve raw bytes when
      // used directly.
      //
      // The `(?:$|\?)` terminator is the dev-proxy bugfix: the
      // earlier `^/photo/[^/]+/raw$` only matched URLs that ended
      // at `/raw`, but `photoUrl()` always appends `?w=1248` (the
      // Lanczos-resized width). With the old regex, every lightbox
      // open 404'd and the user saw a permanent dark backdrop —
      // which itself reads as the "next-photo flash" report.
      // Allowing either end-of-URL or a query-string separator as
      // the terminator lets the `?w=` variant through.
      '^/photo/[^/]+/raw(?:$|\\?)': {
        target: API_PROXY_TARGET,
        changeOrigin: true
      },
      // Round-11: thumbnail proxy. Without this, every grid
      // tile's <img src> in dev mode 404'd (Vite SPA's index.html
      // would be served instead of the WebP bytes), so tiles
      // rendered as blurhash placeholders with no photo ever
      // appearing. Prod is unaffected (Caddy proxies
      // /thumb/{id} unconditionally); this only matters for
      // local vite dev.
      '^/thumb/[^/]+(?:$|\\?)': {
        target: API_PROXY_TARGET,
        changeOrigin: true
      },
      // Zip downloads for the Albums page buttons
      '^/albums/[^/]+/download\\.zip$': {
        target: API_PROXY_TARGET,
        changeOrigin: true
      },
      '^/favorites/download\\.zip$': {
        target: API_PROXY_TARGET,
        changeOrigin: true
      }
    },
    fs: {
      allow: ['..']
    }
  },
  test: {
    include: ['src/**/*.{test,spec}.{js,ts}'],
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./vitest-setup.ts']
  }
});
