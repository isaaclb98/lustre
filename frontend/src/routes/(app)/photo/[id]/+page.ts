/**
 * Photo detail page — universal load function.
 *
 * Runs before the component renders. Pre-fetching here saves
 * one round-trip vs fetching in `onMount`: the component
 * renders with `data.photo` already populated instead of an
 * empty shell that fills in 100-300 ms later.
 *
 * Error handling: rather than calling SvelteKit's `error()`
 * (which short-circuits to the global +error.svelte boundary
 * and bypasses the page's inline error placeholder), we
 * surface the failure via `data.error`. The component then
 * renders the existing `.placeholder.error` block that the
 * e2e tests + users expect. This keeps the user-facing
 * "Photo not found" UX identical between the old client-
 * side fetch path and the new load-driven path.
 *
 * In dev (vite proxy + FastAPI on :18000) and prod (k8s in-
 * cluster service), the URL is the same — `/api/photo/{id}` —
 * because the backend serves both `_app/` and the API on the
 * same origin. No environment branching needed.
 */
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, fetch }) => {
  const id = params.id ?? '';
  const res = await fetch(`/api/photo/${encodeURIComponent(id)}`);
  if (res.status === 404) {
    return { photo: null, error: 'Photo not found' };
  }
  if (!res.ok) {
    return {
      photo: null,
      error: `Failed to load photo (HTTP ${res.status})`
    };
  }
  const photo = await res.json();
  return { photo, error: null };
};