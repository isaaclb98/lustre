/**
 * Typed endpoint wrappers. Each function:
 *   - documents the shape of the response
 *   - opts the dev-mode zod schema for drift detection
 *   - returns the parsed result
 *
 * The generated types in ./types.gen.ts provide compile-time
 * safety; the schemas in ./schemas.ts provide runtime safety.
 * Both come from openapi.json so they don't drift.
 */

import { apiGet, apiPost, apiDelete, apiPatch, photoUrl, thumbUrl } from './client';
import { Z } from './schemas';
import type { components } from './types.gen';
import { GRID_PAGE_SIZE } from './limits';

export type SearchResult = components['schemas']['SearchResult'];
export type SearchResponse = components['schemas']['SearchResponse'];
export type SavedSearch = components['schemas']['SavedSearch'];
export type AlbumSummary = components['schemas']['AlbumSummary'];
export type AlbumDetail = components['schemas']['AlbumDetailResponse'];

// For-you endpoint returns SearchResponse (same shape as /api/random
// and /api/search). The Zod schema for SearchResponse (in schemas.ts)
// is used directly — no separate For You type needed.

// ---------- Search ----------

export type SearchParams = {
  positives?: string[];
  negatives?: string[];
  filename?: string;
  limit?: number;
  offset?: number;
  /** When set, queries the centroid search endpoint. */
  centroid?: string;
  // Round-92: removed `centroidMode` / `sampleK` / `sampleN` from
  // the public SearchParams shape. The per-card "Surprise" button
  // on /albums (which forwarded `mode=sample`) was removed
  // because the in-panel Surprise toggle is the only randomisation
  // entry point on the home page now. The backend still accepts
  // `?mode=sample` for external API consumers, but no UI caller
  // sends it. If you need it from JS, set `qs.set('mode', 'sample')`
  // manually after `search()` returns, or open a PR.
  /** Restrict to one or more `collection` payload values. Empty/undefined = whole library. */
  collections?: string[];
  /**
   * Round-78: Surprise Me mode. When true, the backend samples
   * a 2400-photo deck from a Gumbel-perturbed softmax over the
   * top-N Qdrant candidates and returns it page-by-page via
   * `surpriseSession`. Same query + temperature + pool + seed =
   * same deck (deterministic). The backend also accepts the
   * legacy `diversity` parameter (MMR rerank) for external
   * API consumers, but the two are mutually exclusive.
   */
  surprise?: boolean;
  /**
   * Gumbel noise scale for Surprise Me. 0 = deterministic top-K,
   * 1.0 = roughly uniform. Default 0.05. Ignored unless
   * `surprise=true`.
   */
  surpriseTemperature?: number;
  /**
   * How many top candidates the Gumbel draw is taken from.
   * Larger = more variety, slower first page. Range 200..10000,
   * default 5000. Ignored unless `surprise=true`.
   */
  surprisePool?: number;
  /**
   * Reseed knob. Same query+seed = same deck; bump to roll a new
   * deck. Ignored unless `surprise=true`.
   */
  surpriseSeed?: number;
  /**
   * Opaque session id from a prior Surprise Me response. Pass
   * back with the next `offset` to walk the cached deck without
   * recomputing. Ignored unless `surprise=true`.
   */
  surpriseSession?: string;
};

export function search(params: SearchParams, signal?: AbortSignal) {
  const qs = new URLSearchParams();
  (params.positives ?? []).forEach((p) => qs.append('positives', p));
  (params.negatives ?? []).forEach((p) => qs.append('negatives', p));
  if (params.filename) qs.set('filename', params.filename);
  // Round-87: diversity parameters (diverse, diversity,
  // diversity_depth) are no longer emitted from the UI. The
  // backend still accepts them for external API consumers,
  // but the home page uses Surprise Me as the variety
  // mechanism and the additional-filters UI no longer
  // surfaces the diversity dropdowns.
  // Backend /api/search reads `?collection=` (singular, repeated),
  // not the plural — see `parse_collections` in
  // `search/_result_helpers.py`. Sending the plural here silently
  // no-ops the "Limit to library" filter (the helper parses an
  // empty list and returns the whole library).
  if (params.collections?.length) {
    for (const c of params.collections) qs.append('collection', c);
  }
  if (params.limit !== undefined) qs.set('limit', String(params.limit));
  if (params.offset !== undefined) qs.set('offset', String(params.offset));
  if (params.surprise) {
    qs.set('surprise', 'true');
    if (params.surpriseTemperature !== undefined) {
      qs.set('surprise_temperature', String(params.surpriseTemperature));
    }
    if (params.surprisePool !== undefined) {
      qs.set('surprise_pool', String(params.surprisePool));
    }
    if (params.surpriseSeed !== undefined) {
      qs.set('surprise_seed', String(params.surpriseSeed));
    }
    if (params.surpriseSession) {
      qs.set('surprise_session', params.surpriseSession);
    }
  }
  if (params.centroid) {
    // Round-92: removed the `?mode=` forwarding. The
    // sample-centroid path is no longer reachable from the
    // home page UI (the in-panel Surprise toggle is the only
    // randomisation entry point). The backend's `mode=sample`
    // is still served for external API consumers.
  }
  const base = params.centroid
    ? `/api/centroids/${encodeURIComponent(params.centroid)}/search`
    : '/api/search';
  return apiGet<SearchResponse>(`${base}?${qs.toString()}`, {
    signal,
    schema: Z.SearchResponse,
    schemaName: 'SearchResponse'
  });
}

// ---------- Random / For You ----------

export interface RandomParams {
  /** Session id from a previous /api/random response. Pass it back
   * with the next `offset` to walk forward through the shuffled deck.
   * Omit on the first call to create a new session. */
  session?: string;
  /** Position in the shuffled deck to read from. The first batch is
   * at offset 0; subsequent calls should pass the previous batch's
   * count as the new offset (or use the response's `offset` + count). */
  offset?: number;
  /** Number of photos to return (1..200). Default 30. */
  limit?: number;
  /** Restrict to one or more collections. Empty = whole library. */
  collections?: string[];
  /** Result view: 'grid' (default) or 'feed'. */
  view?: string;
  /** Abort the request. */
  signal?: AbortSignal;
}

export function random(params: RandomParams | number = {}, signal?: AbortSignal) {
  // Back-compat: random(limit) → random({limit: 30})
  const p: RandomParams =
    typeof params === 'number' ? { limit: params } : params;
  const sig = signal ?? p.signal;
  const search = new URLSearchParams();
  if (p.session !== undefined) search.set('session', p.session);
  if (p.offset !== undefined) search.set('offset', String(p.offset));
  if (p.limit !== undefined) search.set('limit', String(p.limit));
  if (p.view !== undefined) search.set('view', p.view);
  // /api/random uses FastAPI's `Query()` which names the wire
  // param after the Python kwarg (`collections`, plural). Keep
  // this in sync with the `/api/random` signature in
  // `search/routers/random.py`.
  if (p.collections) {
    for (const c of p.collections) search.append('collections', c);
  }
  return apiGet<SearchResponse>(`/api/random?${search.toString()}`, {
    signal: sig,
    schema: Z.SearchResponse,
    schemaName: 'SearchResponse (random)'
  });
}

// ---------- For You (Gumbel-max taste sampler) ----------

/**
 * Parameters for forYouFeed.
 *
 * Round-88: the /for-you algorithm swapped from "uniform
 * shuffle the top X% of library by taste" to "Gumbel-max
 * sample from a Qdrant recommend() pool of pool_size
 * candidates." The temperature slider controls the
 * Gumbel noise scale: T=0 is the deterministic top-K,
 * T=1.0 is near-uniform random over the pool.
 *
 * `seed` is a numeric random seed (int). Pass a fresh
 * random int on every page reload so the user sees a
 * different deck; reuse the same value for paginated
 * scroll calls within a single page-mount to walk
 * through the same deck coherently.
 *
 * `session` is the opaque cache key from a prior
 * response. Pass back with offset=N to walk the same
 * deck without recomputing.
 */
export interface ForYouFeedParams {
  /** Photos per page. 1..100. Default 30. */
  limit?: number;
  /** Zero-based offset into the cached Gumbel deck. */
  offset?: number;
  /** Number of recommend() candidates the Gumbel draw is
   * taken from. Default 2000. Range [200, 50_000]. */
  pool_size?: number;
  /** Gumbel temperature. 0.0..1.0. Default 0.05. */
  temperature?: number;
  /** Numeric random seed. See interface docstring. */
  seed?: number;
  /** Opaque cache key from a prior response. */
  session?: string;
  /** Abort the request. */
  signal?: AbortSignal;
}

/**
 * For You — Gumbel-max sample from a Qdrant recommend()
 * pool of `pool_size` candidates. The Gumbel noise scale
 * (temperature) controls how biased the deck is toward
 * high-scoring photos: T=0 is deterministic top-K (curated),
 * T=1.0 is near-uniform random over the pool.
 *
 * The server caches the deck keyed by
 * (fav_ids, dis_ids, temperature, pool_size, seed). Same
 * seed → same deck (coherent pagination), different
 * seed → fresh deck (page reload).
 *
 * See: search/routers/for_you.py for the endpoint wiring.
 */
export function forYouFeed(params: ForYouFeedParams = {}, signal?: AbortSignal) {
  const sig = signal ?? params.signal;
  const search = new URLSearchParams();
  if (params.limit !== undefined) search.set('limit', String(params.limit));
  if (params.offset !== undefined) search.set('offset', String(params.offset));
  if (params.pool_size !== undefined) search.set('pool_size', String(params.pool_size));
  if (params.temperature !== undefined) search.set('temperature', String(params.temperature));
  if (params.seed !== undefined) search.set('seed', String(params.seed));
  if (params.session !== undefined) search.set('session', params.session);
  return apiGet<SearchResponse>(`/api/for-you/feed?${search.toString()}`, {
    signal: sig,
    schema: Z.SearchResponse,
    schemaName: 'SearchResponse (for-you)'
  });
}
export function similarPhotos(
  pointId: string,
  limit = GRID_PAGE_SIZE,
  signal?: AbortSignal
) {
  return apiGet<SearchResponse>(
    `/api/similar/${encodeURIComponent(pointId)}?limit=${limit}`,
    { signal, schema: Z.SearchResponse, schemaName: 'SearchResponse (similar)' }
  );
}

export async function dislikePoint(pointId: string) {
  await apiPost(`/api/dislikes/${encodeURIComponent(pointId)}`);
}

export async function undislikePoint(pointId: string) {
  await apiDelete(`/api/dislikes/${encodeURIComponent(pointId)}`);
}

// ---------- Likes (formerly "favourites") ----------

export async function likePoint(pointId: string) {
  await apiPost(`/api/favorites/${encodeURIComponent(pointId)}`);
}

export async function unlikePoint(pointId: string) {
  await apiDelete(`/api/favorites/${encodeURIComponent(pointId)}`);
}

/** Lightweight list — the Likes album view uses this. */
export function listFavorites(limit = GRID_PAGE_SIZE, offset = 0, signal?: AbortSignal) {
  return apiGet<unknown>(
    `/api/favorites?limit=${limit}&offset=${offset}&as_results=1`,
    { signal, schemaName: 'favorites-list' }
  );
}

/** Lightweight list — the Dislikes album view uses this. */
export function listDislikes(limit = GRID_PAGE_SIZE, offset = 0, signal?: AbortSignal) {
  return apiGet<unknown>(
    `/api/dislikes?limit=${limit}&offset=${offset}&as_results=1`,
    { signal, schemaName: 'dislikes-list' }
  );
}

// ---------- Saved searches ----------

export function listSavedSearches() {
  return apiGet<unknown>('/api/saved-searches', {
    schema: Z.SavedSearchListResponse,
    schemaName: 'SavedSearchListResponse'
  });
}

export function createSavedSearch(body: {
  name: string;
  positives: string[];
  negatives: string[];
}) {
  return apiPost<unknown>('/api/saved-searches', body, {
    schema: Z.SavedSearch,
    schemaName: 'SavedSearch'
  });
}

export async function deleteSavedSearch(id: number) {
  await apiDelete(`/api/saved-searches/${id}`);
}

// ---------- Albums ----------

export function listAlbums() {
  return apiGet<{ albums: AlbumSummary[] }>('/api/albums', {
    schema: Z.AlbumsListResponse,
    schemaName: 'AlbumsListResponse'
  });
}

/**
 * Fetch a single album's metadata + members, with optional
 * limit/offset for pagination. The album detail page walks the
 * members in batches of GRID_PAGE_SIZE so very large albums
 * (hundreds/thousands of photos) still render quickly.
 *
 * The backend's AlbumDetailResponse carries `member_total` so
 * callers can compare against the running member count to drive
 * their own `has_more` flag.
 */
export function getAlbum(
  albumId: number | string,
  limit: number = GRID_PAGE_SIZE,
  offset: number = 0,
  signal?: AbortSignal,
) {
  return apiGet<unknown>(
    `/api/albums/${albumId}?limit=${limit}&offset=${offset}`,
    {
      schema: Z.AlbumDetailResponse,
      schemaName: 'AlbumDetailResponse',
      signal,
    }
  );
}

export async function createAlbum(body: { name: string; description?: string }) {
  await apiPost('/api/albums', body, {
    schema: Z.AlbumSummary,
    schemaName: 'AlbumSummary'
  });
}

export async function updateAlbum(
  albumId: number,
  body: { name?: string; description?: string; cover_favorite_id?: number | null }
) {
  await apiPatch(`/api/albums/${albumId}`, body);
}

export async function deleteAlbum(albumId: number) {
  await apiDelete(`/api/albums/${albumId}`);
}

/** Add a photo to an album. Idempotent — backend no-ops on duplicate. */
export async function addPhotoToAlbum(albumId: number, pointId: string) {
  await apiPost(`/api/albums/${albumId}/members/${encodeURIComponent(pointId)}`);
}

/** Remove a photo from an album. */
export async function removePhotoFromAlbum(albumId: number, pointId: string) {
  await apiDelete(`/api/albums/${albumId}/members/${encodeURIComponent(pointId)}`);
}

/**
 * List every album a given photo belongs to.
 *
 * Used by the per-photo "Add to album" dropdown so the menu can
 * render an "already in this album" indicator on each row and
 * toggle on click instead of always-adding. Returns the album
 * summaries directly; the consumer can read `.id` from each.
 *
 * Returns an empty `albums: []` (never throws) if the photo is in
 * no albums or the backend rejects the query — callers can
 * render an "empty membership" UI without a try/catch.
 */
export async function listAlbumsForFavorite(pointId: string): Promise<AlbumSummary[]> {
  const r = await apiGet<{ favorite_id: string; albums: AlbumSummary[] }>(
    `/api/albums/by-favorite/${encodeURIComponent(pointId)}`,
  );
  return r?.albums ?? [];
}

// ---------- Misc ----------

export function listCollections() {
  return apiGet<{
    collections: { name: string; count: number }[];
  }>('/api/collections', {
    schema: Z.CollectionsList,
    schemaName: 'CollectionsList'
  });
}

export function listCentroids() {
  return apiGet<{
    centroids: { name: string; kind?: string; member_count?: number }[];
  }>('/api/centroids', {
    schema: Z.CentroidList,
    schemaName: 'CentroidList'
  });
}

// ---------- System / admin telemetry (Phase 2) ----------
// Replaces the raw `fetch('/api/system/status')` and
// `fetch('/api/admin/index/status')` calls in the home page.
// The schemas in Z mirror the Pydantic models in
// search/models.py so a backend drift trips the dev-mode
// Zod assertion in client.ts.

export function getSystemStatus(signal?: AbortSignal) {
  return apiGet<{
    model_status: Record<string, unknown>;
    qdrant_count: number;
    index_db_count: number;
    last_refresh: number | null;
  }>('/api/system/status', {
    schema: Z.SystemStatus,
    schemaName: 'SystemStatus',
    signal
  });
}

export function getAdminIndexStatus(signal?: AbortSignal) {
  return apiGet<{
    state: 'idle' | 'running' | 'failed';
    mode?: 'incremental' | 'rebuild' | null;
    job_id?: string | null;
    pid?: number | null;
    started_at?: string | null;
    finished_at?: string | null;
    last_run_at?: string | null;
    last_error?: string | null;
    progress: {
      indexed?: number;
      total?: number;
      current_file?: string;
      eta_seconds?: number;
    };
    points_count?: number | null;
  }>('/api/admin/index/status', {
    schema: Z.IndexerStatus,
    schemaName: 'IndexerStatus',
    signal
  });
}

// ---------- Count-only album endpoints (Phase 2) ----------
// The Albums page header only needs the totals for the
// "Likes" and "Dislikes" album cards. We don't want to ship
// every favorite row just to count them; ?limit=1 gets the
// envelope back and we read .total / .count.

export async function getFavoriteCount(signal?: AbortSignal): Promise<number> {
  const data = await apiGet<{
    favorites: { id: string; path: string; favorited_at?: string }[];
    total: number;
  }>('/api/favorites?limit=1', {
    schema: Z.FavoritesList,
    schemaName: 'FavoritesList',
    signal
  });
  return data.total;
}

export async function getDislikeCount(signal?: AbortSignal): Promise<number> {
  const data = await apiGet<{
    items: unknown[];
    count: number;
  }>('/api/dislikes?limit=1', {
    schema: Z.DislikesList,
    schemaName: 'DislikesList',
    signal
  });
  return data.count;
}

// ---------- Orphan review + confirmed purge ----------
// The orphan lifecycle is two-step. The indexer MARKS points whose
// photo is gone from disk, and points belonging to a source that was
// removed from the source list. A marked point is excluded from
// every read path, so it stops being served immediately — but its
// embedding is still in the collection.
//
// These are the second step: review the count, then confirm the
// purge that removes the embeddings for good. The purge is
// irreversible, which is why it is a separate, explicit call.

export type OrphanSample = components['schemas']['OrphanSample'];
export type OrphanListResponse = components['schemas']['OrphanListResponse'];
export type OrphanPurgeResponse = components['schemas']['OrphanPurgeResponse'];

export function getOrphans(sampleSize = 50, signal?: AbortSignal) {
  return apiGet<OrphanListResponse>(
    `/api/admin/orphans?sample_size=${sampleSize}`,
    {
      schema: Z.OrphanListResponse,
      schemaName: 'OrphanListResponse',
      signal
    }
  );
}

export function purgeOrphans(source?: string, limit?: number) {
  return apiPost<OrphanPurgeResponse>('/api/admin/orphans/purge', {
    confirm: true,
    ...(source ? { source } : {}),
    ...(limit ? { limit } : {})
  });
}

// ---------- Photo URL passthrough ----------

export { photoUrl, thumbUrl };
