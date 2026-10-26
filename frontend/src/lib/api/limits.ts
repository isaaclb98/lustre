/**
 * Fallback page size for API wrappers whose caller doesn't pass a
 * limit. The grid routes size their requests from the column count
 * the grid actually renders — see $lib/grid/geometry — so this
 * value only applies to a call that omits the limit entirely.
 *
 * If a future grid genuinely needs a different page size (e.g.
 * one-page detail view that renders everything), add a separate
 * exported constant alongside this one rather than overriding.
 *
 * 24 = 4 clean rows of 6 (the Large preset at full width).
 */
export const GRID_PAGE_SIZE = 24;

/* --------------------------------------------------------------------------
 * Surprise Me / For-You pool sizing.
 *
 * The two array variants exist because the Gumbel-max sampler
 * behaves differently on the two endpoints:
 *
 *  - The home-page Surprise Me (surprise=true) runs over a
 *    Qdrant cosine top-K; the score distribution has a long
 *    tail, so 5_000 is a useful default that surfaces the bulk
 *    of variety while staying inside the HNSW 30s timeout.
 *
 *  - The /for-you endpoint runs over Qdrant recommend() on a
 *    much denser taste pool, where 2_000 already covers the
 *    top of the relevance curve. The for-you stops include a
 *    500 intermediate step so the user can drop below the
 *    home page's floor when they want even less variety.
 *
 * Pool size cap of 50_000 mirrors the Qdrant HNSW `limit`
 * ceiling — beyond it, the server-side 30s timeout fires
 * before the search returns the candidate set.
 *
 * The DEFAULT_POOL_INDEX_* values index into the matching
 * POOL_STOPS_* array. Both sliders default to ~5_000-equivalent
 * in the user's perceived experience (home = 5_000; for-you
 * index 3 = 2_000 because the taste pool is denser).
 * -------------------------------------------------------------------------- */
export const POOL_SIZE_MIN = 200;
export const POOL_SIZE_MAX = 50_000;
export const TEMPERATURE_MIN = 0;
export const TEMPERATURE_MAX = 1;

export const POOL_STOPS_HOME = [200, 1_000, 5_000, 20_000, 50_000] as const;
export const POOL_STOPS_FOR_YOU = [200, 500, 1_000, 2_000, 10_000, 50_000] as const;
export const DEFAULT_POOL_INDEX_HOME = 2;   // → 5_000
export const DEFAULT_POOL_INDEX_FOR_YOU = 3; // → 2_000
export const DEFAULT_TEMPERATURE = 0.05;

/* --------------------------------------------------------------------------
 * Lightbox slideshow cadence.
 *
 * Mirrors stores/preferences.ts SLIDESHOW_PRESETS — exported
 * here so the settings UI can iterate the same list the store
 * uses to validate user input. The store still owns the
 * *value*; this file owns the *list of legal values*.
 * -------------------------------------------------------------------------- */
export const SLIDESHOW_PRESETS_MS = [1000, 3000, 5000, 10_000, 20_000, 30_000] as const;
export const DEFAULT_SLIDESHOW_MS = 3000;
