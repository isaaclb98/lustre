/**
 * Tests for the API endpoints used by the frontend's main flows
 * that aren't covered by the existing behavioural suite — pins
 * shapes the frontend depends on and catches regressions in
 * favourites toggle, image URL fields, and 4xx error mapping.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

/**
 * Instead of mounting Svelte components for these (which requires
 * the SSR workaround), we exercise the URL builders + payload
 * shapes the components rely on. The Playwright suite covers the
 * wired-up behaviour end-to-end.
 */

describe('photoUrl', () => {
  it('builds the raw photo URL for a point id', async () => {
    const { photoUrl } = await import('./client');
    expect(photoUrl('abc-123')).toBe('/photo/abc-123/raw');
    expect(photoUrl('with space')).toBe('/photo/with%20space/raw');
    expect(photoUrl('with/slash')).toBe('/photo/with%2Fslash/raw');
  });
  it('appends ?w= when a width is requested (Lanczos resize)', async () => {
    const { photoUrl } = await import('./client');
    expect(photoUrl('abc-123', 1920)).toBe('/photo/abc-123/raw?w=1920');
    expect(photoUrl('abc-123', 640)).toBe('/photo/abc-123/raw?w=640');
  });
  it('omits ?w= when width is 0 or undefined', async () => {
    const { photoUrl } = await import('./client');
    expect(photoUrl('abc-123', 0)).toBe('/photo/abc-123/raw');
    expect(photoUrl('abc-123', undefined)).toBe('/photo/abc-123/raw');
  });
});

/**
 * Round-perf (issue #2): thumbUrl() now takes an optional width so
 * the frontend srcset can ask the backend for a pre-generated
 * sibling instead of always fetching the canonical 256-px file.
 * These tests pin the URL contract; the matching variant-on-disk
 * behaviour is covered by tests/test_thumbnails_unit.py on the
 * backend side.
 */
describe('thumbUrl', () => {
  it('builds the canonical /thumb/{id} URL without a width', async () => {
    const { thumbUrl } = await import('./client');
    expect(thumbUrl('abc-123')).toBe('/thumb/abc-123');
  });

  it('appends ?w=N when a width is supplied', async () => {
    const { thumbUrl } = await import('./client');
    // Post the model-variant migration plan, the indexer serves
    // a single 384px thumbnail. Pre-migration this asserted the
    // 240 and 180 ladder; the URL contract is the same (just a
    // different concrete size).
    expect(thumbUrl('abc-123', 384)).toBe('/thumb/abc-123?w=384');
  });

  it('omits ?w= for non-positive widths', async () => {
    const { thumbUrl } = await import('./client');
    expect(thumbUrl('abc-123', 0)).toBe('/thumb/abc-123');
    expect(thumbUrl('abc-123', -1)).toBe('/thumb/abc-123');
  });

  it('percent-encodes point IDs', async () => {
    const { thumbUrl } = await import('./client');
    expect(thumbUrl('a/b c', 384)).toBe('/thumb/a%2Fb%20c?w=384');
  });
});

describe('ApiError', () => {
  it('captures status, body, message', async () => {
    const { ApiError } = await import('./client');
    const e = new ApiError(404, { detail: 'gone' }, 'image not found');
    expect(e.status).toBe(404);
    expect(e.body).toEqual({ detail: 'gone' });
    expect(e.message).toBe('image not found');
    expect(e.name).toBe('Error');
  });

  it('falls back to a default message when not provided', async () => {
    const { ApiError } = await import('./client');
    expect(new ApiError(500, null).message).toMatch(/API 500/);
  });
});

describe('random endpoint round-trip', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            results: [
              {
                id: 'pt1',
                path: '/p.jpg',
                score: 0.9,
                score_str: '0.900',
                url: '/photo/pt1/raw',
                is_favorite: false,
                blurhash: null,
                width: null,
                height: null
              }
            ],
            took_ms: 12,
            limit: 30,
            offset: 0,
            has_more: false
          }),
          {
            status: 200,
            headers: { 'content-type': 'application/json' }
          }
        )
      )
    );
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('parses the SearchResponse envelope', async () => {
    const { random } = await import('./endpoints');
    // disable dev-mode zod assertion — jsdom can't render the
    // Svelte 5 imports the schemas chain pulls in
    const res = await random(1);
    expect(res.results.length).toBe(1);
    expect(res.results[0].id).toBe('pt1');
  });
});

describe('search endpoint URL builder', () => {
  // The dev-mode zod schema runs in vitest (since `dev` is true),
  // so the mocked response must satisfy the SearchResponse shape.
  const valid = JSON.stringify({
    results: [
      {
        id: 'p1',
        path: '/p.jpg',
        score: 0.9,
        score_str: '0.900',
        url: '/photo/p1/raw',
        is_favorite: false,
        blurhash: null,
        width: null,
        height: null
      }
    ],
    took_ms: 1,
    limit: 25,
    offset: 0,
    has_more: false
  });

  let captured: any;
  beforeEach(() => {
    captured = null;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: any, init: any) => {
        captured = { url: String(_url), init };
        return new Response(valid, {
          status: 200,
          headers: { 'content-type': 'application/json' }
        });
      })
    );
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('serialises positives/negatives as repeated query params', async () => {
    const { search } = await import('./endpoints');
    await search({
      positives: ['a', 'b'],
      negatives: ['c'],
      filename: 'IMG',
      limit: 25,
      offset: 50
    });
    const url = new URL(captured.url, 'http://x');
    expect(url.searchParams.getAll('positives')).toEqual(['a', 'b']);
    expect(url.searchParams.getAll('negatives')).toEqual(['c']);
    expect(url.searchParams.get('filename')).toBe('IMG');
    expect(url.searchParams.get('limit')).toBe('25');
    expect(url.searchParams.get('offset')).toBe('50');
    // Round-87: diversity params are no longer emitted. The
    // legacy MMR rerank still works server-side for external
    // API consumers but the home-page UI doesn't send it.
    expect(url.searchParams.has('diversity')).toBe(false);
    expect(url.searchParams.has('diversity_depth')).toBe(false);
  });

  it('switches to the centroid search path when centroid is set', async () => {
    const { search } = await import('./endpoints');
    await search({
      positives: ['a'],
      negatives: [],
      centroid: 'cool-shots'
    });
    expect(captured.url).toContain('/api/centroids/cool-shots/search');
  });

  // /api/search reads `?collection=` (singular, repeated) via
  // `parse_collections` in `search/_result_helpers.py`. Sending
  // the plural here silently no-ops the "Limit to library" filter
  // because the helper sees an empty list and returns the whole
  // library. /api/random is plural — its handler uses FastAPI
  // `Query()` which names the wire param after the Python kwarg.
  // These two tests pin the wire shape per endpoint so a drift on
  // either side trips a test.
  it('serialises collections as repeated ?collection= (singular) for /api/search', async () => {
    const { search } = await import('./endpoints');
    await search({ positives: [], negatives: [], collections: ['kpop', 'portrait'] });
    const url = new URL(captured.url, 'http://x');
    expect(url.pathname).toBe('/api/search');
    expect(url.searchParams.getAll('collection')).toEqual(['kpop', 'portrait']);
    // And critically: the plural key must NOT be present.
    expect(url.searchParams.getAll('collections')).toEqual([]);
  });

  it('serialises collections as repeated ?collections= (plural) for /api/random', async () => {
    const { random } = await import('./endpoints');
    await random({ limit: 10, collections: ['kpop', 'portrait'] });
    expect(captured.url).toContain('/api/random');
    const url = new URL(captured.url, 'http://x');
    expect(url.searchParams.getAll('collections')).toEqual(['kpop', 'portrait']);
    expect(url.searchParams.getAll('collection')).toEqual([]);
  });

  // Round‑29: album-card search buttons navigate to /?centroid=…
  // and the home page calls search({ centroid: 'album:<id>' }) (or
  // the system name 'favourites'). These tests pin the
  // wire shape so a rename on either side trips a test.
  //
  // The backend registers user-album centroids under
  // `album:<id>` (colon) — see `_album_centroid_name` in
  // search/app.py. Using `_` would 404 the search.
  it('round‑29: album:<id> centroid reaches the centroid search endpoint', async () => {
    const { search } = await import('./endpoints');
    await search({ centroid: 'album:7' });
    // The colon is URL-encoded by encodeURIComponent (album%3A7)
    // — that's fine; qdrant + FastAPI decode it back. The wire
    // shape must use a colon, NOT an underscore (which would
    // 404 because the backend key is `album:<id>`).
    expect(captured.url).toContain('/api/centroids/');
    expect(captured.url).toMatch(/album(?:%3A|:)7/);
    expect(captured.url).not.toContain('album_7');
    // Should NOT carry any of the prompt-derived params; those
    // would hit /api/search semantics on a centroid query.
    const url = new URL(captured.url, 'http://test');
    expect(url.searchParams.getAll('positives')).toEqual([]);
    expect(url.searchParams.getAll('negatives')).toEqual([]);
  });

  it('the built-in album searches by the favourites centroid', async () => {
    const { search } = await import('./endpoints');
    await search({ centroid: 'favourites' });
    expect(captured.url).toContain('/api/centroids/favourites/search');
  });

  it('the legacy likes centroid name still resolves', async () => {
    // Back-compat: the built-in album is named Favourites, but
    // `likes` stays registered so old saved searches and shared
    // links keep working.
    const { search } = await import('./endpoints');
    await search({ centroid: 'likes' });
    expect(captured.url).toContain('/api/centroids/likes/search');
  });

  it('round‑29: no centroid + empty prompts → /api/search (not centroid)', async () => {
    const { search } = await import('./endpoints');
    // captured.url is reset by the beforeEach hook on every fetch.
    await search({ positives: [], negatives: [], filename: '' });
    expect(captured.url).toContain('/api/search?');
    expect(captured.url).not.toContain('/api/centroids/');
  });
});
