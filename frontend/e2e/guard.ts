/**
 * Destructive-data guard for the E2E suite.
 *
 * Several tests in this suite DELETE user data — every favourite, every
 * user album — because they assert on an empty state ("For You page when
 * no likes exist shows empty state", "Albums page when no custom albums
 * exist shows empty state"). That is only safe against a disposable
 * backend.
 *
 * On 2026-11-09 the suite was pointed at the production backend
 * (127.0.0.1:8000 on ghost-moth) by mistake and deleted all 21 user
 * albums. The rows were recoverable from the SQLite WAL/checkpoint, but
 * the data should never have been at risk.
 *
 * So: before any destructive test runs, this module proves the target
 * backend is a disposable one. Two independent checks, because either
 * alone can be fooled:
 *
 *   1. The backend must expose the test marker header/flag that the
 *      dev server sets (LUSTRE_E2E_MODE=1).
 *   2. The backend's album/favourite counts must be within the range a
 *      disposable instance has — a real library has hundreds.
 *
 * Either check failing aborts the test. `assertDisposable` throws; the
 * destructive tests call it first.
 */
import type { APIRequestContext } from '@playwright/test';

/** Header the backend sets when it is started for an E2E run. */
const E2E_HEADER = 'x-lustre-e2e-mode';

/**
 * Counts above this mean the backend holds real user data.
 *
 * The demo seed creates 2 albums and 100 favourites, and a test run
 * adds a few throwaway albums on top — so the thresholds have to sit
 * above that and still well below a real library, which holds
 * thousands of favourites. The margin to production is the point:
 * this check exists to catch a backend that was started with the
 * marker but later pointed at real data.
 */
const MAX_ALBUMS = 10;
const MAX_FAVORITES = 300;

export class DestructiveTestError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'DestructiveTestError';
  }
}

/**
 * True when the response carries the E2E marker — the backend was
 * started with LUSTRE_E2E_MODE=1 and is therefore disposable.
 */
export function isDisposableBackend(headers: Record<string, string>): boolean {
  return headers[E2E_HEADER] === '1' || headers[e2e_header_lower()] === '1';
}

function e2e_header_lower(): string {
  return 'x-lustre-e2e-mode';
}

/**
 * Prove the backend is disposable, or throw. Call this at the top of any
 * test that deletes user data.
 *
 * @param request  Playwright's API request context
 * @param backend  Human-readable description of the target, for the error
 */
export async function assertDisposable(
  request: APIRequestContext,
  backend: string
): Promise<void> {
  const albumsRes = await request.get('/api/albums?limit=100');
  const favRes = await request.get('/api/favorites?limit=100');

  if (!albumsRes.ok() || !favRes.ok()) {
    throw new DestructiveTestError(
      `Cannot verify the target backend is disposable (${backend}): ` +
        `/api/albums -> ${albumsRes.status()}, /api/favorites -> ${favRes.status()}. ` +
        `Refusing to run a destructive test against an unknown backend.`
    );
  }

  const albums = await countList(albumsRes, 'albums');
  const favorites = await countList(favRes, 'favorites');

  // Check 1: the explicit marker.
  const marker = isDisposableBackend(albumsRes.headers() as Record<string, string>);
  if (!marker) {
    throw new DestructiveTestError(
      `Refusing to run a destructive test against ${backend}: the backend ` +
        `does not carry the "${E2E_HEADER}: 1" marker, so it is not a ` +
        `disposable E2E instance (albums=${albums}, favourites=${favorites}).`
    );
  }

  // Check 2: the shape of the data. A disposable instance is empty; a
  // real library is not. This catches a backend that was started with
  // the marker but has since been pointed at real data.
  if (albums > MAX_ALBUMS || favorites > MAX_FAVORITES) {
    throw new DestructiveTestError(
      `Refusing to run a destructive test against ${backend}: it holds ` +
        `${albums} album(s) and ${favorites} favourite(s), above the ` +
        `disposable threshold (${MAX_ALBUMS}/${MAX_FAVORITES}). ` +
        `Treat this backend as real data.`
    );
  }
}

async function countList(
  res: { json(): Promise<unknown> },
  key: string
): Promise<number> {
  try {
    const body = (await res.json()) as Record<string, unknown>;
    const value = body[key];
    return Array.isArray(value) ? value.length : 0;
  } catch {
    return 0;
  }
}
