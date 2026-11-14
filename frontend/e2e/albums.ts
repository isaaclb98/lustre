/**
 * Album discovery for the e2e suite.
 *
 * A test that wants an album must ASK for one. The old wall test
 * hardcoded `/albums/5/present`, which only exists in a real
 * library — the e2e backend seeds albums with fresh ids (1 and 2),
 * so the hardcoded id 404'd, the wall never rendered a tile, and
 * the test sat there until its 30s timeout expired.
 *
 * Asking the API is also more honest than a fixture: the test then
 * runs against whatever the backend actually has, and the assertion
 * "an album with photos exists" is stated once, here, instead of
 * being implied by a magic number in six places.
 */
import { expect, type Page } from '@playwright/test';

const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';

/**
 * The albums the backend exposes, with their member counts.
 */
async function albums(page: Page): Promise<{ id: number; member_count: number }[]> {
  const res = await page.request.get(`${APP}/api/albums`);
  expect(res.ok(), 'the albums list must be reachable').toBeTruthy();
  const body = (await res.json()) as {
    albums?: { id: number; member_count: number }[];
  };
  return body.albums ?? [];
}

/**
 * The id of an album that actually has photos.
 *
 * If the backend has none, one is created and given photos. A test
 * that needs an album must not depend on the seed's albums still
 * being there: the destructive tests delete every user album to
 * assert the empty state, and whichever test runs after them would
 * otherwise find nothing and fail for a reason that has nothing to
 * do with what it is testing.
 */
export async function albumWithPhotos(page: Page): Promise<number> {
  const album = (await albums(page)).find((a) => a.member_count > 0);
  if (album) return album.id;

  const created = await page.request.post('/api/albums', {
    data: { name: `e2e-album-${Date.now()}` }
  });
  expect(created.status(), 'the album must be creatable').toBe(200);
  const { id } = await created.json();

  // Members are favourites, so take the ids from the favourites list
  // rather than from the photo list.
  const favRes = await page.request.get(`${APP}/api/favorites?limit=8`);
  expect(favRes.ok(), 'the favourites list must be reachable').toBeTruthy();
  const favBody = (await favRes.json()) as { favorites?: { id: string }[] };
  const favIds = (favBody.favorites ?? []).map((f) => f.id);
  expect(favIds.length, 'the backend must expose some favourites').toBeGreaterThan(0);

  for (const favId of favIds.slice(0, 8)) {
    await page.request.post(`/api/albums/${id}/members/${favId}`);
  }

  return id as number;
}

/**
 * The id of the album with the FEWEST photos.
 *
 * Some assertions need a whole cycle of the album to fit on screen
 * at once — the drift's ring repeats the album, and proving the
 * repeat means seeing more than one cycle. That only works for an
 * album smaller than the wall's window of tiles, so take the
 * smallest one the backend has.
 */
export async function smallestAlbum(page: Page): Promise<number> {
  const withPhotos = (await albums(page)).filter((a) => a.member_count > 0);
  expect(withPhotos, 'the backend must expose an album with photos').toBeTruthy();
  const smallest = withPhotos.reduce((a, b) => (a.member_count <= b.member_count ? a : b));
  return smallest.id;
}

/** The album's own page — the ordinary, non-presentation route. */
export async function albumUrl(page: Page): Promise<string> {
  return `${APP}/albums/${await albumWithPhotos(page)}`;
}

/** The album's presentation wall. */
export async function presentUrl(page: Page): Promise<string> {
  return `${APP}/albums/${await albumWithPhotos(page)}/present`;
}

/** The album's presentation wall, drifting. */
export async function driftUrl(page: Page): Promise<string> {
  return `${APP}/albums/${await albumWithPhotos(page)}/present?motion=drift`;
}