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
 * The id of an album that actually has photos.
 *
 * Throws if the backend exposes no such album — a presentation wall
 * with no photos has nothing to present, so there is no meaningful
 * test to run.
 */
export async function albumWithPhotos(page: Page): Promise<number> {
  const res = await page.request.get(`${APP}/api/albums`);
  expect(res.ok(), 'the albums list must be reachable').toBeTruthy();

  const body = (await res.json()) as {
    albums?: { id: number; member_count: number }[];
  };
  const album = (body.albums ?? []).find((a) => a.member_count > 0);
  expect(album, 'the backend must expose an album with photos').toBeTruthy();
  return album!.id;
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
