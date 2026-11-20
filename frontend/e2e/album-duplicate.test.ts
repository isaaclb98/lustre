/**
 * E2E for the album duplicate.
 *
 * An album is a curation the user built by hand, and the way people
 * use collections is often "same photos, different angle" — a
 * favourites album and a "to print" album, a working set and a
 * shortlist. Before this, making the second one meant adding every
 * photo again.
 *
 * What this pins:
 *
 *   1. The DUPLICATE control is on the album's own page, next to
 *      the other album actions.
 *   2. Pressing it COPIES the album: the copy carries the photos,
 *      the description, and a name of its own ("<name> (copy)").
 *      The name has to differ — `albums.name` is UNIQUE, so a copy
 *      cannot keep the source's name.
 *   3. Pressing it twice does not collide: the second copy is
 *      "<name> (copy 2)".
 *   4. The ORIGINAL is untouched — a duplicate adds a new album, it
 *      does not move the photos out of the one you were looking at.
 *   5. The user LANDS ON THE COPY, and the page shows the COPY.
 *      That last one is the reason this file exists: the album page
 *      loaded its detail with `onMount` only, so the component was
 *      reused across a navigation from one album to another and the
 *      user kept looking at the previous album's photos under the
 *      new album's URL. "Duplicate" navigates to the copy it just
 *      made, so it would land on a page showing the album it was
 *      copied from.
 */
import { test, expect, type Page } from '@playwright/test';

const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';

/** The ids of the favourites, which are what album members are. */
async function favouriteIds(page: Page): Promise<string[]> {
  const res = await page.request.get(`${APP}/api/favorites?limit=8`);
  expect(res.ok(), 'the favourites list must be reachable').toBeTruthy();
  const body = (await res.json()) as { favorites?: { id: string }[] };
  return (body.favorites ?? []).map((f) => f.id);
}

/** A throwaway album with a few photos in it. */
async function createAlbumWithPhotos(
  page: Page,
  name: string
): Promise<{ id: number; name: string; memberIds: string[] }> {
  const res = await page.request.post('/api/albums', {
    data: { name, description: 'natural light only' }
  });
  expect(res.status(), 'the album must be creatable').toBe(200);
  const { id } = (await res.json()) as { id: number };

  const ids = (await favouriteIds(page)).slice(0, 3);
  expect(ids.length, 'the backend must expose some favourites').toBeGreaterThan(0);
  for (const favId of ids) {
    await page.request.post(`/api/albums/${id}/members/${favId}`);
  }
  return { id, name, memberIds: ids };
}

async function deleteAlbum(page: Page, id: number): Promise<void> {
  await page.request.delete(`/api/albums/${id}`);
}

/**
 * Press Duplicate and wait for the copy to be there.
 *
 * The click is asynchronous — the copy is POSTed and the page then
 * navigates to it — so the test has to wait for the copy's NAME to
 * appear in the header before it looks for the copy in the API.
 * Asserting on the API straight after the click races the request.
 */
async function duplicateAndWait(
  page: Page,
  expectedName: string
): Promise<void> {
  await page
    .locator('.head button')
    .filter({ hasText: /^duplicate$/i })
    .first()
    .click();
  await expect(page.locator('.head h1')).toHaveText(expectedName);
}

async function albumDetail(page: Page, id: number) {
  const res = await page.request.get(`${APP}/api/albums/${id}`);
  expect(res.status()).toBe(200);
  return (await res.json()) as {
    id: number;
    name: string;
    description: string;
    member_total: number;
    members: { id: string }[];
  };
}

test.describe('album duplicate', () => {
  test('the album page carries a Duplicate control', async ({ page }) => {
    const album = await createAlbumWithPhotos(page, `e2e-dup-ctl-${Date.now()}`);
    try {
      await page.goto(`${APP}/albums/${album.id}`);
      await page.waitForSelector('.head h1', { timeout: 30000 });

      const dup = page
        .locator('.head button')
        .filter({ hasText: /^duplicate$/i })
        .first();
      await expect(dup).toBeVisible();
      // It names the album it will copy, so the control is legible
      // on hover rather than an unlabelled "Duplicate".
      await expect(dup).toHaveAttribute('title', /Copy album .+ and its photos/);
    } finally {
      await deleteAlbum(page, album.id);
    }
  });

  test('it copies the photos, the description, and names the copy', async ({
    page
  }) => {
    const album = await createAlbumWithPhotos(page, `e2e-dup-copy-${Date.now()}`);
    try {
      await page.goto(`${APP}/albums/${album.id}`);
      await page.waitForSelector('.head h1', { timeout: 30000 });

      await duplicateAndWait(page, `${album.name} (copy)`);

      // The copy exists as its own album, with the photos in it.
      const list = await page.request.get(`${APP}/api/albums`);
      const body = (await list.json()) as { albums: { id: number; name: string }[] };
      const copy = body.albums.find((a) => a.name === `${album.name} (copy)`);
      expect(copy, 'the copy must exist as its own album').toBeTruthy();

      const detail = await albumDetail(page, copy!.id);
      expect(detail.id).not.toBe(album.id);
      expect(detail.member_total).toBe(album.memberIds.length);
      expect(detail.members.map((m) => m.id).sort()).toEqual(
        [...album.memberIds].sort()
      );
      expect(detail.description).toBe('natural light only');

      await deleteAlbum(page, copy!.id);
    } finally {
      await deleteAlbum(page, album.id);
    }
  });

  test('duplicating twice does not collide on the name', async ({ page }) => {
    const album = await createAlbumWithPhotos(page, `e2e-dup-twice-${Date.now()}`);
    try {
      await page.goto(`${APP}/albums/${album.id}`);
      await page.waitForSelector('.head h1', { timeout: 30000 });

      // The first copy is "(copy)". The second one comes from
      // duplicating the ORIGINAL again — duplicating the copy would
      // make "X (copy) (copy)", which is a different thing.
      await duplicateAndWait(page, `${album.name} (copy)`);
      await page.goto(`${APP}/albums/${album.id}`);
      await expect(page.locator('.head h1')).toHaveText(album.name);
      await duplicateAndWait(page, `${album.name} (copy 2)`);

      const list = await page.request.get(`${APP}/api/albums`);
      const body = (await list.json()) as { albums: { id: number; name: string }[] };
      const names = body.albums.map((a) => a.name);
      expect(names).toContain(`${album.name} (copy)`);
      expect(names).toContain(`${album.name} (copy 2)`);

      for (const a of body.albums.filter(
        (x) => x.name.startsWith(`${album.name} (copy`)
      )) {
        await deleteAlbum(page, a.id);
      }
    } finally {
      await deleteAlbum(page, album.id);
    }
  });

  test('the original album is left alone', async ({ page }) => {
    const album = await createAlbumWithPhotos(page, `e2e-dup-orig-${Date.now()}`);
    try {
      await page.goto(`${APP}/albums/${album.id}`);
      await page.waitForSelector('.head h1', { timeout: 30000 });

      await duplicateAndWait(page, `${album.name} (copy)`);

      const original = await albumDetail(page, album.id);
      expect(original.name).toBe(album.name);
      expect(original.member_total).toBe(album.memberIds.length);
      expect(original.members.map((m) => m.id).sort()).toEqual(
        [...album.memberIds].sort()
      );
    } finally {
      await deleteAlbum(page, album.id);
    }
  });

  test('the user lands on the copy, and the page shows the copy', async ({
    page
  }) => {
    const album = await createAlbumWithPhotos(page, `e2e-dup-land-${Date.now()}`);
    try {
      await page.goto(`${APP}/albums/${album.id}`);
      await page.waitForSelector('.head h1', { timeout: 30000 });

      await duplicateAndWait(page, `${album.name} (copy)`);

      // The URL moved to the copy...
      const url = page.url();
      expect(url).not.toContain(`/albums/${album.id}`);

      // ...and the page shows the COPY's name, not the album it was
      // copied from. The album page used to load only on mount, so
      // a navigation between two albums kept the old album's
      // content on screen.
      await expect(page.locator('.head h1')).toHaveText(`${album.name} (copy)`);
    } finally {
      await deleteAlbum(page, album.id);
    }
  });
});
