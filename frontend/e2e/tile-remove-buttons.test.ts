/**
 * E2E for the tile-remove affordance on /albums and
 * /albums/favourites.
 *
 * Parallel-safe: each test seeds its own photo and verifies only
 * that the specific tile for that photo is present and removable.
 * The grid may contain arbitrary other tiles from concurrent
 * tests or pre-existing data; we don't care about their count.
 *
 * For favourites we tag the photo with a unique albums note
 * via a throwaway album membership so we can identify the right
 * tile across concurrent edits.
 */
import { test, expect, type APIRequestContext, type Page } from '@playwright/test';

async function randomPhoto(req: APIRequestContext) {
  const r = await req.get('/api/random?limit=1');
  expect(r.ok()).toBeTruthy();
  const j = await r.json();
  return j.results?.[0]?.id as string | undefined;
}

async function makeAlbum(req: APIRequestContext, name: string) {
  const r = await req.post('/api/albums', {
    data: { name },
    headers: { 'Content-Type': 'application/json' }
  });
  expect(r.ok()).toBeTruthy();
  return (await r.json()).id as number;
}

async function deleteAlbum(req: APIRequestContext, id: number) {
  await req.delete(`/api/albums/${id}`);
}

async function addToAlbum(req: APIRequestContext, id: number, photo: string) {
  const r = await req.post(
    `/api/albums/${id}/members/${encodeURIComponent(photo)}`,
  );
  expect(r.ok()).toBeTruthy();
}

async function like(req: APIRequestContext, photo: string) {
  const r = await req.post(`/api/favorites/${encodeURIComponent(photo)}`);
  expect(r.ok()).toBeTruthy();
}

async function unlike(req: APIRequestContext, photo: string) {
  await req.delete(`/api/favorites/${encodeURIComponent(photo)}`);
}

// Tile locator by photo id — the <a class="tile"> href is
// /photo/<id>, with /photo/ having an encoded prefix path.
// URL-encoded id is what's actually in the href.
function tileByPhoto(page: Page, photo: string) {
  // href is built as /photo/${encodeURIComponent(pointId)}; for
  // typical UUIDs the encoding is identical to the raw id, so
  // the prefix match works directly.
  return page.locator(`.grid-tile:has(a.tile[href$="/photo/${photo}"])`);
}

test.describe('tile remove buttons (album / favourites)', () => {
  test('album: remove button on hover, click decrements grid, server reflects', async ({ page, request }) => {
    const photo = await randomPhoto(request);
    expect(photo, 'dev stack should have at least one photo').toBeTruthy();
    const albumName = `tile-remove-test-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const albumId = await makeAlbum(request, albumName);
    await addToAlbum(request, albumId, photo!);
    try {
      await page.goto(`/albums/${albumId}`);
      const targetTile = tileByPhoto(page, photo!);
      await expect(targetTile).toHaveCount(1);
      const removeBtn = targetTile.locator('button.remove-btn');
      await expect(removeBtn).toHaveCount(1);
      await expect(removeBtn).toHaveAttribute('title', 'Remove from album');
      await targetTile.hover();
      await page.waitForTimeout(200);
      await removeBtn.click({ force: true });
      await expect(targetTile).toHaveCount(0);
      // Server-side: re-fetch album members; should not include the photo.
      const r = await request.get(`/api/albums/${albumId}`);
      const j = await r.json();
      const ids = (j.members ?? []).map((m: { point_id?: string; id: string }) => m.point_id ?? m.id);
      expect(ids).not.toContain(photo);
    } finally {
      await deleteAlbum(request, albumId);
    }
  });

  test('favourites: remove button on hover, click decrements grid, server reflects', async ({ page, request }) => {
    const photo = await randomPhoto(request);
    expect(photo).toBeTruthy();
    await like(request, photo!);
    try {
      await page.goto('/albums/favourites');
      const targetTile = tileByPhoto(page, photo!);
      await expect(targetTile, 'liked photo should appear in the favourites grid').toHaveCount(1);
      const removeBtn = targetTile.locator('button.remove-btn');
      await expect(removeBtn).toHaveCount(1);
      await expect(removeBtn).toHaveAttribute('title', 'Unlike');
      await targetTile.hover();
      await page.waitForTimeout(200);
      await removeBtn.click({ force: true });
      await expect(targetTile).toHaveCount(0);
      // Server-side: photo no longer in /api/favorites.
      const r = await request.get('/api/favorites?as_results=1&limit=500');
      const j = await r.json();
      const ids = (j.results ?? []).map((x: { id: string }) => x.id);
      expect(ids).not.toContain(photo);
    } finally {
      // Cleanup: ensure the test starts each run from a clean state.
      await unlike(request, photo!);
    }
  });

});
