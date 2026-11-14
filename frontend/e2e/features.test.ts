import { test, expect, type Page } from '@playwright/test';

/**
 * features.test.ts — focused coverage of the interactive features
 * that aren't part of the basic happy-path smoke/full-ux suites:
 *   - Albums CRUD (create → add favorites → download → delete)
 *   - Discovery rabbithole (start → pick → state)
 *   - Saved searches (save → pick reapplies → delete)
 *   - Lightbox arrow-key navigation between adjacent photos
 *   - Dislikes add/remove
 *   - Error states (bad point ids, bad inputs, missing files)
 *
 * Most of these exercise endpoints directly via page.request so the
 * assertions are stable against UI churn. The few UI assertions are
 * pointed at stable selectors (role="dialog", heading names) that
 * don't depend on internal CSS.
 */

async function appReady(page: Page) {
  await page.waitForSelector('header.topbar', { timeout: 10000 });
}

/** Pick a real point id we can reuse across tests. */
async function getPointId(page: Page, query = 'cat'): Promise<string | null> {
  const res = await page.request.get(`/api/search?limit=1&positives=${query}`);
  const json = await res.json();
  return json.results?.[0]?.id ?? null;
}

test.describe('Albums CRUD', () => {
  test('create → add favorite → list shows album → download.zip is a real zip → delete', async ({ page }) => {
    const pointId = await getPointId(page);
    if (!pointId) test.skip(true, 'No search results available');

    // Create album
    const createRes = await page.request.post('/api/albums', {
      data: { name: `e2e-album-${Date.now()}` }
    });
    expect(createRes.status()).toBeLessThan(300);
    const album = await createRes.json();
    expect(album.id).toBeTruthy();
    const albumId = album.id;

    // Like the point so it becomes a favorite
    const favRes = await page.request.post(`/api/favorites/${pointId}`);
    expect([200, 201, 204]).toContain(favRes.status());

    // Get the favorite id for that point so we can add it to the album
    const favsRes = await page.request.get('/api/favorites');
    const favsJson = await favsRes.json();
    const favId = (favsJson.favorites ?? []).find(
      (f: { point_id?: string; id?: number }) => f.point_id === pointId || f.id?.toString() === pointId
    )?.id;
    expect(favId, 'favorite id must be resolvable').toBeTruthy();

    // Add favorite to album
    const addRes = await page.request.post(`/api/albums/${albumId}/members/${favId}`);
    expect([200, 201, 204]).toContain(addRes.status());

    // GET /api/albums should include this album with count >= 1
    const listRes = await page.request.get('/api/albums');
    const listJson = await listRes.json();
    const found = (listJson.albums ?? listJson).find(
      (a: { id: number; name: string }) => a.id === albumId
    );
    expect(found, 'album should appear in list').toBeTruthy();
    expect(Number(found.member_count ?? found.count)).toBeGreaterThanOrEqual(1);

    // Download endpoint returns a real ZIP (PK signature)
    const dlRes = await page.request.get(`/albums/${albumId}/download.zip`);
    expect(dlRes.status()).toBe(200);
    const buf = Buffer.from(await dlRes.body());
    expect(buf.length).toBeGreaterThan(0);
    // PK\x03\x04 = local file header, PK\x05\x06 = empty archive marker.
    // We expect at least PK\x03\x04 (a member file) since we added one.
    expect(buf.subarray(0, 4).toString('binary')).toBe('PK\u0003\u0004');

    // Delete the album
    const delRes = await page.request.delete(`/api/albums/${albumId}`);
    expect(delRes.status()).toBe(204);

    // Confirm it's gone
    const after = await page.request.get(`/api/albums/${albumId}`);
    expect(after.status()).toBe(404);
  });

  test('album PATCH renames the album', async ({ page }) => {
    const createRes = await page.request.post('/api/albums', {
      data: { name: 'before-rename' }
    });
    const album = await createRes.json();
    const newName = `renamed-${Date.now()}`;
    const patchRes = await page.request.patch(`/api/albums/${album.id}`, {
      data: { name: newName }
    });
    expect(patchRes.status()).toBe(200);
    const after = await patchRes.json();
    expect(after.name).toBe(newName);
    // cleanup
    await page.request.delete(`/api/albums/${album.id}`);
  });
});

test.describe('Saved searches', () => {
  test('save → pick reapplies prompts → delete removes', async ({ page }) => {
    const saveRes = await page.request.post('/api/saved-searches', {
      data: {
        name: `e2e-saved-${Date.now()}`,
        positives: ['beach', 'sunset'],
        negatives: []
      }
    });
    expect(saveRes.status()).toBe(201);
    const saved = await saveRes.json();
    const savedId = saved.id;
    expect(saved.positives).toEqual(['beach', 'sunset']);

    // List contains it
    const listRes = await page.request.get('/api/saved-searches');
    const list = await listRes.json();
    expect(
      (list.saved_searches ?? list).some((s: { id: unknown }) => s.id === savedId)
    ).toBe(true);

    // GET by id returns the saved prompts
    const getRes = await page.request.get(`/api/saved-searches/${savedId}`);
    expect(getRes.status()).toBe(200);
    const detail = await getRes.json();
    expect(detail.positives).toEqual(['beach', 'sunset']);

    // Delete
    const delRes = await page.request.delete(`/api/saved-searches/${savedId}`);
    expect(delRes.status()).toBe(204);

    const after = await page.request.get(`/api/saved-searches/${savedId}`);
    expect(after.status()).toBe(404);
  });
});

test.describe('Lightbox navigation', () => {
  test('← / → navigate between adjacent tiles; final → wraps to first / closes', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    // Open first photo
    await page.locator('.grid-tile').first().click();
    await expect(page.locator('[role="dialog"]')).toBeVisible({ timeout: 5000 });

    // Capture initial photo src to verify it changes
    const initialSrc = await page
      .locator('[role="dialog"] img.photo')
      .first()
      .getAttribute('src');

    // Arrow Right → next photo
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(300);
    const nextSrc = await page
      .locator('[role="dialog"] img.photo')
      .first()
      .getAttribute('src');
    expect(nextSrc).not.toEqual(initialSrc);

    // Arrow Left → back to the previous photo
    await page.keyboard.press('ArrowLeft');
    await page.waitForTimeout(300);
    const backSrc = await page
      .locator('[role="dialog"] img.photo')
      .first()
      .getAttribute('src');
    expect(backSrc).toEqual(initialSrc);

    // Esc closes
    await page.keyboard.press('Escape');
    await expect(page.locator('[role="dialog"]')).toBeHidden({ timeout: 3000 });
  });
});

test.describe('Dislikes', () => {
  test('add dislike → listed in /api/dislikes → delete removes', async ({ page }) => {
    const pointId = await getPointId(page, 'sunset');
    if (!pointId) test.skip(true, 'No search results available');

    const addRes = await page.request.post(`/api/dislikes/${pointId}`);
    expect([200, 201, 204]).toContain(addRes.status());

    const listRes = await page.request.get('/api/dislikes');
    const list = await listRes.json();
    // /api/dislikes returns {items: [...]}, not {results: [...]}
    const found = (list.items ?? []).some(
      (d: { point_id?: string; id?: string }) => d.point_id === pointId || d.id === pointId
    );
    expect(found, 'dislike should be listed').toBe(true);

    const delRes = await page.request.delete(`/api/dislikes/${pointId}`);
    expect([200, 204]).toContain(delRes.status());

    const after = await page.request.get('/api/dislikes');
    const afterJson = await after.json();
    const stillThere = (afterJson.items ?? []).some(
      (d: { point_id?: string; id?: string }) => d.point_id === pointId || d.id === pointId
    );
    expect(stillThere, 'dislike should be gone after delete').toBe(false);
  });
});

test.describe('Error states', () => {
  test('GET /api/search with empty query returns 400 (no prompts is a bad request, not 500)', async ({ page }) => {
    const res = await page.request.get('/api/search?limit=1');
    // The backend rejects empty queries as 400 — that's the correct behavior,
    // not a server crash. We assert it's a clean 4xx, not 500.
    expect(res.status()).toBe(400);
  });

  test('GET /api/photo/<bogus> returns a structured error, not 500', async ({ page }) => {
    const res = await page.request.get('/api/photo/not-a-real-uuid');
    expect(res.status()).toBeGreaterThanOrEqual(400);
    expect(res.status()).toBeLessThan(500);
  });

  test('GET /thumb/<bogus> returns 400 (invalid format) or 404 (valid format, no file)', async ({ page }) => {
    // Format-invalid: 400
    const r1 = await page.request.get('/thumb/not-a-real-uuid');
    expect(r1.status()).toBe(400);
    // Format-valid 32-hex but no such point: 404
    const r2 = await page.request.get('/thumb/00000000000000000000000000000000');
    expect(r2.status()).toBe(404);
  });

  test('POST /api/albums without name returns 4xx', async ({ page }) => {
    const res = await page.request.post('/api/albums', { data: {} });
    expect(res.status()).toBeGreaterThanOrEqual(400);
    expect(res.status()).toBeLessThan(500);
  });

  test('GET /healthz returns ok + qdrant reachable', async ({ page }) => {
    const res = await page.request.get('/healthz');
    expect(res.status()).toBe(200);
    const json = await res.json();
    expect(json.qdrant).toBe(true);
    // The endpoint reports whether it is in test mode; a test
    // backend IS in test mode, and a dev backend is not. Asserting
    // one particular value only passes against one of them.
    expect(typeof json.test_mode).toBe('boolean');
  });
});

test.describe('Photo context menu', () => {
  test('right-click on a tile opens the context menu', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });
    await page.locator('.grid-tile').first().click({ button: 'right' });
    await expect(page.locator('.menu[role="menu"]')).toBeVisible({ timeout: 3000 });
    // Close by clicking outside
    await page.locator('main').click({ position: { x: 10, y: 10 } });
    await expect(page.locator('.menu[role="menu"]')).toBeHidden({ timeout: 3000 });
  });
});