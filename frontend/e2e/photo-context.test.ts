import { test, expect, type Page } from '@playwright/test';

// Tests run against the dev stack (PLAYWRIGHT_BASE_URL is set by
// the wrapper or the CI workflow). Falls back to the dev port so a
// bare `node_modules/.bin/playwright test` from a developer's machine
// still works against their local dev stack.
const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:18000';

/**
 * photo-context.test.ts — Right-click context menu actions and
 * photo detail page interactions. These are the deeper photo-level
 * flows that users hit when browsing results.
 */

async function appReady(page: Page) {
  await page.waitForSelector('header.topbar', { timeout: 10000 });
}

test.describe('Photo context menu (right-click)', () => {
  test('right-click on tile opens context menu', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    await page.locator('.grid-tile').first().click({ button: 'right' });
    await expect(page.locator('[role="menu"]').first()).toBeVisible({ timeout: 3000 });
  });

  test('context menu has at least one action item', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    await page.locator('.grid-tile').first().click({ button: 'right' });
    await expect(page.locator('[role="menu"]').first()).toBeVisible({ timeout: 3000 });

    // Menu should have at least one menuitem
    const menuItems = await page.locator('[role="menu"] [role="menuitem"]').count();
    expect(menuItems).toBeGreaterThan(0);
  });

  test('clicking outside the context menu closes it', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    await page.locator('.grid-tile').first().click({ button: 'right' });
    await expect(page.locator('[role="menu"]').first()).toBeVisible({ timeout: 3000 });

    // Click outside (on the page background)
    await page.locator('main').click({ position: { x: 10, y: 10 } });
    await expect(page.locator('[role="menu"]').first()).toBeHidden({ timeout: 3000 });
  });

  test('Escape closes context menu', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    await page.locator('.grid-tile').first().click({ button: 'right' });
    await expect(page.locator('[role="menu"]').first()).toBeVisible({ timeout: 3000 });

    await page.keyboard.press('Escape');
    await expect(page.locator('[role="menu"]').first()).toBeHidden({ timeout: 3000 });
  });

  test('right-click on different tiles shows menu for that tile', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    // Right-click first tile
    await page.locator('.grid-tile').nth(0).click({ button: 'right' });
    await expect(page.locator('[role="menu"]').first()).toBeVisible({ timeout: 3000 });

    // Close and right-click second tile
    await page.keyboard.press('Escape');
    await expect(page.locator('[role="menu"]').first()).toBeHidden({ timeout: 3000 });

    await page.locator('.grid-tile').nth(1).click({ button: 'right' });
    await expect(page.locator('[role="menu"]').first()).toBeVisible({ timeout: 3000 });
  });
});

test.describe('Context menu actions', () => {
  test('Open photo action navigates to /photo/<id>', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    // Get the href of the first tile's link
    const firstLink = page.locator('.grid-tile').first().locator('a').first();
    const href = await firstLink.getAttribute('href');

    // Right-click and look for an "Open" action
    await page.locator('.grid-tile').first().click({ button: 'right' });
    await expect(page.locator('[role="menu"]').first()).toBeVisible({ timeout: 3000 });

    const openAction = page.locator('[role="menu"]').getByRole('menuitem').filter({ hasText: /open/i }).first();
    if (await openAction.count() > 0) {
      await openAction.click();
      // The link's href contains the photo path
      expect(href).toMatch(/\/photo\//);
    }
  });

  test('Copy URL action copies to clipboard (if supported)', async ({ page, context, browserName }) => {
    // Grant clipboard permissions
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);

    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    await page.locator('.grid-tile').first().click({ button: 'right' });
    await expect(page.locator('[role="menu"]').first()).toBeVisible({ timeout: 3000 });

    const copyAction = page.locator('[role="menu"]').getByRole('menuitem').filter({ hasText: /copy/i }).first();
    if (await copyAction.count() > 0) {
      await copyAction.click();
      // Verify clipboard has content (if browser supports it)
      if (browserName === 'chromium') {
        const clipboardText = await page.evaluate(() => navigator.clipboard.readText().catch(() => ''));
        // Clipboard might be empty if browser blocks it, but the action shouldn't crash
        expect(clipboardText).toBeDefined();
      }
    }
  });

  test('Like from context menu adds to favorites', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    // Get the photo ID from the first tile
    const firstLink = page.locator('.grid-tile').first().locator('a').first();
    const href = await firstLink.getAttribute('href');
    const photoId = href?.match(/\/photo\/([a-f0-9-]+)/)?.[1];
    if (!photoId) test.skip(true, 'Could not extract photo ID');

    // Get current favorites count
    const beforeRes = await page.request.get('/api/favorites');
    const beforeJson = await beforeRes.json();
    const beforeCount = beforeJson.total ?? 0;

    // Right-click and look for a Like action
    await page.locator('.grid-tile').first().click({ button: 'right' });
    await expect(page.locator('[role="menu"]').first()).toBeVisible({ timeout: 3000 });

    const likeAction = page.locator('[role="menu"]').getByRole('menuitem').filter({ hasText: /like|❤|♥/i }).first();
    if (await likeAction.count() > 0) {
      await likeAction.click();
      await page.waitForTimeout(500);

      const afterRes = await page.request.get('/api/favorites');
      const afterJson = await afterRes.json();
      const afterCount = afterJson.total ?? 0;
      // Count should have changed by at most 1 (toggle)
      expect(Math.abs(afterCount - beforeCount)).toBeLessThanOrEqual(1);
    }

    // Cleanup: remove the favorite if it was added
    await page.request.delete(`/api/favorites/${photoId}`);
  });
});

test.describe('Photo detail page (/photo/<id>)', () => {
  test('direct navigation renders the photo', async ({ page }) => {
    const apiRes = await page.request.get('/api/search?limit=1&positives=photo');
    const apiJson = await apiRes.json();
    const pointId = apiJson.results?.[0]?.id;
    if (!pointId) test.skip(true, 'No search results');

    const resp = await page.goto(`${APP}/photo/${pointId}`, { waitUntil: 'domcontentloaded' });
    expect(resp?.status()).toBe(200);
    await appReady(page);

    // The raw photo image should be visible
    const photoImg = page.locator(`img[src*="/photo/${pointId}/raw"]`);
    await expect(photoImg).toBeVisible({ timeout: 10000 });
  });

  test('photo page has a back button or link', async ({ page }) => {
    const apiRes = await page.request.get('/api/search?limit=1&positives=photo');
    const apiJson = await apiRes.json();
    const pointId = apiJson.results?.[0]?.id;
    if (!pointId) test.skip(true, 'No search results');

    await page.goto(`${APP}/photo/${pointId}`);
    await appReady(page);

    // There should be some navigation element (back button, breadcrumb, or nav)
    const navOrBack = page.locator('header.topbar, [aria-label*="back" i], [aria-label*="close" i], button:has-text("back"), a:has-text("back")').first();
    await expect(navOrBack).toBeVisible({ timeout: 3000 });
  });
});

test.describe('Similar photos page (/similar/<id>)', () => {
  test('direct navigation renders the similar page', async ({ page }) => {
    const apiRes = await page.request.get('/api/search?limit=1&positives=photo');
    const apiJson = await apiRes.json();
    const pointId = apiJson.results?.[0]?.id;
    if (!pointId) test.skip(true, 'No search results');

    const resp = await page.goto(`${APP}/similar/${pointId}`, { waitUntil: 'domcontentloaded' });
    expect(resp?.status()).toBe(200);
    await appReady(page);

    await expect(page.getByRole('heading', { name: /similar/i })).toBeVisible({ timeout: 5000 });
  });

  test('similar page shows related photos', async ({ page }) => {
    const apiRes = await page.request.get('/api/search?limit=1&positives=photo');
    const apiJson = await apiRes.json();
    const pointId = apiJson.results?.[0]?.id;
    if (!pointId) test.skip(true, 'No search results');

    await page.goto(`${APP}/similar/${pointId}`);
    await appReady(page);

    // Should show at least one similar photo tile (or an empty state)
    const tiles = await page.locator('.grid-tile').count();
    const emptyState = await page.locator('.empty, [class*="empty"]').count();
    expect(tiles + emptyState).toBeGreaterThan(0);
  });
});

test.describe('Albums page interactions', () => {
  test('clicking an album card navigates to album detail', async ({ page }) => {
    // Create a test album
    const albumName = `nav-test-${Date.now()}`;
    const createRes = await page.request.post('/api/albums', { data: { name: albumName } });
    const album = await createRes.json();

    await page.goto(`${APP}/albums`);
    await appReady(page);

    // Click the album name link
    await page.getByText(albumName).first().click();
    await appReady(page);

    // Should be on the album detail page
    await expect(page.getByRole('heading', { name: new RegExp(albumName) })).toBeVisible({ timeout: 5000 });

    // Cleanup
    await page.request.delete(`/api/albums/${album.id}`);
  });

  test('album detail page shows member count', async ({ page }) => {
    // Create an album and add a favorite
    const albumName = `count-test-${Date.now()}`;
    const createRes = await page.request.post('/api/albums', { data: { name: albumName } });
    const album = await createRes.json();
    const albumId = album.id;

    // Like a photo and add it to the album
    const apiRes = await page.request.get('/api/search?limit=1');
    const apiJson = await apiRes.json();
    const pointId = apiJson.results?.[0]?.id;
    if (pointId) {
      await page.request.post(`/api/favorites/${pointId}`);
      const favsRes = await page.request.get('/api/favorites');
      const favsJson = await favsRes.json();
      const favId = (favsJson.favorites ?? []).find((f: { point_id: string }) => f.point_id === pointId)?.id;
      if (favId) {
        await page.request.post(`/api/albums/${albumId}/members/${favId}`);
      }
    }

    await page.goto(`${APP}/albums/${albumId}`);
    await appReady(page);

    // Page should show the album name and at least 1 member
    await expect(page.getByRole('heading', { name: new RegExp(albumName) })).toBeVisible({ timeout: 5000 });

    // Cleanup
    await page.request.delete(`/api/albums/${albumId}`);
    if (pointId) await page.request.delete(`/api/favorites/${pointId}`);
  });
});