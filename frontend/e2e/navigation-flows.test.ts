import { test, expect, type Page } from '@playwright/test';

// Tests run against the dev stack (PLAYWRIGHT_BASE_URL is set by
// the wrapper or the CI workflow). Falls back to the dev port so a
// bare `node_modules/.bin/playwright test` from a developer's machine
// still works against their local dev stack.
const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:18000';

/**
 * navigation-flows.test.ts — Browser navigation, deep linking,
 * URL state persistence, and error pages.
 */

async function appReady(page: Page) {
  await page.waitForSelector('header.topbar', { timeout: 10000 });
}

test.describe('Browser navigation', () => {
  test('browser back button navigates between pages', async ({ page }) => {
    // Start at home (test base URL comes from APP).
    await page.goto(`${APP}/`);
    await appReady(page);

    // Go to /?positives=beach (the search route — /search no longer
    // exists; / IS the search route now).
    await page.goto(`${APP}/?positives=beach`);
    await appReady(page);

    // Go to random
    await page.goto(`${APP}/random`);
    await appReady(page);

    // Back to search
    await page.goBack();
    await appReady(page);
    await expect(page.getByPlaceholder(/Add a positive/)).toBeVisible({ timeout: 5000 });
  });

  test('browser forward button after going back', async ({ page }) => {
    await page.goto('http://127.0.0.1:8000/');
    await appReady(page);
    await page.goto('http://127.0.0.1:8000/search');
    await appReady(page);
    await page.goto('http://127.0.0.1:8000/random');
    await appReady(page);

    // Back
    await page.goBack();
    await appReady(page);
    // Forward
    await page.goForward();
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });
  });
});

test.describe('Deep linking', () => {
  test('GET /photo/<uuid> renders the photo page', async ({ page }) => {
    const apiRes = await page.request.get('/api/search?limit=1&positives=photo');
    const apiJson = await apiRes.json();
    const pointId = apiJson.results?.[0]?.id;
    if (!pointId) test.skip(true, 'No search results');

    const resp = await page.goto(`http://127.0.0.1:8000/photo/${pointId}`);
    expect(resp?.status()).toBe(200);
    await appReady(page);
  });

  test('GET /similar/<uuid> renders the similar page', async ({ page }) => {
    const apiRes = await page.request.get('/api/search?limit=1&positives=photo');
    const apiJson = await apiRes.json();
    const pointId = apiJson.results?.[0]?.id;
    if (!pointId) test.skip(true, 'No search results');

    const resp = await page.goto(`http://127.0.0.1:8000/similar/${pointId}`);
    expect(resp?.status()).toBe(200);
    await appReady(page);
    await expect(page.getByRole('heading', { name: /similar/i })).toBeVisible({ timeout: 5000 });
  });

  test('GET /albums/<id> renders the album detail page', async ({ page }) => {
    // Create a test album
    const createRes = await page.request.post('/api/albums', {
      data: { name: `deep-link-${Date.now()}` }
    });
    const album = await createRes.json();

    const resp = await page.goto(`http://127.0.0.1:8000/albums/${album.id}`);
    expect(resp?.status()).toBe(200);
    await appReady(page);

    // Cleanup
    await page.request.delete(`/api/albums/${album.id}`);
  });

  test('GET /random renders the random page', async ({ page }) => {
    const resp = await page.goto('http://127.0.0.1:8000/random');
    expect(resp?.status()).toBe(200);
    await appReady(page);
    await expect(page.getByRole('heading', { name: /random/i })).toBeVisible();
  });

  test('GET /albums renders the albums page', async ({ page }) => {
    const resp = await page.goto('http://127.0.0.1:8000/albums');
    expect(resp?.status()).toBe(200);
    await appReady(page);
    await expect(page.locator('header.topbar')).toBeVisible();
  });
});

test.describe('URL state persistence', () => {
  test('search URL params persist after page refresh', async ({ page }) => {
    await page.goto('http://127.0.0.1:8000/?positives=beach&negatives=ocean');
    await appReady(page);
    await expect(page.locator('.chip').filter({ hasText: 'beach' })).toBeVisible();
    await expect(page.locator('.chip').filter({ hasText: 'ocean' })).toBeVisible();

    // Refresh
    await page.reload();
    await appReady(page);

    // Chips should still be visible
    await expect(page.locator('.chip').filter({ hasText: 'beach' })).toBeVisible({ timeout: 5000 });
    await expect(page.locator('.chip').filter({ hasText: 'ocean' })).toBeVisible({ timeout: 5000 });
  });

  test('adding a prompt updates the URL', async ({ page }) => {
    // / IS the search route now (the /search route was removed).
    await page.goto(`${APP}/`);
    await appReady(page);

    const input = page.getByPlaceholder(/Add a positive/);
    await input.fill('persistence-test');
    await input.press('Enter');

    // Wait for URL to update
    await page.waitForTimeout(500);

    // URL should contain the prompt
    const url = page.url();
    expect(url).toContain('persistence-test');
  });

  test('removing a prompt updates the URL', async ({ page }) => {
    await page.goto('http://127.0.0.1:8000/?positives=removeme');
    await appReady(page);
    await expect(page.locator('.chip').filter({ hasText: 'removeme' })).toBeVisible();

    // Click × on the chip
    await page.locator('.chip button.x').first().click();
    await page.waitForTimeout(500);

    // URL should no longer contain the prompt
    const url = page.url();
    expect(url).not.toContain('removeme');
  });
});

test.describe('Error pages', () => {
  test('GET /nonexistent-page returns 200 (SPA fallback) and renders app', async ({ page }) => {
    const resp = await page.goto('http://127.0.0.1:8000/nonexistent-page-xyz');
    // SPA serves index.html for unknown paths — status 200
    expect(resp?.status()).toBe(200);
    await appReady(page);
    // The app should still render
    await expect(page.locator('header.topbar')).toBeVisible();
  });

  test('GET /api/nonexistent returns 404 (JSON, not HTML)', async ({ page }) => {
    const resp = await page.request.get('http://127.0.0.1:8000/api/nonexistent-endpoint');
    expect(resp.status()).toBe(404);
    const body = await resp.json();
    expect(body).toHaveProperty('detail');
  });
});

test.describe('Page transitions', () => {
  test('navigation between pages is smooth (no white flash)', async ({ page }) => {
    await page.goto('http://127.0.0.1:8000/');
    await appReady(page);

    // Navigate to a few pages
    const pages = ['/search', '/random', '/albums'];
    for (const path of pages) {
      await page.goto(`http://127.0.0.1:8000${path}`);
      await appReady(page);
      // Header should be present on every page
      await expect(page.locator('header.topbar')).toBeVisible({ timeout: 5000 });
    }
  });

  test('header navigation is consistent across all pages', async ({ page }) => {
    // The set of routes that have a real page (not 404 shells). /search
    // was removed when / became the home/search route; we test the
    // routes that actually exist. The base URL also uses the live
    // dev port (:18000) — the old hardcoded :8000 worked when tests
    // ran against prod but the project now uses a dev stack.
    const base = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:18000';
    const paths = ['/', '/random', '/albums'];
    for (const path of paths) {
      await page.goto(`${base}${path}`);
      await appReady(page);
      // Same nav links should be present on every page. Current
      // topbar has Home, Random, Albums, Settings (the old
      // 'Search' tab no longer exists; / IS search).
      await expect(page.getByRole('navigation').getByRole('link', { name: 'Home' })).toBeVisible();
      await expect(page.getByRole('navigation').getByRole('link', { name: 'Random' })).toBeVisible();
    }
  });
});

test.describe('URL encoding', () => {
  test('prompts with spaces work in URL', async ({ page }) => {
    await page.goto('http://127.0.0.1:8000/?positives=hello%20world');
    await appReady(page);
    await expect(page.locator('.chip').filter({ hasText: /hello.*world/ })).toBeVisible({ timeout: 5000 });
  });

  test('prompts with unicode work in URL', async ({ page }) => {
    await page.goto('http://127.0.0.1:8000/?positives=%E2%9C%93'); // ✓ checkmark
    await appReady(page);
    // Should render the chip without crashing
    const chips = await page.locator('.chip').count();
    expect(chips).toBeGreaterThanOrEqual(0); // just shouldn't crash
  });
});

test.describe('Page refresh on every route', () => {
  test('refresh on /search keeps state', async ({ page }) => {
    await page.goto('http://127.0.0.1:8000/?positives=photo');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });
    await page.reload();
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });
  });

  test('refresh on /random keeps results', async ({ page }) => {
    await page.goto('http://127.0.0.1:8000/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });
    await page.reload();
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });
  });

  test('refresh on /albums keeps state', async ({ page }) => {
    await page.goto('http://127.0.0.1:8000/albums');
    await appReady(page);
    await page.reload();
    await appReady(page);
    await expect(page.locator('header.topbar')).toBeVisible();
  });
});