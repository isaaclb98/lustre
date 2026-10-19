import { test, expect, type Page } from '@playwright/test';

/**
 * Full-user-experience tests. These exercise the actual app flows:
 * navigate to a page, type into the composer, click Search, scroll for
 * more, click into a photo, etc. Each test stands up its own context
 * so they're order-independent.
 *
 * Selectors used here:
 *   .grid-tile    — wrapper rendered by SearchGrid for each visible tile
 *   .tile         — the <a> inside the wrapper (what gets clicked)
 *   .chip         — a prompt chip rendered by PromptChips
 *   button.search — the Search submit button (varies, found by role)
 */

async function appReady(page: Page) {
  await page.waitForSelector('header.topbar', { timeout: 10000 });
}

async function gotoSearch(page: Page, positives: string[] = []) {
  // / IS the search route now (the /search route was removed).
  // URL params (positives) drive the search bar.
  const qs = positives.flatMap((p) => `positives=${encodeURIComponent(p)}`).join('&');
  await page.goto(`/${qs ? '?' + qs : ''}`);
  await appReady(page);
}

async function searchFor(page: Page, prompts: string[]) {
  const input = page.locator('.composer-input').first();
  for (const p of prompts) {
    await input.fill(p);
    // Enter submits the prompt as a chip; the Search button fires the query
    await input.press('Enter');
  }
  await page.getByRole('button', { name: /^Search$/ }).first().click();
}

test.describe('Full User Experience E2E Tests', () => {
  test('Homepage loads and shows composer + top-bar tabs', async ({ page }) => {
    await page.goto('/');
    await appReady(page);
    await expect(page).toHaveTitle(/lustre/);
    await expect(page.locator('.composer-input').first()).toBeVisible();
    // Current topbar tabs: Home, Random, For You, Albums, Settings.
    // (The earlier 'Search' tab was removed when /search became the
    // home route — there's no longer a dedicated 'Search' link.)
    await expect(page.getByRole('link', { name: /Home/i }).first()).toBeVisible();
    await expect(page.getByRole('link', { name: /Random/i }).first()).toBeVisible();
    await expect(page.getByRole('link', { name: /For You/i }).first()).toBeVisible();
  });

  test('Search: positive prompt + Search button renders results', async ({ page }) => {
    await gotoSearch(page);
    await searchFor(page, ['beach']);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10_000 });
    expect(await page.locator('.grid-tile').count()).toBeGreaterThan(0);
  });

  test('Photo detail (lightbox) opens from a tile click', async ({ page }) => {
    await gotoSearch(page);
    await searchFor(page, ['cat']);
    const firstTile = page.locator('.grid-tile').first();
    await expect(firstTile).toBeVisible({ timeout: 10_000 });
    await firstTile.click();
    await expect(page.locator('[role="dialog"]')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('.photo')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.locator('[role="dialog"]')).toBeHidden({ timeout: 3000 });
  });

  test('Favorites: like via API then favorites page renders', async ({ page }) => {
    // Get a real point id from the search API
    const apiRes = await page.request.get('/api/search?limit=1&positives=mountain');
    const apiJson = await apiRes.json();
    const pointId = apiJson.results?.[0]?.id;
    if (!pointId) test.skip(true, 'No search results to test against');

    // Like it via the API (UI requires hover to expose the button;
    // exercising the full hover state is fragile in headless Chromium)
    const likeResp = await page.request.post(`/api/favorites/${pointId}`);
    expect([200, 201, 204]).toContain(likeResp.status());

    await page.goto('/favorites');
    await appReady(page);
    const bodyText = await page.textContent('body');
    expect(bodyText).toBeTruthy();
  });

  test('Similar photos from a point ID', async ({ page }) => {
    // Get a point id from the search API
    const apiRes = await page.request.get('/api/search?limit=1&positives=mountain');
    const apiJson = await apiRes.json();
    const pointId = apiJson.results?.[0]?.id;
    if (!pointId) test.skip(true, 'No search results to test against');
    await page.goto(`/similar/${pointId}`);
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10_000 });
  });

  test('Random page loads tiles', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10_000 });
  });

  test('For You page loads (round 33: shuffled pool, no diversity controls)', async ({ page }) => {
    await page.goto('/for-you');
    await appReady(page);
    await expect(page.getByRole('heading', { name: /For you/i })).toBeVisible();
    // Round 33 dropped the diversity controls from this page —
    // the new endpoint takes `top_pct`/`page`/`limit`, no
    // user-facing tuning of diversity rerank. Confirm there's no
    // legacy combobox visible.
    await expect(page.getByRole('combobox', { name: /Diversity mode/i })).toHaveCount(0);
  });

  test('Albums page loads', async ({ page }) => {
    await page.goto('/albums');
    await appReady(page);
    // h1 is exactly 'Albums'; the page also has an h2 'Your albums'
    // for the album grid section. Scope to h1 (exact) so we don't
    // match the h2 in strict mode.
    await expect(page.getByRole('heading', { name: 'Albums', exact: true })).toBeVisible();
  });

  test('Navigation: all main routes render without server error', async ({ page }) => {
    const paths = ['/', '/search', '/random', '/for-you', '/albums', '/favorites'];
    for (const p of paths) {
      const resp = await page.goto(p);
      expect(resp?.status() ?? 0).toBeLessThan(500);
      await appReady(page);
    }
  });

  test('Search page with no prompts renders the For You row (default), no crash', async ({ page }) => {
    // / IS the search route. With no prompts in the URL, the page
    // shows the "For you" recommendations row rather than crashing
    // or showing a blank screen. The earlier expectation of an
    // `.empty` element was based on a now-removed empty-state UI;
    // For You is the current landing default.
    await gotoSearch(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10_000 });
    // The "For you" section heading should be visible.
    await expect(page.getByRole('heading', { name: /For you/i })).toBeVisible();
  });

  test('Image raw endpoint serves JPEG bytes for a search result', async ({ page }) => {
    await gotoSearch(page);
    await searchFor(page, ['cat']);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10_000 });
    // Grab the first photo id (the tile's <a> links to the SPA photo
    // page, not the raw image). The raw image endpoint is
    // /photo/<id>/raw (see photoUrl() in lib/api/client.ts).
    const photoHref = await page.locator('.tile').first().getAttribute('href');
    expect(photoHref).toBeTruthy();
    const id = photoHref!.split('/').pop();
    const rawUrl = `/photo/${encodeURIComponent(id!)}/raw`;
    const resp = await page.request.get(rawUrl);
    expect(resp.status()).toBe(200);
    const ct = resp.headers()['content-type'] || '';
    expect(ct).toMatch(/image\/(jpeg|png|webp)/);
  });

  test('Thumbnail endpoint serves WebP for a search result', async ({ page }) => {
    const apiRes = await page.request.get('/api/search?limit=1&positives=cat');
    const apiJson = await apiRes.json();
    const pointId = apiJson.results?.[0]?.id;
    if (!pointId) test.skip(true, 'No search results to test against');
    const resp = await page.request.get(`/thumb/${pointId}`);
    expect(resp.status()).toBe(200);
    expect(resp.headers()['content-type'] || '').toMatch(/webp|image/);
  });

  test('Search state round-trips through URL params', async ({ page }) => {
    await gotoSearch(page);
    await searchFor(page, ['beach', 'sunset']);
    // URL should now carry positives=beach&positives=sunset (in some order)
    const url = new URL(page.url());
    const positives = url.searchParams.getAll('positives');
    expect(new Set(positives)).toEqual(new Set(['beach', 'sunset']));
  });
});