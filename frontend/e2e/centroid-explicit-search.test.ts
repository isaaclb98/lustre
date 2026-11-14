/**
 * Core E2E: centroid/album search runs ONLY on explicit Search
 * button presses — same contract as the home search (issue #6:
 * "Search runs ONLY on explicit Search button clicks").
 *
 * Regression: filter changes in centroid mode (filename,
 * collections) used to auto-fire the search (debounced for
 * filename). Now they stage into state; the request happens only
 * when the user clicks Search.
 *
 * Method: count centroid search API requests. A staged change
 * must add zero; the Search click must add exactly one, carrying
 * the staged value.
 */
import { test, expect, type Page } from '@playwright/test';

const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';

// The initial centroid search (recommend over ~2M points, possibly
// cold SigLIP2) can take longer than the default 30s test timeout.
test.setTimeout(180_000);

function centroidRequests(page: Page) {
  const urls: string[] = [];
  page.on('request', (r) => {
    const u = r.url();
    if (/\/api\/centroids\/[^/]+\/search/.test(u)) urls.push(u);
  });
  return urls;
}

async function openAlbumSearch(page: Page) {
  // Likes centroid always exists.
  await page.goto(`${APP}/?centroid=${encodeURIComponent('likes')}`);
  await page.waitForSelector('header.topbar', { timeout: 10000 });
  // Initial mount fires the first centroid search. Wait for it to
  // finish (Search button re-enables) rather than a fixed sleep —
  // a cold SigLIP2 encoder can take 15s+.
  const searchBtn = page.locator('.search-actions button', { hasText: 'Search' });
  await expect(searchBtn).toBeEnabled({ timeout: 60000 });
  await page.waitForTimeout(300); // let the grid settle
}

async function openFiltersPanel(page: Page) {
  const body = page.locator('.filters .body');
  if (!(await body.isVisible().catch(() => false))) {
    await page.locator('.filters button.head').click();
    await expect(body).toBeVisible({ timeout: 5000 });
  }
}

test('centroid mode renders a Search button', async ({ page }) => {
  await openAlbumSearch(page);
  const btn = page.locator('.search-actions button', { hasText: 'Search' });
  await expect(btn).toBeVisible();
  await expect(btn).toBeEnabled();
});

test('filename typing does not re-run until Search is pressed', async ({ page }) => {
  await openAlbumSearch(page);
  await openFiltersPanel(page);
  const urls = centroidRequests(page);
  const before = urls.length;

  // The old code debounced filename keystrokes into a reload after
  // 400ms — wait well past that and assert silence.
  await page.locator('.filters input[placeholder="e.g. IMG_2024"]').fill('IMG_2024');
  await page.waitForTimeout(1200);
  expect(urls.length, 'typing must not auto-fire').toBe(before);

  await page.locator('.search-actions button', { hasText: 'Search' }).click();
  await expect
    .poll(() => urls.length, { timeout: 15000 })
    .toBe(before + 1);
  expect(decodeURIComponent(urls[urls.length - 1])).toContain('IMG_2024');
});

test('home mode contract unchanged: Search still required, no auto-fire', async ({ page }) => {
  // Guard the sibling behaviour: home search (non-centroid) must
  // keep its explicit-run contract — filling the filename filter
  // fires nothing until Search.
  await page.goto(`${APP}/`);
  await page.waitForSelector('header.topbar', { timeout: 10000 });
  const urls: string[] = [];
  page.on('request', (r) => {
    if (/\/api\/search\?/.test(r.url())) urls.push(r.url());
  });
  await openFiltersPanel(page);
  await page.locator('.filters input[placeholder="e.g. IMG_2024"]').fill('IMG_2024');
  await page.waitForTimeout(900);
  expect(urls.length).toBe(0);
  await page.locator('.search-actions button', { hasText: 'Search' }).click();
  await expect.poll(() => urls.length, { timeout: 15000 }).toBe(1);
});
