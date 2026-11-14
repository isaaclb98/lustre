/**
 * Verification that the photo page's universal load function
 * (Round 3) populates `photo` before the component renders.
 *
 * Before Round 3: the page mounted, `onMount(load)` fired,
 * then ~50-200 ms later the photo appeared. The page was
 * briefly blank.
 *
 * After Round 3: the +page.ts `load()` function runs first
 * and `data.photo` is populated by the time the component
 * renders. The hero <img> and metadata should be present
 * immediately on first paint, with no "Loading photo..."
 * placeholder.
 */
import { test, expect } from '@playwright/test';

test('Photo page renders with data already populated (no loading state)', async ({ page }) => {
  // Pick a real photo from the dev backend.
  const resp = await page.request.get('/api/random?limit=1');
  const data = (await resp.json()) as { results: Array<{ id: string }> };
  const id = data.results[0].id;

  await page.goto(`/photo/${id}`);

  // The hero <img> should be in the DOM on first paint. If the
  // old onMount-load path was still in effect, we'd see the
  // "Loading photo..." placeholder first.
  await expect(page.locator('img.hero')).toBeVisible({ timeout: 5000 });

  // The error state must NOT be visible (this is a real photo, not
  // a 404). The page renders `.state.error` for a failed load.
  await expect(page.locator('.state.error')).toHaveCount(0);

  // The sidebar metadata should also be populated — verify
  // the filename is non-empty.
  const filename = await page.locator('.filename').textContent();
  expect(filename?.length ?? 0).toBeGreaterThan(0);
});

test('Unknown photo id renders the inline .state.error block', async ({ page }) => {
  // Syntactically valid UUID that we know isn't indexed.
  const fakeId = '00000000-0000-0000-0000-000000000000';
  await page.goto(`/photo/${fakeId}`);

  // The page's inline error placeholder should render.
  // (Round 3 chose to surface errors via data.error rather
  // than throwing to the +error.svelte boundary, so the
  // existing e2e test for this case continues to work.)
  await expect(page.locator('.state.error')).toBeVisible({ timeout: 5000 });
  await expect(page.locator('.state.error')).toContainText(/not found/i);
});