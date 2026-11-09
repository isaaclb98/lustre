import { test, expect, type Page } from '@playwright/test';
const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';

/**
 * Regression: tile chrome (score pill, remove button, fav heart,
 * neg badge) must paint above a LOADED thumbnail, not just during the
 * blurhash phase.
 *
 * The bug: .ph (blurhash) is z:0, .full (thumbnail) is z:1, and the
 * chrome pills had z:auto — so they painted UNDER the thumbnail once
 * it loaded. Fix: chrome gets z:2 (matching .remove-btn, which already
 * documented this stacking requirement).
 *
 * Verification uses computed z-index + visibility rather than
 * elementFromPoint: .remove-btn shares the .fav corner slot at the
 * same z and would win the hit-test even while opacity:0.
 */
async function assertChromeAboveLoadedThumb(page: Page, sel: string) {
  await page.waitForSelector(`.tile.loaded ${sel}`, { timeout: 30000 });
  return page.evaluate((s) => {
    const tile = document.querySelector(`.tile.loaded ${s}`)!.closest('.tile')!;
    const chrome = tile.querySelector(s) as HTMLElement;
    const img = tile.querySelector('img.full') as HTMLImageElement;
    const cs = getComputedStyle(chrome);
    return {
      chromeZ: parseInt(cs.zIndex, 10),
      imgZ: parseInt(getComputedStyle(img).zIndex, 10),
      imgLoaded: img.complete && img.naturalWidth > 0,
      chromeOpacity: parseFloat(cs.opacity),
      chromeVisible: cs.visibility !== 'hidden' && cs.display !== 'none',
    };
  }, sel);
}

test('score pill paints above loaded thumbnail', async ({ page }) => {
  await page.goto(`${APP}/?positives=portrait&diversity=balanced`);
  await page.waitForSelector('.tile.loaded .score', { timeout: 30000 });
  // .score is opacity:0 until hover — hover to reveal it
  await page.locator('.tile.loaded').filter({ has: page.locator('.score') }).first().hover();
  await page.waitForTimeout(350); // opacity transition
  const r = await assertChromeAboveLoadedThumb(page, '.score');
  console.log('score:', JSON.stringify(r));
  expect(r.imgLoaded).toBeTruthy();
  expect(r.chromeZ).toBeGreaterThan(r.imgZ);
  expect(r.chromeVisible).toBeTruthy();
  expect(r.chromeOpacity).toBe(1);
  await page.screenshot({ path: '/tmp/zfix_score.png' });
});

test('remove button paints above loaded thumbnail', async ({ page }) => {
  // The remove button is the chrome that owns the top-right corner
  // slot on the favourites grid: every photo there is a favourite by
  // definition, so the fav heart would mark nothing, and it was
  // covering the control that does something.
  await page.goto(`${APP}/albums/favourites`);
  // It is hover-revealed, so hover the tile to bring it to opacity 1.
  await page.locator('.tile.loaded').filter({ has: page.locator('button.remove-btn') }).first().hover();
  await page.waitForTimeout(350); // opacity transition
  const r = await assertChromeAboveLoadedThumb(page, 'button.remove-btn');
  console.log('remove:', JSON.stringify(r));
  expect(r.imgLoaded).toBeTruthy();
  expect(r.chromeZ).toBeGreaterThan(r.imgZ);
  expect(r.chromeVisible).toBeTruthy();
  expect(r.chromeOpacity).toBe(1);
  await page.screenshot({ path: '/tmp/zfix_remove.png' });
});

test('fav heart paints above loaded thumbnail', async ({ page, request }) => {
  // The fav heart still renders on grids that mix favourites with
  // non-favourites — the home search grid. (The favourites and album
  // grids no longer show it: every photo there is a favourite by
  // definition, so the badge would mark nothing while covering the
  // remove button that does something.)
  //
  // The photo is favourited BEFORE the page is loaded, not after.
  // Favouriting changes the search snapshot key (it digests the
  // favourites set), so a search issued before the favourite and
  // re-issued after it is a different query with a different
  // snapshot — the earlier response is what the browser would
  // replay. Favouriting first means the very first request already
  // reflects it.
  const search = await request.get(`${APP}/api/search?positives=portrait&limit=24&offset=0`);
  expect(search.ok()).toBeTruthy();
  const { results } = await search.json();
  expect(results?.length, 'search should return results').toBeGreaterThan(0);
  const pointId = results[0].id as string;

  const fav = await request.post(`/api/favorites/${encodeURIComponent(pointId)}`);
  expect(fav.ok()).toBeTruthy();
  try {
    await page.goto(`${APP}/?positives=portrait`);
    await page.waitForSelector('.tile.loaded .fav', { timeout: 30000 });
    const r = await assertChromeAboveLoadedThumb(page, '.fav');
    console.log('fav:', JSON.stringify(r));
    expect(r.imgLoaded).toBeTruthy();
    expect(r.chromeZ).toBeGreaterThan(r.imgZ);
    expect(r.chromeVisible).toBeTruthy();
    await page.screenshot({ path: '/tmp/zfix_fav.png' });
  } finally {
    await request.delete(`/api/favorites/${encodeURIComponent(pointId)}`);
  }
});
