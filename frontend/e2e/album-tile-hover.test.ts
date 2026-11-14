/**
 * E2E for the album-page tile chrome and the hover motion.
 *
 * 1. The favourite heart is gone from album tiles. The top-right
 *    corner belongs to the remove button alone — on an album grid
 *    every photo is a favourite by definition, so the heart marked
 *    nothing while sitting on top of the control that does something.
 * 2. The remove button is revealed on hover only.
 * 3. Hovering a tile zooms the PHOTO inside it, and the tile itself
 *    does not move. A tile that lifts leaves a gap under itself and
 *    overlaps the row above; the zoom reads as the same "leaning in"
 *    without anything on the page shifting.
 *
 * Playwright's `Desktop Chrome` device reports hover:hover and
 * pointer:fine, which is what the hover-reveal CSS keys off. A
 * headless Chrome started without that emulation reports hover:none
 * and would match the touch fallback instead, so the assertions
 * here would be measuring the wrong branch.
 */
import { test, expect, type Page } from '@playwright/test';

import { albumWithPhotos } from './albums';

const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';

async function firstLoadedTile(page: Page) {
  await page.waitForSelector('.tile.loaded', { timeout: 30000 });
  return page.locator('.tile.loaded').first();
}

test.describe('album tile chrome + hover motion', () => {
  test('album grid: no favourite heart, remove button hidden until hover', async ({ page }) => {
    const albumId = await albumWithPhotos(page);
    await page.goto(`${APP}/albums/${albumId}`);
    const tile = await firstLoadedTile(page);

    // No heart on album tiles.
    await expect(tile.locator('.fav')).toHaveCount(0);

    // The remove button exists but is invisible at rest.
    const removeBtn = tile.locator('button.remove-btn');
    await expect(removeBtn).toHaveCount(1);
    await expect(removeBtn).toHaveAttribute('title', 'Remove from album');
    await expect(removeBtn).toHaveCSS('opacity', '0');

    // It is hidden by opacity, so it must not be clickable either.
    await expect(removeBtn).toHaveCSS('pointer-events', 'none');

    // Hovering reveals it.
    await tile.hover();
    await expect(removeBtn).toHaveCSS('opacity', '1');
  });

  test('album grid: hover zooms the photo, the tile does not move', async ({ page }) => {
    const albumId = await albumWithPhotos(page);
    await page.goto(`${APP}/albums/${albumId}`);
    const tile = await firstLoadedTile(page);

    const before = await tile.evaluate((el) => {
      const r = el.getBoundingClientRect();
      const img = el.querySelector('img.full') as HTMLImageElement;
      return {
        tileTop: Math.round(r.top),
        tileLeft: Math.round(r.left),
        tileWidth: Math.round(r.width),
        tileHeight: Math.round(r.height),
        // The image carries an explicit `transform: scale(1)` so it has
        // a baseline to transition FROM — computed as an identity
        // matrix, not `none`.
        imgScale: new DOMMatrix(getComputedStyle(img).transform).a
      };
    });

    // Rest: no zoom, no movement.
    expect(before.imgScale).toBeCloseTo(1, 3);

    await tile.hover();
    // Let the 100ms transition settle.
    await page.waitForTimeout(300);

    const after = await tile.evaluate((el) => {
      const r = el.getBoundingClientRect();
      const img = el.querySelector('img.full') as HTMLImageElement;
      const tileStyle = getComputedStyle(el);
      return {
        tileTop: Math.round(r.top),
        tileLeft: Math.round(r.left),
        tileWidth: Math.round(r.width),
        tileHeight: Math.round(r.height),
        tileTransform: tileStyle.transform,
        tileTransitionDuration: tileStyle.transitionDuration,
        imgScale: new DOMMatrix(getComputedStyle(img).transform).a,
        imgTransitionDuration: getComputedStyle(img).transitionDuration
      };
    });

    // The photo is scaled up.
    expect(after.imgScale).toBeGreaterThan(1);
    expect(after.imgScale).toBeLessThanOrEqual(1.1);

    // The tile itself is unmoved — no jump, no lift.
    expect(after.tileTransform).toBe('none');
    expect(after.tileTop).toBe(before.tileTop);
    expect(after.tileLeft).toBe(before.tileLeft);
    expect(after.tileWidth).toBe(before.tileWidth);
    expect(after.tileHeight).toBe(before.tileHeight);

    // The motion is snappy.
    expect(after.imgTransitionDuration).toContain('0.1s');
  });

  test('hovering one tile does not disturb its neighbours', async ({ page }) => {
    const albumId = await albumWithPhotos(page);
    await page.goto(`${APP}/albums/${albumId}`);
    const tile = await firstLoadedTile(page);

    const neighbourBefore = await page
      .locator('.tile.loaded')
      .nth(1)
      .evaluate((el) => {
        const r = el.getBoundingClientRect();
        return { top: Math.round(r.top), left: Math.round(r.left) };
      });

    await tile.hover();
    await page.waitForTimeout(300);

    const neighbourAfter = await page
      .locator('.tile.loaded')
      .nth(1)
      .evaluate((el) => {
        const r = el.getBoundingClientRect();
        return { top: Math.round(r.top), left: Math.round(r.left) };
      });

    expect(neighbourAfter).toEqual(neighbourBefore);
  });
});
