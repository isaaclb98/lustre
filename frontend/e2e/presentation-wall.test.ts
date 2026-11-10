/**
 * E2E for the presentation wall — the album shown as a gallery.
 *
 * The wall is a gapless square grid in a random order. Two things
 * matter and both are easy to regress:
 *
 *  1. The wall's own look: no gutter, square corners, no border
 *     between tiles, so the photos touch pixel-to-pixel.
 *  2. That look is SCOPED. The gapless treatment belongs to the
 *     presentation route only — every other route keeps the
 *     app-wide grid (14px gutter, rounded corners, a border). A
 *     change that leaks the wall's gap into the rest of the app
 *     would silently restyle every grid in the product.
 *
 * The random order is also asserted: re-opening the wall must
 * produce a different order, or the "reshuffle on every open"
 * promise is broken.
 */
import { test, expect, type Page } from '@playwright/test';

const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';

/**
 * The photo ids in the order the wall rendered them.
 *
 * Only the VIRTUALISED window is in the DOM, so this is the visible
 * slice, not the whole album. After a reshuffle the visible window
 * holds DIFFERENT photos (the same photos in a different order, and
 * the window is a slice of that order) — so this list is good for
 * comparing ORDER, and the album's membership count is what proves
 * the photo set is unchanged.
 */
async function renderedOrder(page: Page): Promise<string[]> {
  return page.evaluate(() =>
    Array.from(document.querySelectorAll('.tile')).map(
      (t) => (t.getAttribute('href') ?? '').split('/photo/').pop() ?? ''
    )
  );
}

/** The album's member count, as the wall's bar reports it. */
async function albumCount(page: Page): Promise<number> {
  const text = (await page.locator('.present .count').innerText()) ?? '';
  const m = text.match(/(\d+)/);
  return m ? Number(m[1]) : 0;
}

test.describe('presentation wall', () => {
  test('renders the album as a gapless square grid', async ({ page }) => {
    await page.goto(`${APP}/albums/5/present`);
    await page.waitForSelector('.tile', { timeout: 30000 });

    const geom = await page.evaluate(() => {
      const row = document.querySelector('.grid-row')!;
      const tiles = Array.from(row.querySelectorAll('.tile')) as HTMLElement[];
      const cs = getComputedStyle(row);
      const tcs = getComputedStyle(tiles[0]);
      const a = tiles[0].getBoundingClientRect();
      const b = tiles[1].getBoundingClientRect();
      return {
        gap: cs.gap,
        radius: tcs.borderRadius,
        borderStyle: tcs.borderStyle,
        borderWidth: tcs.borderWidth,
        gapBetweenTiles: Math.round(b.x - (a.x + a.width)),
        width: Math.round(a.width),
        height: Math.round(a.height)
      };
    });

    // No gutter: the row's own gap AND the space between two
    // neighbours' boxes are both zero.
    expect(geom.gap).toBe('0px');
    expect(geom.gapBetweenTiles).toBe(0);

    // Square corners, and no border to separate the photos.
    expect(geom.radius).toBe('0px');
    expect(geom.borderStyle).toBe('none');

    // Square tiles.
    expect(geom.width).toBe(geom.height);
  });

  test('the wall look does not leak into other routes', async ({ page }) => {
    // The normal album page must keep the app-wide grid.
    await page.goto(`${APP}/albums/5`);
    await page.waitForSelector('.tile', { timeout: 30000 });

    const geom = await page.evaluate(() => {
      const row = document.querySelector('.grid-row')!;
      const tiles = Array.from(row.querySelectorAll('.tile')) as HTMLElement[];
      const cs = getComputedStyle(row);
      const tcs = getComputedStyle(tiles[0]);
      const a = tiles[0].getBoundingClientRect();
      const b = tiles[1].getBoundingClientRect();
      return {
        gap: cs.gap,
        radius: tcs.borderRadius,
        borderStyle: tcs.borderStyle,
        gapBetweenTiles: Math.round(b.x - (a.x + a.width))
      };
    });

    expect(geom.gap).not.toBe('0px');
    expect(geom.gapBetweenTiles).toBeGreaterThan(0);
    expect(geom.radius).not.toBe('0px');
    expect(geom.borderStyle).not.toBe('none');
  });

  test('reshuffles on every open', async ({ page }) => {
    await page.goto(`${APP}/albums/5/present`);
    await page.waitForSelector('.tile', { timeout: 30000 });
    const first = await renderedOrder(page);
    expect(first.length).toBeGreaterThan(0);

    const countBefore = await albumCount(page);
    expect(countBefore).toBeGreaterThan(0);

    // A fresh open of the same album must be a different order.
    await page.goto(`${APP}/albums/5/present`);
    await page.waitForSelector('.tile', { timeout: 30000 });
    const second = await renderedOrder(page);

    expect(second).not.toEqual(first);
    // The same album — the reshuffle reorders, it does not change
    // which photos are in it.
    expect(await albumCount(page)).toBe(countBefore);
  });

  test('"Shuffle again" reorders the wall', async ({ page }) => {
    await page.goto(`${APP}/albums/5/present`);
    await page.waitForSelector('.tile', { timeout: 30000 });
    const before = await renderedOrder(page);

    const countBefore = await albumCount(page);

    await page.getByRole('button', { name: /shuffle again/i }).click();
    await page.waitForTimeout(500);
    const after = await renderedOrder(page);

    expect(after).not.toEqual(before);
    expect(await albumCount(page)).toBe(countBefore);
  });

  test('the bar sits below the app TopBar, not under it', async ({ page }) => {
    // The app's TopBar is sticky at the top of the viewport. The
    // wall's own bar is sticky too, so if it sticks at top:0 it
    // slides UNDER the TopBar as soon as the wall is scrolled and
    // the two overlap. This pins the bar below it.
    await page.goto(`${APP}/albums/5/present`);
    await page.waitForSelector('.tile', { timeout: 30000 });

    for (const scroll of [0, 400, 1200]) {
      await page.evaluate((y) => window.scrollTo(0, y), scroll);
      await page.waitForTimeout(300);
      const overlap = await page.evaluate(() => {
        const bar = document.querySelector('.present .bar')!.getBoundingClientRect();
        const top = document.querySelector('.topbar')!.getBoundingClientRect();
        return bar.top < top.bottom && bar.bottom > top.top;
      });
      expect(overlap, `the wall's bar must not overlap the TopBar at scroll ${scroll}`).toBe(
        false
      );
    }
  });

  test('is read-only — no remove affordance on the wall', async ({ page }) => {
    await page.goto(`${APP}/albums/5/present`);
    await page.waitForSelector('.tile', { timeout: 30000 });

    await expect(page.locator('button.remove-btn')).toHaveCount(0);
    await expect(page.locator('button.vote-btn')).toHaveCount(0);
  });
});
