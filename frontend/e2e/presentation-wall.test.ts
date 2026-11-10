/**
 * E2E for the presentation wall — the album shown as a gallery.
 *
 * The wall is a standalone foreground surface: it breaks out of
 * the app's layout hierarchy, so no nav bar, no page shell and no
 * floating controls are in its tree at all. It has no buttons of
 * its own either, and its tiles are pictures rather than targets.
 *
 * Three things matter and all are easy to regress:
 *
 *  1. The wall is ON ITS OWN. The app chrome must not appear on
 *     it — a single stray TopBar or shell element means the
 *     layout split broke and the wall is back inside the app.
 *  2. The wall's own look: no gutter, square corners, no border
 *     between tiles, and no background bleeding in at the edges.
 *  3. That look is SCOPED. The gapless treatment belongs to the
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
      (t) => t.getAttribute('data-photo-id') ?? ''
    )
  );
}

/** The album's member count, as the wall reports it. */
async function albumCount(page: Page): Promise<number> {
  return page.evaluate(() => {
    const el = document.querySelector('[data-photo-count]');
    return Number(el?.getAttribute('data-photo-count') ?? 0);
  });
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

  test('is on its own — no app chrome, no controls', async ({ page }) => {
    await page.goto(`${APP}/albums/5/present`);
    await page.waitForSelector('.tile', { timeout: 30000 });

    // The app chrome is not in the wall's layout tree at all. It
    // is not hidden with CSS — the elements do not exist.
    await expect(page.locator('.topbar')).toHaveCount(0);
    await expect(page.locator('.app-shell')).toHaveCount(0);
    await expect(page.locator('.shell')).toHaveCount(0);

    // No buttons of any kind — the wall's own, the grid's, the
    // lightbox's. Nothing on the surface can be pressed.
    await expect(page.locator('button')).toHaveCount(0);
    await expect(page.locator('.present .bar')).toHaveCount(0);

    // The tiles are pictures, not links: nothing to click, nothing
    // to focus, nothing to navigate to.
    await expect(page.locator('a.tile')).toHaveCount(0);
    await expect(page.locator('.tile[href]')).toHaveCount(0);

    // No lightbox, and no right-click context menu.
    await expect(page.locator('.overlay')).toHaveCount(0);
    await page.locator('.tile').first().click({ button: 'right' });
    await page.waitForTimeout(300);
    await expect(page.locator('.menu.glass-strong')).toHaveCount(0);
  });

  test('fills the viewport — no background bleeding in at the edges', async ({ page }) => {
    await page.setViewportSize({ width: 1270, height: 900 });
    await page.goto(`${APP}/albums/5/present`);
    await page.waitForSelector('.tile', { timeout: 30000 });

    const edges = await page.evaluate(() => {
      const tiles = Array.from(
        document.querySelectorAll('.grid-row')[0].querySelectorAll('.tile')
      ) as HTMLElement[];
      const wrapper = document.querySelector('.grid-wrapper')!.getBoundingClientRect();
      const first = tiles[0].getBoundingClientRect();
      const last = tiles[tiles.length - 1].getBoundingClientRect();
      return {
        left: Math.round(first.left - wrapper.left),
        right: Math.round(wrapper.right - last.right),
        scrollWidth: document.documentElement.scrollWidth,
        // clientWidth excludes the vertical scrollbar; innerWidth
        // includes it. The wall fills the content box.
        clientWidth: document.documentElement.clientWidth
      };
    });

    // The first photo starts at the content box's left edge and
    // the last one ends at its right edge.
    expect(edges.left).toBe(0);
    expect(edges.right).toBe(0);
    // And the page does not scroll sideways.
    expect(edges.scrollWidth).toBe(edges.clientWidth);
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

  test('the app chrome is still on the ordinary routes', async ({ page }) => {
    // The counterpart to the wall's own-standing test: the layout
    // split moved the chrome into a route group, so the ordinary
    // pages must still have it.
    await page.goto(`${APP}/albums/5`);
    await page.waitForSelector('.topbar', { timeout: 30000 });
    await expect(page.locator('.app-shell')).toHaveCount(1);
    await expect(page.locator('.shell')).toHaveCount(1);
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

  test('is read-only — nothing on the wall can be changed', async ({ page }) => {
    await page.goto(`${APP}/albums/5/present`);
    await page.waitForSelector('.tile', { timeout: 30000 });

    // The grid's own mutation affordances are absent.
    await expect(page.locator('button.remove-btn')).toHaveCount(0);
    await expect(page.locator('button.vote-btn')).toHaveCount(0);
    // And so is the favourite badge, which on an album page marks
    // every photo as a favourite and therefore says nothing.
    await expect(page.locator('.fav')).toHaveCount(0);
  });

  test('Escape leaves the wall', async ({ page }) => {
    await page.goto(`${APP}/albums/5/present`);
    await page.waitForSelector('.tile', { timeout: 30000 });

    await page.keyboard.press('Escape');
    await page.waitForURL(/\/albums\/5$/, { timeout: 10000 });
    // Back on the album page, with the app chrome.
    await expect(page.locator('.topbar')).toHaveCount(1);
  });
});
