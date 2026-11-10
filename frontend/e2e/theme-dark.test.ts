/**
 * E2E for the optional dark theme.
 *
 * The theme is a deployment-level choice, not a user setting: the
 * backend stamps `data-theme` onto the <html> of the shell it
 * serves, and the SPA reads it back (see search/theme.py and
 * frontend/src/lib/theme.ts). One static build serves either theme,
 * so switching is a redeploy, not a rebuild.
 *
 * Two halves have to agree, and this file pins both:
 *
 *   1. The MARKER. The shell the server sends carries the theme,
 *      and the page the browser renders is that theme. This is what
 *      makes the dark theme work in a deployment that was built
 *      once.
 *   2. The RENDERING. The theme decides the palette AND the grid
 *      geometry: the dark theme's grid has no gutter, square
 *      corners and no border, so the photos touch and the chrome
 *      recedes. A theme that changed only the colours would leave
 *      the gap in place, and the grid would read as the light
 *      theme with a dark background.
 *
 * The test reads the theme the SERVER was started with and asserts
 * against that, so it passes whether the suite runs against a light
 * or a dark deployment. It then flips the marker at runtime and
 * asserts the same rendering rules apply — the frontend's half of
 * the contract, exercised in a browser.
 */
import { test, expect, type Page } from '@playwright/test';

const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';

/** The theme the shell the server sends declares. */
async function servedTheme(page: Page): Promise<string> {
  const html = await (await page.request.get(`${APP}/`)).text();
  const match = html.match(/<html[^>]*\sdata-theme="([^"]*)"/);
  expect(match, 'the served shell must carry a data-theme marker').not.toBeNull();
  return match![1];
}

/** The theme the browser's document actually carries. */
function documentTheme(page: Page): Promise<string | null> {
  return page.evaluate(
    () => document.documentElement.getAttribute('data-theme')
  );
}

/** A custom property's computed value on the document root. */
function rootToken(page: Page, name: string): Promise<string> {
  return page.evaluate(
    (token) =>
      getComputedStyle(document.documentElement).getPropertyValue(token).trim(),
    name
  );
}

/**
 * The id of an album that actually has photos.
 *
 * Discovered from the API rather than hardcoded: the id depends on
 * what the backend was seeded with, and the CI's demo data does not
 * use the same ids as a local library. A hardcoded id makes the test
 * pass on one dataset and time out on another.
 */
async function albumWithPhotos(page: Page): Promise<number> {
  const res = await page.request.get(`${APP}/api/albums`);
  expect(res.ok(), 'the albums list must be reachable').toBeTruthy();
  const body = (await res.json()) as {
    albums?: { id: number; member_count: number }[];
  };
  const album = (body.albums ?? [])
    .filter((a) => a.member_count > 0)
    .sort((a, b) => b.member_count - a.member_count)[0];
  expect(album, 'the backend must expose an album with photos').toBeTruthy();
  return album!.id;
}

/**
 * The grid's geometry, measured off the DOM.
 *
 * The row's own `gap` and the space between two neighbours' boxes
 * are both read: the CSS gap and the actual rendered space can
 * disagree (a padding on the tile would show up in one and not the
 * other), and it is the rendered space that decides whether the
 * photos touch.
 */
async function gridGeometry(page: Page) {
  return page.evaluate(() => {
    const row = document.querySelector('.grid-row');
    if (!row) return null;
    const tiles = Array.from(row.querySelectorAll('.tile')) as HTMLElement[];
    if (tiles.length < 2) return null;
    const cs = getComputedStyle(row);
    const tcs = getComputedStyle(tiles[0]);
    const a = tiles[0].getBoundingClientRect();
    const b = tiles[1].getBoundingClientRect();
    return {
      gap: cs.gap,
      gapBetweenTiles: Math.round(b.x - (a.x + a.width)),
      radius: tcs.borderRadius,
      borderWidth: tcs.borderWidth,
      borderStyle: tcs.borderStyle,
      width: Math.round(a.width),
      height: Math.round(a.height)
    };
  });
}

/** Wait for a grid with at least two tiles, and return its geometry. */
async function waitForGrid(page: Page) {
  await page.waitForSelector('.grid-row .tile', { timeout: 30000 });
  await page.waitForFunction(
    () => document.querySelectorAll('.grid-row .tile').length >= 2,
    undefined,
    { timeout: 30000 }
  );
  const geom = await gridGeometry(page);
  expect(geom, 'the grid must render at least two tiles').not.toBeNull();
  return geom!;
}

test.describe('dark theme', () => {
  test('the served shell declares the theme and the page renders it', async ({
    page
  }) => {
    const theme = await servedTheme(page);
    expect(['light', 'dark']).toContain(theme);

    const albumId = await albumWithPhotos(page);
    await page.goto(`${APP}/albums/${albumId}`);
    await page.waitForSelector('.tile', { timeout: 30000 });

    // The marker the server sent is the marker the browser has.
    expect(await documentTheme(page)).toBe(theme);
  });

  test('the theme decides the palette', async ({ page }) => {
    const albumId = await albumWithPhotos(page);
    await page.goto(`${APP}/albums/${albumId}`);
    await page.waitForSelector('.tile', { timeout: 30000 });

    const theme = await documentTheme(page);
    const pageBackground = await rootToken(page, '--bg-0');
    const accent = await rootToken(page, '--accent');

    if (theme === 'dark') {
      // The dark palette: a near-black page and one accent.
      expect(pageBackground.toLowerCase()).toBe('#0a0a0b');
      expect(accent.toLowerCase()).toBe('#4f8ff0');
    } else {
      // The light palette, unchanged.
      expect(pageBackground.toLowerCase()).toBe('#fafaf7');
      expect(accent.toLowerCase()).toBe('#2563eb');
    }
  });

  test('the theme decides the grid geometry', async ({ page }) => {
    const albumId = await albumWithPhotos(page);
    await page.goto(`${APP}/albums/${albumId}`);
    const geom = await waitForGrid(page);
    const theme = await documentTheme(page);

    if (theme === 'dark') {
      // No gutter: the row's own gap AND the space between two
      // neighbours are both zero, so the photos touch.
      expect(geom.gap).toBe('0px');
      expect(geom.gapBetweenTiles).toBe(0);
      // Square corners, and no border to separate the photos.
      expect(geom.radius).toBe('0px');
      expect(geom.borderWidth).toBe('0px');
    } else {
      // The light theme keeps its 14px gutter and rounded tiles.
      expect(geom.gap).not.toBe('0px');
      expect(geom.gapBetweenTiles).toBeGreaterThan(0);
      expect(geom.radius).not.toBe('0px');
      expect(geom.borderWidth).not.toBe('0px');
    }

    // Square tiles in both themes.
    expect(geom.width).toBe(geom.height);
  });

  test('switching the marker switches the rendering', async ({ page }) => {
    const albumId = await albumWithPhotos(page);
    await page.goto(`${APP}/albums/${albumId}`);
    await waitForGrid(page);

    // Flip to whichever theme is NOT currently active, so this
    // asserts the same thing whether the suite runs against a light
    // or a dark deployment: the marker decides, and the rendering
    // follows it.
    const current = (await documentTheme(page)) ?? 'light';
    const target = current === 'dark' ? 'light' : 'dark';

    const before = await gridGeometry(page);
    const beforeBackground = await rootToken(page, '--bg-0');

    // Flip the marker the way the backend would stamp it.
    await page.evaluate((theme) => {
      document.documentElement.setAttribute('data-theme', theme);
    }, target);

    // The palette follows immediately — the CSS keys off the same
    // attribute.
    const expectedBackground =
      target === 'dark' ? '#0a0a0b' : '#fafaf7';
    await expect
      .poll(async () => rootToken(page, '--bg-0'))
      .toBe(expectedBackground);

    // The geometry follows the store: the gap is part of the
    // theme, so the grid has to re-derive it.
    await expect
      .poll(async () => (await gridGeometry(page))?.gap)
      .toBe(target === 'dark' ? '0px' : '14px');

    const after = await gridGeometry(page);
    if (target === 'dark') {
      expect(after!.gapBetweenTiles).toBe(0);
      expect(after!.radius).toBe('0px');
      expect(after!.borderWidth).toBe('0px');
    } else {
      expect(after!.gapBetweenTiles).toBeGreaterThan(0);
      expect(after!.radius).not.toBe('0px');
      expect(after!.borderWidth).not.toBe('0px');
    }

    // And the theme it came from was what it was — the change is
    // scoped to the marker, not to the page.
    expect(beforeBackground.toLowerCase()).not.toBe(
      expectedBackground.toLowerCase()
    );
    if (current === 'dark') {
      expect(before.gap).toBe('0px');
    } else {
      expect(before.gap).not.toBe('0px');
    }
  });

  test('the dark theme leaves the presentation wall alone', async ({ page }) => {
    // The wall was already square, gapless and borderless. The dark
    // theme must not change its behaviour — it is a standalone
    // read-only surface, and its own rules are its own.
    const albumId = await albumWithPhotos(page);
    await page.goto(`${APP}/albums/${albumId}/present`);
    await page.waitForSelector('.tile', { timeout: 30000 });

    const geom = await gridGeometry(page);
    expect(geom).not.toBeNull();
    expect(geom!.gap).toBe('0px');
    expect(geom!.gapBetweenTiles).toBe(0);
    expect(geom!.radius).toBe('0px');

    // Still standalone: no app chrome, no controls.
    await expect(page.locator('.topbar')).toHaveCount(0);
    await expect(page.locator('.app-shell')).toHaveCount(0);
    await expect(page.locator('button')).toHaveCount(0);
  });
});
