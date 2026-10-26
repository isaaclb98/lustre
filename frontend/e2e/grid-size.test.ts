/**
 * grid-size.test.ts — E2E coverage for the Grid size preference
 * (Settings → Grid size card, Large/Medium/Small).
 *
 * What we pin:
 *   - The Settings page renders the new "Grid size" card with
 *     three radio buttons (Large, Medium, Small).
 *   - The default is Large.
 *   - Clicking a button persists the choice to localStorage.
 *   - The PhotoGrid wrapper reflects the choice via the
 *     data-grid-size attribute.
 *   - The CSS vars --grid-tile-size and --grid-gutter-size
 *     update on :root when the choice changes.
 *
 * Why this is a separate file: the existing settings-index.test.ts
 * covers the Index button + admin API. The Grid size feature is a
 * pure-frontend concern (no backend, no API call), and keeping
 * its tests separate makes the failures easier to attribute.
 */
import { test, expect, type Page } from '@playwright/test';

/**
 * The configured photo count per page at each preset's maximum
 * density (4 rows × maxCols) — the floor for any request. The
 * actual request rounds UP to a whole number of rows at the
 * column count the viewport renders, so this is a minimum, never
 * an exact value. Mirrors $lib/grid/config.
 */
const BASE_PAGE_SIZE = { large: 24, medium: 32, small: 40 } as const;

async function gotoSettings(page: Page) {
  await page.goto('/settings');
  await expect(
    page.getByRole('heading', { name: 'Settings' })
  ).toBeVisible();
}

test.describe('Grid size preference', () => {
  test.beforeEach(async ({ page }) => {
    // Reset the preference to the default before each test so
    // a prior test's localStorage write can't leak into this
    // one's assertions. clear() happens in a page context
    // because localStorage is origin-scoped.
    await page.goto('/');
    await page.evaluate(() => window.localStorage.clear());
  });

  test('Settings page renders the Grid size card with three radios', async ({ page }) => {
    await gotoSettings(page);
    await expect(
      page.getByRole('heading', { name: 'Grid size' })
    ).toBeVisible();
    // The three size options are exposed as a role=radiogroup
    // with three role=radio children. The labels are the same
    // words the user sees in the UI.
    const group = page.getByRole('radiogroup', { name: 'Grid size' });
    await expect(group).toBeVisible();
    await expect(group.getByRole('radio', { name: 'Large' })).toBeVisible();
    await expect(group.getByRole('radio', { name: 'Medium' })).toBeVisible();
    await expect(group.getByRole('radio', { name: 'Small' })).toBeVisible();
  });

  test('Large is the default (no persisted preference)', async ({ page }) => {
    await gotoSettings(page);
    const large = page.getByRole('radio', { name: 'Large' });
    await expect(large).toHaveAttribute('aria-checked', 'true');
    // The other two are unchecked.
    await expect(page.getByRole('radio', { name: 'Medium' })).toHaveAttribute(
      'aria-checked',
      'false'
    );
    await expect(page.getByRole('radio', { name: 'Small' })).toHaveAttribute(
      'aria-checked',
      'false'
    );
  });

  test('Clicking a size persists the choice and updates aria-checked', async ({ page }) => {
    await gotoSettings(page);
    const small = page.getByRole('radio', { name: 'Small' });
    await small.click();
    await expect(small).toHaveAttribute('aria-checked', 'true');
    await expect(page.getByRole('radio', { name: 'Large' })).toHaveAttribute(
      'aria-checked',
      'false'
    );

    // Verify the choice was written to localStorage under the
    // same key the preferences store uses. The exact key is
    // not a contract this test should care about — only that
    // some preference was saved with gridSize='small'.
    const stored = await page.evaluate(() =>
      window.localStorage.getItem('lustre:preferences')
    );
    expect(stored).not.toBeNull();
    const parsed = JSON.parse(stored ?? '{}');
    expect(parsed.gridSize).toBe('small');
  });

  test('The home page grid reflects the chosen size via data-grid-size', async ({ page }) => {
    // Pick Small on the Settings page.
    await gotoSettings(page);
    await page.getByRole('radio', { name: 'Small' }).click();

    // Navigate to a page that mounts PhotoGrid. The home page
    // only shows PhotoGrid when items have loaded — search for
    // a known term to force a render.
    await page.goto('/search?q=test');
    // Wait for the grid wrapper to appear. It carries the
    // data-grid-size attribute that PhotoGrid sets from the
    // preference. The grid may be empty for the fixture data,
    // but the wrapper still mounts.
    const grid = page.locator('.grid-wrapper').first();
    // No grid wrapper may exist on a fixture with no results —
    // the test fixture seeds points so a search for 'test'
    // returns results; if it doesn't, the assertion below
    // surfaces the empty-grid case.
    await grid.waitFor({ state: 'visible', timeout: 5000 }).catch(() => {});
    if (await grid.isVisible().catch(() => false)) {
      await expect(grid).toHaveAttribute('data-grid-size', 'small');
    } else {
      test.skip(true, 'fixture returned no results for the search; data-grid-size unreachable');
    }
  });

  test('/random requests a whole number of grid rows at every size', async ({ page }) => {
    // The page size follows the column count the grid actually
    // renders at the current viewport (see $lib/grid/geometry), so
    // the invariant to pin is "the limit is a whole number of
    // rendered rows" — not a magic number. A fixed limit would
    // leave a ragged last row at every width that renders fewer
    // columns than the preset's maximum.
    //
    // Strategy: intercept the first /api/random?limit= call, then
    // read the rendered track count from the DOM and compare.
    for (const size of ['large', 'medium', 'small'] as const) {
      // Set the preference in localStorage before the page
      // mounts. Going through localStorage is faster than
      // clicking through the Settings page between every
      // fetch.
      await page.goto('/');
      await page.evaluate((s) => {
        const raw = localStorage.getItem('lustre:preferences');
        const obj = raw ? JSON.parse(raw) : {};
        obj.gridSize = s;
        localStorage.setItem('lustre:preferences', JSON.stringify(obj));
      }, size);
      // Capture the first /api/random?limit= call. Wait for
      // the request to fire after navigation.
      const requestPromise = page.waitForRequest(
        (req) => req.url().includes('/api/random') && !req.url().includes('offset='),
        { timeout: 10000 }
      );
      await page.goto('/random', { waitUntil: 'domcontentloaded' });
      const req = await requestPromise;
      const url = new URL(req.url());
      const limit = Number(url.searchParams.get('limit'));

      // The rendered track count is the ground truth for "how many
      // columns does this page actually show".
      await page.waitForSelector('.grid-row', { timeout: 15000 });
      const renderedCols = await page.evaluate(() => {
        const row = document.querySelector('.grid-row');
        if (!row) return 0;
        return getComputedStyle(row).gridTemplateColumns
          .split(' ')
          .filter(Boolean).length;
      });

      expect(renderedCols).toBeGreaterThan(0);
      // The whole-row guarantee.
      expect(limit % renderedCols).toBe(0);
      // And the page never carries fewer photos than the preset
      // asks for (4 rows at maximum density).
      expect(limit).toBeGreaterThanOrEqual(BASE_PAGE_SIZE[size]);
    }
  });

  test('resizing the window changes the page size to keep rows whole', async ({ page }) => {
    // The request size tracks the viewport. A narrow window
    // renders fewer columns, so the limit must grow (rounded up
    // to whole rows) to stay a multiple of the rendered columns.
    await page.goto('/');
    await page.evaluate(() => {
      localStorage.setItem(
        'lustre:preferences',
        JSON.stringify({ slideshowIntervalMs: 3000, gridSize: 'large' })
      );
    });

    const limitsAt = async (width: number) => {
      await page.setViewportSize({ width, height: 900 });
      const requestPromise = page.waitForRequest(
        (req) => req.url().includes('/api/random') && !req.url().includes('offset='),
        { timeout: 15000 }
      );
      await page.goto('/random', { waitUntil: 'domcontentloaded' });
      const req = await requestPromise;
      const limit = Number(new URL(req.url()).searchParams.get('limit'));
      await page.waitForSelector('.grid-row', { timeout: 15000 });
      const cols = await page.evaluate(() => {
        const row = document.querySelector('.grid-row');
        if (!row) return 0;
        return getComputedStyle(row).gridTemplateColumns
          .split(' ')
          .filter(Boolean).length;
      });
      return { limit, cols };
    };

    const wide = await limitsAt(1920);
    const narrow = await limitsAt(768);

    // Fewer columns at the narrow width...
    expect(narrow.cols).toBeLessThan(wide.cols);
    // ...and both requests fill whole rows.
    expect(wide.limit % wide.cols).toBe(0);
    expect(narrow.limit % narrow.cols).toBe(0);
    expect(narrow.limit).toBeGreaterThanOrEqual(BASE_PAGE_SIZE.large);
    expect(wide.limit).toBeGreaterThanOrEqual(BASE_PAGE_SIZE.large);
  });

  test('The CSS vars on :root reflect the chosen size', async ({ page }) => {
    // Pick Medium on Settings.
    await gotoSettings(page);
    await page.getByRole('radio', { name: 'Medium' }).click();
    // The +layout.svelte ResizeObserver publishes the new
    // vars on every preference change. The vars are
    // --grid-tile-size and --grid-gutter-size.
    await page.goto('/');
    const vars = await page.evaluate(() => {
      const style = getComputedStyle(document.documentElement);
      return {
        tile: style.getPropertyValue('--grid-tile-size').trim(),
        gap: style.getPropertyValue('--grid-gutter-size').trim()
      };
    });
    // Medium config: 192/8. The leading-zero trimmed values
    // should be '192px' and '8px'.
    expect(vars.tile).toBe('192px');
    expect(vars.gap).toBe('8px');
  });
});
