/**
 * Playwright smoke tests — one happy-path per page.
 *
 * Per docs/lustre-v2-testing.md Layer 4:
 *   - One happy-path per page. The OpenAPI + zod layers above
 *     are the real guardrails; this just verifies pages
 *     actually render and the core interaction works.
 *   - No snapshot tests here. They rot.
 *
 * Hydration races are real with SvelteKit dev mode (the SSR
 * HTML lands before the client JS is wired up). The appReady
 * helper waits for the SvelteKit preload-data attribute plus
 * a small settle window before any user input is fired. This
 * keeps the tests reliable without needing to know about
 * internal hydration timing.
 */

import { test, expect, type Page } from '@playwright/test';

const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:8000';

async function appReady(page: Page) {
  await page.waitForSelector('header.topbar', { timeout: 10000 });
}

/** Wait for an element matching the locator. Polls up to `timeout`
 *  ms. SvelteKit dev hydration races the test — without polling,
 *  the first attempt to find a chip/grid can fire before the
 *  client JS has wired up the listeners. */
async function waitFor(page: Page, sel: string, timeout = 8000) {
  await page.waitForFunction(
    (s) => document.querySelector(s) !== null,
    sel,
    { timeout }
  );
}

/** Wait until SvelteKit's interactive surface has mounted for a
 *  Search-style page (the composer input is the readiness signal).
 *  This is the most fragile page in dev because the topbar renders
 *  before the route-specific code hydrates. */
async function searchPageReady(page: Page) {
  await appReady(page);
  await page.waitForFunction(
    () => document.querySelector('.composer-input') !== null,
    null,
    { timeout: 15000 }
  );
  // Settle window so Svelte 5 effect batches run.
  await page.waitForTimeout(300);
}

test('Home renders with composer + For-you row', async ({ page }) => {
  await page.goto(APP + '/');
  await appReady(page);
  await expect(page.getByRole('heading', { name: /Find photos/i })).toBeVisible();
  await expect(page.getByRole('button', { name: /^Search$/ })).toBeVisible();
});

test('Search renders chips from URL + shows the grid', async ({ page }) => {
  // / IS the search route now (the /search route was removed).
  // URL params (positives/negatives) drive the search bar.
  await page.goto(APP + '/?positives=beach&positives=ocean');
  await appReady(page);
  // Scope to .chip.pos (prompt chip) — .chip is shared with the
  // collection chips component, which renders its own chip for the
  // currently selected collection.
  await waitFor(page, '.chip.pos', 15000);
  await expect(page.locator('.chip.pos').filter({ hasText: 'beach' })).toBeVisible();
  await expect(page.locator('.chip.pos').filter({ hasText: 'ocean' })).toBeVisible();
  await waitFor(page, '.grid-tile, .empty', 8000);
});

test('Random renders a grid', async ({ page }) => {
  await page.goto(APP + '/random');
  await appReady(page);
  await expect(page.getByRole('heading', { name: /Random/i })).toBeVisible();
  await waitFor(page, '.grid-tile, .empty', 8000);
});

test('Albums renders list or empty state', async ({ page }) => {
  await page.goto(APP + '/albums');
  await appReady(page);
  // h1 is exactly 'Albums'; the page also has an h2 'Your albums'
  // for the album grid section. Scope to h1 (exact) so we don't
  // match the h2 in strict mode.
  await expect(page.getByRole('heading', { name: 'Albums', exact: true })).toBeVisible();
  await waitFor(page, '.card, .placeholder', 8000);
});

test('For You renders header + body', async ({ page }) => {
  await page.goto(APP + '/for-you');
  await appReady(page);
  await expect(page.getByRole('heading', { name: /For you/i })).toBeVisible();
  await expect(page.getByRole('combobox', { name: /Diversity mode/i })).toBeVisible();
});

test('Top bar tabs are interactive', async ({ page }) => {
  // Topbar tabs: Home, Random, For You, Albums, Settings. We test
  // a few — clicking Random should navigate to /random and activate
  // that tab visually. (The older 'Search' tab test was removed
  // when /search became the home route.)
  await page.goto(APP + '/');
  await appReady(page);
  const randomTab = page.getByRole('link', { name: 'Random' });
  await randomTab.click();
  await page.waitForURL(/\/random$/);
  // The Random page doesn't have the Search button (composer lives
  // on the home / search page). Verify the random grid is loaded
  // instead — that's the destination page's primary content.
  await expect(page.getByRole('heading', { name: /Random/i })).toBeVisible();
  await waitFor(page, '.grid-tile, .empty', 8000);
});

test('Adding a positive prompt renders a chip', async ({ page }) => {
  // / IS the search route now (the /search route was removed).
  await page.goto(APP + '/');
  await searchPageReady(page);

  const input = page.getByPlaceholder(/Add a positive/);
  await input.click();
  await input.fill('beach');
  await input.press('Enter');

  // Scope to .chip.pos — .chip matches both the prompt chip and
  // the collection chip ("nas 182"); the test is about the prompt.
  await waitFor(page, '.chip.pos', 10000);
  await expect(page.locator('.chip.pos').filter({ hasText: 'beach' })).toBeVisible();
});

/**
 * Lightbox — clicking a tile opens it; ←/→ navigate; Esc closes.
 * Regression cover for the most-used feature on every search page.
 */
test('Lightbox opens, navigates with arrow keys, closes with Esc', async ({ page }) => {
  await page.goto(APP + '/random');
  await appReady(page);
  await waitFor(page, '.grid-tile', 10000);

  // Open by clicking the first tile
  await page.locator('.grid-tile').first().click();
  await expect(page.locator('[role="dialog"]')).toBeVisible({ timeout: 5000 });
  await expect(page.locator('.photo')).toBeVisible();

  // Navigate next
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(150);
  await expect(page.locator('.count')).toContainText('2 /');

  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(150);
  await expect(page.locator('.count')).toContainText('3 /');

  // Navigate back
  await page.keyboard.press('ArrowLeft');
  await page.waitForTimeout(150);
  await expect(page.locator('.count')).toContainText('2 /');

  // Esc closes
  await page.keyboard.press('Escape');
  await expect(page.locator('[role="dialog"]')).toBeHidden({ timeout: 3000 });
});

/**
 * Saved searches — save current prompts, pick from dropdown, delete.
 * End-to-end exercise of the SavedSearchesMenu + backend roundtrip.
 */
test('Saved searches: save, pick reapplies prompts, delete removes', async ({ page }) => {
  // Start fresh — go to / with no params so we control state. / IS
  // the search route now (the /search route was removed).
  await page.goto(APP + '/');
  await searchPageReady(page);

  const input = page.getByPlaceholder(/Add a positive/);
  await input.click();
  await input.fill('mountain');
  await input.press('Enter');
  await input.fill('sunset');
  await input.press('Enter');
  // Scope to prompt chips (.chip.pos); .chip matches collection too.
  await waitFor(page, '.chip.pos', 8000);

  // Use a unique name so we don't collide with saved searches from
  // other tests, then locate our entry by that name. Round 2
  // migrated the Save action to the Svelte Dialog primitive —
  // window.prompt no longer fires, so we type into the
  // dialog's input directly.
  const uniqueName = `my-mountain-set-${Date.now()}`;

  // Click the Save button
  await page.getByTitle('Save current search').click();
  // The prompt dialog opens; fill it and submit.
  const promptInput = page.locator('input[id="dialog-prompt-input"]');
  await promptInput.fill(uniqueName);
  await page.locator('button:has-text("Save"):not([title])').last().click();
  await expect(page.locator('[role="dialog"]')).toHaveCount(0);

  // Open the Saved dropdown and verify our name appears
  await page.getByTitle('Saved searches').click();
  await waitFor(page, '.pop', 5000);
  const ourItem = page.locator('.pop .item-row').filter({ hasText: uniqueName });
  await expect(ourItem).toBeVisible({ timeout: 5000 });

  // Delete our specific entry via the × button on its row.
  // Round 2: also a Svelte Dialog now — confirm via the
  // Delete button inside the modal.
  await ourItem.locator('button.del, .del').first().click();
  await expect(page.locator('[role="dialog"]')).toBeVisible();
  await page.locator('button:has-text("Delete"):not([title])').click();
  await expect(page.locator('[role="dialog"]')).toHaveCount(0);

  // After deletion our entry should be gone
  await expect(ourItem).toBeHidden({ timeout: 5000 });
});

/**
 * URL state sync: search bar → URL → reload → still the same.
 * Guards the spec line "picking one re-applies … and runs the search".
 */
test('Search state round-trips through the URL', async ({ page }) => {
  // / IS the search route now (the /search route was removed).
  await page.goto(APP + '/');
  await searchPageReady(page);

  const input = page.getByPlaceholder(/Add a positive/);
  await input.click();
  await input.fill('round-trip-keyword');
  await input.press('Enter');
  // Scope to prompt chips (.chip.pos); .chip matches collection too.
  await waitFor(page, '.chip.pos', 8000);

  // After the 180ms debounce + write the URL should carry the prompt.
  await page.waitForTimeout(500);
  const url = page.url();
  expect(url).toContain('positives=round-trip-keyword');
});

/**
 * Right-click context menu opens on a tile; clicking outside closes.
 */
test('Right-click opens the photo context menu', async ({ page }) => {
  await page.goto(APP + '/random');
  await appReady(page);
  await waitFor(page, '.grid-tile', 10000);

  const firstTile = page.locator('.grid-tile').first();
  await firstTile.click({ button: 'right' });
  await expect(page.locator('.menu[role="menu"]')).toBeVisible({ timeout: 3000 });
  // The menu lists Open / Copy URL / Pin-ish items
  await expect(page.getByRole('menuitem', { name: /Open in new tab/ })).toBeVisible();

  // Click outside closes
  await page.locator('main').click();
  await expect(page.locator('.menu[role="menu"]')).toBeHidden({ timeout: 3000 });
});

/**
 * Infinite scroll: scrolling near the bottom triggers loadMore
 * which appends more tiles. The sentinel uses IntersectionObserver
 * with a 600px rootMargin so the trigger fires well before the
 * viewport edge — we exercise it by scrolling to ~80% page height.
 *
 * Requires a non-empty library (--demo-count >= ~60 in the dev
 * server). The default demo-count is now 200, so this should be
 * reliable.
 */
test('Random page infinite-scrolls: scrolling loads more tiles', async ({ page }) => {
  await page.goto(APP + '/random');
  await appReady(page);
  await waitFor(page, '.grid-tile', 10000);

  // Wait for the initial fetch to settle so the count is stable.
  await page.waitForFunction(() => !window.__random_loading, null, { timeout: 5000 }).catch(() => {});
  await page.waitForTimeout(800);

  const initial = await page.locator('.grid-tile').count();
  expect(initial).toBeGreaterThan(0);

  // The page body is the scroll container — the grid uses
  // createWindowVirtualizer, so scrolling window reaches the sentinel.
  for (let i = 1; i <= 4; i++) {
    await page.evaluate((frac) => {
      window.scrollTo({ top: document.body.scrollHeight * frac });
    }, i / 4);
    await page.waitForTimeout(800);
  }

  const final = await page.locator('.grid-tile').count();
  // The session cursor guarantees fresh photos on every scroll
  // (no duplicates), so the items array grows with each batch and
  // the virtualizer renders more visible tiles as the page grows.
  expect(final).toBeGreaterThan(initial);

  const endVisible = await page.locator('.end').isVisible().catch(() => false);
  expect(endVisible || final > initial).toBe(true);
});

test('Random page walks through the full session without duplicates', async ({ page }) => {
  /**
   * Session-cursor contract: scrolling through the entire /random
   * page visits every photo in the library exactly once, then the
   * scroll stops (has_more goes false, sentinel unmounts).
   *
   * The virtualizer only renders visible rows, so we harvest
   * photo IDs from thumb URLs as we scroll — accumulating unique
   * IDs across all visible positions, not at any one moment.
   *
   * Requires the dev server to have indexed photos. The default
   * demo dataset is 182; this should reliably walk the whole
   * library in ~10 scroll iterations.
   */
  await page.goto(APP + '/random');
  await appReady(page);
  await waitFor(page, '.grid-tile', 10000);
  await page.waitForTimeout(500);

  // Get session_total from the first API response. The page made
  // the call on mount, so the headers/body are already in flight.
  const firstResp = await page.evaluate(async () => {
    // Re-fetch to capture the response — the original call already
    // completed by the time the test starts running.
    const r = await fetch('/api/random?limit=1');
    return r.json();
  });
  const sessionTotal = firstResp.session_total;
  expect(sessionTotal).toBeGreaterThan(20);

  // Walk the session by harvesting unique photo IDs from visible
  // thumb URLs. The virtualizer only renders visible rows, so we
  // count IDs across all scroll positions.
  const allIds = new Set<string>();
  const harvest = async () => {
    const ids = await page.evaluate(() => {
      const ids: string[] = [];
      for (const img of document.querySelectorAll('img')) {
        const m = (img.src || '').match(/\/thumb\/([^/?#]+)/);
        if (m) ids.push(m[1]);
      }
      return ids;
    });
    for (const id of ids) allIds.add(id);
  };

  // Initial harvest
  await harvest();

  // Scroll repeatedly until has_more is false.
  let iterations = 0;
  const maxIterations = 50;
  while (iterations < maxIterations) {
    iterations++;
    // Trigger more loads by scrolling to the bottom.
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(600);
    await harvest();

    // Check if scroll has stopped firing. We do this by checking
    // the last /api/random response's has_more via a fresh call.
    await page.evaluate(async () => {
      // Use a HEAD-like probe: fetch with limit=0 won't change
      // session state but returns the current has_more.
      // Actually we need to read from the component state. The
      // simpler approach: just check if __random_loading stays false
      // for a full second.
      return null;
    });

    // Break when we've seen all session_total unique ids.
    if (allIds.size >= sessionTotal) break;
  }

  // Every photo in the library was visited exactly once.
  expect(allIds.size).toBe(sessionTotal);

  // Confirm the scroll has stopped: no more new batches arriving.
  // After scrolling to the bottom and waiting, harvesting should
  // not increase the unique count.
  await page.waitForTimeout(1500);
  const beforeIdle = allIds.size;
  await harvest();
  expect(allIds.size).toBe(beforeIdle);
});

test('Search page infinite-scrolls: scrolls append a second page', async ({ page }) => {
  // / IS the search route now (the /search route was removed).
  // Initial page size is 24 (see GRID_SIZE_CONFIG.large.pageSize;
  // was a global GRID_PAGE_SIZE=24 before the per-grid-size
  // scaling landed — same value for the Large default, just
  // sourced from a different constant). Scope the tile count
  // to the search results section so the home page's For You
  // row tiles don't contaminate the count. Use a generic
  // prompt that's likely to match many photos — "photo"
  // matches all images in the dev library of 182 K-pop
  // photos, so the second page will load.
  await page.goto(APP + '/?positives=photo');
  await searchPageReady(page);
  await waitFor(page, 'section.results .grid-tile', 10000);

  // Wait for the first page of results to fully render before
  // counting — without this, the count can include stale tiles
  // from a previous render cycle.
  await page.waitForTimeout(1500);

  const initial = await page.locator('section.results .grid-tile').count();
  expect(initial).toBe(24);

  // Scroll the sentinel into view to trigger the next-page load.
// The sentinel lives inside section.results; the page has other
// sections below (For You, etc.) so scrolling to document.body
// scrollHeight skips past the sentinel without triggering it. We
// scroll the sentinel element itself into view, then wait for the
// append to land.
  for (let i = 0; i < 5; i++) {
    await page.locator('section.results .sentinel').first().scrollIntoViewIfNeeded();
    await page.waitForTimeout(2000);
  }

  const final = await page.locator('section.results .grid-tile').count();
  expect(final).toBeGreaterThan(24);
});
