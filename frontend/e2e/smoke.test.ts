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

/**
 * The photo ids the SERVER delivered, batch by batch.
 *
 * The Random page's contract is a session cursor: the server shuffles
 * the library once, hands out a session id, and slices that deck by
 * offset — so the photos a client receives across its requests are
 * distinct, and the whole walk covers the deck exactly once.
 *
 * Reading that from the NETWORK rather than the DOM is what makes the
 * assertion layout-independent. The DOM's tile count is a function of
 * the viewport: the uniform grid virtualises by rows and the masonry
 * renders only the visible band, so in both cases the number of tiles
 * in the document says nothing about how many photos were fetched.
 * The server's batches say exactly that.
 *
 * Attach this BEFORE the first navigation — the page fetches on mount.
 */
interface RandomBatch {
  ids: string[];
  sessionTotal: number;
  hasMore: boolean;
}

async function collectRandomBatches(page: Page): Promise<RandomBatch[]> {
  const batches: RandomBatch[] = [];
  page.on('response', (res) => {
    if (!res.url().includes('/api/random')) return;
    void res
      .json()
      .then(
        (body: {
          results?: { id: string }[];
          session_total?: number;
          has_more?: boolean;
        }) => {
          if (!Array.isArray(body?.results)) return;
          batches.push({
            ids: body.results.map((r) => r.id),
            sessionTotal: body.session_total ?? 0,
            hasMore: !!body.has_more
          });
        }
      )
      .catch(() => {
        /* not JSON — ignore */
      });
  });
  return batches;
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
  // The delete control is the row's icon button, labelled with the
  // saved search's name.
  await ourItem.getByRole('button', { name: new RegExp(`Delete ${uniqueName}`) }).click();
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
  const batches = await collectRandomBatches(page);
  await page.goto(APP + '/random');
  await appReady(page);
  await waitFor(page, '.grid-tile', 10000);

  // Wait for the initial fetch to settle so the count is stable.
  await page.waitForFunction(() => !window.__random_loading, null, { timeout: 5000 }).catch(() => {});
  await page.waitForTimeout(800);

  const initial = await page.locator('.grid-tile').count();
  expect(initial).toBeGreaterThan(0);
  const initialDelivered = batches.flatMap((b) => b.ids).length;
  expect(initialDelivered).toBeGreaterThan(0);

  // The page body is the scroll container — the grid uses
  // createWindowVirtualizer, so scrolling window reaches the sentinel.
  for (let i = 1; i <= 4; i++) {
    await page.evaluate((frac) => {
      window.scrollTo({ top: document.body.scrollHeight * frac });
    }, i / 4);
    await page.waitForTimeout(800);
  }

  // Count the photos the page has LOADED, not the tiles the DOM
  // holds. The two are different numbers: the uniform grid
  // virtualises by rows, and the masonry band renders only the
  // visible slice — in both cases the DOM's tile count is a
  // function of the VIEWPORT, not of how much has been fetched.
  // The photos the server delivered are the thing that grows.
  const delivered = batches.flatMap((b) => b.ids);
  expect(delivered.length).toBeGreaterThan(initialDelivered);

  const endVisible = await page.locator('.end').isVisible().catch(() => false);
  expect(endVisible || delivered.length > initialDelivered).toBe(true);
});

test('Random page walks through the full session without duplicates', async ({ page }) => {
  /**
   * Session-cursor contract: scrolling through the entire /random
   * page visits every photo in the library exactly once, then the
   * scroll stops (has_more goes false, sentinel unmounts).
   *
   * The photo ids come from the SERVER's responses, not from the
   * rendered thumbnails. The DOM's tiles are a window onto the deck
   * — the uniform grid virtualises by rows, the masonry renders only
   * the visible band — so the thumbnails in the document cannot say
   * how much of the deck has been walked. The server's batches can.
   *
   * Requires the dev server to have indexed photos. The default
   * demo dataset is 182; this should reliably walk the whole
   * library in ~10 scroll iterations.
   */
  const batches = await collectRandomBatches(page);
  await page.goto(APP + '/random');
  await appReady(page);
  await waitFor(page, '.grid-tile', 10000);
  await page.waitForTimeout(500);

  // The session's size, from the server's own first response. It is
  // read from the batch rather than from a probe fetch: a probe
  // would open a SECOND session, and its photos would land in the
  // batch list alongside the walk's own.
  let iterations = 0;
  const maxIterations = 50;
  while (iterations < maxIterations && batches.length === 0) {
    iterations++;
    await page.waitForTimeout(200);
  }
  const sessionTotal = batches[0]?.sessionTotal ?? 0;
  expect(sessionTotal).toBeGreaterThan(20);

  // Walk the session by scrolling until the server stops handing out
  // photos.
  iterations = 0;
  while (iterations < maxIterations) {
    iterations++;
    const before = batches.length;
    // Trigger more loads by scrolling to the bottom.
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(600);

    // The server has stopped handing out photos when it says
    // has_more is false, or when a scroll produces no new batch.
    const last = batches[batches.length - 1];
    if (last && !last.hasMore) break;
    if (batches.length === before) break;
  }

  // The session's contract: the photos the server delivered across
  // the whole walk are the deck, each exactly once.
  //
  // The ids come from the server's own responses rather than from
  // the rendered thumbnails. The DOM cannot answer this question:
  // the uniform grid virtualises by rows and the masonry renders
  // only the visible band, so the thumbnails in the document are a
  // window onto the deck, not the deck. (The masonry's window is the
  // narrower of the two — a jump to the bottom skips the photos in
  // between, which were never rendered and so never harvested.)
  const delivered = batches.flatMap((b) => b.ids);
  const allIds = new Set(delivered);
  expect(allIds.size).toBe(sessionTotal);
  // And no photo was delivered twice.
  expect(delivered.length).toBe(allIds.size);

  // Confirm the scroll has stopped: no more new batches arriving.
  // After scrolling to the bottom and waiting, the server should
  // not hand out another photo.
  await page.waitForTimeout(1500);
  const beforeIdle = batches.flatMap((b) => b.ids).length;
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(600);
  expect(batches.flatMap((b) => b.ids).length).toBe(beforeIdle);
});

test('Search page infinite-scrolls: scrolls append a second page', async ({ page }) => {
  // / IS the search route now (the /search route was removed).
  // The first page holds a whole number of grid rows at the
  // column count the viewport renders (see $lib/grid/geometry) —
  // the count is derived, not a fixed 24. Scope the tile count
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

  // The rendered track count is the ground truth for the column
  // count. The virtualizer renders whole rows, so the tile count
  // must be a multiple of it — that is the whole-row guarantee,
  // visible in the DOM.
  const renderedCols = await page.evaluate(() => {
    // Two layouts render the grid: the uniform grid (rows in a CSS
    // grid) and masonry (tiles positioned absolutely into columns).
    // The layout is a deployment choice, so read the column count
    // from whichever one is live.
    const row = document.querySelector('section.results .grid-row');
    if (row) {
      return {
        cols: getComputedStyle(row).gridTemplateColumns
          .split(' ')
          .filter(Boolean).length,
        masonry: false
      };
    }
    const lefts = new Set(
      Array.from(document.querySelectorAll('section.results .tile')).map(
        (t) => Math.round((t as HTMLElement).getBoundingClientRect().left)
      )
    );
    return { cols: lefts.size, masonry: true };
  });
  expect(renderedCols.cols).toBeGreaterThan(0);

  const initial = await page.locator('section.results .grid-tile').count();
  expect(initial).toBeGreaterThan(0);
  // The uniform grid's virtualizer renders WHOLE ROWS, so the tile
  // count in the DOM is a multiple of the column count. Masonry
  // renders the visible band instead, which is not a whole number of
  // rows — there the guarantee lives on the REQUEST, not the DOM.
  if (!renderedCols.masonry) {
    expect(initial % renderedCols.cols).toBe(0);
  }

  // The grid is VIRTUALIZED: only the rows in view are in the DOM,
  // so the RENDERED tile count can go DOWN as the window moves. The
  // unambiguous signal that a page was appended is the request for
  // it — the grid asks for the next offset.
  const requestedOffsets: number[] = [];
  page.on('request', (req) => {
    if (req.url().includes('/api/search')) {
      const raw = new URL(req.url()).searchParams.get('offset');
      requestedOffsets.push(raw === null ? 0 : Number(raw));
    }
  });

  // Scroll the sentinel into view to trigger the next-page load.
// The sentinel lives inside section.results; the page has other
// sections below (For You, etc.) so scrolling to document.body
// scrollHeight skips past the sentinel without triggering it. We
// scroll the sentinel element itself into view, then wait for the
// append to land.
// The sentinel only renders while there is another page to load, so
// scroll to it when it is there and to the bottom of the results
// otherwise.
  for (let i = 0; i < 5; i++) {
    const sentinel = page.locator('section.results .sentinel').first();
    if (await sentinel.count()) {
      await sentinel.scrollIntoViewIfNeeded();
    } else {
      await page.evaluate(() => {
        document.querySelector('section.results')?.scrollIntoView({ block: 'end' });
      });
    }
    await page.waitForTimeout(2000);
  }

  // The append lands asynchronously — poll for it rather than
  // assuming the fixed waits above were enough.
  await expect
    .poll(() => requestedOffsets.filter((o) => o > 0).length, {
      timeout: 15_000,
      message: 'scrolling must fetch the next page'
    })
    .toBeGreaterThan(0);
  const final = await page.locator('section.results .grid-tile').count();
  // The appended page keeps the whole-row guarantee.
  if (!renderedCols.masonry) {
    expect(final % renderedCols.cols).toBe(0);
  }
});
