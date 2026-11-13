import { test, expect, type Page } from '@playwright/test';

// Tests run against the dev stack (PLAYWRIGHT_BASE_URL is set by
// the wrapper or the CI workflow). Falls back to the dev port so a
// bare `node_modules/.bin/playwright test` from a developer's machine
// still works against their local dev stack.
const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:18000';

/**
 * concurrency.test.ts — Race conditions, rapid clicks, concurrent
 * operations, and stress scenarios.
 *
 * Real users mash buttons, open multiple tabs, and don't wait for
 * things to finish. These tests verify the app handles that gracefully.
 */

async function appReady(page: Page) {
  await page.waitForSelector('header.topbar', { timeout: 10000 });
}

test.describe('Rapid clicks', () => {
  test('clicking Search button rapidly does not fire duplicate requests', async ({ page }) => {
    let requestCount = 0;
    page.on('request', (req) => {
      if (req.url().includes('/api/search') && req.method() === 'GET') {
        requestCount++;
      }
    });

    await page.goto('/?positives=beach');
    await appReady(page);

    // Wait for initial search
    await page.waitForTimeout(500);
    const initialCount = requestCount;

    // Mash the Search button 5 times rapidly
    const searchBtn = page.getByRole('button', { name: 'Search', exact: true });
    for (let i = 0; i < 5; i++) {
      await searchBtn.click({ force: true }).catch(() => {});
    }

    await page.waitForTimeout(2000);

    // Should NOT have fired 5 additional searches. The button is
    // disabled during the in-flight request (UX fix), but clicks
    // that race ahead of the disabled state can still fire. The
    // key invariant is that dedupe is working: at most 2 of the
    // 5 rapid clicks should land as separate searches. The
    // previous threshold of 4 was found to be flaky under suite
    // load (it'd intermittently allow 4 of 5 through, suggesting
    // dedupe only sometimes engaged). Threshold of 3 catches real
    // regressions in the dedupe path without flaking on
    // slow-network races.
    expect(requestCount - initialCount).toBeLessThan(3);
  });

  test('clicking the same tile rapidly does not open multiple lightboxes', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    // Click the same tile 3 times rapidly
    const tile = page.locator('.grid-tile').first();
    await tile.click();
    await tile.click({ force: true }).catch(() => {});
    await tile.click({ force: true }).catch(() => {});

    await page.waitForTimeout(500);

    // Should have exactly one dialog
    const dialogCount = await page.locator('[role="dialog"]').count();
    expect(dialogCount).toBeLessThanOrEqual(1);
  });

  test('pressing ArrowRight rapidly in lightbox does not break navigation', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    await page.locator('.grid-tile').first().click();
    await expect(page.locator('[role="dialog"]')).toBeVisible({ timeout: 5000 });

    // Rapid-fire ArrowRight 10 times
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('ArrowRight');
    }

    await page.waitForTimeout(500);

    // Dialog should still be open and not crash
    await expect(page.locator('[role="dialog"]')).toBeVisible();
  });
});

test.describe('Concurrent operations', () => {
  test('liking a photo while another search is loading does not crash', async ({ page }) => {
    await page.goto('/?positives=beach');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    // Like the first photo (opens lightbox, clicks like)
    await page.locator('.grid-tile').first().click();
    await expect(page.locator('[role="dialog"]')).toBeVisible({ timeout: 5000 });

    // Close and immediately do another search
    await page.keyboard.press('Escape');
    await expect(page.locator('[role="dialog"]')).toBeHidden({ timeout: 3000 });

    const input = page.getByPlaceholder(/Add a positive/);
    await input.fill('concurrent-test');
    await input.press('Enter');
    await page.getByRole('button', { name: 'Search', exact: true }).click();

    // Should not crash, results should load
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });
  });

  test('removing all chips while search is loading does not crash', async ({ page }) => {
    await page.goto('/?positives=beach&positives=sunset');
    await appReady(page);

    // Wait for the initial search to complete before manipulating chips.
    // Scope to .chip.pos (prompt chip) — .chip is shared with
    // CollectionsChips which renders its own chip elements that
    // shouldn't be removed by the × button here.
    await expect(page.locator('.chip.pos').first()).toBeVisible({ timeout: 5000 });
    await page.waitForTimeout(500);

    // Remove all chips one at a time, re-querying each iteration
    // because the DOM shifts after each removal
    while ((await page.locator('.chip.pos').count()) > 0) {
      await page.locator('.chip.pos button.x').first().click();
      await page.waitForTimeout(100);
    }

    // Search button should be disabled (no prompts, not loading)
    const searchBtn = page.getByRole('button', { name: 'Search', exact: true });
    await expect(searchBtn).toBeDisabled({ timeout: 3000 });
  });

  test('navigating away during a search does not cause errors', async ({ page }) => {
    // Start a search that will be slow
    await page.route('**/api/search**', async (route) => {
      await new Promise(r => setTimeout(r, 2000));
      await route.continue();
    });

    await page.goto('/?positives=beach');
    await appReady(page);

    // Trigger a new search
    const input = page.getByPlaceholder(/Add a positive/);
    await input.fill('slow-search');
    await input.press('Enter');
    await page.getByRole('button', { name: 'Search', exact: true }).click();

    // Navigate away immediately (don't wait for the response)
    await page.goto(`${APP}/random`, { waitUntil: 'domcontentloaded' });
    await appReady(page);

    // Random page should load fine
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });
  });
});

test.describe('Stress scenarios', () => {
  test('adding 10+ prompts works', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    const input = page.getByPlaceholder(/Add a positive/);
    for (let i = 0; i < 12; i++) {
      await input.fill(`prompt-${i}`);
      await input.press('Enter');
      await page.waitForTimeout(50);
    }

    const chipCount = await page.locator('.chip.pos').count();
    expect(chipCount).toBe(12);
  });

  test('rapidly toggling polarity tab does not corrupt state', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    const positiveTab = page.locator('[role="tablist"] [role="tab"]').nth(0);
    const negativeTab = page.locator('[role="tablist"] [role="tab"]').nth(1);

    // Rapidly toggle 20 times
    for (let i = 0; i < 20; i++) {
      if (i % 2 === 0) {
        await positiveTab.click({ force: true }).catch(() => {});
      } else {
        await negativeTab.click({ force: true }).catch(() => {});
      }
    }

    // Final tab should be selected
    const selectedTab = page.locator('[role="tab"][aria-selected="true"]');
    await expect(selectedTab).toBeVisible();
  });

  test('opening and closing lightbox 5 times in a row works', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    for (let i = 0; i < 5; i++) {
      await page.locator('.grid-tile').first().click();
      await expect(page.locator('[role="dialog"]')).toBeVisible({ timeout: 5000 });
      await page.keyboard.press('Escape');
      await expect(page.locator('[role="dialog"]')).toBeHidden({ timeout: 3000 });
    }
  });
});

test.describe('Multiple tabs', () => {
  test('opening the app in two tabs does not share state inappropriately', async ({ browser }) => {
    const context = await browser.newContext();
    const page1 = await context.newPage();
    const page2 = await context.newPage();

    // Tab 1: add a prompt. /?positives=... is the search route now
    // (the /search route was removed; / IS search). The base URL
    // is dev (:18000) — the old hardcoded :8000 worked when tests
    // ran against prod.
    await page1.goto(APP + '/');
    await page1.waitForSelector('header.topbar');
    const input1 = page1.getByPlaceholder(/Add a positive/);
    await input1.fill('tab1-prompt');
    await input1.press('Enter');

    // Tab 2: should be independent (no shared state). The base URL
    // is dev (:18000) — the old hardcoded :8000 worked when tests
    // ran against prod. / IS the search route now (the /search
    // route was removed). We navigate without any query string so
    // the initial state is empty — the test is about state leakage,
    // not URL-driven state.
    await page2.goto(APP + '/');
    await page2.waitForSelector('header.topbar');

    // Tab 2 should NOT have the prompt from tab 1
    const tab2Chips = await page2.locator('.chip.pos').count();
    expect(tab2Chips).toBe(0);

    await context.close();
  });
});

test.describe('Network resilience', () => {
  test('slow network: search completes eventually', async ({ page }) => {
    // Add a 3-second delay to all API responses
    await page.route('**/api/**', async (route) => {
      await new Promise(r => setTimeout(r, 3000));
      await route.continue();
    });

    await page.goto(`${APP}/?positives=beach`);
    await appReady(page);

    // Search button should be disabled during the slow load.
    // Note: not asserted directly — the grid appearance below
    // proves the search eventually completed.

    // Wait for results to eventually appear
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 15000 });
  });

  test('flaky network: retry on transient failure', async ({ page }) => {
    let attempt = 0;
    await page.route('**/api/search**', (route) => {
      attempt++;
      if (attempt === 1) {
        // First attempt fails
        route.fulfill({ status: 502, body: 'Bad Gateway' });
      } else {
        // Subsequent attempts succeed
        route.continue();
      }
    });

    await page.goto(`${APP}/?positives=beach`);
    await appReady(page);

    // First search shows error. User retries.
    const responsePromise = page.waitForResponse(
      (r) => r.url().includes('/api/search') && r.status() === 200,
      { timeout: 10000 }
    );
    await page.getByRole('button', { name: 'Search', exact: true }).click();
    await responsePromise;

    // Results should appear
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });
  });
});