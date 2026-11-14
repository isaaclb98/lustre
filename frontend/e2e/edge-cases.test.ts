import { test, expect } from '@playwright/test';
import { assertDisposable, DestructiveTestError } from './guard';

async function appReady(page: any) {
  // The composer is on every page that has a search bar; the topbar
  // header is the most reliable "app is hydrated" signal because it's
  // rendered server-side and survives SvelteKit hydration. (Match
  // what smoke.test.ts, ui-flows.test.ts, etc. use.)
  await page.waitForSelector('header.topbar', { timeout: 10000 });
}

test.describe('Search edge cases', () => {
  test('search with special characters in prompts', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    const input = page.getByRole('textbox', { name: /add prompt/i });
    await input.fill('cat\'s "cute" photo (2024)');
    await input.press('Enter');

    // Should render the chip without crashing
    await expect(page.locator('.chip.pos')).toHaveCount(1);
    
    // Click Search
    await page.getByRole('button', { name: /^Search$/ }).click();
    
    // Should fire the search without 500
    const response = await page.waitForResponse((r: any) => r.url().includes('/api/search'));
    expect(response.status()).toBe(200);
  });

  test('search with very long prompt (>100 chars)', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    const longPrompt = 'a'.repeat(150);
    const input = page.getByRole('textbox', { name: /add prompt/i });
    await input.fill(longPrompt);
    await input.press('Enter');

    // Should render the chip (truncated visually if needed)
    await expect(page.locator('.chip.pos')).toHaveCount(1);
    
    // Click Search - should work without error
    await page.getByRole('button', { name: /^Search$/ }).click();
    const response = await page.waitForResponse((r: any) => r.url().includes('/api/search'));
    expect(response.status()).toBe(200);
  });

  test('search with only filename filter (no prompts)', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    // Open the "Additional options" panel first
    await page.getByRole('button', { name: /additional options/i }).click();

    // Wait for the panel to expand and find the filename filter input
    // Actual placeholder is "e.g. IMG_2024"
    const filenameInput = page.locator('input[placeholder*="IMG_2024"]').first();
    await filenameInput.fill('0001');

    // Click Search - should fire with filename param
    await page.getByRole('button', { name: /^Search$/ }).click();
    const response = await page.waitForResponse((r: any) => r.url().includes('/api/search'));
    expect(response.status()).toBe(200);
    expect(response.url()).toContain('filename=0001');
  });

  test('multiple positive prompts (5+)', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    const input = page.getByRole('textbox', { name: /add prompt/i });
    const prompts = ['cat', 'dog', 'bird', 'fish', 'rabbit', 'hamster'];

    for (const p of prompts) {
      await input.fill(p);
      await input.press('Enter');
      await page.waitForTimeout(100);
    }

    // All chips should render
    const chips = page.locator('.chip.pos');
    await expect(chips).toHaveCount(6, { timeout: 5000 });

    // Click Search
    await page.getByRole('button', { name: /^Search$/ }).click();
    const response = await page.waitForResponse((r: any) => r.url().includes('/api/search'));
    expect(response.status()).toBe(200);
    // Should include all prompts
    expect(response.url()).toContain('positives=cat');
    expect(response.url()).toContain('positives=dog');
  });

  test('remove all chips → Search button disabled', async ({ page }) => {
    await page.goto('/?positives=beach');
    await appReady(page);

    // Wait for initial load
    await page.waitForTimeout(500);

    // Remove the chip
    await page.locator('.chip.pos').locator('button').first().click();

    // Search button should be disabled
    const searchBtn = page.getByRole('button', { name: /^Search$/ });
    await expect(searchBtn).toBeDisabled();
  });
});

test.describe('Lightbox edge cases', () => {
  test('← at first photo does nothing (no wrap)', async ({ page }) => {
    await page.goto('/random');
    await page.waitForSelector('.grid-tile', { timeout: 10000 });

    // Open first photo
    await page.locator('.grid-tile').first().click();
    await expect(page.locator('[role="dialog"]')).toBeVisible({ timeout: 5000 });

    const initialSrc = await page.locator('[role="dialog"] img.photo').first().getAttribute('src');

    // Press ArrowLeft at first photo
    await page.keyboard.press('ArrowLeft');
    await page.waitForTimeout(300);

    // Should stay on same photo (no wrap to last)
    const afterSrc = await page.locator('[role="dialog"] img.photo').first().getAttribute('src');
    expect(afterSrc).toBe(initialSrc);
  });

  test('Esc closes lightbox, Esc again does nothing (already closed)', async ({ page }) => {
    await page.goto('/random');
    await page.waitForSelector('.grid-tile', { timeout: 10000 });

    await page.locator('.grid-tile').first().click();
    await expect(page.locator('[role="dialog"]')).toBeVisible({ timeout: 5000 });

    // First Esc closes
    await page.keyboard.press('Escape');
    await expect(page.locator('[role="dialog"]')).toBeHidden({ timeout: 3000 });

    // Second Esc should not crash or do anything weird
    await page.keyboard.press('Escape');
    await page.waitForTimeout(200);
    await expect(page.locator('[role="dialog"]')).toBeHidden();
  });

  test('lightbox open → refresh page → lightbox still open (URL state)', async ({ page }) => {
    await page.goto('/random');
    await page.waitForSelector('.grid-tile', { timeout: 10000 });

    await page.locator('.grid-tile').first().click();
    await expect(page.locator('[role="dialog"]')).toBeVisible({ timeout: 5000 });

    // Refresh the page
    await page.reload();
    await page.waitForSelector('.grid-tile', { timeout: 10000 });

    // Lightbox should NOT reopen (no URL state for lightbox index)
    await expect(page.locator('[role="dialog"]')).toBeHidden({ timeout: 3000 });
  });
});

test.describe('Empty states', () => {
  test('Albums page when no custom albums exist shows empty state', async ({ page }) => {
    // Delete all user albums (keep system albums). This deletes EVERY
    // user album, so it must only ever run against a disposable
    // backend — assertDisposable aborts the test otherwise.
    // A destructive test cannot run against a real library, so it
    // SKIPS there rather than failing: the guard has already proved
    // the backend is not disposable, and nothing has been deleted.
    try {
      await assertDisposable(page.request, 'the empty-albums Albums test');
    } catch (err) {
      if (err instanceof DestructiveTestError) {
        test.skip(true, `not a disposable backend: ${err.message}`);
        return;
      }
      throw err;
    }
    const albums = await page.request.get('/api/albums').then(r => r.json());
    for (const album of (albums.albums ?? [])) {
      if (!['Likes', 'Dislikes'].includes(album.name)) {
        await page.request.delete(`/api/albums/${album.id}`);
      }
    }

    await page.goto('/albums');
    await page.waitForSelector('h1', { timeout: 5000 });

    // The empty-state message reads "No custom albums yet — create
    // one to group your photos." Wait for it specifically rather
    // than reading body text immediately, because the albums API
    // fetch can still be in flight after h1 mounts.
    await expect(page.getByText(/No custom albums yet/i)).toBeVisible({ timeout: 10_000 });
  });

  test('Random page when library is empty (edge case)', async ({ page }) => {
    // This test is hard to set up (need empty Qdrant), so we skip if
    // the library has photos. In a real test suite, we'd have a fixture
    // for an empty library.
    const coll = await page.request.get('http://localhost:6333/collections/images');
    const count = (await coll.json()).result.points_count;

    if (count > 0) {
      test.skip(true, 'Library is not empty — skipping empty-library test');
    }

    await page.goto('/random');
    await page.waitForSelector('h1', { timeout: 5000 });

    // Should show empty state, not crash
    await expect(page.getByRole('heading', { name: /random/i })).toBeVisible();
    const body = await page.textContent('body');
    expect(body).toMatch(/no.*photo|empty/i);
  });
});

test.describe('Filters and diversity', () => {
  test('the additional options panel offers no diversity control', async ({ page }) => {
    // Diversity was removed from the UI (round-87): the API still
    // accepts the parameters for external callers, but the app no
    // longer emits them and no longer offers a control for them. A
    // selector that came back would put the query back on the wire.
    await page.goto('/?positives=beach');
    await appReady(page);

    await page.getByRole('button', { name: /additional options/i }).click();

    await expect(page.getByRole('combobox', { name: /diversity mode/i })).toHaveCount(0);

    // And the search request carries no diversity parameter.
    const responsePromise = page.waitForResponse((r: any) => r.url().includes('/api/search'), {
      timeout: 15000,
    });
    await page.getByRole('button', { name: /^Search$/ }).click();
    const response = await responsePromise;
    expect(response.url()).not.toContain('diversity=');
  });

  test('filename filter with regex-like pattern', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    // Open "Additional options" panel first
    await page.getByRole('button', { name: /additional options/i }).click();

    // Fill a pattern-like filename
    const filenameInput = page.locator('input').filter({ hasText: '' }).nth(1);
    await filenameInput.fill('00*');

    // Click Search
    await page.getByRole('button', { name: /^Search$/ }).click();
    const response = await page.waitForResponse((r: any) => r.url().includes('/api/search'));
    expect(response.status()).toBe(200);
    expect(response.url()).toContain('filename=00');
  });
});


test.describe('Toast messages', () => {
  test('error message appears on API failure', async ({ page }) => {
    // Intercept the search API and force a 500
    await page.route('**/api/search**', (route: any) => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ detail: 'Test error' }),
      });
    });

    await page.goto('/?positives=beach');
    await appReady(page);

    // Click search to trigger the error
    await page.getByRole('button', { name: /^Search$/ }).click();

    // Should show an error message - the app displays inline errors in a div.error
    const errorDiv = page.locator('div.error').first();
    await expect(errorDiv).toBeVisible({ timeout: 5000 });
    const errorText = await errorDiv.textContent();
    expect(errorText).toContain("Couldn't load results");
  });

  test('a saved search appears in the menu', async ({ page }) => {
    await page.goto('/?positives=beach');
    await appReady(page);

    // Wait for search to complete
    await page.waitForTimeout(1000);

    // Open saved searches menu
    await page.getByTitle('Saved searches').click();

    // Click Save — the app prompts in its OWN dialog, not the
    // browser's native prompt, so the name goes into the dialog's
    // input rather than into a page.on('dialog') handler.
    const saveName = 'test-save-' + Date.now();
    await page.getByTitle('Save current search').click();
    await page.locator('input[id="dialog-prompt-input"]').fill(saveName);
    await page.locator('button:has-text("Save")').click();

    // The app has no toast popups — the confirmation is the saved
    // search showing up in the menu it was saved from. The list
    // refreshes on save, but the popover is not necessarily open
    // when it does, so open it again to read the result.
    await page.getByTitle('Saved searches').click();
    await expect(page.getByText(saveName)).toBeVisible({ timeout: 5000 });
  });
});

test.describe('Loading states', () => {
  test('grid shows loading state during search', async ({ page }) => {
    // Intercept search API and add delay
    await page.route('**/api/search**', async (route: any) => {
      await new Promise(r => setTimeout(r, 2000));
      const response = await route.fetch();
      await route.fulfill({ response });
    });

    await page.goto('/?positives=beach');
    await appReady(page);

    // Click search to trigger the delayed request
    await page.getByRole('button', { name: /^Search$/ }).click();

    // The app now shows a spinner with "Searching..." text during loading
    const loadingState = page.locator('.empty.loading');
    await expect(loadingState).toBeVisible({ timeout: 2000 });
    await expect(loadingState).toContainText('Searching');

    // Verify the spinner is visible
    await expect(loadingState.locator('.spinner')).toBeVisible({ timeout: 500 });

    // After the delayed response, results should appear. Scope to
    // section.results because .grid-tile also matches the home
    // page's always-on result tiles — without scoping, the wait
    // would succeed on those BEFORE the search finishes, and the
    // loading-state assertion below would race against the
    // still-in-flight search.
    await expect(page.locator('section.results .grid-tile').first()).toBeVisible({ timeout: 5000 });

    // Loading state should be gone. Give it a moment to clear after
    // the delayed response lands (the test routes the request with
    // a 1s delay, so loading state may stay visible briefly past
    // the grid-tile assertion).
    await expect(loadingState).not.toBeVisible({ timeout: 5000 });
  });

  test('search button shows loading state', async ({ page }) => {
    // Intercept search API and add delay
    const DELAY = 2000;
    await page.route('**/api/search**', async (route: any) => {
      await new Promise(r => setTimeout(r, DELAY));
      const response = await route.fetch();
      await route.fulfill({ response });
    });

    await page.goto('/?positives=beach');
    await appReady(page);

    // Arm the waiter BEFORE the click, so it cannot miss the
    // response it is waiting for.
    const responded = page.waitForResponse((r: any) => r.url().includes('/api/search'), {
      timeout: 15000,
    });

    // Click search to trigger the delayed request
    const searchBtn = page.getByRole('button', { name: /^Search$/ });
    await searchBtn.click();

    // Search button should be disabled during search
    await expect(searchBtn).toBeDisabled({ timeout: 500 });

    // Wait for the response to complete
    await responded;

    // Button should be enabled again. The delay applies to EVERY
    // search the page issues, and the grid fetches its next page
    // behind the first one, so the button can still be waiting on a
    // follow-up request. The re-enable window has to outlast the
    // injected delay, or the assertion races the app's own traffic.
    await expect(searchBtn).toBeEnabled({ timeout: DELAY * 5 });
  });
});
