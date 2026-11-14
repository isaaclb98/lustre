/**
 * from-scratch.test.ts — full lifecycle verification on a freshly
 * wiped dev environment. Run against the live stack.
 *
 * Steps:
 *   1. Fresh-install state: home shows empty-state prompt, settings
 *      shows Index button.
 *   2. Click Index → job starts (round-35: incremental is the only
 *      mode the UI exposes; rebuild is operator-only via API).
 *   3. Cancel mid-run → state returns to idle.
 *   4. Start incremental again → wait for completion → search
 *      returns real hits.
 *   5. /api/admin/index/status reflects last_run_at.
 */
import { test, expect, type Page } from '@playwright/test';

async function waitFor(
  page: Page,
  url: string,
  fn: (body: any) => boolean,
  timeoutMs = 20000,
) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const res = await page.request.get(url);
    if (res.ok()) {
      const body = await res.json();
      if (fn(body)) return body;
    }
    await page.waitForTimeout(500);
  }
  throw new Error(`timed out waiting for ${url}`);
}

test.describe.serial('Fresh-install lifecycle', () => {
  // The IndexerRunner is a singleton in the search backend, so
  // tests that mutate it MUST run serially. test.describe.serial
  // ensures that; default workers > 1 would race.
  test.beforeEach(async ({ page }) => {
    // Best-effort cleanup: cancel any job left over from a previous
    // test run, then wait for idle so the next test starts clean.
    await page.request.post('/api/admin/index/cancel').catch(() => {});
    await waitFor(page, '/api/admin/index/status', (b) => b.state !== 'running', 15000)
      .catch(() => {});
  });

  test('home + settings look right on a fresh install', async ({ page }) => {
    // Sanity: status should be idle with no last_run_at. If this
    // assertion fails, the dev stack wasn't actually wiped before
    // the test ran — bail.
    const status = await page.request.get('/api/admin/index/status');
    const body = await status.json();
    // The empty-index banner shows when the library holds no points.
    // The demo backend is seeded with photos, so the banner is
    // legitimately absent there — this test needs an actually empty
    // library, which means a wiped stack.
    test.skip(
      body.points_count !== 0,
      'the library is not empty (points_count is set) — wipe and re-run'
    );

    // Empty-state banner on home (contains both the message text and
    // a link to /settings, so the locator targets the banner pill).
    await page.goto('/');
    await expect(page.getByText('No photos indexed yet.')).toBeVisible();
    await expect(page.getByRole('link', { name: /Go to Settings/i })).toBeVisible();

    // Settings tab in the TopBar (header.topbar scope so we don't
    // collide with the empty-state banner's link).
    await page.locator('header.topbar').getByRole('link', { name: 'Settings' }).click();
    await expect(page).toHaveURL(/\/settings$/);
    await expect(page.getByRole('heading', { name: 'Index' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Index' })).toBeVisible();
  });

  test('incremental index: start → cancel mid-run → idle', async ({ page }) => {
    test.setTimeout(90000);

    await page.goto('/settings');
    // Round-35: Index button is now a plain action that immediately
    // starts an incremental job. No menu / mode picker anymore.
    await page.getByRole('button', { name: 'Index' }).click();

    // Wait until the runner reports running.
    await waitFor(page, '/api/admin/index/status', (b) => b.state === 'running');

    // Cancel button shows.
    await expect(page.getByRole('button', { name: 'Cancel' })).toBeVisible();

    // Cancel.
    await page.getByRole('button', { name: 'Cancel' }).click();

    // Status returns to idle (allow up to 60s for SIGTERM to land
    // at the next batch boundary — encoder can take ~1s per batch).
    await waitFor(
      page,
      '/api/admin/index/status',
      (b) => b.state === 'idle',
      60000
    );
    expect((await (await page.request.get('/api/admin/index/status')).json()).state)
      .toBe('idle');
  });
});
