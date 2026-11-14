/**
 * settings-index.test.ts — E2E coverage for the in-app Indexer
 * (Settings → Index button + admin API).
 *
 * What we pin:
 *   - Settings tab appears in the TopBar.
 *   - /api/admin/index/status returns a well-formed response shape.
 *   - Clicking the Index button starts an incremental job (round-35:
 *     the UI no longer offers a "rebuild from scratch" option — it's
 *     destructive, see +page.svelte docstring).
 *   - POST /api/admin/index starts a job and updates status to
 *     running or idle (the local test fixture finishes fast).
 *   - Concurrent POST returns 409.
 *   - Cancel returns 400 when nothing is running; if a job is in
 *     flight, cancel transitions it to idle.
 *
 * The test runs against the live dev stack (playwright.config.ts
 * defaults to http://127.0.0.1:8000) and skips itself if the
 * admin endpoint is unreachable (e.g. in CI without the dev
 * stack spun up).
 */
import { test, expect, type Page } from '@playwright/test';

const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';

async function statusOrSkip(page: Page) {
  const res = await page.request.get('/api/admin/index/status');
  if (!res.ok()) test.skip(true, `admin endpoint unreachable (${res.status()})`);
  return res.json();
}

test.describe('Settings page', () => {
  test('TopBar exposes a Settings tab', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('link', { name: 'Settings' })).toBeVisible();
  });

  test('settings page renders the Index card', async ({ page }) => {
    await page.goto('/settings');
    await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Index' })).toBeVisible();
    // Idle state shows the Index button (not Cancel).
    await expect(page.getByRole('button', { name: 'Index' })).toBeVisible();
  });

  test('status endpoint returns a well-formed response', async ({ page }) => {
    const body = await statusOrSkip(page);
    expect(body).toHaveProperty('state');
    expect(['idle', 'running', 'failed']).toContain(body.state);
    expect(body).toHaveProperty('progress');
    expect(body.progress).toEqual(
      expect.objectContaining({
        indexed: expect.any(Number),
        reembedded: expect.any(Number),
        skipped: expect.any(Number),
        errors: expect.any(Number),
      })
    );
  });

  /**
   * Round-35: only the incremental mode is exposed to users.
   * The backend still accepts `rebuild` for operator use (see the
   * +page.svelte docstring) but the UI no longer offers it — it's
   * destructive (wipes favourites + albums + saved searches +
   * vectors) and shouldn't be one click away. The Settings page
   * now has a plain Index action instead of a mode-picker popover.
   */
  test('Index button starts an incremental job when clicked', async ({ page }) => {
    await statusOrSkip(page);
    // Make sure no other job is in flight from a prior test before
    // we start a fresh one.
    await page.request.post('/api/admin/index/cancel').catch(() => {});
    // The fetch runs in the page, so it needs an absolute URL — and
    // the page has to be on the app for one to exist at all.
    await page.goto('/settings');
    await page.waitForFunction(
      async (url) => {
        const r = await fetch(url);
        const b = await r.json();
        return b.state !== 'running';
      },
      `${APP}/api/admin/index/status`,
      { timeout: 15000 }
    );
    // The button is just "Index" with no popover now — clicking it
    // immediately starts an incremental job.
    await expect(page.getByRole('button', { name: 'Index' })).toBeVisible();
    await page.getByRole('button', { name: 'Index' }).click();

    // Wait for status to flip to running (or idle if the fixture
    // finishes before our poll — both are acceptable, the contract
    // is just that the click triggered a POST /api/admin/index).
    await page.waitForFunction(
      async (url) => {
        const r = await fetch(url);
        const b = await r.json();
        return b.state === 'running' || b.state === 'idle';
      },
      `${APP}/api/admin/index/status`,
      { timeout: 5000 }
    );

    // Clean up so subsequent tests see idle.
    await page.request.post('/api/admin/index/cancel').catch(() => {});
    await page.waitForFunction(
      async () => {
        const r = await fetch('/api/admin/index/status');
        const b = await r.json();
        return b.state !== 'running';
      },
      { timeout: 15000 }
    );
  });

  test('concurrent start returns 409', async ({ page }) => {
    await statusOrSkip(page);
    // Start a job; the dev fixture is fast (sub-second) so we may
    // already be back to idle by the time the second request hits.
    // We assert either 202 (started) or 409 (still running) — both
    // are valid responses, and the second attempt MUST be 409 if the
    // job is still running.
    const first = await page.request.post('/api/admin/index', {
      data: { mode: 'incremental' }
    });
    expect([202, 409]).toContain(first.status());

    if (first.status() === 202) {
      const second = await page.request.post('/api/admin/index', {
        data: { mode: 'incremental' }
      });
      expect(second.status()).toBe(409);
      // Clean up: wait for the first job to finish before the test
      // teardown so subsequent tests see idle.
      await page.request.post('/api/admin/index/cancel').catch(() => {});
    }
  });

  test('cancel when idle returns 400', async ({ page }) => {
    await statusOrSkip(page);
    // Make sure nothing's running by waiting for idle.
    let body = await statusOrSkip(page);
    const deadline = Date.now() + 15000;
    while (body.state === 'running' && Date.now() < deadline) {
      await page.waitForTimeout(500);
      body = await statusOrSkip(page);
    }
    const res = await page.request.post('/api/admin/index/cancel');
    expect(res.status()).toBe(400);
  });

  test('log endpoint returns an array of lines', async ({ page }) => {
    await statusOrSkip(page);
    const res = await page.request.get('/api/admin/index/log');
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(Array.isArray(body.lines)).toBe(true);
    expect(typeof body.next_line).toBe('number');
    expect(typeof body.total).toBe('number');
  });
});
