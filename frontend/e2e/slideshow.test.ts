/**

 * E2E for the Slideshow feature.
 *
 * Slideshow lives inside the Lightbox action bar as a Play/Pause
 * button (no entry point on PhotoGrid). The button is visible
 * whenever the Lightbox has two or more photos — opening a
 * Lightbox does NOT auto-start the slideshow; the user has to
 * press Play. Once playing, the timer advances the photo every
 * `slideshowIntervalMs` (default 4000) and wraps around. Manual
 * prev/next nav pauses the timer.
 *
 * These tests run against the prod stack (PLAYWRIGHT_BASE_URL,
 * default :8000). They will fail until the container image is
 * rebuilt with the slideshow commit — that's expected for a new
 * feature, not a regression.
 */
import { test, expect, type Page } from '@playwright/test';

const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:8000';

test.beforeEach(async ({ page }) => {
  // Photo URLs go through the vite dev proxy when running against
  // :5173, but the regex in vite.config.ts breaks on query strings
  // (`?w=1408`) so images never load. Re-route those to the
  // backend directly. The replacement is of the whole ORIGIN —
  // swapping a bare host for one that carries its own scheme would
  // produce `http://http://…`. When the target already IS the
  // origin the page is on, the rule is a no-op.
  const origin = new URL(APP).origin;
  const reroute = async (route: import('@playwright/test').Route) => {
    const url = route.request().url();
    if (url.startsWith(origin)) {
      await route.continue();
      return;
    }
    await route.continue({ url: url.replace(new URL(url).origin, origin) });
  };
  await page.route('**/photo/*/raw**', reroute);
  await page.route('**/thumb/**', reroute);
});

async function appReady(page: Page) {
  await page.waitForSelector('header.topbar', { timeout: 10_000 });
}

async function gridReady(page: Page) {
  await appReady(page);
  await page.waitForSelector('.grid-tile', { timeout: 10_000 });
}

async function lightboxOpen(page: Page) {
  await expect(page.locator('[role="dialog"]')).toBeVisible({ timeout: 5_000 });
  await page.waitForFunction(
    () => {
      const img = document.querySelector('.overlay img.photo');
      return img instanceof HTMLImageElement && img.complete && img.naturalWidth > 0;
    },
    { timeout: 10_000 }
  );
}

test('PhotoGrid has no Slideshow button — entry point is inside the Lightbox', async ({
  page
}) => {
  await page.goto(APP + '/random');
  await gridReady(page);
  await expect(page.locator('.slideshow-btn')).toHaveCount(0);
});

test('Lightbox shows a Play slideshow button by default (no auto-start)', async ({
  page
}) => {
  await page.goto(APP + '/random');
  await gridReady(page);
  await page.locator('.grid-tile').first().click();
  await lightboxOpen(page);

  // Initial state: not playing. The button title and label
  // advertise Play, and ariaPressed reads false.
  const playBtn = page.locator(
    '.bar.glass-strong button[title^="Play slideshow"]'
  );
  await expect(playBtn).toBeVisible({ timeout: 2_000 });
  await expect(playBtn).toHaveAttribute('aria-pressed', 'false');

  // The pause variant is not in the DOM yet.
  await expect(
    page.locator('.bar.glass-strong button[title^="Pause slideshow"]')
  ).toHaveCount(0);
});

test('Clicking Play starts auto-advance from the current photo', async ({
  page
}) => {
  await page.goto(APP + '/random');
  await gridReady(page);
  await page.locator('.grid-tile').first().click();
  await lightboxOpen(page);

  // Counter starts at "1 / N".
  const counter = page.locator('.bar.glass-strong .count');
  const beforeIdx = Number.parseInt(
    ((await counter.textContent()) ?? '').split('/')[0].trim(),
    10
  );
  expect(beforeIdx).toBeGreaterThanOrEqual(1);

  await page
    .locator('.bar.glass-strong button[title^="Play slideshow"]')
    .click();
  await expect(
    page.locator('.bar.glass-strong button[title^="Pause slideshow"]')
  ).toBeVisible({ timeout: 2_000 });

  // Default intervalMs is 3000 (3s) — within ~3.5s the counter
  // should advance.
  await page.waitForTimeout(3500);
  const after = (await counter.textContent()) ?? '';
  const afterIdx = Number.parseInt(after.split('/')[0].trim(), 10);
  expect(afterIdx).not.toBe(beforeIdx);
});

test('Clicking Pause stops the timer and flips the button to Play', async ({
  page
}) => {
  await page.goto(APP + '/random');
  await gridReady(page);
  await page.locator('.grid-tile').first().click();
  await lightboxOpen(page);

  await page
    .locator('.bar.glass-strong button[title^="Play slideshow"]')
    .click();
  await expect(
    page.locator('.bar.glass-strong button[title^="Pause slideshow"]')
  ).toBeVisible({ timeout: 2_000 });

  await page
    .locator('.bar.glass-strong button[title^="Pause slideshow"]')
    .click();
  await expect(
    page.locator('.bar.glass-strong button[title^="Play slideshow"]')
  ).toBeVisible({ timeout: 2_000 });

  // Counter should NOT advance while paused.
  const counter = page.locator('.bar.glass-strong .count');
  const before = (await counter.textContent()) ?? '';
  await page.waitForTimeout(4500);
  const after = (await counter.textContent()) ?? '';
  expect(after).toBe(before);
});

test('Manual prev/next nav pauses the slideshow', async ({ page }) => {
  await page.goto(APP + '/random');
  await gridReady(page);
  await page.locator('.grid-tile').first().click();
  await lightboxOpen(page);

  await page
    .locator('.bar.glass-strong button[title^="Play slideshow"]')
    .click();
  await expect(
    page.locator('.bar.glass-strong button[title^="Pause slideshow"]')
  ).toBeVisible({ timeout: 2_000 });

  // Press ArrowRight — that should both advance the photo AND
  // flip the Play/Pause button back to Play.
  await page.keyboard.press('ArrowRight');
  await expect(
    page.locator('.bar.glass-strong button[title^="Play slideshow"]')
  ).toBeVisible({ timeout: 2_000 });

  // Counter should NOT advance after the manual nav (timer was
  // paused by the keyboard handler).
  const counter = page.locator('.bar.glass-strong .count');
  const before = (await counter.textContent()) ?? '';
  await page.waitForTimeout(4500);
  const after = (await counter.textContent()) ?? '';
  expect(after).toBe(before);
});

test('Lightbox hides the Slideshow button when there is only one photo', async ({
  page
}) => {
  // /photo/{id} is a dedicated single-photo layout — no PhotoGrid,
  // no toolbar, and the Lightbox (if present) only ever wraps one
  // item so there's no point showing a Slideshow affordance.
  await page.goto(APP + '/random');
  await gridReady(page);
  const firstId = await page.evaluate(() => {
    const link = document.querySelector(
      'a[href^="/photo/"]'
    ) as HTMLAnchorElement | null;
    return link?.href.split('/photo/')[1] ?? null;
  });
  expect(firstId).toBeTruthy();
  await page.goto(APP + `/photo/${firstId}`);
  await appReady(page);
  // No Slideshow button on the dedicated photo page.
  await expect(page.locator('.slideshow-btn')).toHaveCount(0);
  // And no Play slideshow button inside any dialog (the page
  // shows its own photo viewer, not the Lightbox).
  await expect(
    page.locator('[role="dialog"] button[title^="Play slideshow"]')
  ).toHaveCount(0);
});

test('Settings page exposes the slideshow duration presets', async ({
  page
}) => {
  await page.goto(APP + '/settings');
  await appReady(page);
  // Wait for the page to render the slideshow card. It might
  // be after the Index card's status load, but the card itself
  // is independent of /api/admin/index/status.
  const radiogroup = page.locator('[role="radiogroup"][aria-label="Slideshow photo duration"]');
  await expect(radiogroup).toBeVisible({ timeout: 5_000 });

  // Six preset chips (1s, 3s, 5s, 10s, 20s, 30s), default 3s
  // selected. The 1s preset was added in this round so the user
  // can pin a fast cadence for quick browsing.
  const presets = radiogroup.locator('button.preset');
  await expect(presets).toHaveCount(6);

  // The default (3s) starts active — aria-checked reflects the
  // current setting and data-active drives the visual highlight.
  const active = radiogroup.locator('button.preset[data-active="true"]');
  await expect(active).toHaveCount(1);
  await expect(active.first()).toHaveText('3s');
  await expect(active.first()).toHaveAttribute('aria-checked', 'true');
});

test('Selecting a different preset from Settings changes the Lightbox cadence', async ({
  page
}) => {
  await page.goto(APP + '/settings');
  await appReady(page);
  const radiogroup = page.locator('[role="radiogroup"][aria-label="Slideshow photo duration"]');
  await expect(radiogroup).toBeVisible({ timeout: 5_000 });

  // Pick 5s — should now be the active preset.
  await radiogroup.locator('button.preset', { hasText: '5s' }).click();
  await expect(
    radiogroup.locator('button.preset[data-active="true"]')
  ).toHaveText('5s');

  // Open the lightbox from the grid and verify the new cadence
  // is in effect: counter should NOT advance within 3.5s (the
  // old default), but SHOULD advance by ~5s.
  await page.goto(APP + '/random');
  await gridReady(page);
  await page.locator('.grid-tile').first().click();
  await lightboxOpen(page);
  await page
    .locator('.bar.glass-strong button[title^="Play slideshow"]')
    .click();
  await expect(
    page.locator('.bar.glass-strong button[title^="Pause slideshow"]')
  ).toBeVisible({ timeout: 2_000 });

  const counter = page.locator('.bar.glass-strong .count');
  const before = (await counter.textContent()) ?? '';

  // Wait 3.5s — under the new 5s cadence, the timer should NOT
  // have fired yet. If the preference didn't apply, this would
  // also be where the old 3s default would fire — so this
  // single check distinguishes "preference took effect" from
  // "preference silently ignored".
  await page.waitForTimeout(3500);
  expect((await counter.textContent()) ?? '').toBe(before);

  // Wait another 2.5s (total ~6s) — the 5s timer has fired at
  // least once, the slideshow has advanced.
  await page.waitForTimeout(2500);
  const after = (await counter.textContent()) ?? '';
  expect(after).not.toBe(before);
});

test('Pressing Space toggles Play/Pause when no button is focused', async ({
  page
}) => {
  await page.goto(APP + '/random');
  await gridReady(page);
  await page.locator('.grid-tile').first().click();
  await lightboxOpen(page);

  // Confirm Space toggles Play → Pause → Play when focus is
  // somewhere non-interactive. We blur any active element so
  // the keydown target is body — without this, clicking the
  // tile in step 2 leaves focus on the tile (an <a> inside
  // PhotoTile), which would route Space to that element.
  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
  });

  await page.keyboard.press('Space');
  await expect(
    page.locator('.bar.glass-strong button[title^="Pause slideshow"]')
  ).toBeVisible({ timeout: 1_000 });

  // Counter should advance within ~3.5s (default 3s cadence).
  const counter = page.locator('.bar.glass-strong .count');
  const beforeIdx = Number.parseInt(
    ((await counter.textContent()) ?? '').split('/')[0].trim(),
    10
  );
  await page.waitForTimeout(3500);
  const afterIdx = Number.parseInt(
    ((await counter.textContent()) ?? '').split('/')[0].trim(),
    10
  );
  expect(afterIdx).not.toBe(beforeIdx);

  // Space again pauses it.
  await page.keyboard.press('Space');
  await expect(
    page.locator('.bar.glass-strong button[title^="Play slideshow"]')
  ).toBeVisible({ timeout: 1_000 });
  const paused = (await counter.textContent()) ?? '';
  await page.waitForTimeout(3500);
  expect((await counter.textContent()) ?? '').toBe(paused);
});

test('Pressing Space with the Play/Pause button focused fires it once (no double-toggle)', async ({
  page
}) => {
  await page.goto(APP + '/random');
  await gridReady(page);
  await page.locator('.grid-tile').first().click();
  await lightboxOpen(page);

  // Focus the Play button directly.
  const playBtn = page.locator(
    '.bar.glass-strong button[title^="Play slideshow"]'
  );
  await playBtn.focus();
  await page.keyboard.press('Space');

  // Browser native click handler fires togglePlay once. Our
  // window keydown handler must NOT also fire — otherwise the
  // two toggles would net out to no-op and the timer would
  // never start.
  await expect(
    page.locator('.bar.glass-strong button[title^="Pause slideshow"]')
  ).toBeVisible({ timeout: 1_000 });

  // Confirm the slideshow actually advanced — if double-fire
  // had happened, playing would be back to false and the timer
  // would never fire.
  const counter = page.locator('.bar.glass-strong .count');
  const before = (await counter.textContent()) ?? '';
  await page.waitForTimeout(3500);
  expect((await counter.textContent()) ?? '').not.toBe(before);
});

test('Play/Pause button tooltip advertises the Space keyboard shortcut', async ({
  page
}) => {
  await page.goto(APP + '/random');
  await gridReady(page);
  await page.locator('.grid-tile').first().click();
  await lightboxOpen(page);
  const playBtn = page.locator(
    '.bar.glass-strong button[title^="Play slideshow"]'
  );
  await expect(playBtn).toHaveAttribute('title', 'Play slideshow (Space)');
  await playBtn.click();
  await expect(
    page.locator('.bar.glass-strong button[title^="Pause slideshow"]')
  ).toHaveAttribute('title', 'Pause slideshow (Space)');
});