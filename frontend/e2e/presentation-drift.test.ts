/**
 * E2E for the DRIFT wall — the album as a wall that moves itself.
 *
 * The still wall is a page you scroll; the drift is the same wall
 * with a time axis. The photos, the gapless square field, the random
 * order and the read-only tiles are all the still wall's — what is
 * new is that the wall scrolls UPWARD on its own, forever, without
 * anything being clicked.
 *
 * What this pins:
 *
 *  1. The drift MOVES. The wall's own track transform changes over
 *     time — that is the whole feature, and it is what a
 *     screenshot or a static render could never prove.
 *  2. It can be STOPPED, and stopped means stopped: the transform
 *     stops changing while paused. A pause button that only hides
 *     the motion would look identical for a single frame.
 *  3. It is PAUSABLE, which means it carries exactly one control —
 *     the pause button — while the still wall carries none. The
 *     wall's other promises (gapless, square, standalone, no app
 *     chrome, read-only) are unchanged.
 *  4. `prefers-reduced-motion` parks it: the wall does not move on
 *     its own, and the control says so.
 *
 * The wall is not a scroll surface: the document must not scroll,
 * because the wall is exactly one screen and it moves itself.
 */
import { test, expect, type ElementHandle, type Page } from '@playwright/test';

const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';
import { driftUrl, smallestAlbum } from './albums';

/**
 * The vertical position of one photo on the drift wall, in pixels.
 *
 * NOT the track's transform. The track is anchored to the wall's
 * origin, so its transform SAWTOOTHS: it slides up continuously and
 * then jumps back down by one row height whenever the rendered
 * window advances a row. A sampler straddling that jump sees the
 * wall "move downward" — and with a 700ms window on a 16s row
 * period that is a 4% flake per assertion.
 *
 * A single PHOTO's position is monotonic instead: it moves up at
 * the drift speed until it leaves the top of the wall. The handle
 * is taken once, on the row in the middle of the screen, so it
 * refers to the same element across both samples — that element
 * stays in the rendered window for as long as it takes to cross the
 * whole screen.
 */
async function photoY(page: Page, handle: ElementHandle<SVGElement | HTMLElement>) {
  const box = await handle.boundingBox();
  return box ? box.y : NaN;
}

/**
 * A handle on a photo near the middle of the screen.
 *
 * The middle row is far from both ends of the rendered window, so
 * the element it names survives the whole measurement.
 */
async function middlePhoto(page: Page) {
  const handle = await page.evaluateHandle(() => {
    const rows = Array.from(document.querySelectorAll('.grid-row'));
    const mid = rows[Math.floor(rows.length / 2)];
    return mid?.querySelector('.tile') ?? null;
  });
  return handle.asElement();
}

test.describe('presentation drift', () => {
  test('the wall moves itself', async ({ page }) => {
    await page.goto(await driftUrl(page));
    await page.waitForSelector('.tile', { timeout: 30000 });

    const photo = await middlePhoto(page);
    expect(photo, 'the drift wall must render photos').not.toBeNull();

    const before = await photoY(page, photo);
    expect(Number.isNaN(before), 'the photo must be on screen').toBe(false);

    // The wall drifts at 16px/s, so 700ms is 11px — a change any
    // browser can see.
    await page.waitForTimeout(700);
    const after = await photoY(page, photo);

    expect(after).not.toBe(before);
    // Upward is a SMALLER y: the wall's content moves toward the
    // top of the screen.
    expect(after).toBeLessThan(before);
  });

  test('the pause control stops the wall, and resumes it', async ({ page }) => {
    await page.goto(await driftUrl(page));
    await page.waitForSelector('.tile', { timeout: 30000 });

    const pause = page.locator('.drift-pause');
    await expect(pause).toHaveCount(1);
    await expect(pause).toHaveAttribute('aria-label', 'Pause the drift');

    await pause.click();
    await expect(pause).toHaveAttribute('aria-label', 'Resume the drift');

    const photo = await middlePhoto(page);
    const parked = await photoY(page, photo);
    await page.waitForTimeout(900);
    expect(await photoY(page, photo)).toBe(parked);

    await pause.click();
    await expect(pause).toHaveAttribute('aria-label', 'Pause the drift');

    await page.waitForTimeout(700);
    expect(await photoY(page, photo)).toBeLessThan(parked);
  });

  test('it keeps the still wall’s look — gapless and square', async ({ page }) => {
    await page.goto(await driftUrl(page));
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
        gapBetweenTiles: Math.round(b.x - (a.x + a.width)),
        width: Math.round(a.width),
        height: Math.round(a.height)
      };
    });

    expect(geom.gap).toBe('0px');
    expect(geom.gapBetweenTiles).toBe(0);
    expect(geom.radius).toBe('0px');
    expect(geom.borderStyle).toBe('none');
    expect(geom.width).toBe(geom.height);
  });

  test('it is on its own — no app chrome, and one control', async ({ page }) => {
    await page.goto(await driftUrl(page));
    await page.waitForSelector('.tile', { timeout: 30000 });

    await expect(page.locator('.topbar')).toHaveCount(0);
    await expect(page.locator('.app-shell')).toHaveCount(0);
    await expect(page.locator('.shell')).toHaveCount(0);

    // Exactly one button: the pause. The still wall has none, but
    // content that moves on its own has to be pausable.
    await expect(page.locator('button')).toHaveCount(1);

    // Still read-only: no way to favourite, dislike or re-file.
    await expect(page.locator('button.remove-btn')).toHaveCount(0);
    await expect(page.locator('button.vote-btn')).toHaveCount(0);
    await expect(page.locator('.fav')).toHaveCount(0);
    await expect(page.locator('a.tile')).toHaveCount(0);
  });

  test('the document does not scroll — the wall is the screen', async ({ page }) => {
    await page.goto(await driftUrl(page));
    await page.waitForSelector('.tile', { timeout: 30000 });

    const scroll = await page.evaluate(() => ({
      y: window.scrollY,
      scrollHeight: document.documentElement.scrollHeight,
      clientHeight: document.documentElement.clientHeight
    }));

    expect(scroll.y).toBe(0);
    expect(scroll.scrollHeight).toBe(scroll.clientHeight);
  });

  test('prefers-reduced-motion parks the wall', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto(await driftUrl(page));
    await page.waitForSelector('.tile', { timeout: 30000 });

    // The control says the wall is stopped, and the wall is.
    await expect(page.locator('.drift-pause')).toHaveAttribute(
      'aria-label',
      'Resume the drift'
    );

    const photo = await middlePhoto(page);
    const parked = await photoY(page, photo);
    await page.waitForTimeout(900);
    expect(await photoY(page, photo)).toBe(parked);
  });

  test('the wall loops the album without a seam', async ({ page }) => {
    // The ring is the whole trick: the wall shows the album's photos
    // in a loop, so the sequence must be PERIODIC — tile i and tile
    // i+N are the same photo, where N is the album's size — and the
    // first cycle must cover the album exactly once with no photo
    // repeated inside it.
    //
    // The wall's order is the route's RANDOM order, so it cannot be
    // compared against the API's insertion order: only the
    // periodicity is a property of the wall itself.
    //
    // A cycle only fits on screen if the album is SMALLER than the
    // window of tiles, so this needs an album with fewer photos than
    // the wall shows — the smallest one the backend has.
    const albumId = await smallestAlbum(page);
    const album = await (
      await page.request.get(`${APP}/api/albums/${albumId}?limit=1000`)
    ).json();
    const size = album.members.length;

    await page.goto(`${APP}/albums/${albumId}/present?motion=drift`);
    await page.waitForSelector('.tile', { timeout: 30000 });

    const ids = await page.evaluate(() =>
      Array.from(document.querySelectorAll('.tile')).map((t) =>
        t.getAttribute('data-photo-id')
      )
    );

    // Enough tiles to see more than one cycle.
    expect(ids.length).toBeGreaterThan(size);

    // The first cycle covers the album exactly once.
    expect(new Set(ids.slice(0, size)).size).toBe(size);

    // And the sequence repeats with the album's period.
    for (let i = size; i < ids.length; i++) {
      expect(ids[i], `tile ${i} must repeat the album's cycle`).toBe(
        ids[i - size]
      );
    }
  });

  test('Escape leaves the wall', async ({ page }) => {
    await page.goto(await driftUrl(page));
    await page.waitForSelector('.tile', { timeout: 30000 });

    await page.keyboard.press('Escape');
    await page.waitForURL(/\/albums\/\d+$/, { timeout: 10000 });
    await expect(page.locator('.topbar')).toHaveCount(1);
  });
});