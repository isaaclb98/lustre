/**
 * E2E for the album card's delete affordance.
 *
 * The albums list is a directory of covers, and a text "Delete"
 * button on every card put the most destructive action in the
 * loudest position on the page. The delete now lives in the card's
 * top-right corner as a minus, revealed on hover — the same
 * affordance the photo tile uses to remove a photo from an album
 * (PhotoTile's .remove-btn), so "take this out of the collection"
 * looks like one gesture everywhere in the app.
 *
 * What this pins:
 *
 *   1. The affordance MOVED and CHANGED SHAPE. No card carries a
 *      text Delete button any more; user-album cards carry a corner
 *      minus instead. The built-in Favourites album carries neither
 *      — it cannot be deleted.
 *   2. It is HIDDEN until the card is hovered (or the button itself
 *      is focused). A control that is always visible is the thing
 *      this change exists to remove.
 *   3. It is HIDDEN SAFELY. opacity: 0 alone leaves the button
 *      swallowing pointer events, so a click in the corner would
 *      delete the album without the user ever seeing the control
 *      that did it.
 *   4. It still GUARDS the delete. Clicking the minus opens the
 *      confirmation; it does not delete on click.
 *
 * The test reads the theme the SERVER was started with and asserts
 * the geometry against that, so the same file passes against a
 * light or a dark deployment.
 */
import { test, expect, type Page } from '@playwright/test';

const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';

/**
 * The theme the running document carries.
 *
 * Read from the DOM rather than by fetching the shell: the marker
 * IS the server's stamp, so the browser's copy of it is the thing
 * the contract is about — and reading it does not depend on how the
 * server frames its HTML response.
 */
async function servedTheme(page: Page): Promise<string> {
  await page.goto(`${APP}/albums`);
  await page.waitForSelector('.card', { timeout: 30000 });
  const theme = await page.evaluate(() =>
    document.documentElement.getAttribute('data-theme')
  );
  expect(theme, 'the document must carry a data-theme marker').not.toBeNull();
  return theme!;
}

/**
 * The corner minus on the first user-album card.
 *
 * The built-in Favourites card is rendered first and has no minus,
 * so the first card that HAS one is a user album.
 */
function firstRemoveButton(page: Page) {
  return page.locator('.card-remove').first();
}

async function createThrowawayAlbum(page: Page): Promise<number> {
  const res = await page.request.post('/api/albums', {
    data: { name: `e2e-card-delete-${Date.now()}` }
  });
  expect(res.status()).toBe(200);
  const album = await res.json();
  return album.id;
}

async function deleteAlbum(page: Page, id: number): Promise<void> {
  const res = await page.request.delete(`/api/albums/${id}`);
  expect(res.status()).toBe(204);
}

test.describe('album card delete affordance', () => {
  test('cards carry a corner minus, not a text Delete button', async ({
    page
  }) => {
    await page.goto(`${APP}/albums`);
    await page.waitForSelector('.card', { timeout: 30000 });

    // The old affordance is gone: no card carries a text button.
    await expect(page.locator('.card button', { hasText: /^delete$/i })).toHaveCount(0);

    // The new one is there, on every user-album card.
    const cards = page.locator('.card');
    const cardCount = await cards.count();
    const minusCount = await page.locator('.card-remove').count();
    // Favourites is the one card without a minus, so there is one
    // fewer minus than there are cards.
    expect(minusCount).toBe(cardCount - 1);

    // It is an icon button with a name, not a glyph.
    const btn = firstRemoveButton(page);
    await expect(btn.locator('svg')).toHaveCount(1);
    await expect(btn).toHaveAttribute('aria-label', /Delete .+/);
  });

  test('the built-in Favourites card has no minus', async ({ page }) => {
    await page.goto(`${APP}/albums`);
    await page.waitForSelector('.card', { timeout: 30000 });

    // The Favourites card is the one that links to /albums/favourites.
    const favourites = page
      .locator('.card')
      .filter({ has: page.locator('a[href$="/albums/favourites"]') })
      .first();
    await expect(favourites).toHaveCount(1);
    await expect(favourites.locator('.card-remove')).toHaveCount(0);
  });

  test('it is hidden until the card is hovered', async ({ page }) => {
    await page.goto(`${APP}/albums`);
    await page.waitForSelector('.card', { timeout: 30000 });

    const card = page
      .locator('.card')
      .filter({ has: page.locator('.card-remove') })
      .first();
    const btn = card.locator('.card-remove');

    // At rest: invisible AND inert. The inert half matters — a
    // button hidden with opacity alone still swallows clicks.
    await expect(btn).toHaveCSS('opacity', '0');
    await expect(btn).toHaveCSS('pointer-events', 'none');

    // Hovering the card reveals it.
    await card.hover();
    await expect(btn).toHaveCSS('opacity', '1');
    await expect(btn).toHaveCSS('pointer-events', 'auto');
  });

  test('focusing it reveals it too', async ({ page }) => {
    await page.goto(`${APP}/albums`);
    await page.waitForSelector('.card', { timeout: 30000 });

    const btn = firstRemoveButton(page);
    await btn.focus();
    // A keyboard user must be able to see the control they tabbed to.
    await expect(btn).toHaveCSS('opacity', '1');
  });

  test('it sits in the card\'s top-right corner', async ({ page }) => {
    await page.goto(`${APP}/albums`);
    await page.waitForSelector('.card', { timeout: 30000 });

    const card = page
      .locator('.card')
      .filter({ has: page.locator('.card-remove') })
      .first();
    const box = await card.boundingBox();
    const btnBox = await card.locator('.card-remove').boundingBox();
    expect(box).not.toBeNull();
    expect(btnBox).not.toBeNull();

    // Right of the card's midpoint, near the top, inset a little.
    const centre = box!.x + box!.width / 2;
    expect(btnBox!.x + btnBox!.width / 2).toBeGreaterThan(centre);
    expect(btnBox!.y).toBeLessThan(box!.y + box!.height / 2);
    // Inset from the corner, not flush against it.
    expect(box!.x + box!.width - (btnBox!.x + btnBox!.width)).toBeGreaterThan(0);
    expect(btnBox!.y - box!.y).toBeGreaterThan(0);
  });

  test('the theme decides its shape, not just its colour', async ({ page }) => {
    const theme = await servedTheme(page);
    await page.goto(`${APP}/albums`);
    await page.waitForSelector('.card', { timeout: 30000 });

    const btn = firstRemoveButton(page);
    const radius = await btn.evaluate(
      (el) => getComputedStyle(el).borderRadius
    );

    if (theme === 'dark') {
      // The dark theme has no rounded corners at all.
      expect(radius).toBe('0px');
    } else {
      // The light theme keeps the chip round, like the tile's.
      expect(radius).not.toBe('0px');
    }
  });

  test('clicking it asks first, and does not delete on click', async ({
    page
  }) => {
    const albumId = await createThrowawayAlbum(page);

    try {
      await page.goto(`${APP}/albums`);
      await page.waitForSelector('.card', { timeout: 30000 });

      const card = page.locator('.card').filter({ hasText: `e2e-card-delete-` }).first();
      await card.hover();
      await card.locator('.card-remove').click();

      // The confirmation opens.
      const dialog = page.locator('[role=dialog], dialog').first();
      await expect(dialog).toBeVisible();
      await expect(dialog).toContainText('Delete album');

      // And nothing was deleted yet.
      const still = await page.request.get(`/api/albums/${albumId}`);
      expect(still.status()).toBe(200);

      // Dismissing leaves the album alone.
      await page
        .locator('[role=dialog] button, dialog button', { hasText: /^cancel$/i })
        .first()
        .click();
      await expect(dialog).toBeHidden();

      const after = await page.request.get(`/api/albums/${albumId}`);
      expect(after.status()).toBe(200);
    } finally {
      await deleteAlbum(page, albumId);
    }
  });
});
