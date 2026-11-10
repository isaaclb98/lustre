/**
 * E2E for where the album rename action lives.
 *
 * Renaming an album belongs on the album's own page. The albums
 * list is a directory — every card carries the same pair of
 * actions, and naming an album is something you do when you are
 * looking at it, not when you are scanning a list of names.
 *
 * So:
 *  1. The albums list has no Rename button on any card.
 *  2. The album's own page has exactly one, and it renames.
 *  3. The built-in Favourites album has none — it is not
 *     renameable at all.
 *
 * The rename itself is a write, so the test creates its own
 * throwaway album, renames it, and deletes it again. It asserts
 * against the API for the server-side truth rather than trusting
 * the page's own state.
 */
import { test, expect, type Page } from '@playwright/test';

const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';

/** The Rename control, wherever it is on the page. */
function renameButton(page: Page) {
  return page.locator('button', { hasText: /^rename$/i });
}

async function createThrowawayAlbum(page: Page): Promise<number> {
  const res = await page.request.post('/api/albums', {
    data: { name: `e2e-rename-${Date.now()}` }
  });
  expect(res.status()).toBe(200);
  const album = await res.json();
  return album.id;
}

async function deleteAlbum(page: Page, id: number): Promise<void> {
  const res = await page.request.delete(`/api/albums/${id}`);
  expect(res.status()).toBe(204);
}

test.describe('album rename lives on the album page', () => {
  test('the albums list has no Rename button', async ({ page }) => {
    await page.goto(`${APP}/albums`);
    await page.waitForSelector('.card', { timeout: 30000 });

    // Not one card may offer a rename.
    await expect(renameButton(page)).toHaveCount(0);
  });

  test('the album page has exactly one Rename button', async ({ page }) => {
    await page.goto(`${APP}/albums/5`);
    await page.waitForSelector('.tile', { timeout: 30000 });

    await expect(renameButton(page)).toHaveCount(1);
  });

  test('renaming from the album page updates the album', async ({ page }) => {
    const albumId = await createThrowawayAlbum(page);
    const newName = `e2e-renamed-${Date.now()}`;

    try {
      await page.goto(`${APP}/albums/${albumId}`);
      await page.waitForSelector('.head', { timeout: 30000 });

      await renameButton(page).click();

      // The prompt opens pre-filled with the current name, so the
      // user edits rather than retypes.
      const input = page.locator('dialog input, [role=dialog] input').first();
      await expect(input).toBeVisible();
      await expect(input).toHaveValue(/^e2e-rename-/);

      await input.fill(newName);
      await page
        .locator('dialog button, [role=dialog] button', { hasText: /^rename$/i })
        .first()
        .click();

      // The header follows the new name...
      await expect(page.locator('.head h1, .head h2, h1').first()).toHaveText(newName);
      // ...and the server agrees. The page's own state is not the
      // proof — the album's name is persisted server-side.
      const after = await page.request.get(`/api/albums/${albumId}`);
      expect(after.status()).toBe(200);
      expect((await after.json()).name).toBe(newName);
    } finally {
      await deleteAlbum(page, albumId);
    }
  });

  test('the built-in Favourites album has no Rename button', async ({ page }) => {
    await page.goto(`${APP}/albums/favourites`);
    await page.waitForSelector('.head', { timeout: 30000 });

    await expect(renameButton(page)).toHaveCount(0);
  });
});
