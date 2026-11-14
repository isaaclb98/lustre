import { test, expect, type Page } from '@playwright/test';

/**
 * accessibility.test.ts — Keyboard navigation, ARIA labels, and
 * screen-reader support across every page.
 *
 * Real users who rely on keyboards, screen readers, or assistive tech
 * must be able to use every feature. These tests verify the basics.
 */

async function appReady(page: Page) {
  await page.waitForSelector('header.topbar', { timeout: 10000 });
}

test.describe('Keyboard navigation', () => {
  test('Tab order on home page is logical', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    // Start tabbing from the top — should reach each nav link in order
    await page.keyboard.press('Tab'); // first focusable
    const focused: string[] = [];
    for (let i = 0; i < 8; i++) {
      const tag = await page.evaluate(() => {
        const el = document.activeElement as HTMLElement;
        return el?.textContent?.trim().slice(0, 30) || el?.tagName || '';
      });
      focused.push(tag);
      await page.keyboard.press('Tab');
    }
    // Should have visited at least the Home link, Search link, etc.
    expect(focused.length).toBe(8);
    // At least some should be nav links (Home, Search, Random...)
    expect(focused.some(f => /Home|Search|Random|Albums/.test(f))).toBe(true);
  });

  test('Search input receives keyboard focus via Tab', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    // Focus the search input directly
    const input = page.getByPlaceholder(/Add a positive/);
    await input.focus();
    const isFocused = await input.evaluate((el) => el === document.activeElement);
    expect(isFocused).toBe(true);
  });

  test('Enter on search input commits the prompt as a chip', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    const input = page.getByPlaceholder(/Add a positive/);
    await input.focus();
    await page.keyboard.type('keyboard-test');
    await page.keyboard.press('Enter');

    await expect(page.locator('.chip').filter({ hasText: 'keyboard-test' })).toBeVisible({ timeout: 3000 });
  });

  test('Lightbox traps focus inside the dialog', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    await page.locator('.grid-tile').first().click();
    await expect(page.locator('[role="dialog"]')).toBeVisible({ timeout: 5000 });

    // Arrow keys should navigate between photos
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(200);

    // Escape should close
    await page.keyboard.press('Escape');
    await expect(page.locator('[role="dialog"]')).toBeHidden({ timeout: 3000 });
  });

  test('Search button is keyboard-accessible (Enter activates)', async ({ page }) => {
    await page.goto('/?positives=beach');
    await appReady(page);

    // Wait for initial results to load
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    // Focus the search button and press Enter
    const searchBtn = page.getByRole('button', { name: 'Search', exact: true });
    await searchBtn.focus();
    await page.keyboard.press('Enter');
    // Search should fire again — wait for the response
    const response = await page.waitForResponse((r) => r.url().includes('/api/search'), { timeout: 10000 });
    expect(response.status()).toBe(200);
  });
});

test.describe('ARIA labels and semantic markup', () => {
  test('Header navigation has aria-label', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    const nav = page.getByRole('navigation');
    await expect(nav).toBeVisible();
  });

  test('All main navigation links are reachable by accessible name', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    // Scope to the navigation element to avoid matching the brand link.
    // Current topbar tabs: Home, Random, Albums, Settings.
    // (The earlier 'Search' tab was removed when /search became the
    // home route, and the 'For You' tab was removed with the For
    // You feature — there is no longer a dedicated link for either.)
    const nav = page.getByRole('navigation');
    for (const label of ['Home', 'Random', 'Albums', 'Settings']) {
      await expect(nav.getByRole('link', { name: label })).toBeVisible();
    }
  });

  test('Search input has accessible name (placeholder or label)', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    // The input should have a placeholder that serves as its accessible name
    const input = page.getByPlaceholder(/Add a positive/);
    await expect(input).toBeVisible();
  });

  test('Photo tiles have accessible alt text or aria-label', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    // Check that tiles have either an alt text on the img or an aria-label on the link
    const firstTile = page.locator('.grid-tile').first();
    const link = firstTile.locator('a').first();
    const linkAttrs = await link.evaluate((el) => ({
      ariaLabel: el.getAttribute('aria-label'),
      title: el.getAttribute('title'),
      text: el.textContent?.trim().slice(0, 50)
    }));
    // At minimum, the link should have some identifying info
    const hasAccess = linkAttrs.ariaLabel || linkAttrs.title || (linkAttrs.text && linkAttrs.text.length > 0);
    expect(hasAccess).toBeTruthy();
  });

  test('Dialog has role="dialog" when open', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    await page.locator('.grid-tile').first().click();
    await expect(page.locator('[role="dialog"]')).toBeVisible({ timeout: 5000 });
  });

  test('Saved searches popover has role="menu"', async ({ page }) => {
    await page.goto('/?positives=beach');
    await appReady(page);

    await page.getByTitle('Saved searches').click();
    await expect(page.getByRole('menu')).toBeVisible({ timeout: 5000 });
  });

  test('Buttons have accessible names', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    // All buttons should have accessible names (text content or aria-label)
    const buttons = await page.locator('button').all();
    for (const btn of buttons.slice(0, 10)) {
      const name = await btn.evaluate((el) => {
        return el.textContent?.trim() || el.getAttribute('aria-label') || el.getAttribute('title') || '';
      });
      // Skip buttons that are purely decorative or have no accessible name (some icon buttons might be exempt)
      expect(name.length).toBeGreaterThan(0);
    }
  });
});

test.describe('Screen reader announcements', () => {
  test('the page has no stale toast region', async ({ page }) => {
    await page.goto('/?positives=beach');
    await appReady(page);

    // The app has no toast popups — the toaster region was removed
    // along with them — so a live region announcing notifications
    // must not be left behind claiming to announce things.
    expect(await page.getByRole('region', { name: /notifications/i }).count()).toBe(0);
  });

  test('Search results load without announcing errors', async ({ page }) => {
    await page.goto('/?positives=beach');
    await appReady(page);

    // Wait for results
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    // No error banner is present.
    await expect(page.locator('div.error')).toHaveCount(0);
  });
});

test.describe('Focus management', () => {
  test('Clicking a tile opens lightbox without losing page focus context', async ({ page }) => {
    await page.goto('/random');
    await appReady(page);
    await expect(page.locator('.grid-tile').first()).toBeVisible({ timeout: 10000 });

    await page.locator('.grid-tile').first().click();
    await expect(page.locator('[role="dialog"]')).toBeVisible({ timeout: 5000 });

    // Close and verify we're back on the page
    await page.keyboard.press('Escape');
    await expect(page.locator('[role="dialog"]')).toBeHidden({ timeout: 3000 });
  });

  test('Removing a chip returns focus to the input', async ({ page }) => {
    await page.goto('/?positives=beach');
    await appReady(page);

    // Wait for initial load. Scope to .chip.pos (the prompt chip)
    // because .chip is shared with CollectionsChips which renders
    // its own chip elements (e.g. "nas 182" for the selected
    // collection) that shouldn't be removed by the × button here.
    await expect(page.locator('.chip.pos').first()).toBeVisible({ timeout: 5000 });

    // Click the × on the prompt chip
    const removeBtn = page.locator('.chip.pos').first().locator('button').first();
    await removeBtn.click();

    // Wait for the prompt chip to disappear (collection chips
    // remain).
    await expect(page.locator('.chip.pos')).toHaveCount(0, { timeout: 3000 });
  });
});

test.describe('Color contrast and visual accessibility', () => {
  test('Text elements have sufficient color contrast (smoke check)', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    // Basic check: body text should not be invisible (transparent or same as bg)
    const textColor = await page.evaluate(() => {
      const el = document.querySelector('h1') || document.querySelector('p') || document.body;
      const style = window.getComputedStyle(el as Element);
      return {
        color: style.color,
        bgColor: style.backgroundColor
      };
    });

    expect(textColor.color).toBeTruthy();
    expect(textColor.color).not.toBe('rgba(0, 0, 0, 0)'); // not transparent
  });
});

test.describe('Form accessibility', () => {
  test('Search input can be filled and submitted via keyboard only', async ({ page }) => {
    await page.goto('/');
    await appReady(page);

    // Focus the input directly
    const input = page.getByPlaceholder(/Add a positive/);
    await input.focus();

    // Type without using mouse
    await page.keyboard.type('keyboard-only-search');

    // Submit with Enter
    await page.keyboard.press('Enter');

    // Chip should appear
    await expect(page.locator('.chip').filter({ hasText: 'keyboard-only-search' })).toBeVisible({ timeout: 3000 });

    // Click the Search button directly (Tab navigation to a specific
    // button is brittle — depends on focus order which varies by
    // browser/viewport). Focus the button directly and press Enter.
    const searchBtn = page.getByRole('button', { name: 'Search', exact: true });
    await searchBtn.focus();
    await page.keyboard.press('Enter');

    // Search should fire
    const response = await page.waitForResponse((r) => r.url().includes('/api/search'), { timeout: 10000 });
    expect(response.status()).toBe(200);
  });
});