/**
 * Verification that PhotoTile's `sizes` attribute is computed
 * from the viewport (Round 5), and that the srcset advertises
 * the post-migration single 384 variant (Round 4 of the
 * model-variant plan).
 *
 * Before Round 5: every PhotoTile had `sizes="(max-width:
 * 600px) 120px, (max-width: 1200px) 180px, 240px"` regardless
 * of viewport. So a 4K monitor with 4 columns at ~480 px each
 * still told it "I'm 240 px", and the browser picked the
 * 240w variant — blurry at 2x DPR.
 *
 * After Round 5: `sizes` derives from `tileSizeGuess()`, which
 * computes the actual rendered tile width from the viewport
 * width and the PhotoGrid column math. The variants it picks
 * are still 120/180/240/384 — same ladder as before, but now
 * the largest advertised size is the actual tile width
 * (capped at 384, the only variant the indexer emits).
 *
 * Round 4 of the model-variant plan collapses the thumbnail
 * pipeline to a single 384px asset. The srcset is therefore
 * `src="/thumb/{id}?w=384 384w"` (one entry, the 384 variant),
 * and the `src` defaults to `?w=384` for non-eager tiles.
 */
import { test, expect, type Page } from '@playwright/test';

/**
 * The deployment's thumbnail long edge, as the shell publishes it.
 *
 * It is a deployment-level number — the dev stack renders 512,
 * production 384 — and the frontend reads it off the shell rather
 * than hardcoding it. The test reads it the same way, so it holds
 * on both.
 */
async function deploymentThumbnailSize(page: Page): Promise<number> {
  const raw = await page.evaluate(() =>
    document.documentElement.getAttribute('data-thumbnail-size')
  );
  const parsed = Number(raw);
  expect(Number.isFinite(parsed) && parsed > 0).toBe(true);
  return parsed;
}

test('PhotoTile sizes attribute reflects viewport-derived tile width', async ({ page }) => {
  // The PhotoGrid uses `auto-fill minmax(180px, 1fr)`, so the
  // The tile width is EXACT in both grids — the masonry places
  // tiles at the ladder's fixed width, and the uniform grid's
  // tracks are fixed too (auto-fit collapses EMPTY tracks; it
  // never resizes a filled one). What changes with the viewport
  // is the COLUMN COUNT, not the tile.
  //
  // So `sizes` is ONE width in both layouts, and it is the tile's
  // real width. The old breakpoint ladder assumed viewport-
  // stretching tracks, and it over-advertised the dense view: a
  // 144px tile claimed 240px, and the browser fetched a 240w
  // source for a 144px slot — two thirds more pixels than the
  // tile can show.
  await page.setViewportSize({ width: 900, height: 800 });
  await page.goto('/random');
  await page.waitForSelector('.grid-tile img.full');
  await page.waitForTimeout(800);

  const sizes = await page.locator('.grid-tile img.full').first().getAttribute('sizes');

  // Sanity: the attribute is set (it would be `undefined` if
  // eager-only and the first tile wasn't eager).
  expect(sizes).toBeTruthy();
  // It is ONE exact width — no breakpoint ladder, because the
  // tile width does not depend on the viewport.
  expect(sizes).toMatch(/^\d+px$/);
  //
  // And it is the tile's REAL width, which is the ladder's rung
  // for the layout the deployment serves. The masonry's ladder is
  // a rung above the square grid's (360/240/180 vs 240/180/144).
  const tileWidth = await page.evaluate(() => {
    const raw = getComputedStyle(document.documentElement)
      .getPropertyValue('--grid-tile-size')
      .trim();
    return Number(raw.slice(0, -2));
  });
  expect(sizes).toBe(`${tileWidth}px`);
  // The srcset is a LADDER: the deployment's smaller variants plus
  // the canonical, each advertised with the width it really is.
  const thumbSize = await deploymentThumbnailSize(page);
  const srcset = await page.locator('.grid-tile img.full').first().getAttribute('srcset');
  if (srcset) {
    expect(srcset).toContain(`w=${thumbSize}`);
  }
});

test('the srcset is a ladder of the deployment\'s smaller variants', async ({ page }) => {
  // The DENSE view's assets.
  //
  // The canonical thumbnail is sized for the LARGEST tile (360px
  // in the masonry). A 180px tile does not need it: handing it
  // one wastes bandwidth and decode time for pixels the browser
  // scales straight back down.
  //
  // So the deployment publishes a ladder of smaller variants, the
  // backend serves them from pre-generated siblings, and the
  // srcset lists them all so the browser picks the smallest that
  // covers the tile at the device's DPR.
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/random');
  await page.waitForSelector('.grid-tile img.full');
  await page.waitForTimeout(800);

  const variants = await page.evaluate(() => {
    const raw = document.documentElement.getAttribute('data-thumbnail-variants');
    return raw ? raw.split(',').map(Number).filter((n) => n > 0) : [];
  });
  const thumbSize = await deploymentThumbnailSize(page);
  const srcset = await page.locator('.grid-tile img.full').first().getAttribute('srcset');
  expect(srcset).toBeTruthy();

  // Every variant the deployment publishes appears in the srcset,
  // and so does the canonical — it is the fallback the router
  // serves when a variant is missing.
  //
  // A deployment with NO ladder (the default) has a single-entry
  // srcset: just the canonical. That is the correct behaviour for
  // one — there is nothing smaller to pick.
  for (const v of variants) {
    expect(srcset).toContain(`?w=${v} `);
  }
  expect(srcset).toContain(`?w=${thumbSize} `);

  // The ladder is ordered smallest first, so the browser walks it
  // upward and takes the smallest asset that covers the tile.
  const widths = (srcset as string)
    .split(',')
    .map((entry) => Number(entry.trim().match(/w=(\d+)/)![1]));
  const sorted = [...widths].sort((a, b) => a - b);
  expect(widths).toEqual(sorted);
});

test('PhotoTile non-eager tile uses the deployment thumbnail width', async ({ page }) => {
  // Non-eager tiles (any tile past the first 3 in the first row)
  // use `src={thumbUrl(pointId, thumbnailSize)}` — the deployment's
  // own thumbnail size. Pre-migration these used
  // `src={thumbUrl(pointId)}` (no width), which routed through the
  // canonical 256px fallback in the router.
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/random');
  await page.waitForSelector('.grid-tile img.full');
  await page.waitForTimeout(800);

  // Pick a tile that's not in the first 3 (eager). On 1280px
  // viewport that's roughly any tile past column 3.
  const tiles = page.locator('.grid-tile img.full');
  const count = await tiles.count();
  if (count > 5) {
    const src = await tiles.nth(5).getAttribute('src');
    // Non-eager tiles use the deployment's thumbnail width.
    const thumbSize = await deploymentThumbnailSize(page);
    expect(src).toContain(`w=${thumbSize}`);
  } else {
    // Not enough tiles loaded — skip via assertion.
    expect(count).toBeGreaterThan(5);
  }
});
