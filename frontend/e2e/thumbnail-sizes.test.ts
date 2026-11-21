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
import { test, expect } from '@playwright/test';

test('PhotoTile sizes attribute reflects viewport-derived tile width', async ({ page }) => {
  // The PhotoGrid uses `auto-fill minmax(180px, 1fr)`, so the
  // minimum tile width is 180 px regardless of viewport. The
  // `sizes` attribute must reflect that, not a hardcoded value.
  // Before Round 5: every tile had sizes="... 240px" — the
  // browser would always pick the 240w variant even on phones
  // where the tile renders at 180 px. After Round 5: the
  // largest advertised size derives from the actual tile width
  // (180, 240, 384). After Round 4 of the model-variant plan:
  // the single thumbnail variant is 384, so the maximum cap
  // here is 384 (was 240).
  await page.setViewportSize({ width: 900, height: 800 });
  await page.goto('/random');
  await page.waitForSelector('.grid-tile img.full');
  await page.waitForTimeout(800);

  const sizes = await page.locator('.grid-tile img.full').first().getAttribute('sizes');

  // Sanity: the attribute is set (it would be `undefined` if
  // eager-only and the first tile wasn't eager).
  expect(sizes).toBeTruthy();
  // The attribute reflects an actual rendered CSS width — not
  // a hardcoded literal. We just verify it ends in `px` and
  // contains one of the variant widths (180, 240, or 384).
  //
  // The masonry advertises ONE width — its tile width is exact and
  // viewport-independent — and that width comes from the masonry's
  // own size ladder (360/240/192), which is a rung above the
  // square grid's. So the accepted set depends on the layout the
  // deployment serves.
  const masonry = await page.evaluate(() =>
    document.documentElement.getAttribute('data-masonry') === '1'
  );
  if (masonry) {
    expect(sizes).toMatch(/^(192|240|360)px$/);
  } else {
    expect(sizes).toMatch(/(180|240|384)px/);
  }
  // And the srcset advertises the single 384 variant — the
  // post-migration shape. The actual URL is `/thumb/{id}?w=384`.
  const srcset = await page.locator('.grid-tile img.full').first().getAttribute('srcset');
  if (srcset) {
    expect(srcset).toContain('w=384');
  }
});

test('PhotoTile non-eager tile uses ?w=384', async ({ page }) => {
  // Non-eager tiles (any tile past the first 3 in the first row)
  // use `src={thumbUrl(pointId, 384)}` per Round 4. Pre-migration
  // these used `src={thumbUrl(pointId)}` (no width), which routed
  // through the canonical 256px fallback in the router.
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
    // Non-eager tiles use `?w=384` per Round 4.
    expect(src).toContain('w=384');
  } else {
    // Not enough tiles loaded — skip via assertion.
    expect(count).toBeGreaterThan(5);
  }
});
