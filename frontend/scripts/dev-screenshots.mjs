// scripts/dev-screenshots.mjs — capture UI screenshots of the dev stack
// for visual review. Run from the project repo (where @playwright/test
// is in node_modules):
//
//   ~/projects/image-search/scripts/dev.sh up
//   # NOTE: the local clone path stays ~/projects/image-search/
//   until Phase C rotates the host state. The repo was renamed
//   to github.com/isaaclb98/lustre (Sep 2026) but the directory
//   on this machine is still image-search/ until the deploy
//   rotation lands.
//   # index + create test albums (one-time):
//   curl -X POST -H 'Content-Type: application/json' \
//     -d '{"mode":"rebuild"}' http://127.0.0.1:18000/api/admin/index
//   for n in "K-pop favorites" "Landscapes" "Portraits"; do
//     curl -sS -X POST -H 'Content-Type: application/json' \
//       -d "{\"name\":\"$n\"}" http://127.0.0.1:18000/api/albums > /dev/null
//   done
//   node scripts/dev-screenshots.mjs
//
// Output: /tmp/dev-screenshots/*.png (one per page, plus the dropdown-open variants)
//
// The Dropdown primitive is what this script is most useful for
// verifying — both consumers (photo page + lightbox) get a capture.

import { chromium } from '@playwright/test';
import { mkdir } from 'node:fs/promises';

const BASE = process.env.DEV_BASE_URL ?? 'http://127.0.0.1:18000';
const OUT = process.env.DEV_SCREENSHOT_DIR ?? '/tmp/dev-screenshots';

const VIEWPORT = { width: 1440, height: 900 };

async function shot(page, name, opts = {}) {
  await page.screenshot({ path: `${OUT}/${name}.png`, ...opts });
  console.log(`  saved ${OUT}/${name}.png${opts.fullPage ? ' (full)' : ''}`);
}

async function getFirstPhotoId(page) {
  const r = await page.request.get(`${BASE}/api/random?limit=1`);
  const json = await r.json();
  return json.results?.[0]?.id ?? null;
}

async function newPage(browser) {
  const ctx = await browser.newContext({ viewport: VIEWPORT });
  const page = await ctx.newPage();
  // Hydration helper: wait for topbar + a settle window. The SvelteKit
  // dev mode ships SSR HTML first; client JS wires up async. Without
  // this, early clicks fire before listeners exist.
  const ready = async () => {
    await page.waitForSelector('header.topbar', { timeout: 10000 });
    await page.waitForTimeout(500);
  };
  return { ctx, page, ready };
}

await mkdir(OUT, { recursive: true });

const browser = await chromium.launch();

{
  const { page, ready } = await newPage(browser);

  console.log('1. home /');
  await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
  await ready();
  await shot(page, '01-home-grid');

  console.log('2. /random');
  await page.goto(`${BASE}/random`, { waitUntil: 'networkidle' });
  await ready();
  await shot(page, '02-random-grid');

  console.log('3. /for-you');
  await page.goto(`${BASE}/for-you`, { waitUntil: 'networkidle' });
  await ready();
  await shot(page, '03-for-you-grid');

  console.log('4. /albums');
  await page.goto(`${BASE}/albums`, { waitUntil: 'networkidle' });
  await ready();
  await shot(page, '04-albums-list');

  const photoId = await getFirstPhotoId(page);
  if (!photoId) {
    console.error('no photos indexed — did the dev indexer run?');
    await browser.close();
    process.exit(1);
  }

  console.log(`5. /photo/${photoId.slice(0, 8)}... (closed)`);
  await page.goto(`${BASE}/photo/${photoId}`, { waitUntil: 'networkidle' });
  await ready();
  await shot(page, '05-photo-page');

  console.log('6. /photo/{id} with add-to-album dropdown OPEN');
  const photoAddBtn = page.locator('button:visible', { hasText: /add to album/i }).first();
  await photoAddBtn.click();
  await page.waitForTimeout(1500); // ensureAlbumsLoaded + menu render
  await shot(page, '06-photo-page-dropdown-open', {
    clip: { x: 1000, y: 0, width: 440, height: 600 }
  });

  await page.context().close();
}

{
  const { page, ready } = await newPage(browser);

  console.log('7. lightbox + add-to-album dropdown (via /random)');
  await page.goto(`${BASE}/random`, { waitUntil: 'networkidle' });
  await ready();
  const tile = page.locator('.grid-tile').first();
  if (await tile.count() === 0) {
    console.error('no .grid-tile elements on /random');
  } else {
    await tile.click({ force: true });
    await page.waitForTimeout(1500); // lightbox opens
    await shot(page, '07-lightbox');

    const lbAddBtn = page.locator('[role="dialog"] button[aria-haspopup="menu"]', {
      hasText: /add to album/i
    }).first();
    if (await lbAddBtn.count() > 0) {
      await lbAddBtn.click();
      await page.waitForTimeout(1500);
      await shot(page, '08-lightbox-dropdown-open');
    } else {
      console.error('lightbox add-to-album button not found');
    }
  }

  await page.context().close();
}

await browser.close();
console.log('done.');
