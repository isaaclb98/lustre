/**
 * scripts/screenshot-grid-density.ts — drives the vite dev server
 * at http://127.0.0.1:5173 (with API_PROXY_TARGET=http://127.0.0.1:8000)
 * and takes screenshots of the Settings page and a search results
 * page at each of the three grid sizes.
 *
 * Run with: npx tsx scripts/screenshot-grid-density.ts
 *
 * Output: /home/isaac/.hermes/cache/scratch/grid-density/*.png
 *
 * The vite dev server is required (not vite preview) because the
 * dev server has the API proxy in vite.config.ts. The preview
 * server has no proxy so /api/* 500s.
 */
import { chromium } from 'playwright';
import * as fs from 'node:fs';
import * as path from 'node:path';

const OUT_DIR = '/home/isaac/.hermes/cache/scratch/grid-density';
const PREVIEW_URL = 'http://127.0.0.1:5173';
const VIEWPORT = { width: 1600, height: 1000 };

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: VIEWPORT,
    deviceScaleFactor: 1
  });
  const page = await context.newPage();

  page.on('pageerror', (err) => console.error('PAGEERROR:', err.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error') console.error('CONSOLE.ERROR:', msg.text());
  });

  // Helper: set the gridSize preference directly in localStorage
  // without going through the Settings page. Used for the
  // populated-grid shots where the user clicks persist + a
  // reload is the cleanest way to see the new size take effect.
  async function setGridSize(size: 'large' | 'medium' | 'small') {
    await page.evaluate((s) => {
      const raw = localStorage.getItem('lustre:preferences');
      const obj = raw ? JSON.parse(raw) : {};
      obj.gridSize = s;
      localStorage.setItem('lustre:preferences', JSON.stringify(obj));
    }, size);
  }

  async function gridVars() {
    return await page.evaluate(() => {
      const style = getComputedStyle(document.documentElement);
      return {
        tile: style.getPropertyValue('--grid-tile-size').trim(),
        gap: style.getPropertyValue('--grid-gutter-size').trim(),
        rowMax: style.getPropertyValue('--grid-row-max-width').trim(),
        width: style.getPropertyValue('--grid-width').trim()
      };
    });
  }

  // Reset preference to a known state.
  await page.goto(`${PREVIEW_URL}/`);
  await page.evaluate(() => window.localStorage.clear());

  // === 1. Settings page (default = Large) ===
  await page.goto(`${PREVIEW_URL}/settings`, { waitUntil: 'networkidle' });
  await page.getByRole('heading', { name: 'Grid size' }).waitFor();
  await page.screenshot({
    path: path.join(OUT_DIR, '01-settings-default-large.png'),
    fullPage: false
  });
  console.log('shot 01: settings default (Large)');

  // === 2. Settings page after clicking Medium ===
  await page.getByRole('radio', { name: 'Medium' }).click();
  await page.waitForTimeout(200);
  await page.screenshot({
    path: path.join(OUT_DIR, '02-settings-medium-active.png'),
    fullPage: false
  });
  console.log('shot 02: settings Medium active');

  // === 3. Settings page after clicking Small ===
  await page.getByRole('radio', { name: 'Small' }).click();
  await page.waitForTimeout(200);
  await page.screenshot({
    path: path.join(OUT_DIR, '03-settings-small-active.png'),
    fullPage: false
  });
  console.log('shot 03: settings Small active');

  // === 4-6. Populated grid at each size via /random.
  // /random loads a populated grid on mount (no Search click
  // required), so we can see the density change immediately.
  await setGridSize('large');
  await page.goto(`${PREVIEW_URL}/random`, { waitUntil: 'networkidle' });
  // Wait for at least one tile to render.
  await page.locator('.grid-wrapper [class*="tile"], .grid-wrapper img').first()
    .waitFor({ timeout: 15000 })
    .catch(() => console.warn('  no tiles rendered for Large'));
  // Let blurhash placeholders swap to images.
  await page.waitForTimeout(1500);
  const v4 = await gridVars();
  console.log('shot 04 gridVars (Large):', v4);
  await page.screenshot({
    path: path.join(OUT_DIR, '04-random-large.png'),
    fullPage: false
  });

  // Switch to Medium.
  await setGridSize('medium');
  await page.goto(`${PREVIEW_URL}/random`, { waitUntil: 'networkidle' });
  await page.locator('.grid-wrapper [class*="tile"], .grid-wrapper img').first()
    .waitFor({ timeout: 15000 })
    .catch(() => console.warn('  no tiles rendered for Medium'));
  await page.waitForTimeout(1500);
  const v5 = await gridVars();
  console.log('shot 05 gridVars (Medium):', v5);
  await page.screenshot({
    path: path.join(OUT_DIR, '05-random-medium.png'),
    fullPage: false
  });

  // Switch to Small.
  await setGridSize('small');
  await page.goto(`${PREVIEW_URL}/random`, { waitUntil: 'networkidle' });
  await page.locator('.grid-wrapper [class*="tile"], .grid-wrapper img').first()
    .waitFor({ timeout: 15000 })
    .catch(() => console.warn('  no tiles rendered for Small'));
  await page.waitForTimeout(1500);
  const v6 = await gridVars();
  console.log('shot 06 gridVars (Small):', v6);
  await page.screenshot({
    path: path.join(OUT_DIR, '06-random-small.png'),
    fullPage: false
  });

  // === 7. Annotated comparison: read the data-grid-size
  //    attribute and column count from the rendered grid. ===
  const meta = await page.evaluate(() => {
    const wrapper = document.querySelector('.grid-wrapper');
    const dataAttr = wrapper?.getAttribute('data-grid-size') ?? 'none';
    const tile = document.querySelector('.grid-row > *') as HTMLElement | null;
    const tileRect = tile?.getBoundingClientRect();
    const rowEl = document.querySelector('.grid-row') as HTMLElement | null;
    const rowRect = rowEl?.getBoundingClientRect();
    const colCount = rowEl?.children.length ?? 0;
    return {
      dataAttr,
      tileSize: tileRect ? Math.round(tileRect.width) : null,
      rowWidth: rowRect ? Math.round(rowRect.width) : null,
      colCount
    };
  });
  console.log('shot 07 grid metadata (Small):', meta);

  await browser.close();
  console.log(`\nDone. Output: ${OUT_DIR}`);
  const files = fs.readdirSync(OUT_DIR).filter((f) => f.endsWith('.png'));
  for (const f of files) {
    const stat = fs.statSync(path.join(OUT_DIR, f));
    console.log(`  ${f}  ${(stat.size / 1024).toFixed(1)} KB`);
  }
}

main().catch((err) => {
  console.error('FATAL:', err);
  process.exit(1);
});
