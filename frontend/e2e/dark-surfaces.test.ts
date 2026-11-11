/**
 * The dark theme separates its surfaces WITHOUT outlines.
 *
 * The dark theme used to draw a 1px 10%-white hairline around every
 * block — 57 of them on the settings page alone — which read as a
 * wireframe rather than as a set of surfaces.
 *
 * The hairline was standing in for a tonal step that was too small
 * to see on its own: the page sat at oklch L 14.5% and the panel at
 * L 20.5%, so the panel could not read as a distinct slab and the
 * outline had to do the work. Meanwhile a 10% white line on
 * near-black is a BRIGHT graphic element, where the light theme's
 * 6% black line on cream is nearly nothing — the same token, the
 * opposite effect.
 *
 * So the dark theme now separates surfaces by LIGHTNESS and paints
 * no outline around them. Two things are pinned here:
 *
 *   1. The symptom — a panel paints no border at all. The panels
 *      are exactly the surfaces that used to carry the outline.
 *   2. The mechanism — the panel is a real step above the page. If
 *      the step collapses again, the hairline has to come back, and
 *      this test fails before the wireframe does.
 *
 * A photo badge is deliberately NOT covered: the little white chips
 * that sit ON a photo (the album card's remove button) keep their
 * hairline in both themes, because a badge on a photograph has to
 * separate from whatever is behind it and cannot use the page's
 * tonal step to do it.
 */
import { test, expect, type Page } from '@playwright/test';

const APP = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';

/** Every border the page actually PAINTS, with the element it is on. */
async function paintedBorders(page: Page) {
  return page.evaluate(() => {
    const out: { sel: string; colour: string; width: string; bg: string }[] = [];
    for (const el of Array.from(document.querySelectorAll('*'))) {
      const cs = getComputedStyle(el);
      for (const side of ['Top', 'Right', 'Bottom', 'Left'] as const) {
        const width = cs[`border${side}Width`];
        if (width === '0px') continue;
        const colour = cs[`border${side}Color`];
        const m = colour.match(
          /rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)/
        );
        if (!m) continue;
        const alpha = m[4] === undefined ? 1 : parseFloat(m[4]);
        // A transparent border paints nothing, however wide it is.
        if (alpha < 0.01) continue;
        const cls =
          typeof el.className === 'string' && el.className.trim()
            ? el.className.trim().split(/\s+/)[0]
            : '';
        out.push({
          sel: el.tagName.toLowerCase() + (cls ? `.${cls}` : ''),
          colour: `${m[1]},${m[2]},${m[3]}`,
          width,
          bg: cs.backgroundColor
        });
      }
    }
    return out;
  });
}

/**
 * The perceptual lightness of a colour, 0-100 (CIE L*).
 */
function lightness(colour: string): number {
  const [r, g, b] = colour.split(',').map(Number);
  const lin = [r, g, b].map((v) => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  const y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
  return y > 0.008856 ? 116 * Math.cbrt(y) - 16 : 903.3 * y;
}

/**
 * Flip the marker the way the backend stamps it, and wait for it to
 * land.
 *
 * The wait is not just for the marker: a custom property on the root
 * changes the instant the attribute does, but the DESCENDANTS' computed
 * styles pick it up on the next style recalc. Measuring straight
 * after the flip reads the old border colours — the root already says
 * "transparent" while every element below it still says "6% black".
 * Two frames is enough for that recalc to run.
 */
async function goDark(page: Page) {
  await page.evaluate(() => {
    document.documentElement.setAttribute('data-theme', 'dark');
  });
  await expect
    .poll(async () =>
      page.evaluate(() =>
        getComputedStyle(document.documentElement)
          .getPropertyValue('--bg-0')
          .trim()
      )
    )
    .toBe('#0a0a0b');
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
      )
  );
}

test.describe('dark theme surfaces', () => {
  test('a dark-theme panel paints no border', async ({ page }) => {
    await page.goto(`${APP}/albums`);
    await page.waitForSelector('.card', { timeout: 30000 });
    await goDark(page);

    // The panel colour, as the browser resolves it.
    const panelRgb = await page.evaluate(() => {
      const raw = getComputedStyle(document.documentElement)
        .getPropertyValue('--bg-1')
        .trim();
      const m = raw.match(/^#?([\da-f]{6})$/i);
      if (!m) return null;
      const n = parseInt(m[1], 16);
      return `${(n >> 16) & 255},${(n >> 8) & 255},${n & 255}`;
    });
    expect(panelRgb, 'the dark theme must define --bg-1').not.toBeNull();

    const borders = await paintedBorders(page);

    // The panels are the surfaces that used to carry the outline.
    // They must now separate by their own lightness.
    const panels = borders.filter((b) => {
      const m = b.bg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      if (!m) return false;
      return `${m[1]},${m[2]},${m[3]}` === panelRgb;
    });

    // The panel has to be there for this to mean anything: on a light
    // deployment the panel colour belongs to the dark theme, so
    // nothing matches and the assertion below would pass on an
    // empty set.
    const panelCount = await page.evaluate((rgb) => {
      const [r, g, b] = rgb.split(',').map(Number);
      let n = 0;
      for (const el of Array.from(document.querySelectorAll('*'))) {
        const cs = getComputedStyle(el);
        const m = cs.backgroundColor.match(
          /rgba?\((\d+),\s*(\d+),\s*(\d+)/
        );
        if (!m) continue;
        if (+m[1] === r && +m[2] === g && +m[3] === b) n++;
      }
      return n;
    }, panelRgb);
    expect(panelCount, 'the dark theme must render panels to check').toBeGreaterThan(0);

    expect(
      panels.map((b) => `${b.sel} ${b.width} ${b.colour}`),
      'a dark-theme panel must not paint a border — it separates by ' +
        'lightness, and an outline around it is the wireframe'
    ).toEqual([]);
  });

  test('the dark theme paints no neutral border at all', async ({ page }) => {
    await page.goto(`${APP}/albums`);
    await page.waitForSelector('.card', { timeout: 30000 });
    await goDark(page);

    const borders = await paintedBorders(page);

    // The only borders the dark theme paints are chromatic — the
    // accent, which marks state. A neutral one would be an edge, and
    // the dark theme has none.
    const neutral = borders.filter((b) => {
      const [r, g, b2] = b.colour.split(',').map(Number);
      return Math.max(r, g, b2) - Math.min(r, g, b2) < 12;
    });

    expect(
      neutral.map((b) => `${b.sel} ${b.width} ${b.colour}`),
      'the dark theme must not paint a neutral border anywhere'
    ).toEqual([]);
  });

  test("the dark theme's controls carry the glass film", async ({ page }) => {
    await page.goto(`${APP}/settings`);
    await page.waitForSelector('.preset', { timeout: 30000 });
    await goDark(page);

    const film = await page.evaluate(() => {
      const el = document.querySelector('.preset');
      if (!el) return null;
      const cs = getComputedStyle(el);
      return {
        image: cs.backgroundImage,
        blur: cs.backdropFilter,
        shadow: cs.boxShadow
      };
    });
    expect(film, 'the settings page must render a preset control').not.toBeNull();

    // The film is FLAT. A gradient across a small control is the old
    // iOS recipe — a shiny dome with a bright band on top and a dark
    // band at the bottom — and that is what made the first attempt
    // look old-school rather than like glass. Modern dark glass has
    // almost no contrast in it, so the film must not fall off.
    expect(film!.image).toContain('linear-gradient');
    const alphas = [
      ...film!.image.matchAll(/rgba?\([^)]*?,\s*([\d.]+)\)/g)
    ].map((m) => parseFloat(m[1]));
    expect(alphas.length).toBeGreaterThan(1);
    expect(
      new Set(alphas).size,
      'the film must be flat — a gradient across it is the old iOS dome'
    ).toBe(1);

    // And it frosts whatever is behind it.
    expect(film!.blur, 'the film must frost what is behind it').not.toBe(
      'none'
    );

    // No rim. A rim is a shadow layer, not a border, so the "no
    // neutral border" test above never saw it — this one does. The
    // control separates by the tone of its own surface; nothing is
    // drawn around it.
    const lightLayers = [
      ...film!.shadow.matchAll(/rgba?\((\d+),\s*(\d+),\s*(\d+)/g)
    ]
      .map((m) => [+m[1], +m[2], +m[3]])
      .filter(([r, g, b]) => Math.min(r, g, b) > 180);
    expect(
      lightLayers,
      'the glass must not carry a rim — a light lip on a control reads ' +
        'as an edge, which is what the dark theme stopped drawing'
    ).toEqual([]);
  });

  test('the panel is a real step above the page', async ({ page }) => {
    await page.goto(`${APP}/albums`);
    await page.waitForSelector('.card', { timeout: 30000 });
    await goDark(page);

    const [pageColour, panelColour] = await page.evaluate(() => {
      const root = getComputedStyle(document.documentElement);
      const toRgb = (token: string) => {
        const raw = root.getPropertyValue(token).trim();
        const m = raw.match(/^#?([\da-f]{6})$/i);
        if (!m) return '0,0,0';
        const n = parseInt(m[1], 16);
        return `${(n >> 16) & 255},${(n >> 8) & 255},${n & 255}`;
      };
      return [toRgb('--bg-0'), toRgb('--bg-1')];
    });

    // A panel that is barely lighter than the page cannot read as a
    // surface on its own — which is why the hairline had to be
    // standing in for it. 8% of perceptual lightness is the floor
    // for a panel to separate on its own; the old ramp was 6%.
    const step = lightness(panelColour) - lightness(pageColour);
    expect(
      step,
      `panel ${panelColour} is only ${step.toFixed(1)}% L above page ` +
        `${pageColour} — too close to read as a surface without an outline`
    ).toBeGreaterThan(8);
  });
});
