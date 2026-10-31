/**
 * Favicon regression tests.
 *
 * A favicon is loaded by the browser as its OWN SVG document, not as a
 * child of the page. Two consequences bite, and both are invisible to
 * the eye in the source file:
 *
 *  1. `currentColor` resolves against the SVG document's own initial
 *     `color` (black) — it does NOT inherit the page's colour, and it
 *     does NOT track the browser's tab colour. A glyph stroked in
 *     `currentColor` is therefore black-on-black on dark chrome and
 *     vanishes at 16px.
 *
 *  2. The favicon has to carry its own opaque background so it reads on
 *     both light and dark tab strips without relying on the browser
 *     picking a tone for it.
 *
 * The third assertion pins the favicon to the SAME path data as the
 * in-app brand mark (Icon.svelte's 'sparkle'), so the tab icon and the
 * topbar logo cannot drift into two different approximations of a
 * sparkle.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const faviconSvg = readFileSync(
  path.resolve(import.meta.dirname, '..', '..', 'static', 'favicon.svg'),
  'utf-8'
);

/** Markup only — the file's prose comments legitimately mention these
 *  terms, and the assertions below are about what the browser renders. */
const faviconMarkup = faviconSvg.replace(/<!--[\s\S]*?-->/g, '');

const iconSvelte = readFileSync(
  path.resolve(import.meta.dirname, 'components', 'Icon.svelte'),
  'utf-8'
);

function pathData(svg: string): string[] {
  return [...svg.matchAll(/<path[^>]*\bd="([^"]+)"/g)].map((m) => m[1]);
}

function brandMarkPaths(): string[] {
  const entry = iconSvelte.match(/'sparkle':\s*'([^']+)'/);
  expect(entry, "Icon.svelte must define a 'sparkle' glyph").toBeTruthy();
  return pathData(entry![1]);
}

describe('favicon', () => {
  it('does not rely on currentColor for its glyph', () => {
    expect(faviconMarkup).not.toContain('currentColor');
  });

  it('paints an opaque background of its own', () => {
    // A rect with a solid (fully opaque) fill — no rgba() with alpha,
    // no 'none', no currentColor.
    const bg = faviconMarkup.match(/<rect[^>]*>/)?.[0] ?? '';
    expect(bg, 'favicon must have a background rect').toBeTruthy();
    const fill = bg.match(/fill="([^"]+)"/)?.[1];
    expect(fill).toBeTruthy();
    expect(fill).toMatch(/^#[0-9a-fA-F]{6}$/);
  });

  it('is black on white, in one variant', () => {
    // The favicon is a single black-on-white mark: no theme switch, no
    // accent colour. The white tile supplies the contrast, so the icon
    // reads the same on light and dark chrome.
    const bg = faviconMarkup.match(/<rect[^>]*>/)?.[0] ?? '';
    expect(bg.match(/fill="([^"]+)"/)?.[1]?.toLowerCase()).toBe('#ffffff');

    const glyphFills = [...faviconMarkup.matchAll(/<path[^>]*\bfill="([^"]+)"/g)].map(
      (m) => m[1].toLowerCase()
    );
    expect(glyphFills.length).toBeGreaterThan(0);
    for (const f of glyphFills) {
      expect(f).toBe('#000000');
    }
  });

  it('has no theme-dependent variant', () => {
    // A prefers-color-scheme block inside a favicon would produce a
    // second look; there is only one.
    expect(faviconMarkup).not.toContain('prefers-color-scheme');
    expect(faviconMarkup).not.toContain('@media');
  });

  it('uses the same path data as the in-app brand mark', () => {
    expect(pathData(faviconMarkup)).toEqual(brandMarkPaths());
  });

  it('is valid SVG with a viewBox', () => {
    expect(faviconMarkup).toMatch(/^<svg[^>]*xmlns="http:\/\/www\.w3\.org\/2000\/svg"/);
    expect(faviconMarkup).toMatch(/viewBox="0 0 \d+ \d+"/);
  });

  it('is well-formed XML', () => {
    // XML forbids a double hyphen inside a comment. A browser that
    // hits a malformed SVG favicon renders nothing at all — the tab
    // shows the default page icon — while the file still parses fine
    // for anything that only greps it.
    const doc = new DOMParser().parseFromString(faviconSvg, 'image/svg+xml');
    expect(
      doc.querySelector('parsererror'),
      'favicon.svg must be well-formed XML'
    ).toBeNull();
  });
});
