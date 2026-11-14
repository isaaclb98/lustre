/**
 * Unit tests for the lightbox preload-window arithmetic.
 * Mirrors the repo convention (lightbox-pagination.test.ts): pure
 * logic tested without mounting Svelte; the mounted behaviour is
 * covered by the Playwright E2E.
 */
import { describe, it, expect } from 'vitest';
import {
  MANUAL_PRELOAD_WINDOW,
  SLIDESHOW_PRELOAD_WINDOW,
  preloadIndices,
  type PreloadWindow
} from './preload-window';

const MANUAL = MANUAL_PRELOAD_WINDOW;
const SLIDESHOW = SLIDESHOW_PRELOAD_WINDOW;

describe('window shape', () => {
  it('manual navigation leans forward: 1 behind, 2 ahead', () => {
    expect(MANUAL.prev).toBe(1);
    expect(MANUAL.next).toBe(2);
  });

  it('the slideshow keeps a symmetric, deeper window', () => {
    expect(SLIDESHOW.prev).toBe(SLIDESHOW.next);
    expect(SLIDESHOW.next).toBeGreaterThan(MANUAL.next);
  });
});

describe('preloadIndices (no wrap — manual navigation)', () => {
  it('warms one behind and two ahead of the cursor', () => {
    expect(preloadIndices(5, 20, MANUAL, false)).toEqual([6, 7, 4]);
  });

  it('orders forward indices first so their fetches start first', () => {
    expect(preloadIndices(5, 20, MANUAL, false).slice(0, 2)).toEqual([6, 7]);
  });

  it('drops the behind index at the start of the list', () => {
    expect(preloadIndices(0, 20, MANUAL, false)).toEqual([1, 2]);
  });

  it('drops the ahead indices at the end of the list', () => {
    expect(preloadIndices(18, 20, MANUAL, false)).toEqual([19, 17]);
  });

  it('never includes the cursor itself', () => {
    const idxs = preloadIndices(7, 20, MANUAL, false);
    expect(idxs).not.toContain(7);
  });

  it('emits each index once when the window is wider than the list', () => {
    const idxs = preloadIndices(0, 3, SLIDESHOW, false);
    expect(idxs.sort()).toEqual([1, 2]);
    expect(new Set(idxs).size).toBe(idxs.length);
  });

  it('returns nothing for a single photo', () => {
    expect(preloadIndices(0, 1, MANUAL, false)).toEqual([]);
  });

  it('returns nothing for an empty list', () => {
    expect(preloadIndices(0, 0, MANUAL, false)).toEqual([]);
  });
});

describe('preloadIndices (wrap — slideshow)', () => {
  it('wraps past the last photo back to the first', () => {
    expect(preloadIndices(19, 20, MANUAL, true)).toEqual([0, 1, 18]);
  });

  it('wraps before the first photo to the last', () => {
    expect(preloadIndices(0, 20, MANUAL, true)).toEqual([1, 2, 19]);
  });

  it('reaches every other photo when the window covers the whole list', () => {
    const idxs = preloadIndices(0, 4, SLIDESHOW, true);
    expect(idxs.sort()).toEqual([1, 2, 3]);
  });
});

describe('the forward run stays warm', () => {
  it('the second photo of a → run was warmed before the first press', () => {
    // At cursor 5 the window already contains 6 and 7 — so pressing
    // → twice lands on two photos that were both cached up front.
    const warm = preloadIndices(5, 20, MANUAL, false);
    expect(warm).toContain(6);
    expect(warm).toContain(7);
  });

  it('each new cursor keeps two ahead warm as the run continues', () => {
    const len = 20;
    for (let cursor = 0; cursor < len - 2; cursor++) {
      const warm = preloadIndices(cursor, len, MANUAL, false);
      expect(warm).toContain(cursor + 1);
      expect(warm).toContain(cursor + 2);
    }
  });
});

describe('custom windows', () => {
  const custom: PreloadWindow = { prev: 0, next: 1 };
  it('honours a zero behind side', () => {
    expect(preloadIndices(5, 20, custom, false)).toEqual([6]);
  });

  it('honours a zero ahead side, nearest-behind first', () => {
    expect(preloadIndices(5, 20, { prev: 2, next: 0 }, false)).toEqual([4, 3]);
  });
});
