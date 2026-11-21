/**
 * tests for the masonry layout — the Pinterest-style packing the photo
 * grid uses when the masonry mode is on.
 */
import { describe, expect, it } from 'vitest';
import {
  aspectFor,
  isClamped,
  layoutMasonry,
  DEFAULT_MAX_ASPECT,
  DEFAULT_MIN_ASPECT
} from './masonry';

const OPTS = { tile: 240, columns: 3 };

describe('aspectFor', () => {
  it('uses the photo\'s own ratio', () => {
    expect(aspectFor({ width: 1000, height: 500 })).toBeCloseTo(0.5);
    expect(aspectFor({ width: 1000, height: 1500 })).toBeCloseTo(1.5);
  });

  it('falls back to a square when the dimensions are unknown', () => {
    expect(aspectFor({})).toBe(1);
    expect(aspectFor({ width: null, height: null })).toBe(1);
    expect(aspectFor({ width: 0, height: 0 })).toBe(1);
    expect(aspectFor(null)).toBe(1);
    expect(aspectFor(undefined)).toBe(1);
  });

  it('clamps an extreme ratio', () => {
    // A 5:1 banner is very wide: its height/width ratio is 0.2, so
    // it clamps to the SHORTEST tile. A 1:5 sliver is very tall:
    // its ratio is 5, so it clamps to the TALLEST tile.
    expect(aspectFor({ width: 5000, height: 1000 })).toBe(DEFAULT_MIN_ASPECT);
    expect(aspectFor({ width: 1000, height: 5000 })).toBe(DEFAULT_MAX_ASPECT);
  });

  it('does not clamp the ratios the real library actually contains', () => {
    // Measured across 11,485 photos of the library: the ratio runs
    // from 0.325 to 2.996. A clamp that bit inside that range would
    // CROP those photos — the tile's box would not be the photo's
    // shape, and `object-fit: cover` would cut it off.
    expect(aspectFor({ width: 3000, height: 975 })).toBeCloseTo(0.325);
    expect(aspectFor({ width: 1000, height: 2996 })).toBeCloseTo(2.996);
    // The bounds themselves.
    expect(DEFAULT_MIN_ASPECT).toBeLessThanOrEqual(0.325);
    expect(DEFAULT_MAX_ASPECT).toBeGreaterThanOrEqual(2.996);
  });
});

describe('isClamped', () => {
  it('is false for a photo inside the bounds', () => {
    expect(isClamped({ width: 1000, height: 1500 })).toBe(false);
    expect(isClamped({ width: 1000, height: 500 })).toBe(false);
  });

  it('is false when the dimensions are unknown', () => {
    // An unknown-dimension photo falls back to a square, and a
    // square is its own shape — nothing to clamp.
    expect(isClamped({})).toBe(false);
    expect(isClamped(null)).toBe(false);
    expect(isClamped({ width: 0, height: 0 })).toBe(false);
  });

  it('is true for a photo outside the bounds', () => {
    expect(isClamped({ width: 10000, height: 1000 })).toBe(true);
    expect(isClamped({ width: 1000, height: 10000 })).toBe(true);
  });
});

describe('layoutMasonry', () => {
  it('places the first photos across the columns', () => {
    const { placements } = layoutMasonry(
      [
        { width: 1000, height: 1000 },
        { width: 1000, height: 1000 },
        { width: 1000, height: 1000 }
      ],
      OPTS
    );
    expect(placements).toHaveLength(3);
    expect(placements.map((p) => p.x)).toEqual([0, 240, 480]);
    expect(placements.map((p) => p.y)).toEqual([0, 0, 0]);
  });

  it('gives every tile its own height', () => {
    const { placements } = layoutMasonry(
      [
        { width: 1000, height: 500 },
        { width: 1000, height: 1500 }
      ],
      { ...OPTS, columns: 1 }
    );
    expect(placements[0].height).toBe(120);
    expect(placements[1].height).toBe(360);
    expect(placements[0].width).toBe(240);
  });

  it('packs into the shortest column, not round-robin', () => {
    // A tall photo lands in column 0, so the next photo goes to
    // column 1 (the shortest), not column 1 by round-robin luck.
    const { placements } = layoutMasonry(
      [
        { width: 1000, height: 2000 },
        { width: 1000, height: 1000 },
        { width: 1000, height: 1000 }
      ],
      { ...OPTS, columns: 2 }
    );
    // The tall photo is 480 tall; the square next to it is 240.
    expect(placements[0].x).toBe(0);
    expect(placements[1].x).toBe(240);
    expect(placements[1].y).toBe(0);
    // The third photo goes to column 1 again — it is still the
    // shortest column, not column 0 where the tall photo stands.
    expect(placements[2].x).toBe(240);
    expect(placements[2].y).toBe(240);
  });

  it('keeps the columns level', () => {
    const { placements, height } = layoutMasonry(
      [
        { width: 1000, height: 1500 },
        { width: 1000, height: 500 },
        { width: 1000, height: 1000 },
        { width: 1000, height: 1000 }
      ],
      OPTS
    );
    // Every tile sits above the layout's bottom edge.
    for (const p of placements) {
      expect(p.y + p.height).toBeLessThanOrEqual(height);
    }
    expect(height).toBeGreaterThan(0);
  });

  it('reports the layout height', () => {
    const { height } = layoutMasonry([{ width: 100, height: 100 }], {
      ...OPTS,
      columns: 1
    });
    expect(height).toBe(240);
  });

  it('is empty for no photos', () => {
    const { placements, height } = layoutMasonry([], OPTS);
    expect(placements).toHaveLength(0);
    expect(height).toBe(0);
  });

  it('is deterministic', () => {
    const photos = [
      { width: 1000, height: 1500 },
      { width: 800, height: 600 },
      { width: 1200, height: 1200 },
      { width: 640, height: 480 },
      { width: 900, height: 1600 }
    ];
    const a = layoutMasonry(photos, OPTS);
    const b = layoutMasonry(photos, OPTS);
    expect(a.placements).toEqual(b.placements);
    expect(a.height).toBe(b.height);
  });
});
