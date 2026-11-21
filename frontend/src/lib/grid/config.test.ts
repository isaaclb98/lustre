/**
 * Unit tests for the grid configuration.
 *
 * The grid's numbers live in grid/config.ts. Two facts are pinned
 * here:
 *
 *   - there is no gutter and no rounded corner: the grid's photos
 *     touch, edge to edge, in every theme and at every size. The
 *     geometry is not a function of the theme any more, so the
 *     config carries no theme parameter at all;
 *   - the masonry ladder starts one rung above the square ladder,
 *     and both keep their row width.
 */
import { describe, it, expect } from 'vitest';
import {
  GRID_SIZE_CONFIG,
  GRID_TILE_RADIUS,
  MASONRY_GRID_SIZE_CONFIG,
  gridConfigFor,
  wallConfig
} from './config';
import type { GridSize } from './config';

const SIZES: GridSize[] = ['large', 'medium', 'small'];

function rowWidthOf(c: { tile: number; maxCols: number }): number {
  return c.maxCols * c.tile;
}

describe('gridConfigFor', () => {
  it('returns the square ladder untouched', () => {
    for (const size of SIZES) {
      expect(gridConfigFor(size)).toEqual(GRID_SIZE_CONFIG[size]);
    }
  });

  it('has no gutter to drop — the photos touch', () => {
    // There is no `gap` on the config any more: the gutter is not a
    // value the grid carries, so it cannot be theme-dependent.
    for (const size of SIZES) {
      expect('gap' in gridConfigFor(size)).toBe(false);
    }
  });

  it('has square corners — the radius is zero', () => {
    expect(GRID_TILE_RADIUS).toBe(0);
  });

  it('does not mutate the shared preset', () => {
    const before = GRID_SIZE_CONFIG.large.tile;
    gridConfigFor('large');
    expect(GRID_SIZE_CONFIG.large.tile).toBe(before);
  });
});

describe('the masonry ladder', () => {
  it('starts one rung above the square ladder', () => {
    // A masonry tile is not cropped to a square: the square grid
    // shows a centre CROP blown up to the tile width, masonry shows
    // the whole photo at 1:1. So at the same width a masonry photo
    // reads SMALLER, and the ladder starts one rung higher.
    expect(MASONRY_GRID_SIZE_CONFIG.large.tile).toBeGreaterThan(
      GRID_SIZE_CONFIG.large.tile
    );
    expect(MASONRY_GRID_SIZE_CONFIG.medium.tile).toBe(
      GRID_SIZE_CONFIG.large.tile
    );
    expect(MASONRY_GRID_SIZE_CONFIG.small.tile).toBe(
      GRID_SIZE_CONFIG.medium.tile
    );
  });

  it('keeps the density ordering: a bigger tile means fewer columns', () => {
    for (const size of SIZES) {
      expect(MASONRY_GRID_SIZE_CONFIG[size].maxCols).toBeLessThanOrEqual(
        GRID_SIZE_CONFIG[size].maxCols
      );
    }
    expect(MASONRY_GRID_SIZE_CONFIG.large.maxCols).toBeLessThan(
      MASONRY_GRID_SIZE_CONFIG.medium.maxCols
    );
  });

  it('keeps the row width where the square grid keeps it', () => {
    // The square ladder's row spans ~1440px at every size: a bigger
    // tile buys fewer columns. The masonry ladder must not sprawl
    // across a wide monitor.
    for (const size of SIZES) {
      expect(rowWidthOf(MASONRY_GRID_SIZE_CONFIG[size])).toBeLessThan(1700);
      expect(rowWidthOf(MASONRY_GRID_SIZE_CONFIG[size])).toBeGreaterThan(1400);
    }
  });

  it('is used by gridConfigFor when masonry is on', () => {
    for (const size of SIZES) {
      expect(gridConfigFor(size, true)).toEqual(
        MASONRY_GRID_SIZE_CONFIG[size]
      );
    }
  });

  it('does not mutate the shared preset', () => {
    const before = MASONRY_GRID_SIZE_CONFIG.large.tile;
    gridConfigFor('large', true);
    expect(MASONRY_GRID_SIZE_CONFIG.large.tile).toBe(before);
  });
});

describe('wallConfig', () => {
  it('is the app grid’s own shape', () => {
    // The wall used to differ by its gutter alone. With the app grid
    // gapless too, the wall IS the app grid's shape.
    for (const size of SIZES) {
      expect(wallConfig(gridConfigFor(size))).toEqual(gridConfigFor(size));
    }
  });

  it('keeps the tile size — the wall renders the photos the size the user chose', () => {
    for (const size of SIZES) {
      expect(wallConfig(gridConfigFor(size)).tile).toBe(
        GRID_SIZE_CONFIG[size].tile
      );
    }
  });
});
