/**
 * Unit tests for the theme-aware grid configuration.
 *
 * The grid's numbers live in grid/config.ts, and the theme decides
 * two of them: the gutter and the tile radius. The dark theme's
 * grid has NO gutter — its photos touch pixel-to-pixel, like the
 * presentation wall's — and no radius.
 *
 * What we pin:
 *   - the light theme's numbers are unchanged by the theming (the
 *     light theme is the default and must not shift),
 *   - the dark theme's gutter is 0 and its radius is 0,
 *   - the tile size and the column cap are the user's grid-size
 *     preference in BOTH themes — the theme changes how the photos
 *     sit together, not how big they are or how many fit.
 */
import { describe, it, expect } from 'vitest';
import {
  GRID_SIZE_CONFIG,
  GRID_TILE_RADIUS,
  gridConfigFor,
  tileRadiusFor,
  wallConfig
} from './config';
import type { GridSize } from './config';

const SIZES: GridSize[] = ['large', 'medium', 'small'];

describe('gridConfigFor', () => {
  it('leaves the light theme exactly as the size preset defines it', () => {
    for (const size of SIZES) {
      const config = gridConfigFor(size, 'light');
      expect(config).toEqual(GRID_SIZE_CONFIG[size]);
      // The same numbers, and the same gap.
      expect(config.gap).toBe(GRID_SIZE_CONFIG[size].gap);
      expect(config.tile).toBe(GRID_SIZE_CONFIG[size].tile);
      expect(config.maxCols).toBe(GRID_SIZE_CONFIG[size].maxCols);
    }
  });

  it('drops the gutter entirely in the dark theme', () => {
    for (const size of SIZES) {
      expect(gridConfigFor(size, 'dark').gap).toBe(0);
    }
  });

  it('keeps the tile size and column cap from the size preset in the dark theme', () => {
    for (const size of SIZES) {
      const dark = gridConfigFor(size, 'dark');
      expect(dark.tile).toBe(GRID_SIZE_CONFIG[size].tile);
      expect(dark.maxCols).toBe(GRID_SIZE_CONFIG[size].maxCols);
      expect(dark.rows).toBe(GRID_SIZE_CONFIG[size].rows);
    }
  });

  it('does not mutate the shared preset', () => {
    const before = GRID_SIZE_CONFIG.large.gap;
    gridConfigFor('large', 'dark');
    expect(GRID_SIZE_CONFIG.large.gap).toBe(before);
  });
});

describe('tileRadiusFor', () => {
  it('keeps the rounded tile in the light theme', () => {
    expect(tileRadiusFor('light')).toBe(GRID_TILE_RADIUS);
  });

  it('is square in the dark theme', () => {
    expect(tileRadiusFor('dark')).toBe(0);
  });
});

describe('wallConfig', () => {
  it('drops the gutter whatever the theme', () => {
    for (const size of SIZES) {
      expect(wallConfig(gridConfigFor(size, 'light')).gap).toBe(0);
      expect(wallConfig(gridConfigFor(size, 'dark')).gap).toBe(0);
    }
  });

  it('keeps the tile size — the wall renders the photos the size the user chose', () => {
    for (const size of SIZES) {
      expect(wallConfig(gridConfigFor(size, 'dark')).tile).toBe(
        GRID_SIZE_CONFIG[size].tile
      );
    }
  });
});
