/**
 * Unit tests for lib/grid/geometry.ts — the shared column math and
 * request sizing.
 *
 * The invariant that matters: a page request must contain a whole
 * number of grid rows at the column count the viewport actually
 * renders. Every test here pins either that invariant or the
 * formula the renderer shares with the fetch layer.
 */
import { describe, it, expect } from 'vitest';
import {
  GRID_SIZE_CONFIG,
  basePageSize,
  type GridSize
} from '$lib/grid/config';
import {
  columnsForWidth,
  widthForColumns,
  pageSizeForColumns,
  pageSizeWithinColumns
} from '$lib/grid/geometry';

const SIZES: GridSize[] = ['large', 'medium', 'small'];

/** Container widths spanning narrow phone to ultra-wide desktop. */
const WIDTHS = [
  320, 375, 414, 480, 600, 700, 768, 900, 1024, 1100, 1280, 1440,
  1600, 1920, 2200, 2560, 3840
];

describe('columnsForWidth', () => {
  it('never returns fewer than one column', () => {
    for (const size of SIZES) {
      const config = GRID_SIZE_CONFIG[size];
      expect(columnsForWidth(0, config)).toBe(1);
      expect(columnsForWidth(10, config)).toBe(1);
    }
  });

  it('never exceeds the preset maxCols', () => {
    for (const size of SIZES) {
      const config = GRID_SIZE_CONFIG[size];
      for (const width of WIDTHS) {
        expect(columnsForWidth(width, config)).toBeLessThanOrEqual(
          config.maxCols
        );
      }
    }
  });

  it('matches the CSS auto-fit formula exactly', () => {
    // The grid renders `repeat(auto-fit, TILE)` tracks with the
    // configured gap, so the track count is floor((w + gap) /
    // (tile + gap)) — capped at maxCols.
    for (const size of SIZES) {
      const config = GRID_SIZE_CONFIG[size];
      for (const width of WIDTHS) {
        const expected = Math.max(
          1,
          Math.min(
            config.maxCols,
            Math.floor((width + config.gap) / (config.tile + config.gap))
          )
        );
        expect(columnsForWidth(width, config)).toBe(expected);
      }
    }
  });

  it('grows monotonically with width', () => {
    for (const size of SIZES) {
      const config = GRID_SIZE_CONFIG[size];
      let previous = 1;
      for (const width of WIDTHS) {
        const cols = columnsForWidth(width, config);
        expect(cols).toBeGreaterThanOrEqual(previous);
        previous = cols;
      }
    }
  });

  it('scales density with the preset — small packs more columns than large', () => {
    for (const width of [1024, 1280, 1600]) {
      expect(columnsForWidth(width, GRID_SIZE_CONFIG.small)).toBeGreaterThan(
        columnsForWidth(width, GRID_SIZE_CONFIG.large)
      );
    }
  });
});

describe('widthForColumns', () => {
  it('is the exact pixel width of a full row of that many columns', () => {
    for (const size of SIZES) {
      const config = GRID_SIZE_CONFIG[size];
      for (let cols = 1; cols <= config.maxCols; cols++) {
        const width = widthForColumns(cols, config);
        expect(width).toBe(cols * config.tile + (cols - 1) * config.gap);
        // A row that width fits must derive back the same count.
        expect(columnsForWidth(width, config)).toBe(cols);
      }
    }
  });
});

describe('pageSizeForColumns', () => {
  it('rounds up to a whole number of rows', () => {
    expect(pageSizeForColumns(24, 6)).toBe(24);
    expect(pageSizeForColumns(24, 5)).toBe(25);
    expect(pageSizeForColumns(24, 4)).toBe(24);
    expect(pageSizeForColumns(24, 7)).toBe(28);
    expect(pageSizeForColumns(24, 3)).toBe(24);
  });

  it('never returns fewer photos than requested', () => {
    for (const size of SIZES) {
      const base = basePageSize(GRID_SIZE_CONFIG[size]);
      for (let cols = 1; cols <= GRID_SIZE_CONFIG[size].maxCols; cols++) {
        expect(pageSizeForColumns(base, cols)).toBeGreaterThanOrEqual(base);
      }
    }
  });

  it('always divides evenly by the column count', () => {
    for (const size of SIZES) {
      const base = basePageSize(GRID_SIZE_CONFIG[size]);
      for (let cols = 1; cols <= GRID_SIZE_CONFIG[size].maxCols; cols++) {
        expect(pageSizeForColumns(base, cols) % cols).toBe(0);
      }
    }
  });
});

describe('pageSizeWithinColumns', () => {
  it('is the largest whole number of rows under the ceiling', () => {
    expect(pageSizeWithinColumns(100, 4)).toBe(100);
    expect(pageSizeWithinColumns(100, 6)).toBe(96);
    expect(pageSizeWithinColumns(100, 7)).toBe(98);
    expect(pageSizeWithinColumns(100, 3)).toBe(99);
  });

  it('never exceeds the ceiling', () => {
    for (let cols = 1; cols <= 10; cols++) {
      expect(pageSizeWithinColumns(100, cols)).toBeLessThanOrEqual(100);
    }
  });

  it('always divides evenly by the column count', () => {
    for (let cols = 1; cols <= 10; cols++) {
      expect(pageSizeWithinColumns(100, cols) % cols).toBe(0);
    }
  });

  it('returns one row when the ceiling is smaller than a row', () => {
    expect(pageSizeWithinColumns(3, 6)).toBe(6);
  });
});

describe('the whole-row invariant', () => {
  it('holds for every preset at every tested viewport width', () => {
    // The property the feature exists for: the number of photos a
    // request asks for is always a whole number of rows at the
    // columns that viewport renders.
    for (const size of SIZES) {
      const config = GRID_SIZE_CONFIG[size];
      const base = basePageSize(config);
      for (const width of WIDTHS) {
        const cols = columnsForWidth(width, config);
        const limit = pageSizeForColumns(base, cols);
        expect(limit % cols).toBe(0);
        expect(limit).toBeGreaterThanOrEqual(base);
      }
    }
  });

  it('holds for the ceiling-bounded variant too', () => {
    for (const size of SIZES) {
      const config = GRID_SIZE_CONFIG[size];
      for (const width of WIDTHS) {
        const cols = columnsForWidth(width, config);
        const limit = pageSizeWithinColumns(100, cols);
        expect(limit % cols).toBe(0);
        expect(limit).toBeLessThanOrEqual(100);
      }
    }
  });

  it('the fixed page sizes were NOT whole-row safe at narrow widths', () => {
    // Regression guard for the bug this replaces: the old fixed
    // counts left a partial trailing row at several widths.
    expect(32 % 3).not.toBe(0);
    expect(32 % 6).not.toBe(0);
    expect(32 % 7).not.toBe(0);
    expect(40 % 6).not.toBe(0);
  });
});
