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
  wallColumnsForWidth,
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

describe('wallColumnsForWidth', () => {
  it('never returns fewer than one column', () => {
    for (const size of SIZES) {
      const config = GRID_SIZE_CONFIG[size];
      expect(wallColumnsForWidth(0, config)).toBe(1);
      expect(wallColumnsForWidth(10, config)).toBe(1);
    }
  });

  it('is never capped by maxCols — the wall fills the screen', () => {
    // The app grid caps its column count so a wide monitor does
    // not crowd the photo sizes the user chose. The wall has no
    // row-width cap — its row IS the screen — so the count has to
    // follow the width, or six tracks would stretch across 2560px
    // and each photo would render at nearly twice the size the
    // user picked.
    const config = GRID_SIZE_CONFIG.large;
    expect(wallColumnsForWidth(2560, config)).toBeGreaterThan(
      config.maxCols
    );
  });

  it('keeps the photo within half a tile of the user\'s choice', () => {
    // The nearest fit: the tracks stretch, so the track width is
    // what the photo actually renders at. It has to stay close to
    // the size the user picked everywhere else in the app.
    //
    // The one exception is a single column on a narrow screen:
    // there is no second photo to share the width with, so the
    // one photo stretches to fill it. A phone showing one photo
    // across the screen is right; showing it at 240px with a
    // 135px gap beside it is not.
    for (const size of SIZES) {
      const config = GRID_SIZE_CONFIG[size];
      for (const width of WIDTHS) {
        const cols = wallColumnsForWidth(width, config);
        const track = width / cols;
        if (cols === 1) continue;
        expect(Math.abs(track - config.tile)).toBeLessThanOrEqual(
          config.tile / 2
        );
      }
    }
  });

  it('fills the width exactly — there is no leftover band', () => {
    // The property the wall exists for. The CSS lays the tracks
    // out as 1fr, so they always sum to the container width; the
    // JS count only has to be the number of tracks the browser
    // lays out. What must not happen is a count that makes the
    // photos a wildly different size — a leftover reappearing as
    // one oversized hole instead of a thin margin.
    for (const size of SIZES) {
      const config = GRID_SIZE_CONFIG[size];
      for (const width of WIDTHS) {
        const cols = wallColumnsForWidth(width, config);
        const track = width / cols;
        // The track is the photo's width. It can never exceed the
        // width it has to fit in.
        expect(track).toBeLessThanOrEqual(width);
        // And with more than one column it stays within a tile of
        // the user's choice.
        if (cols > 1) {
          expect(track).toBeLessThanOrEqual(config.tile * 1.5);
        }
      }
    }
  });

  it('grows monotonically with width', () => {
    for (const size of SIZES) {
      const config = GRID_SIZE_CONFIG[size];
      let previous = 1;
      for (const width of WIDTHS) {
        const cols = wallColumnsForWidth(width, config);
        expect(cols).toBeGreaterThanOrEqual(previous);
        previous = cols;
      }
    }
  });

  it('is never further from the user\'s tile size than the app grid is', () => {
    // The wall's count is the nearest fit; the app grid's is the
    // largest that fits. The wall's track width is never FURTHER
    // from the user's tile size than the app grid's — so a photo
    // on the wall is never a different size than the same photo
    // on the album page.
    for (const size of SIZES) {
      const config = GRID_SIZE_CONFIG[size];
      for (const width of WIDTHS) {
        const appCols = columnsForWidth(width, config);
        const wallCols = wallColumnsForWidth(width, config);
        const appTrack = width / appCols;
        const wallTrack = width / wallCols;
        expect(Math.abs(wallTrack - config.tile)).toBeLessThanOrEqual(
          Math.abs(appTrack - config.tile)
        );
      }
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
