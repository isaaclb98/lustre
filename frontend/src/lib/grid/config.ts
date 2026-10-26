/**
 * Grid size configuration — the numbers behind the user's
 * "Large / Medium / Small" choice.
 *
 * This module owns the static numbers only: no DOM, no store, no
 * imports. Everything that needs to reason about the grid — the CSS
 * vars, the layout-level ResizeObserver, the virtualizer's row
 * slicing, the request sizing — reads from here so those four
 * consumers can never drift apart.
 *
 * Tile size: 240/192/144 px. Large matches the Round-55 default;
 * medium and small pack progressively more photos per row.
 *
 * Gap: 14/8/4 px. Larger photos get more breathing room (the
 * Round-38 design intent — 14px reads as "personal photo library",
 * tighter gaps read as utilitarian). Smaller photos need a tighter
 * gap to read as a continuous wall instead of fragmenting into a
 * grid of small squares.
 *
 * Max cols: 6/8/10. The CSS grid uses `auto-fit` with the tile
 * size as the track width, so a narrow viewport shows fewer cols;
 * the max-cols cap kicks in only on wide monitors. This is the
 * per-size density promise: "Large is 6 cols wide, Small is 10
 * cols wide" regardless of viewport.
 *
 * Rows: 4. How many rows of photos one page holds at maximum
 * density. The *photo* count a request asks for depends on the
 * viewport (see grid/geometry.ts) — the row count is the stable
 * part of the promise, and every page holds at least this many
 * rows once aligned to whole rows.
 */
export type GridSize = 'large' | 'medium' | 'small';

export interface GridSizeConfig {
  tile: number;
  gap: number;
  maxCols: number;
  /** Rows of photos a page holds at maximum density. */
  rows: number;
}

export const GRID_SIZE_CONFIG: Record<GridSize, GridSizeConfig> = {
  large: { tile: 240, gap: 14, maxCols: 6, rows: 4 },
  medium: { tile: 192, gap: 8, maxCols: 8, rows: 4 },
  small: { tile: 144, gap: 4, maxCols: 10, rows: 4 }
};

export const DEFAULT_GRID_SIZE: GridSize = 'large';

/**
 * Photos per page before row alignment — `rows × maxCols`. The
 * request size is this number rounded up to a whole number of rows
 * at the column count the viewport actually renders (Large 24,
 * Medium 32, Small 40 at their design widths).
 */
export function basePageSize(config: GridSizeConfig): number {
  return config.rows * config.maxCols;
}
