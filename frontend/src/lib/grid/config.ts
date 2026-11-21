/**
 * Grid size configuration — the numbers behind the user's
 * "Large / Medium / Small" choice.
 *
 * This module owns the static numbers only: no DOM, no store, no
 * runtime imports. Everything that needs to reason about the grid —
 * the CSS vars, the layout-level ResizeObserver, the virtualizer's
 * row slicing, the request sizing — reads from here so those four
 * consumers can never drift apart.
 *
 * Tile size: 240/192/144 px. Large matches the Round-55 default;
 * medium and small pack progressively more photos per row.
 *
 * There is NO GUTTER. The grid's photos touch, edge to edge, the
 * way the presentation wall's do — that is now the only rule, so
 * the geometry is not a function of the theme: one ladder, one
 * shape, whatever the palette is doing.
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
  maxCols: number;
  /** Rows of photos a page holds at maximum density. */
  rows: number;
}

export const GRID_SIZE_CONFIG: Record<GridSize, GridSizeConfig> = {
  large: { tile: 240, maxCols: 6, rows: 4 },
  medium: { tile: 192, maxCols: 8, rows: 4 },
  small: { tile: 144, maxCols: 10, rows: 4 }
};

/**
 * The masonry ladder — the same three choices, one rung higher.
 *
 * A masonry tile is NOT cropped to a square: the photo is shown at
 * its own aspect ratio, and the tile's height follows from it. So
 * the tile width no longer implies the photo's bulk.
 *
 * The square grid shows a portrait as a centre square CROP blown
 * up to the tile's width — the crop is magnified past 1:1 (a 3:4
 * portrait loses a third of its width). Masonry shows the whole
 * photo at 1:1, so at the same tile width the photo reads
 * SMALLER, even though more of it is visible.
 *
 * The masonry ladder therefore starts one rung higher: the 240px
 * the square grid calls "large" is masonry's "medium", and the
 * square grid's "medium" (192) is masonry's "small". Masonry's
 * "large" is a rung above both — 360px, four columns.
 *
 * Why 360 and four columns: the ladder's other promise is that
 * the ROW keeps its width whatever the tile size, so the grid does
 * not sprawl across a wide monitor. The square grid's large row is
 * 6 x 240 = 1440px. Masonry's large row is 4 x 360 = 1440px —
 * the same width, two thirds of the columns, half again the photo.
 */
export const MASONRY_GRID_SIZE_CONFIG: Record<GridSize, GridSizeConfig> = {
  large: { tile: 360, maxCols: 4, rows: 4 },
  medium: { tile: 240, maxCols: 6, rows: 4 },
  small: { tile: 192, maxCols: 8, rows: 4 }
};

/**
 * The hover zoom: how far the photo inside a tile scales up when
 * the pointer is over it.
 *
 * The zoom is applied to the IMAGE, not the tile. Scaling the
 * tile would push it over its neighbours and break the grid's
 * rhythm, so the tile stays put and the photo grows inside the
 * clipped frame instead.
 */
export const GRID_HOVER_ZOOM = 1.04;

/**
 * Corner radius of a tile, in px.
 *
 * Published as `--grid-tile-radius` so the tile and the loading
 * skeleton — rendered by two different components — cannot drift
 * apart in shape.
 *
 * Zero: the grid's photos touch, and a rounded corner would leave
 * the background showing through at every intersection.
 */
export const GRID_TILE_RADIUS = 0;

/**
 * The grid configuration for a size.
 *
 * The layout is a parameter rather than a lookup so this module
 * stays free of the DOM: it owns numbers, and the caller owns the
 * rest. Masonry has its own ladder (see MASONRY_GRID_SIZE_CONFIG),
 * and every consumer that must agree on these numbers (the
 * layout's published CSS vars, the grid's virtualizer, the column
 * math, the request sizing) calls this one function.
 */
export function gridConfigFor(size: GridSize, masonry = false): GridSizeConfig {
  return masonry ? MASONRY_GRID_SIZE_CONFIG[size] : GRID_SIZE_CONFIG[size];
}

/**
 * The presentation wall's grid config: the same ladder as the app
 * grid.
 *
 * The wall used to differ by its gutter alone — it is a square,
 * gapless field by design. With the app grid gapless too, the wall
 * is the app grid's own shape, and the only thing it still
 * overrides is the row's width cap (a presentation fills the
 * screen rather than keeping the app's row width).
 *
 * Because there is no gutter, the column math in grid/geometry.ts
 * collapses to floor(width / tile), and a full row is exactly
 * cols x tile px — the row's max width and the CSS track count
 * agree with the JS without a correction term.
 */
export function wallConfig(base: GridSizeConfig): GridSizeConfig {
  return base;
}

/**
 * Motion timing for the grid's own transitions, in ms.
 *
 * The app-wide `--t-fast` / `--ease-out` tokens are referenced
 * across the codebase but were never defined, so any transition
 * that names one currently resolves to nothing and the browser
 * applies the initial `0s` duration — the animations do not run.
 * The grid carries its own pair rather than depending on tokens
 * that do not exist, and rather than switching on transitions
 * across the whole app as a side effect of a grid change.
 */
export const GRID_MOTION_MS = 100;

/**
 * Easing for the grid's transitions. The app's historical
 * `--ease-out` curve (cubic-bezier(0.16, 0.84, 0.32, 1)), which
 * is what the pre-redesign tokens had.
 */
export const GRID_MOTION_EASE = 'cubic-bezier(0.16, 0.84, 0.32, 1)';

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
