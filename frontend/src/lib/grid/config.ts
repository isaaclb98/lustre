/**
 * Grid size configuration — the numbers behind the user's
 * "Large / Medium / Small" choice, and the arrangement the grid
 * uses to place them.
 *
 * This module owns the static numbers only: no DOM, no store, no
 * runtime imports. Everything that needs to reason about the grid —
 * the CSS vars, the layout-level ResizeObserver, the virtualizer's
 * row slicing, the request sizing — reads from here so those four
 * consumers can never drift apart.
 *
 * Tile size: 240/180/144 px. Large matches the Round-55 default;
 * medium and small pack progressively more photos per row.
 *
 * Every rung's row is 1440px wide: 6x240, 8x180, 10x144. That is
 * the ladder's other promise (see below), and it is why the middle
 * rung is 180 and not 192 — 192x8 is 1536, which would make the
 * row a different width from the other two. A row that changed
 * width with the size preference resized every piece of chrome
 * keyed to it, including the settings page's own width.
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

/**
 * The grid's arrangement.
 *
 * 'grid' — the uniform square grid: one tile size, one row height,
 * rows sliced by the virtualizer.
 *
 * 'masonry' — Pinterest-style: every photo keeps its own aspect
 * ratio and the tiles are packed into the shortest column. The tile
 * WIDTH is fixed and the height varies.
 *
 * 'justified' — the Flickr / Google Photos / Immich layout: every
 * ROW has one height and the photos in it are scaled to fill the
 * row's width exactly. The tile HEIGHT is fixed per row and the
 * width varies.
 *
 * The three are different arrangements of the same photos, and the
 * size ladder is per-arrangement: the square grid and the masonry
 * both have a tile WIDTH, while the justified layout has a row
 * HEIGHT instead.
 */
export type GridLayout = 'grid' | 'masonry' | 'justified';

export interface GridSizeConfig {
  tile: number;
  maxCols: number;
  /** Rows of photos a page holds at maximum density. */
  rows: number;
}

export const GRID_SIZE_CONFIG: Record<GridSize, GridSizeConfig> = {
  large: { tile: 240, maxCols: 6, rows: 4 },
  medium: { tile: 180, maxCols: 8, rows: 4 },
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
 * square grid's "medium" (180) is masonry's "small". Masonry's
 * "large" is a rung above both — 360px, four columns.
 *
 * Why 360 and four columns: the ladder's other promise is that
 * the ROW keeps its width whatever the tile size, so the grid does
 * not sprawl across a wide monitor. The square grid's large row is
 * 6 x 240 = 1440px. Masonry's large row is 4 x 360 = 1440px —
 * the same width, two thirds the columns, half again the photo.
 *
 * That 1440px row is why the masonry's small rung is 180 and not
 * 192: 8x192 is 1536, so a 192 tile would make the SMALL view
 * wider than the large one — the opposite of what a denser view
 * should do. Every rung in both ladders is now 1440px.
 */
export const MASONRY_GRID_SIZE_CONFIG: Record<GridSize, GridSizeConfig> = {
  large: { tile: 360, maxCols: 4, rows: 4 },
  medium: { tile: 240, maxCols: 6, rows: 4 },
  small: { tile: 180, maxCols: 8, rows: 4 }
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
 * The grid's numbers for the layout that is actually live.
 *
 * The three layouts measure different things: the square grid and
 * the masonry have a tile WIDTH, the justified layout has a row
 * HEIGHT. The consumers that only care about "how big is a photo
 * here, and how many of them are in a row" — the CSS vars the
 * chrome is sized against, the skeleton placeholder — do not care
 * which of the two it is, so they read this one function.
 *
 * The justified ladder's numbers are chosen so that the row width
 * the layout promises is the same 1440px the other two keep:
 * 4 x 360, 6 x 240, 8 x 180.
 */
export function layoutGridConfig(
  size: GridSize,
  layout: GridLayout
): GridSizeConfig {
  if (layout === 'justified') {
    const justified = JUSTIFIED_GRID_SIZE_CONFIG[size];
    return {
      tile: justified.rowHeight,
      maxCols: justified.cols,
      rows: justified.rows
    };
  }
  return gridConfigFor(size, layout === 'masonry');
}

/**
 * The justified layout's ladder.
 *
 * A justified layout has no fixed tile width — that is the whole
 * point of it — so the size knob is the ROW HEIGHT instead. The
 * row WIDTH is the grid's own width, capped at the ladder's promise.
 *
 * The three row heights are the masonry ladder's three tile widths,
 * and they put the SAME number of photos on a row (measured on the
 * real library at a 1440px row: 4.4 / 6.7 / 8.7 photos per row for
 * heights 360 / 240 / 180, against the masonry's 4 / 6 / 8 columns).
 * So switching between the masonry and the justified layout changes
 * the ARRANGEMENT and not the density — which is what makes the two
 * comparable at a glance.
 *
 * Immich's own desktop row height is 235, which lands between this
 * ladder's medium (240) and large (360).
 */
export interface JustifiedGridSizeConfig {
  /** The target height of a row, in px. */
  rowHeight: number;
  /** The width a row fills, in px. Never more than this. */
  maxWidth: number;
  /**
   * The photos a row is expected to hold.
   *
   * Used only to size a page request — the actual count per row
   * varies with the photos' aspect ratios, so this is the average.
   */
  cols: number;
  /** Rows of photos one page holds. */
  rows: number;
}

export const JUSTIFIED_GRID_SIZE_CONFIG: Record<GridSize, JustifiedGridSizeConfig> = {
  large: { rowHeight: 360, maxWidth: 1440, cols: 4, rows: 4 },
  medium: { rowHeight: 240, maxWidth: 1440, cols: 6, rows: 4 },
  small: { rowHeight: 180, maxWidth: 1440, cols: 8, rows: 4 }
};

/**
 * The justified layout's numbers for a size.
 *
 * Separate from gridConfigFor() because the two ladders measure
 * different things: the square grid and the masonry have a tile
 * WIDTH, the justified layout has a row HEIGHT.
 */
export function justifiedConfigFor(size: GridSize): JustifiedGridSizeConfig {
  return JUSTIFIED_GRID_SIZE_CONFIG[size];
}

/**
 * Photos per page before row alignment, for the justified layout.
 *
 * The same shape as basePageSize(): rows of photos, the row count
 * being the stable part of the promise.
 */
export function justifiedBasePageSize(config: JustifiedGridSizeConfig): number {
  return config.rows * config.cols;
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
