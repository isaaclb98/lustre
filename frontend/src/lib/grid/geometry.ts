/**
 * Grid geometry — the single source of truth for how many columns the
 * photo grid renders and how many photos a page request asks for.
 *
 * The grid renders `repeat(auto-fit, TILE)` tracks in a row as wide as
 * the grid's container, so the column count is a pure function of the
 * container width and the active grid size:
 *
 *   cols = min(maxCols, floor((width + GAP) / (TILE + GAP)))
 *
 * Three consumers depend on that number and must never disagree:
 *
 *   - the layout, which publishes --grid-width from it,
 *   - the grid itself, which slices items into rows of `cols`,
 *   - the fetch layer, which sizes its requests.
 *
 * They agree by construction because they all read the same numbers
 * from grid/config.ts and the same formula from this file.
 *
 * ## Which width to measure
 *
 * The layout's ResizeObserver watches `.shell` and publishes
 * `--grid-width` from its content box. The grid's wrapper is sized
 * from that var, so the number of tracks the grid actually renders
 * is the number that fits in `--grid-width` — which is the column
 * count derived from the shell's content width, capped at the
 * preset's maxCols.
 *
 * So the fetch layer measures the SAME box the layout measures:
 * `.shell`'s content box. Reading it directly (subtracting padding
 * and border from the bounding rect) reproduces the ResizeObserver's
 * contentRect.width exactly, and it works before the grid has
 * mounted — which matters, because the first request is issued
 * before any tile exists.
 *
 * ## Whole rows
 *
 * A page has to contain a whole number of rows — when it doesn't,
 * the grid ends on a half-empty row. The configured photo count
 * (grid/config.ts) is therefore rounded UP to the next multiple of
 * the column count the viewport actually renders, so
 * `limit % cols === 0` always holds and a page never carries fewer
 * photos than the grid size asks for.
 */
import {
  GRID_SIZE_CONFIG,
  basePageSize,
  type GridSize,
  type GridSizeConfig
} from '$lib/grid/config';

/**
 * The element the photo grid is laid out inside. The layout's
 * ResizeObserver watches the same box and the grid wrapper is
 * sized from it, so this is the width the column count is derived
 * from — on every route.
 */
export const GRID_CONTAINER_SELECTOR = '.shell';

/** Columns the CSS grid renders inside `width` px of container. */
export function columnsForWidth(width: number, config: GridSizeConfig): number {
  return Math.max(
    1,
    Math.min(
      config.maxCols,
      Math.floor((width + config.gap) / (config.tile + config.gap))
    )
  );
}

/** Pixel width of a row of `cols` columns. */
export function widthForColumns(cols: number, config: GridSizeConfig): number {
  return cols * config.tile + (cols - 1) * config.gap;
}

/**
 * Smallest multiple of `cols` that is at least `count`. Rounding
 * up (never down) keeps the promise that a page never carries
 * fewer photos than the grid size asks for — the extra photos fill
 * the last row instead of leaving it ragged.
 */
export function pageSizeForColumns(count: number, cols: number): number {
  return Math.ceil(count / cols) * cols;
}

/**
 * Largest multiple of `cols` that does not exceed `ceiling`.
 * For endpoints with a hard ceiling (the Similar page's "up to N
 * neighbours"), the count has to fit under it and still land on a
 * whole number of rows.
 */
export function pageSizeWithinColumns(ceiling: number, cols: number): number {
  const rows = Math.floor(ceiling / cols);
  return rows > 0 ? rows * cols : cols;
}

/**
 * Content width of the grid's container, in CSS pixels — the box
 * the layout measures via ResizeObserver and the grid renders
 * into. Measuring the same box the renderer uses (rather than
 * recomputing the layout from the viewport) keeps the fetch layer
 * and the grid on the same basis, so the number of columns a
 * request is sized for is the number of columns that render.
 *
 * Returns 0 outside the browser (SSR), where callers fall back to
 * the configured base count.
 */
export function gridContainerWidth(): number {
  if (typeof document === 'undefined') return 0;
  const el = document.querySelector<HTMLElement>(GRID_CONTAINER_SELECTOR);
  if (!el) return 0;
  const rect = el.getBoundingClientRect();
  const cs = getComputedStyle(el);
  const inset =
    px(cs.paddingLeft) +
    px(cs.paddingRight) +
    px(cs.borderLeftWidth) +
    px(cs.borderRightWidth);
  return Math.max(0, rect.width - inset);
}

function px(value: string): number {
  const parsed = parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

/**
 * Columns the grid renders right now for the given grid size.
 * Falls back to the preset's maximum when the container can't be
 * measured (SSR) — the design width, where the base page size is
 * already a whole number of rows.
 */
export function currentColumns(size: GridSize): number {
  const config = GRID_SIZE_CONFIG[size];
  const width = gridContainerWidth();
  if (width <= 0) return config.maxCols;
  return columnsForWidth(width, config);
}

/**
 * Photos to request for one page: the configured page count
 * rounded up to a whole number of grid rows at the current column
 * count. Measured per request, so it follows the viewport and the
 * grid-size preference.
 */
export function currentPageSize(size: GridSize): number {
  const config = GRID_SIZE_CONFIG[size];
  const base = basePageSize(config);
  const width = gridContainerWidth();
  if (width <= 0) return base;
  return pageSizeForColumns(base, columnsForWidth(width, config));
}

/**
 * Photos to request when the endpoint has a hard ceiling (the
 * Similar page's "up to N neighbours"): the largest whole number
 * of rows that still fits under the ceiling.
 */
export function currentPageSizeWithin(size: GridSize, ceiling: number): number {
  const config = GRID_SIZE_CONFIG[size];
  const width = gridContainerWidth();
  const cols = width <= 0 ? config.maxCols : columnsForWidth(width, config);
  return pageSizeWithinColumns(ceiling, cols);
}
