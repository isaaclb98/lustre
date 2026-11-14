/**
 * Drift — the arithmetic of a moving presentation wall.
 *
 * The drift is the album's wall scrolling itself upward, forever.
 * Three numbers decide how it looks, and all three live here so the
 * component that draws it and the tests that pin it cannot disagree:
 *
 *   - how fast the wall moves (DRIFT_SPEED_PX_PER_SEC),
 *   - which rows are on screen at a given moment (driftWindowRows),
 *   - which photos a row holds (driftRowItems).
 *
 * ## Why the photos are a RING, not a list
 *
 * The wall has to loop without a seam. A plain list of the album's
 * photos cannot loop: the last row would be a half-empty row (the
 * album does not divide into whole rows) and the photos would stop.
 *
 * So the wall shows a RING — the album's photos end-to-start, forever.
 * Row r holds ring[r*cols .. r*cols+cols-1], and the ring wraps at the
 * album's length. Every row is therefore full, and the only place the
 * album's end meets its start is inside one row, where the photos are
 * already in a random order and the meeting is invisible.
 *
 * The cost is that the ring repeats every album-length of photos,
 * which at a drift of a few pixels a second is minutes of viewing.
 *
 * ## The wall's coordinates
 *
 * The wall is an infinite strip of rows. Row `i` occupies the wall
 * coordinate `i * rowHeight`, for every integer `i` — the strip has no
 * top and no bottom. `topOffsetPx` says how far into that strip the
 * viewport's top edge sits, and it grows for as long as the wall
 * drifts. A row's position ON SCREEN is therefore
 * `i * rowHeight - topOffsetPx`.
 *
 * The row INDEX is absolute and keeps counting as the wall drifts.
 * That is what lets the ring wrap without a row's contents ever
 * changing: row 12 always holds the same photos, wherever it is.
 */

/**
 * How fast the wall drifts upward, in CSS pixels per second.
 *
 * Slow on purpose. The wall is a thing you look at from across the
 * room, not a thing you scroll: at 16px/s a photo crosses the whole
 * screen in about a minute, which reads as the wall breathing rather
 * than moving.
 */
export const DRIFT_SPEED_PX_PER_SEC = 16;

/**
 * Rows rendered beyond the viewport, above and below it.
 *
 * The wall moves, so a row has to exist before it scrolls into view
 * — otherwise its photos pop in at the bottom edge as a half-loaded
 * row. Two rows of lead time is roughly half a minute of drift, which
 * is far longer than a thumbnail takes to arrive.
 */
export const DRIFT_OVERSCAN_ROWS = 2;

/** The rows of the wall that are on screen at a given moment. */
export interface DriftWindow {
  /** The first rendered row, counting from the wall's origin. */
  first: number;
  /** One past the last rendered row — the range is half-open. */
  last: number;
}

/**
 * The rows of the wall that are on screen at a given scroll offset.
 *
 * `topOffsetPx` is where the viewport's top edge sits in the wall's
 * own coordinates; `rowHeight` is the height of one row; `viewportPx`
 * is how much of the wall is visible.
 *
 * The returned range is half-open, `[first, last)`, and the indices
 * are absolute: they keep counting as the wall drifts, which is what
 * lets the ring wrap without the rows' contents ever changing.
 *
 * The range covers the viewport plus `overscanRows` rows of lead on
 * each side, so a row is rendered before it scrolls into view and
 * lingers after it scrolls out.
 */
export function driftWindowRows(
  topOffsetPx: number,
  rowHeight: number,
  viewportPx: number,
  overscanRows: number = DRIFT_OVERSCAN_ROWS
): DriftWindow {
  if (rowHeight <= 0) return { first: 0, last: 0 };
  const first = Math.floor(topOffsetPx / rowHeight) - overscanRows;
  const visible = Math.ceil(viewportPx / rowHeight) + 1;
  return { first, last: first + visible + overscanRows * 2 };
}

/**
 * The photos in one row of the drifting wall.
 *
 * `rowIndex` is the row's absolute index in the wall. The photos come
 * from the album's list as a RING: the row takes the `columns` photos
 * that follow the previous row's, wrapping at the end of the album.
 *
 * Wrapping mid-row is what makes every row full. It also means the
 * album's end meets its start inside a row — invisible, because the
 * photos are in a random order and nothing about a row says where the
 * album's boundary was.
 */
export function driftRowItems<T>(
  items: readonly T[],
  columns: number,
  rowIndex: number
): T[] {
  const n = items.length;
  if (n === 0 || columns <= 0) return [];
  const start = (((rowIndex * columns) % n) + n) % n;
  const row: T[] = [];
  for (let c = 0; c < columns; c++) {
    row.push(items[(start + c) % n]);
  }
  return row;
}
