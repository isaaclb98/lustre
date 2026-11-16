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
 * The bounds of the drift's SPEED control.
 *
 * The floor is not zero: a wall that cannot move is what the PAUSE
 * control is for, and a speed of zero would leave the wall parked
 * while the pause button still said "Pause" — two controls, one
 * state, and no way to tell which one the user had actually set.
 *
 * The ceiling is fast enough to read as a scroll rather than a
 * breath, and slow enough that a photo is on screen for a while
 * before it leaves. At 64px/s a 900px-tall screen takes 14 seconds
 * to cross.
 */
export const DRIFT_SPEED_MIN_PX_PER_SEC = 4;
export const DRIFT_SPEED_MAX_PX_PER_SEC = 64;

/** The step the speed control moves by, in px/s. */
export const DRIFT_SPEED_STEP_PX_PER_SEC = 4;

/**
 * The widths a photo may be fetched at.
 *
 * The app's thumbnail is 384px, which is the right size for a tile
 * in a grid. A surface the user can ZOOM has to fetch something
 * bigger when the tile outgrows that — and the bigger asset is
 * produced by the server, resized from the original and cached on
 * disk.
 *
 * The steps are quantised for a reason: an unquantised width would
 * make every column count ask for a slightly different size, and
 * every one of those is a separate cached copy of the same photo.
 * A handful of steps keeps the number of variants bounded while
 * staying close enough to the real tile size that the photo looks
 * sharp.
 */
export const DRIFT_PHOTO_WIDTHS = [384, 768, 1280, 1920] as const;

/**
 * The width to fetch a photo at for a tile `tileWidth` px wide.
 *
 * The smallest step that still covers the tile — so the photo is
 * never scaled UP by the browser, only down. Below the thumbnail's
 * own size the answer is the thumbnail.
 */
export function quantizePhotoWidth(tileWidth: number): number {
  if (!Number.isFinite(tileWidth) || tileWidth <= 0) {
    return DRIFT_PHOTO_WIDTHS[0];
  }
  for (const step of DRIFT_PHOTO_WIDTHS) {
    if (tileWidth <= step) return step;
  }
  // Wider than the biggest step: ask for the biggest one. The
  // server never upscales, so the photo comes back at its own size
  // and the browser softens it — a soft photo, never a broken one.
  return DRIFT_PHOTO_WIDTHS[DRIFT_PHOTO_WIDTHS.length - 1];
}

/**
 * The bounds of the drift's ZOOM control — how many photos sit
 * across the wall.
 *
 * One column is the closest the wall can come to a single photo at
 * a time; the upper bound is a wall of thumbnails rather than
 * photos, which is a different surface and not a zoom.
 */
export const DRIFT_MIN_COLUMNS = 1;
export const DRIFT_MAX_COLUMNS = 24;

/**
 * The smallest a tile may be, in CSS pixels, when the user is
 * choosing the column count themselves.
 *
 * Below this the wall stops being a wall: the photos become
 * postage stamps and the gapless field turns into noise. The
 * thumbnails are 384px, so this is already generous.
 */
export const DRIFT_MIN_TILE_PX = 120;

/**
 * The speed control's value, clamped into range.
 *
 * The slider is the only way in, but the value also arrives from
 * the browser (a restored form state, a dragged thumb that overshoots
 * the end of the track), so the range is enforced in one place.
 *
 * NaN — the value that is not a number at all — falls back to the
 * default. Infinity and -Infinity are ordinary out-of-range values
 * and clamp to the ends like any other.
 */
export function clampDriftSpeed(value: number): number {
  if (Number.isNaN(value)) return DRIFT_SPEED_PX_PER_SEC;
  return Math.min(
    DRIFT_SPEED_MAX_PX_PER_SEC,
    Math.max(DRIFT_SPEED_MIN_PX_PER_SEC, Math.round(value))
  );
}

/**
 * The column count the zoom control may take, clamped into range.
 *
 * `max` is the ceiling the surface itself allows — it depends on
 * the viewport, so it is computed by the caller and passed in.
 */
export function clampDriftColumns(value: number, max: number): number {
  if (Number.isNaN(value)) return DRIFT_MIN_COLUMNS;
  const hi = Math.max(DRIFT_MIN_COLUMNS, Math.floor(max));
  return Math.min(hi, Math.max(DRIFT_MIN_COLUMNS, Math.round(value)));
}

/**
 * The most columns the wall may show at `surfaceWidth`.
 *
 * Two ceilings apply, and the LARGER one wins: the surface's own
 * limit (a tile must stay at least `DRIFT_MIN_TILE_PX` wide) and the
 * absolute ceiling. The auto column count is folded in as well, so
 * the wall's own default is always reachable on the slider — a
 * control that cannot return to where it started is a control that
 * traps the user.
 */
export function driftMaxColumns(
  surfaceWidth: number,
  autoColumns: number
): number {
  const byTile =
    surfaceWidth > 0
      ? Math.floor(surfaceWidth / DRIFT_MIN_TILE_PX)
      : DRIFT_MAX_COLUMNS;
  return Math.max(
    DRIFT_MIN_COLUMNS,
    autoColumns,
    Math.min(DRIFT_MAX_COLUMNS, byTile)
  );
}

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
