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
 * The speeds the drift's SPEED control offers, in px/s.
 *
 * A linear scale cannot span this range usably. The speeds worth
 * having are the slow ones — a wall you can watch a photo in — and
 * they all live at the bottom: on a linear 4..2048 slider, the
 * entire range from "a breath" to "a brisk scroll" would sit in the
 * first 3% of the travel, and the other 97% would be unreachable
 * dead space past the top of the useful band.
 *
 * So the control is a LADDER: each rung is a fixed RATIO above the
 * one below, so every notch feels like the same amount of "faster"
 * wherever on the scale it is. These are the camera's shutter
 * speeds, which solve exactly this problem, and the ratio alternates
 * 1.5x, 1.33x (a third of a stop) so the ladder lands on the
 * numbers people already have in their heads: 6, 8, 12, 24, 48, 96,
 * 192, 384, 768, 1536.
 *
 * The floor is not zero: a wall that cannot move is what the PAUSE
 * control is for, and a speed of zero would leave the wall parked
 * while the pause button still said "Pause" — two controls, one
 * state, and no way to tell which one the user had actually set.
 *
 * The ceiling is where the wall stops being watchable and becomes a
 * flick: at 2048px/s a photo crosses a 900px screen in under half a
 * second. It is deliberately far past the useful band — the ladder
 * makes the fast rungs cheap to reach, and it is better to have a
 * top the user outgrows than one they hit.
 */
export const DRIFT_SPEED_LADDER = [
  4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
  1536, 2048
] as const;

/**
 * How fast the wall drifts upward, in CSS pixels per second, when
 * nothing has been chosen.
 *
 * Slow on purpose. The wall is a thing you look at from across the
 * room, not a thing you scroll: at 16px/s a photo crosses the whole
 * screen in about a minute, which reads as the wall breathing rather
 * than moving. It is the middle of the useful band — slow enough to
 * watch a photo arrive, fast enough that the wall never looks
 * broken.
 */
export const DRIFT_SPEED_PX_PER_SEC = 16;

/** The slowest speed the wall can drift at. */
export const DRIFT_SPEED_MIN_PX_PER_SEC = DRIFT_SPEED_LADDER[0];

/** The fastest speed the wall can drift at. */
export const DRIFT_SPEED_MAX_PX_PER_SEC =
  DRIFT_SPEED_LADDER[DRIFT_SPEED_LADDER.length - 1];

/**
 * How many rungs the speed control has.
 *
 * The slider's own unit is the RUNG, not the px/s: a slider whose
 * value is the raw speed would have to be linear, and linear is
 * exactly what the ladder exists to avoid. So the slider counts
 * rungs and the component converts.
 */
export const DRIFT_SPEED_STEP_COUNT = DRIFT_SPEED_LADDER.length;

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
 * The rung of the speed ladder a speed falls on.
 *
 * The ladder is discrete, so a speed that is not exactly on it has to
 * be resolved to the rung it belongs to. The NEAREST rung wins, and
 * a tie goes to the SLOWER one — dragging toward "faster" should not
 * jump two rungs at the boundary, and rounding down is the
 * conservative direction.
 *
 * A value that is not a number falls back to the default speed.
 */
export function driftSpeedRung(value: number): number {
  if (Number.isNaN(value)) return driftSpeedRung(DRIFT_SPEED_PX_PER_SEC);
  // An infinity is out of range, not equidistant from every rung:
  // every distance would be infinite, and the first rung would win.
  if (value <= DRIFT_SPEED_LADDER[0]) return 0;
  if (value >= DRIFT_SPEED_LADDER[DRIFT_SPEED_LADDER.length - 1]) {
    return DRIFT_SPEED_LADDER.length - 1;
  }
  let best = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < DRIFT_SPEED_LADDER.length; i++) {
    const distance = Math.abs(DRIFT_SPEED_LADDER[i] - value);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = i;
    }
  }
  return best;
}

/**
 * The speed a rung of the ladder means, in px/s.
 *
 * A rung outside the ladder clamps to its ends, so a slider whose
 * thumb overshoots cannot produce a speed the wall cannot run at.
 */
export function driftSpeedFromRung(rung: number): number {
  if (Number.isNaN(rung)) return DRIFT_SPEED_PX_PER_SEC;
  const index = Math.min(
    DRIFT_SPEED_LADDER.length - 1,
    Math.max(0, Math.round(rung))
  );
  return DRIFT_SPEED_LADDER[index];
}

/**
 * The speed control's value, resolved onto the ladder.
 *
 * The slider is the only way in, but the value also arrives from the
 * browser (a restored form state, a dragged thumb that overshoots
 * the end of the track), so the range is enforced in one place.
 *
 * The result is always a speed the ladder actually offers — there is
 * no in-between value, because there is no way to ask for one.
 */
export function clampDriftSpeed(value: number): number {
  if (Number.isNaN(value)) return DRIFT_SPEED_PX_PER_SEC;
  return driftSpeedFromRung(driftSpeedRung(value));
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
 * Rows rendered beyond the viewport, above and below it, at the
 * wall's default speed.
 *
 * The wall moves, so a row has to exist before it scrolls into view
 * — otherwise its photos pop in at the bottom edge as a half-loaded
 * row. Two rows of lead time is roughly half a minute of drift, which
 * is far longer than a thumbnail takes to arrive.
 */
export const DRIFT_OVERSCAN_ROWS = 2;

/**
 * How much drift a photo must be able to survive before it is shown.
 *
 * The overscan has to cover the time it takes a photo to ARRIVE, and
 * a fixed number of rows only does that at the default speed. At the
 * ladder's top rung — 2048px/s — two rows of lead is a quarter of a
 * second, and a photo that has not arrived by then is a blank tile
 * flickering past at 30 frames a second.
 *
 * So the lead is measured in SECONDS and converted into rows: the
 * wall always keeps the same amount of drift in reserve, whatever
 * speed it is running at. Two seconds is comfortably longer than a
 * thumbnail takes to arrive over the network.
 */
export const DRIFT_LEAD_SECONDS = 2;

/**
 * The rows of lead the wall needs at a given speed.
 *
 * The row count is the time the lead represents divided by the time
 * one row takes to pass, which is the row height over the speed. A
 * slow wall needs a single row to cover its lead; a fast one needs
 * many, and gets them.
 *
 * The floor is `DRIFT_OVERSCAN_ROWS`, so the wall is never rendered
 * with less lead than it has always had.
 */
export function driftOverscanRows(
  speedPxPerSec: number,
  rowHeight: number
): number {
  if (rowHeight <= 0 || !Number.isFinite(speedPxPerSec)) {
    return DRIFT_OVERSCAN_ROWS;
  }
  const speed = Math.max(0, speedPxPerSec);
  const rowsForLead = Math.ceil((speed * DRIFT_LEAD_SECONDS) / rowHeight);
  return Math.max(DRIFT_OVERSCAN_ROWS, rowsForLead);
}

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
