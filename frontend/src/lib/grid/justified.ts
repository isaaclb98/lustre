/**
 * Justified rows — the layout Immich uses.
 *
 * Immich has no masonry. `masonry` appears nowhere in its codebase
 * (web, server, or mobile): what it renders is the FLICKR /
 * GOOGLE-PHOTOS "justified" layout, where every ROW has one height
 * and the photos in that row are scaled to fill the row's width
 * exactly.
 *
 * The difference from Pinterest masonry is the axis:
 *
 *   masonry  — the tile WIDTH is fixed, the height varies per photo
 *   justified — the tile HEIGHT is fixed per row, the width varies
 *
 * So a justified row holds however many photos fit its width at that
 * height: a landscape photo takes a wide slot, a portrait a narrow
 * one, and the row still ends flush at both edges. Every row is
 * exactly the container's width. That is why Immich's grid has no
 * ragged bottom edge and no ragged right edge — the raggedness
 * lives in the row HEIGHTS, which vary by the tolerance.
 *
 * ## The algorithm
 *
 * A port of `immich-app/justified-layout` (the Rust/WASM build
 * Immich ships by default). The Rust's core, in its own terms:
 *
 *   1. Walk the photos, accumulating their aspect ratios into a row.
 *      The accumulator carries the spacing as well, so it stays
 *      comparable with the target row aspect.
 *   2. Close the row when the accumulated ratio overshoots the
 *      target — i.e. when the photo would take the row FURTHER from
 *      the target than the row without it. A row that grows past
 *      the tolerance's ceiling closes too.
 *   3. A closed row's height is (rowWidth - spacing) / sum(ratios),
 *      so the row's photos exactly fill the width.
 *   4. Each photo's width is its own ratio times that height, so its
 *      aspect ratio is preserved EXACTLY. Nothing is ever cropped.
 *   5. The LAST row takes the previous row's height when its own
 *      would exceed the ceiling — the widow case, where the final
 *      row is short and would otherwise collapse to a sliver.
 *
 * The port is verified against Immich's own WASM build on 400 real
 * photos — see justified.test.ts.
 */

/** What the layout needs to know about a photo: its dimensions. */
export interface JustifiedSource {
  width?: number | null;
  height?: number | null;
}

/** A tile's placement in the justified layout. */
export interface JustifiedPlacement {
  /** Left edge, in px, from the grid's left. */
  x: number;
  /** Top edge, in px, from the grid's top. */
  y: number;
  /** Tile width, in px. */
  width: number;
  /** Tile height, in px. */
  height: number;
}

/**
 * One row of a justified layout.
 *
 * The rows are what the renderer works with: the layout's tiles are
 * grouped into rows of ONE height, and the rows are ordered by their
 * top edge. A renderer that virtualizes by row can therefore find
 * the visible rows with a binary search — the placements alone are
 * not enough, because a row's photos are interleaved with the
 * previous row's in the placement list.
 */
export interface JustifiedRow {
  /** The row's top edge, in px, from the grid's top. */
  top: number;
  /** The row's height, in px. Every photo in the row is this tall. */
  height: number;
  /**
   * The row's own width, in px — the sum of its photos' widths.
   *
   * A row that FILLS the grid's width is exactly that wide. The
   * last row usually is not: the widow rule gives it the previous
   * row's height, and at that height its photos cover less than
   * the full width. Immich leaves that row short and left-aligned
   * rather than stretching it, and so does this.
   */
  width: number;
  /** The index of the row's first photo, in the source array. */
  start: number;
  /** How many photos the row holds. */
  count: number;
}

/** The numbers a justified layout is computed from. */
export interface JustifiedOptions {
  /** The width every row fills, in px. */
  rowWidth: number;
  /** The target height of a row, in px. */
  rowHeight: number;
  /** The horizontal gap between photos in a row, in px. */
  spacing?: number;
  /**
   * How far a row's height may stray from the target, as a
   * fraction of it. Immich uses 0.5.
   */
  heightTolerance?: number;
}

/** Immich's own height tolerance (`heightTolerance: 0.5`). */
export const DEFAULT_HEIGHT_TOLERANCE = 0.5;

/**
 * The aspect ratio (width / height) of a photo.
 *
 * A photo with no known dimensions falls back to a square, which is
 * what the uniform grid showed anyway.
 */
export function aspectFor(source: JustifiedSource | null | undefined): number {
  const w = source?.width;
  const h = source?.height;
  if (!w || !h || w <= 0 || h <= 0) return 1;
  return w / h;
}

/**
 * Lay photos out in justified rows.
 *
 * Returns the placement of every photo and the total height of the
 * layout. The placements are in the grid's own coordinate space:
 * (0, 0) is the grid's top-left corner.
 */
export function layoutJustified(
  sources: readonly (JustifiedSource | null | undefined)[],
  options: JustifiedOptions
): {
  placements: JustifiedPlacement[];
  rows: JustifiedRow[];
  height: number;
  width: number;
} {
  const rowWidth = Math.max(1, options.rowWidth);
  const rowHeight = Math.max(1, options.rowHeight);
  const spacing = Math.max(0, options.spacing ?? 0);
  const tolerance = options.heightTolerance ?? DEFAULT_HEIGHT_TOLERANCE;

  const ratios = sources.map((source) => aspectFor(source));
  if (ratios.length === 0) {
    return { placements: [], rows: [], height: 0, width: rowWidth };
  }

  // The aspect ratio a full row of photos would have.
  const targetRowAspect = rowWidth / rowHeight;
  // The row height's floor and ceiling. A row whose photos are so
  // tall that it would exceed the ceiling is clamped to it, and one
  // whose photos are so short that it would fall below the floor
  // closes early.
  const minRowHeight = Math.max(rowHeight * (1 - tolerance), 0);
  const maxRowHeight = rowHeight * (1 + tolerance);
  // The widest a row may grow before it closes on that account
  // alone.
  const maxRowAspect =
    minRowHeight > 0 ? rowWidth / minRowHeight : Number.POSITIVE_INFINITY;
  // The spacing, expressed in the same units as an aspect ratio.
  const spacingAspect = spacing / rowHeight;

  const placements: JustifiedPlacement[] = [];
  // The rows, in order. Each records where its photos sit in the
  // source array and how tall it is, which is what the renderer
  // needs to slice the layout by row.
  const rows: JustifiedRow[] = [];

  // The row being built, and where it starts.
  let rowStart = 0;
  let rowTop = 0;
  // The accumulated aspect ratio of the photos seen since the row
  // started, WITH the spacing between them — the value compared
  // against the target.
  let cumulative = 0;
  // The same accumulator, carried across to the next iteration. It
  // is what a closed row's height is derived from.
  let rowAspect = 0;
  // How far the row is from the target — the closest it has been.
  // A photo has to beat this to close the row.
  let bestDiff = targetRowAspect;
  // The widest row, which is the layout's width.
  let maxWidth = 0;

  /** Write the boxes of the row [start, end) at the given height. */
  const writeRow = (start: number, end: number, height: number) => {
    let x = 0;
    for (let i = start; i < end; i++) {
      const width = ratios[i] * height;
      placements.push({ x, y: rowTop, width, height });
      x += width + spacing;
    }
    // The row's width is its photos' widths plus the gaps between
    // them (n-1 gaps for n photos).
    let rowPixels = spacing * Math.max(0, end - start - 1);
    for (let i = start; i < end; i++) rowPixels += ratios[i] * height;
    if (rowPixels > maxWidth) maxWidth = rowPixels;
    rows.push({ top: rowTop, height, width: rowPixels, start, count: end - start });
  };

  /**
   * The height a row of photos takes so that it exactly fills the
   * width, clamped to the ceiling.
   *
   * The row's photos are scaled to a common height, so the sum of
   * their widths (plus the gaps between them) is the row's width:
   *
   *   height = (rowWidth - spacing) / sum(aspect ratios)
   *
   * The accumulator carries the spacing as an aspect ratio, so it
   * is removed here to leave the photos' own ratios.
   */
  const rowHeightFor = (count: number) => {
    const totalAspect = rowAspect - spacingAspect * count;
    if (totalAspect <= 0) return 0;
    const gaps = Math.max(0, count - 1);
    const base = (rowWidth - spacing * gaps) / totalAspect;
    return Math.min(base, maxRowHeight);
  };

  for (let i = 0; i < ratios.length; i++) {
    cumulative += ratios[i];

    // Does this row have to close before the photo?
    const tooWide = cumulative > maxRowAspect;
    const furtherFromTarget = Math.abs(cumulative - targetRowAspect) > bestDiff;
    if ((tooWide || furtherFromTarget) && i > 0) {
      // Close the row WITHOUT this photo, then start a new one
      // with it.
      writeRow(rowStart, i, rowHeightFor(i - rowStart));
      rowTop += rows[rows.length - 1].height + spacing;

      rowStart = i;
      cumulative = ratios[i];
    }

    bestDiff = Math.abs(cumulative - targetRowAspect);
    cumulative += spacingAspect;
    rowAspect = cumulative;
  }

  // The last row. It takes the PREVIOUS row's height when its own
  // would exceed the ceiling — the widow case, where the final row
  // is short and would otherwise collapse to a sliver.
  const count = ratios.length - rowStart;
  const totalAspect = rowAspect - spacingAspect * count;
  const gaps = Math.max(0, count - 1);
  const ownHeight =
    totalAspect > 0 ? (rowWidth - spacing * gaps) / totalAspect : rowHeight;
  const lastHeight =
    ownHeight > maxRowHeight ? previousRowHeight(placements, rowHeight) : ownHeight;
  writeRow(rowStart, ratios.length, lastHeight);

  // The layout's height is every row's height plus the gaps between
  // them — no trailing gap after the last row.
  const height = rowTop + lastHeight;
  return { placements, rows, height, width: maxWidth };
}

/** The height of the last closed row, for the widow row to match. */
function previousRowHeight(
  placements: readonly JustifiedPlacement[],
  fallback: number
): number {
  for (let i = placements.length - 1; i >= 0; i--) {
    if (placements[i].height > 0) return placements[i].height;
  }
  return fallback;
}
