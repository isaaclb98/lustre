/**
 * grid/skeleton.ts — the shape of the loading placeholder band.
 *
 * The skeleton is the band of placeholders the grid shows while the
 * NEXT page of photos is on the wire. It has to have the same shape
 * as the layout it stands in for, or the grid visibly changes shape
 * when the real photos arrive.
 *
 * It used not to. The skeleton was built from the UNIFORM grid's
 * CSS grid — square tiles in `repeat(auto-fit, --grid-tile-size)`
 * tracks — and reused for every layout. Under the justified layout
 * that is wrong twice over:
 *
 *   - `--grid-tile-size` is the justified layout's ROW HEIGHT, not
 *     a tile width, so the placeholder tracks are square blocks the
 *     size of a row rather than the flush, variable-width tiles a
 *     justified row actually holds;
 *   - the CSS fits fewer tracks than the skeleton renders tiles
 *     into, so the extra tiles WRAP onto a second line and the row
 *     doubles in height with a ragged edge.
 *
 * Measured on the dev stack at 1440x900, justified layout: the
 * skeleton rendered 4 tiles into 3 tracks, producing 720px rows of
 * 360x360 squares where the real grid above had 381px rows of
 * variable-width tiles flush to the grid's width. The band was
 * 2164px tall — twice the height of the three rows it stood in for
 * — and looked like a different grid entirely.
 *
 * So the placeholders are laid out by the SAME engines the real
 * photos are: `layoutJustified` and `layoutMasonry`. One shape
 * comes out — a list of absolutely positioned boxes — and the grid
 * renders them with the same markup it uses for the real tiles.
 *
 * The uniform grid keeps its CSS-grid skeleton: there the CSS
 * tracks and the JS column count agree by construction (see
 * grid/geometry.ts), so its placeholders already match its tiles.
 */
import { layoutJustified } from './justified';
import { layoutMasonry } from './masonry';

export interface SkeletonBox {
  /** Left edge, in px, from the grid's left. */
  x: number;
  /** Top edge, in px, from the grid's top. */
  y: number;
  /** Box width, in px. */
  width: number;
  /** Box height, in px. */
  height: number;
}

export interface SkeletonLayout {
  /** The placeholder boxes, in the grid's own coordinate space. */
  boxes: SkeletonBox[];
  /** How tall the band is, in px. */
  height: number;
}

/**
 * The aspect ratios the placeholder photos are drawn at.
 *
 * The skeleton stands in for photos whose real ratios are not known
 * yet, so it uses a plausible MIX. A single repeated ratio would
 * give the justified layout rows of identical tiles — the one shape
 * a real justified row never has, since its photos fill the row's
 * width exactly.
 */
export const PLACEHOLDER_ASPECTS: readonly number[] = [
  0.75, 1.3333, 1, 0.6667, 1.5, 0.8
];

/** Placeholder photos with the ratios above, cycling. */
export function placeholderSources(
  count: number
): { width: number; height: number }[] {
  const out: { width: number; height: number }[] = [];
  for (let i = 0; i < count; i++) {
    const ratio = PLACEHOLDER_ASPECTS[i % PLACEHOLDER_ASPECTS.length];
    out.push({ width: Math.round(ratio * 100), height: 100 });
  }
  return out;
}

/**
 * Placeholder boxes for the JUSTIFIED layout.
 *
 * The same engine, the same options and the same row width the real
 * rows use, so the placeholders sit exactly where the photos will:
 * flush to the grid's width, one height per row.
 *
 * The rows shown are the FIRST `rows` rows of a longer layout, so
 * none of them is the layout's last row — which the widow rule
 * would leave short. A skeleton whose last row is short would show
 * a ragged edge the real grid does not have.
 */
export function skeletonJustified(opts: {
  rowWidth: number;
  rowHeight: number;
  rows: number;
}): SkeletonLayout {
  const sources = placeholderSources(opts.rows * 6 + 6);
  const layout = layoutJustified(sources, {
    rowWidth: opts.rowWidth,
    rowHeight: opts.rowHeight
  });
  const boxes: SkeletonBox[] = [];
  let height = 0;
  for (const row of layout.rows.slice(0, opts.rows)) {
    for (let i = row.start; i < row.start + row.count; i++) {
      const placement = layout.placements[i];
      if (!placement) continue;
      boxes.push({
        x: placement.x,
        y: row.top,
        width: placement.width,
        height: row.height
      });
    }
    height = Math.max(height, row.top + row.height);
  }
  return { boxes, height };
}

/**
 * Placeholder boxes for the MASONRY layout.
 *
 * The masonry fills column by column in source order, so the first
 * `columns * rows` placements are the top `rows` bands of the
 * layout. Taking them in SOURCE order (not by height) is what keeps
 * the band coherent: sorting by height would mix a tall column's
 * second photo into a band it does not belong to.
 */
export function skeletonMasonry(opts: {
  tile: number;
  columns: number;
  rows: number;
}): SkeletonLayout {
  const count = Math.max(1, opts.columns * opts.rows);
  const layout = layoutMasonry(placeholderSources(count), {
    tile: opts.tile,
    columns: opts.columns
  });
  const boxes: SkeletonBox[] = [];
  let height = 0;
  for (const placement of layout.placements.slice(0, count)) {
    boxes.push({
      x: placement.x,
      y: placement.y,
      width: placement.width,
      height: placement.height
    });
    height = Math.max(height, placement.y + placement.height);
  }
  return { boxes, height };
}
