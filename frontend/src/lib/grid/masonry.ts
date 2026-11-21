/**
 * Masonry layout — Pinterest-style packing.
 *
 * The uniform grid gives every tile the same height, which means a
 * portrait photo is cropped to a square. Masonry keeps each photo's
 * own aspect ratio and packs the tiles into the shortest column, so
 * the columns stay level and no photo is cut.
 *
 * The layout is computed in JS rather than left to CSS on purpose:
 *
 *  - The tile HEIGHTS have to be known before the image loads, or the
 *    grid would reflow every time a thumbnail arrives. The photo's
 *    dimensions come from the API (the indexer stored the source's
 *    width and height in the payload), so the height is known up
 *    front and the tile can be sized in the same paint as the layout.
 *  - The uniform grid's virtualizer slices items into ROWS of a fixed
 *    height, which a variable-height layout cannot be expressed in.
 *    Masonry computes the positions of every item itself, so the
 *    positions ARE the index: the visible slice is a filter over
 *    them.
 *
 * The packing rule is the shortest column, not a round-robin: a tall
 * portrait in one column must not push the next photo down a column
 * that has plenty of room. That is what makes the columns end level.
 */
/**
 * What the layout needs to know about a photo: its dimensions.
 *
 * Structural, so the API's result shape satisfies it without a cast
 * — the search and album endpoints both return `width` and `height`
 * from the indexer's payload.
 */
export interface MasonrySource {
  width?: number | null;
  height?: number | null;
}

/** A tile's placement in the masonry layout. */
export interface MasonryPlacement {
  /** Left edge, in px, from the grid's left. */
  x: number;
  /** Top edge, in px, from the grid's top. */
  y: number;
  /** Tile width, in px. */
  width: number;
  /** Tile height, in px. */
  height: number;
}

/** The numbers a masonry layout is computed from. */
export interface MasonryOptions {
  /** The width every tile is rendered at, in px. */
  tile: number;
  /** How many columns the grid has. */
  columns: number;
  /** The narrowest a tile may get, as a height/width ratio. */
  minAspect?: number;
  /** The tallest a tile may get, as a height/width ratio. */
  maxAspect?: number;
}

/**
 * The default clamps.
 *
 * A panorama (a 5:1 banner) or an extreme portrait would make a
 * single tile absurdly long, and the column it lands in would tower
 * over the rest. The clamp bounds that.
 *
 * The bound is deliberately WIDE, because a clamp that bites CROPS.
 * The tile's box is drawn at the clamped ratio while the photo inside
 * is `object-fit: cover` — so a box whose ratio is not the photo's
 * ratio cuts the photo off. The masonry's whole promise is that no
 * photo is cropped.
 *
 * These numbers are measured, not guessed: across 11,485 photos of
 * the real library the height/width ratio runs from 0.325 (a wide
 * landscape) to 2.996 (a tall portrait), and this bound crops none
 * of them. The previous [0.5, 2] bound cropped 35.
 */
export const DEFAULT_MIN_ASPECT = 0.25;
export const DEFAULT_MAX_ASPECT = 4;

/**
 * The height/width ratio a photo is laid out at.
 *
 * A photo with no known dimensions (the indexer did not record them,
 * or the payload is legacy) falls back to a square, which is what
 * the uniform grid showed anyway.
 */
export function aspectFor(
  source: MasonrySource | null | undefined,
  minAspect: number = DEFAULT_MIN_ASPECT,
  maxAspect: number = DEFAULT_MAX_ASPECT
): number {
  const w = source?.width;
  const h = source?.height;
  if (!w || !h || w <= 0 || h <= 0) return 1;
  return Math.min(maxAspect, Math.max(minAspect, h / w));
}

/**
 * Whether a photo's ratio falls outside the layout's bounds.
 *
 * A clamped photo is laid out in a box that is NOT its own shape, so
 * `object-fit: cover` would cut part of it off. The caller uses this
 * to switch such a tile to `contain` — the photo then shows whole,
 * with the tile's own background at the edges, instead of losing
 * part of itself.
 *
 * With the default bounds this is rare: it needs a photo beyond 4:1
 * or 1:4, and the library has none.
 */
export function isClamped(
  source: MasonrySource | null | undefined,
  minAspect: number = DEFAULT_MIN_ASPECT,
  maxAspect: number = DEFAULT_MAX_ASPECT
): boolean {
  const w = source?.width;
  const h = source?.height;
  if (!w || !h || w <= 0 || h <= 0) return false;
  const ratio = h / w;
  return ratio < minAspect || ratio > maxAspect;
}

/**
 * Pack photos into columns, each photo at its own aspect ratio.
 *
 * Returns the placement of every photo and the total height of the
 * layout. The placements are in the grid's own coordinate space:
 * (0, 0) is the grid's top-left corner.
 */
export function layoutMasonry(
  sources: readonly (MasonrySource | null | undefined)[],
  options: MasonryOptions
): { placements: MasonryPlacement[]; height: number } {
  const tile = Math.max(1, options.tile);
  const columns = Math.max(1, Math.floor(options.columns));
  const minAspect = options.minAspect ?? DEFAULT_MIN_ASPECT;
  const maxAspect = options.maxAspect ?? DEFAULT_MAX_ASPECT;

  // The running height of each column.
  const columnHeights = new Array<number>(columns).fill(0);
  const placements: MasonryPlacement[] = [];

  for (const source of sources) {
    const height = Math.round(tile * aspectFor(source, minAspect, maxAspect));

    // The shortest column wins. Ties go to the leftmost, so the
    // layout is deterministic — the same photos always land in the
    // same places.
    let column = 0;
    for (let c = 1; c < columns; c++) {
      if (columnHeights[c] < columnHeights[column]) column = c;
    }

    const x = column * tile;
    const y = columnHeights[column];
    placements.push({ x, y, width: tile, height });
    columnHeights[column] = y + height;
  }

  // The layout's height is the tallest column.
  let height = 0;
  for (const h of columnHeights) height = Math.max(height, h);

  return { placements, height };
}
