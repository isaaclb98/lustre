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
  /** The gutter between tiles, in px. */
  gap: number;
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
 * single tile taller than the screen, and the column it lands in
 * would tower over the rest. The clamp keeps the grid readable
 * without discarding the photo's shape entirely — 0.5 is a 2:1
 * landscape and 2 is a 1:2 portrait.
 */
export const DEFAULT_MIN_ASPECT = 0.5;
export const DEFAULT_MAX_ASPECT = 2;

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
  const gap = Math.max(0, options.gap);
  const columns = Math.max(1, Math.floor(options.columns));
  const minAspect = options.minAspect ?? DEFAULT_MIN_ASPECT;
  const maxAspect = options.maxAspect ?? DEFAULT_MAX_ASPECT;

  // The running height of each column, including the gutter that
  // will separate its last tile from the next one.
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

    const x = column * (tile + gap);
    const y = columnHeights[column];
    placements.push({ x, y, width: tile, height });
    columnHeights[column] = y + height + gap;
  }

  // The layout's height is the tallest column, minus the trailing
  // gutter that no tile actually occupies.
  let height = 0;
  for (const h of columnHeights) height = Math.max(height, h);
  if (placements.length > 0) height = Math.max(0, height - gap);

  return { placements, height };
}
