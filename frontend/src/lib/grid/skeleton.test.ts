/**
 * Tests for the loading placeholder band's shape.
 *
 * The band has to have the same shape as the layout it stands in
 * for. The bug these pin: the skeleton was built from the UNIFORM
 * grid's CSS grid and reused for every layout, so under the
 * justified layout it rendered square tiles where the real grid
 * had flush, variable-width rows — and the CSS fit fewer tracks
 * than the skeleton rendered tiles into, so the extra tiles wrapped
 * onto a second line and the band doubled in height.
 */
import { describe, expect, it } from 'vitest';
import {
  PLACEHOLDER_ASPECTS,
  placeholderSources,
  skeletonJustified,
  skeletonMasonry
} from './skeleton';
import { layoutJustified } from './justified';
import { layoutMasonry } from './masonry';

const ROW_WIDTH = 1366;
const ROW_HEIGHT = 360;

describe('placeholderSources', () => {
  it('cycles through a MIX of ratios', () => {
    const sources = placeholderSources(12);
    const ratios = sources.map((s) => s.width / s.height);
    // A single repeated ratio would give the justified layout rows
    // of identical tiles — the one shape a real row never has.
    expect(new Set(ratios).size).toBeGreaterThan(1);
    expect(ratios[0]).toBeCloseTo(PLACEHOLDER_ASPECTS[0], 3);
    expect(ratios.length).toBe(12);
  });

  it('produces a whole number of requests', () => {
    expect(placeholderSources(0)).toEqual([]);
    expect(placeholderSources(7)).toHaveLength(7);
  });
});

describe('skeletonJustified', () => {
  it('renders tiles that FILL the row width', () => {
    const layout = skeletonJustified({
      rowWidth: ROW_WIDTH,
      rowHeight: ROW_HEIGHT,
      rows: 3
    });

    expect(layout.boxes.length).toBeGreaterThan(0);

    // Group the boxes by their top edge — each group is a row.
    const rows = new Map<number, typeof layout.boxes>();
    for (const box of layout.boxes) {
      const row = rows.get(box.y) ?? [];
      row.push(box);
      rows.set(box.y, row);
    }

    expect(rows.size).toBe(3);
    for (const row of rows.values()) {
      // The row's tiles span the whole width: that is what a
      // justified row does, and what the square-tile skeleton
      // failed to do.
      const left = Math.min(...row.map((b) => b.x));
      const right = Math.max(...row.map((b) => b.x + b.width));
      expect(right - left).toBeCloseTo(ROW_WIDTH, 0);
      // And they all share the row's height.
      expect(new Set(row.map((b) => b.height)).size).toBe(1);
    }
  });

  it('uses the SAME row heights the real layout uses', () => {
    // A justified row's height is computed from its photos' ratios,
    // so it STRAYS from the target within the layout's tolerance
    // (0.5 of it, Immich's own). The placeholder row must have the
    // height the real row will have — not the nominal target, which
    // no real row ever hits exactly.
    const real = layoutJustified(placeholderSources(24), {
      rowWidth: ROW_WIDTH,
      rowHeight: ROW_HEIGHT
    });
    const skeleton = skeletonJustified({
      rowWidth: ROW_WIDTH,
      rowHeight: ROW_HEIGHT,
      rows: 3
    });
    const realHeights = real.rows.slice(0, 3).map((r) => r.height);
    const skeletonHeights = [...new Set(skeleton.boxes.map((b) => b.height))];
    expect(skeletonHeights).toHaveLength(realHeights.length);
    for (let i = 0; i < realHeights.length; i++) {
      expect(skeletonHeights[i]).toBeCloseTo(realHeights[i], 0);
      // And within the layout's own tolerance of the target.
      expect(
        Math.abs(skeletonHeights[i] - ROW_HEIGHT)
      ).toBeLessThanOrEqual(ROW_HEIGHT * 0.5);
    }
  });

  it('is as tall as the rows it stands in for', () => {
    const layout = skeletonJustified({
      rowWidth: ROW_WIDTH,
      rowHeight: ROW_HEIGHT,
      rows: 3
    });
    // Three rows of 360px, plus the small gap between rows. The
    // OLD skeleton was 2164px for the same three rows, because the
    // wrapped tiles doubled every row.
    expect(layout.height).toBeLessThan(ROW_HEIGHT * 3 + 40);
    expect(layout.height).toBeGreaterThan(ROW_HEIGHT * 2);
  });

  it('leaves NO row short', () => {
    // The widow rule gives a layout's LAST row the previous row's
    // height, which leaves it narrower than the grid. The skeleton
    // shows the FIRST rows of a longer layout, so none of them is
    // the last — every one fills the width.
    const layout = skeletonJustified({
      rowWidth: ROW_WIDTH,
      rowHeight: ROW_HEIGHT,
      rows: 3
    });
    const widths = layout.boxes.map((b) => b.x + b.width);
    expect(Math.max(...widths)).toBeCloseTo(ROW_WIDTH, 0);
  });

  it('does not wrap its tiles onto a second line', () => {
    // The regression: the CSS grid fitted 3 tracks and the skeleton
    // rendered 4 tiles, so the fourth wrapped and doubled the row.
    const layout = skeletonJustified({
      rowWidth: ROW_WIDTH,
      rowHeight: ROW_HEIGHT,
      rows: 3
    });
    const tops = new Set(layout.boxes.map((b) => b.y));
    // One top edge per row, and no two rows share one.
    expect(tops.size).toBe(3);
    for (const top of tops) {
      const inRow = layout.boxes.filter((b) => b.y === top);
      expect(inRow.length).toBeGreaterThan(0);
      // Every tile in a row shares the row's height, so a tile
      // hanging below the others would show up as a taller box.
      expect(new Set(inRow.map((b) => b.height)).size).toBe(1);
    }
  });

  it('agrees with the real layout it stands in for', () => {
    // The placeholder band's first row must be the SAME shape as
    // the real layout's first row — same row height, same width.
    const real = layoutJustified(placeholderSources(24), {
      rowWidth: ROW_WIDTH,
      rowHeight: ROW_HEIGHT
    });
    const skeleton = skeletonJustified({
      rowWidth: ROW_WIDTH,
      rowHeight: ROW_HEIGHT,
      rows: 3
    });
    expect(skeleton.boxes[0].height).toBeCloseTo(real.rows[0].height, 0);
    expect(skeleton.boxes[0].y).toBeCloseTo(real.rows[0].top, 0);
  });
});

describe('skeletonMasonry', () => {
  it('renders tiles at the masonry tile width', () => {
    const layout = skeletonMasonry({
      tile: 360,
      columns: 4,
      rows: 3
    });
    expect(layout.boxes.length).toBe(12);
    for (const box of layout.boxes) {
      expect(box.width).toBeCloseTo(360, 0);
    }
  });

  it('gives the tiles VARIED heights', () => {
    // A masonry tile's height follows its photo's ratio. Square
    // placeholders would be the uniform grid's shape again.
    const layout = skeletonMasonry({
      tile: 360,
      columns: 4,
      rows: 3
    });
    const heights = new Set(layout.boxes.map((b) => Math.round(b.height)));
    expect(heights.size).toBeGreaterThan(1);
  });

  it('packs the tiles into the columns it was given', () => {
    const layout = skeletonMasonry({
      tile: 360,
      columns: 4,
      rows: 3
    });
    const lefts = new Set(layout.boxes.map((b) => Math.round(b.x)));
    expect(lefts.size).toBe(4);
  });

  it('agrees with the real masonry layout', () => {
    const real = layoutMasonry(placeholderSources(12), {
      tile: 360,
      columns: 4
    });
    const skeleton = skeletonMasonry({
      tile: 360,
      columns: 4,
      rows: 3
    });
    expect(skeleton.boxes[0].y).toBeCloseTo(real.placements[0].y, 0);
    expect(skeleton.boxes[0].height).toBeCloseTo(
      real.placements[0].height,
      0
    );
  });
});
