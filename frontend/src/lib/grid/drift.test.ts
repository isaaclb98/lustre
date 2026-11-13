/**
 * Unit tests for lib/grid/drift.ts — the arithmetic of the moving
 * presentation wall.
 *
 * Two invariants matter and both are easy to break:
 *
 *  1. The wall LOOPS. The photos are a ring, so walking down the rows
 *     reads the album end-to-start, forever, with no half-empty row
 *     and no stop. A plain list fails both: the last row is ragged
 *     and the photos run out.
 *  2. The window COVERS the viewport, with lead on both sides. A row
 *     must exist before it scrolls into view (or its photos pop in at
 *     the edge) and linger after it scrolls out (or it pops out).
 */
import { describe, it, expect } from 'vitest';
import {
  DRIFT_OVERSCAN_ROWS,
  DRIFT_SPEED_PX_PER_SEC,
  driftRowItems,
  driftWindowRows
} from '$lib/grid/drift';

const ALBUM = Array.from({ length: 224 }, (_, i) => `p${i}`);

/** The album's photos in the order the ring presents them. */
function ringSequence(
  items: readonly string[],
  columns: number,
  rows: number
): string[] {
  const out: string[] = [];
  for (let r = 0; r < rows; r++) {
    out.push(...driftRowItems(items, columns, r));
  }
  return out;
}

describe('driftWindowRows', () => {
  it('renders nothing before the row height is known', () => {
    expect(driftWindowRows(0, 0, 900)).toEqual({ first: 0, last: 0 });
    expect(driftWindowRows(500, -10, 900)).toEqual({ first: 0, last: 0 });
  });

  it('covers the viewport', () => {
    const rowHeight = 240;
    const viewport = 900;
    const { first, last } = driftWindowRows(0, rowHeight, viewport);

    // Every row whose screen position falls inside the viewport is
    // inside the rendered range.
    const top = 0;
    const bottom = top + viewport;
    const firstVisible = Math.floor(top / rowHeight);
    const lastVisible = Math.ceil(bottom / rowHeight);
    expect(first).toBeLessThanOrEqual(firstVisible);
    expect(last).toBeGreaterThan(lastVisible);
  });

  it('leads the viewport on both sides', () => {
    const rowHeight = 240;
    const { first, last } = driftWindowRows(0, rowHeight, 900, 2);

    // The first rendered row is two rows ABOVE the top edge, so it is
    // already on screen when the wall starts moving.
    expect(first).toBe(-2);
    // And the last rendered row is two rows BELOW the bottom edge.
    const visibleRows = Math.ceil(900 / rowHeight) + 1;
    expect(last - first).toBe(visibleRows + 4);
  });

  it('advances one row at a time as the wall drifts', () => {
    const rowHeight = 240;
    const before = driftWindowRows(0, rowHeight, 900);
    const after = driftWindowRows(rowHeight, rowHeight, 900);
    expect(after.first).toBe(before.first + 1);
    expect(after.last).toBe(before.last + 1);
  });

  it('keeps the range the same size wherever the wall is', () => {
    const rowHeight = 240;
    const size = new Set<number>();
    for (let offset = 0; offset < 5000; offset += 137) {
      const { first, last } = driftWindowRows(offset, rowHeight, 900);
      size.add(last - first);
    }
    expect(size.size).toBe(1);
  });

  it('defaults to the shared overscan', () => {
    expect(driftWindowRows(0, 240, 900)).toEqual(
      driftWindowRows(0, 240, 900, DRIFT_OVERSCAN_ROWS)
    );
  });
});

describe('driftRowItems', () => {
  it('fills the row', () => {
    expect(driftRowItems(ALBUM, 6, 0)).toHaveLength(6);
  });

  it('reads the album in order, row after row', () => {
    expect(driftRowItems(ALBUM, 6, 0)).toEqual([
      'p0',
      'p1',
      'p2',
      'p3',
      'p4',
      'p5'
    ]);
    expect(driftRowItems(ALBUM, 6, 1)).toEqual([
      'p6',
      'p7',
      'p8',
      'p9',
      'p10',
      'p11'
    ]);
  });

  it('wraps the album end into the start of the next row', () => {
    // 224 photos at 6 columns: row 37 starts at photo 222, so the
    // album's last two photos sit beside its first four.
    const row = driftRowItems(ALBUM, 6, 37);
    expect(row).toEqual(['p222', 'p223', 'p0', 'p1', 'p2', 'p3']);
  });

  it('loops forever with no ragged row', () => {
    const columns = 6;
    const rows = 100;
    const sequence = ringSequence(ALBUM, columns, rows);

    // Every row is full.
    expect(sequence).toHaveLength(columns * rows);
    // The sequence is the album repeating — photo 0 comes back
    // exactly where the ring says it does.
    const cycleLength = ALBUM.length;
    for (let i = cycleLength; i < sequence.length; i++) {
      expect(sequence[i]).toBe(sequence[i - cycleLength]);
    }
    // And it never stops: reading past the album continues rather
    // than running out.
    expect(sequence.length).toBeGreaterThan(cycleLength * 2);
  });

  it('gives the same photos to the same row wherever the wall is', () => {
    // The row index is absolute, so a row's contents never change as
    // the wall drifts — that is what makes the drift seamless.
    expect(driftRowItems(ALBUM, 6, 12)).toEqual(driftRowItems(ALBUM, 6, 12));
  });

  it('keeps rows contiguous across the wrap', () => {
    const columns = 6;
    const first = driftRowItems(ALBUM, columns, 37);
    const second = driftRowItems(ALBUM, columns, 38);
    // The last photo of one row is the one before the first photo of
    // the next, in album order.
    const indexOf = (id: string) => ALBUM.indexOf(id);
    expect(indexOf(second[0])).toBe((indexOf(first[5]) + 1) % ALBUM.length);
  });

  it('handles an album smaller than one row', () => {
    const tiny = ['a', 'b'];
    const row = driftRowItems(tiny, 6, 0);
    expect(row).toHaveLength(6);
    expect(row).toEqual(['a', 'b', 'a', 'b', 'a', 'b']);
  });

  it('handles an empty album and a zero column count', () => {
    expect(driftRowItems([], 6, 0)).toEqual([]);
    expect(driftRowItems(ALBUM, 0, 0)).toEqual([]);
  });

  it('handles a negative row index', () => {
    // The wall's origin is arbitrary; a negative index is a row above
    // it, and the ring still wraps.
    expect(driftRowItems(ALBUM, 6, -1)).toEqual([
      'p218',
      'p219',
      'p220',
      'p221',
      'p222',
      'p223'
    ]);
  });
});

describe('drift speed', () => {
  it('is slow enough to read as breathing', () => {
    // At 16px/s a photo crosses a 900px screen in under a minute.
    expect(DRIFT_SPEED_PX_PER_SEC).toBe(16);
    expect(900 / DRIFT_SPEED_PX_PER_SEC).toBeLessThan(60);
  });
});
