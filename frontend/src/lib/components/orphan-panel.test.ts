/**
 * tests/orphan-panel.test.ts — the orphan panel's decision logic.
 *
 * The panel's job is to tell the operator what a purge would remove
 * and to make the destructive action require a deliberate click.
 * These are pure functions extracted from the component, tested
 * without mounting Svelte.
 *
 * The count that matters is the SUM of both halves. A favourite
 * whose photo was deleted from disk is an orphan even when its
 * Qdrant point was never marked (the indexer may not have run since
 * the deletion) — so a panel that only looked at the Qdrant count
 * would show "nothing to purge" while orphaned favourites sat in
 * the database.
 */

type Counts = Record<string, number>;

/** The user-state rows a purge would remove, as a flat total. */
function userDataCount(counts: Counts | undefined): number {
  return Object.values(counts ?? {}).reduce((sum, n) => sum + n, 0);
}

/**
 * The breakdown, labelled by what the user sees rather than by
 * what the tables are called — an album cover reset is not a
 * deletion, and "Album covers" says that better than
 * "album_covers_reset" does.
 */
function userDataBreakdown(counts: Counts | undefined): { label: string; count: number }[] {
  const c = counts ?? {};
  const labels: [string, string][] = [
    ['favorites', 'Favourites'],
    ['dislikes', 'Dislikes'],
    ['album_memberships', 'Album entries'],
    ['album_covers_reset', 'Album covers'],
  ];
  return labels
    .filter(([key]) => (c[key] ?? 0) > 0)
    .map(([key, label]) => ({ label, count: c[key] ?? 0 }));
}

/** Is there anything a purge would actually remove? */
function hasWork(markedOrphans: number, counts: Counts | undefined): boolean {
  return markedOrphans > 0 || userDataCount(counts) > 0;
}

import { describe, it, expect } from 'vitest';

describe('orphan panel counts', () => {
  it('sums the user-state counts', () => {
    expect(
      userDataCount({ favorites: 12, dislikes: 3, album_memberships: 5, album_covers_reset: 1 })
    ).toBe(21);
  });

  it('treats missing counts as zero', () => {
    expect(userDataCount(undefined)).toBe(0);
    expect(userDataCount({})).toBe(0);
  });

  it('labels the breakdown by what the user sees', () => {
    const rows = userDataBreakdown({ favorites: 12, album_covers_reset: 1 });
    expect(rows).toEqual([
      { label: 'Favourites', count: 12 },
      { label: 'Album covers', count: 1 },
    ]);
  });

  it('omits the categories with nothing in them', () => {
    const rows = userDataBreakdown({ favorites: 1, dislikes: 0, album_memberships: 0 });
    expect(rows).toEqual([{ label: 'Favourites', count: 1 }]);
  });
});

describe('orphan panel purge button', () => {
  it('is enabled when there are marked orphans', () => {
    expect(hasWork(5, {})).toBe(true);
  });

  it('is enabled when only the user state is orphaned', () => {
    // The case the Qdrant-only count would miss: the photo was
    // deleted from disk but the indexer has not run since, so its
    // point was never marked.
    expect(hasWork(0, { favorites: 2 })).toBe(true);
  });

  it('is disabled when there is nothing to purge', () => {
    expect(hasWork(0, {})).toBe(false);
    expect(hasWork(0, undefined)).toBe(false);
  });
});
