/**
 * preload-window — which photos the Lightbox warms around the cursor.
 *
 * The window is ASYMMETRIC: more photos ahead of the cursor than
 * behind it. The lightbox is browsed forward — a user opens a photo
 * and presses →, →, →. With a symmetric ±1 window the SECOND press
 * of a burst waits on a fetch that only started when the FIRST press
 * landed, because that photo was only pulled into the window at that
 * moment. Warming two ahead puts the second photo of a burst in the
 * cache before the first press, so a run of arrows is warm end to end.
 *
 * Pure module: the composable owns the Svelte state, this owns the
 * arithmetic. Unit-tested without mounting Svelte.
 */

export type PreloadWindow = {
  /** Photos BEFORE the cursor — the ← direction. */
  prev: number;
  /** Photos AFTER the cursor — the → direction. */
  next: number;
};

/**
 * Manual navigation: one behind, two ahead.
 *
 * The behind side stays at 1 because stepping back is the exception,
 * not the rule — the cost of one extra warm photo is a fetch the
 * user usually never needs.
 */
export const MANUAL_PRELOAD_WINDOW: PreloadWindow = { prev: 1, next: 2 };

/**
 * Slideshow: three each way.
 *
 * The shortest cadence a user can pick is 1s, and the auto-advance
 * moves on its own — so the forward runway has to be deep enough that
 * the tick never outruns the network. The behind side keeps the same
 * depth because the slideshow wraps at the ends (the last photo leads
 * back to the first), so "behind" is reachable there too.
 */
export const SLIDESHOW_PRELOAD_WINDOW: PreloadWindow = { prev: 3, next: 3 };

/**
 * The item indices to preload for a cursor at `cursor` in a list of
 * `len` items.
 *
 * Forward indices come first: the → direction is the hot path, so
 * those fetches start before the ← ones.
 *
 * With `wrap`, the window wraps around the ends (the slideshow loops,
 * so the photo after the last one IS the first one). Without it the
 * window stops at the ends — prev/next are disabled there, so warming
 * across the boundary would be wasted bandwidth.
 *
 * A window wider than the list yields each index once.
 *
 * The cursor is never in its own window — the displayed photo is
 * already loaded, so warming it again would be a wasted node.
 */
export function preloadIndices(
  cursor: number,
  len: number,
  win: PreloadWindow,
  wrap: boolean
): number[] {
  if (len <= 1) return [];
  const out: number[] = [];
  const seen = new Set<number>();
  const push = (i: number) => {
    if (i < 0 || i >= len || i === cursor || seen.has(i)) return;
    seen.add(i);
    out.push(i);
  };
  for (let d = 1; d <= win.next; d++) {
    push(wrap ? (((cursor + d) % len) + len) % len : cursor + d);
  }
  for (let d = 1; d <= win.prev; d++) {
    push(wrap ? (((cursor - d) % len) + len) % len : cursor - d);
  }
  return out;
}
