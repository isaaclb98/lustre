/**
 * usePhotoPreloader — extracts the Lightbox's adjacent-photo
 * preload effect into a reusable composable.
 *
 * Behaviour preserved from Lightbox.svelte:
 *   - Preloads the adjacent photos around the cursor. The window is
 *     ASYMMETRIC — 1 behind and 2 ahead for manual navigation, ±3
 *     while the slideshow runs (see ./preload-window.ts).
 *   - During slideshow, wraps around the edges (Prev from
 *     the first loops to the last, and vice versa). Outside
 *     of slideshow, respects the grid boundaries — wrap-around
 *     would be wasted bandwidth because prev/next buttons are
 *     disabled at the edges.
 *   - Skips entirely when items.length <= 1 (single photo).
 *   - Skips entirely when the user opted into data-saver mode
 *     (Network Information API's `saveData`, or
 *     `prefers-reduced-data: reduce`).
 *   - Excludes the currently-displayed photo (`photoSrc`).
 *   - Evicts entries that fell out of the window so a fast
 *     scrub through 1000 photos doesn't mount 1000 hidden
 *     <img> nodes.
 *   - Uses a SvelteSet for the live list so the template's
 *     `{#each [...preloadedSrcs] as preloadSrc (preloadSrc)}`
 *     block reactively adds/removes the hidden <img> nodes.
 *
 * Reactive deps: `idx` (the cursor), `isPlaying` (window
 * expansion), and `items.length`. The composable re-reads
 * `lightboxWidth()` per effect run (not subscribed) so a
 * window resize doesn't trigger re-preloads; a fresh preload
 * happens on the next nav.
 */
import { SvelteSet } from 'svelte/reactivity';
import {
  MANUAL_PRELOAD_WINDOW,
  SLIDESHOW_PRELOAD_WINDOW,
  preloadIndices,
  type PreloadWindow
} from './preload-window';

export type PreloadOptions = {
  getItemsLength: () => number;
  getCursor: () => number;
  getIsPlaying: () => boolean;
  /** The current photo's URL — excluded from the preload
   * window. (Already displayed; no need to preload.) */
  getPhotoSrc: () => string;
  /** Build the photo URL for a given item id. The
   * composable doesn't know what an "item" is, so it
   * delegates the URL shape. */
  getItemUrl: (id: string) => string;
  /** Get the id of the item at index `i`. */
  getItemIdAt: (i: number) => string | null | undefined;
  /** Read the current window width (used as the photo URL's
   * `?w=` query). Re-read on every effect run, not
   * subscribed to. */
  getWidth: () => number;
};

export type PreloadController = {
  /** The set of currently-preloaded URLs. The Lightbox
   * template iterates this to mount hidden <img> nodes. */
  preloadedSrcs: SvelteSet<string>;
};

export function usePhotoPreloader(opts: PreloadOptions): PreloadController {
  // eslint-disable-next-line svelte/no-unnecessary-state-wrap -- reassigned on every effect run; SvelteSet alone doesn't track the variable reference
  let preloadedSrcs: SvelteSet<string> = $state(new SvelteSet());

  function isDataSaver(): boolean {
    const conn =
      typeof navigator !== 'undefined'
        ? (navigator as Navigator & {
            connection?: { saveData?: boolean };
          }).connection
        : undefined;
    return (
      !!conn?.saveData ||
      (typeof window !== 'undefined' &&
        !!window.matchMedia?.('(prefers-reduced-data: reduce)').matches)
    );
  }

  $effect(() => {
    // Touch the deps so Svelte tracks them.
    const cursor = opts.getCursor();
    const isPlaying = opts.getIsPlaying();
    const len = opts.getItemsLength();
    const photoSrc = opts.getPhotoSrc();

    // Skip on single-photo lightbox.
    if (len <= 1) {
      if (preloadedSrcs.size > 0) preloadedSrcs = new SvelteSet();
      return;
    }

    // Skip when the user opted into data-saver mode.
    if (isDataSaver()) {
      if (preloadedSrcs.size > 0) preloadedSrcs = new SvelteSet();
      return;
    }

    // The window is asymmetric: 1 behind, 2 ahead for manual
    // navigation; ±3 while the slideshow is running so a 1s
    // cadence doesn't race the network.
    const win: PreloadWindow = isPlaying ? SLIDESHOW_PRELOAD_WINDOW : MANUAL_PRELOAD_WINDOW;
    // The want set is local — never bound to $state, so a
    // plain Set is fine. (The exposed preloadedSrcs is a
    // SvelteSet, mutated via reassignment of the variable.)
    // eslint-disable-next-line svelte/prefer-svelte-reactivity
    const want = new Set<string>();
    // Wrap-around only while the slideshow is running —
    // manual prev at the first photo and next at the last
    // are intentionally disabled, so preloading the wrap
    // would be wasted bandwidth.
    for (const i of preloadIndices(cursor, len, win, isPlaying)) {
      const id = opts.getItemIdAt(i);
      if (id) want.add(opts.getItemUrl(id));
    }

    // Evict entries that fell out of the window. Without this,
    // a fast scrub through a 1000-photo grid would mount 1000
    // hidden <img> nodes; the browser would keep them all
    // cached and we'd hold 1000 decoded bitmaps in memory.
    let changed = false;
    const next = new SvelteSet<string>();
    for (const src of want) {
      if (preloadedSrcs.has(src) || src === photoSrc) next.add(src);
      else {
        next.add(src);
        changed = true;
      }
    }
    for (const src of preloadedSrcs) {
      if (!next.has(src)) changed = true;
    }
    if (changed) preloadedSrcs = next;
  });

  return {
    get preloadedSrcs() {
      return preloadedSrcs;
    }
  };
}
