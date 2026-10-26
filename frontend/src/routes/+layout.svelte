<script lang="ts">
  import '../lib/styles/tokens.css';
  import '../lib/styles/global.css';
  import TopBar from '$lib/components/TopBar.svelte';
  import ScrollToTop from '$lib/components/ScrollToTop.svelte';
  import Dialog from '$lib/components/Dialog.svelte';
  import { onNavigate } from '$app/navigation';
  import { onMount } from 'svelte';
  import { preferences } from '$lib/stores/preferences';
  import { GRID_SIZE_CONFIG } from '$lib/grid/config';
  import { columnsForWidth, widthForColumns } from '$lib/grid/geometry';

  let { children } = $props();

  // Round-37: dropped the photo-derived backdrop tint. The previous
  // design pushed the first-visible row's blurhash to a global
  // backdrop element on every IntersectionObserver tick, which
  // caused two visible problems:
  //   - the backdrop colour shifted continuously as the user
  //     scrolled (the anchor row changed, the blurhash changed,
  //     the wash followed);
  //   - the photo's saturated regions bled through the 90px blur
  //     as radiating colour bands.
  // Per-panel colour still comes from `.glass-tint::before` (each
  // tile's surrounding glass picks up a soft-light sample from the
  // photo inside). The page-level backdrop goes back to a single
  // flat dark base. Design preference: prefer elegant mathematical
  // relationships over hardcoded values; "no tint" is the most
  // elegant relationship for a flat backdrop.

  // Round-8: Cmd/Ctrl+K focuses the search composer input.
  // The convention is shared with GitHub, Linear, Vercel, and
  // most search-first apps. Implementation: a single
  // document-level keydown listener installed on mount, scoped
  // to modifier+K. We avoid stealing the key when the user is
  // already typing in an input/textarea (the conventional case
  // where Ctrl+K should pass through to the browser — though
  // most browsers have no default Ctrl+K binding, some users
  // map it to extensions).
  //
  // The shortcut only fires on routes that have a search
  // composer — currently / (home) and /search. On other routes
  // (lightbox is open, etc.) it's a no-op so we don't grab the
  // keystroke from a context that doesn't need it.
  function focusSearchInput() {
    const input = document.querySelector<HTMLInputElement>(
      '.composer-input'
    );
    if (input) {
      input.focus();
      // Select existing text so the user can type to replace
      // it (matches GitHub/Linear behaviour). If the input is
      // empty, this is a no-op.
      input.select();
    }
  }
  function onGlobalKey(e: KeyboardEvent) {
    if (!(e.metaKey || e.ctrlKey)) return;
    if (e.key !== 'k' && e.key !== 'K') return;
    // Don't fire if the user is already typing in a non-search
    // input (e.g. an album-rename modal's text field). The
    // composer input IS a text input — but we WANT to focus
    // it, so this guard is intentionally narrow: only skip
    // when the focused element is outside the composer.
    if (
      e.target instanceof HTMLElement &&
      e.target.tagName !== 'BODY' &&
      !e.target.closest('.composer')
    ) {
      return;
    }
    e.preventDefault();
    focusSearchInput();
  }
  onMount(() => {
    document.addEventListener('keydown', onGlobalKey);
    return () => document.removeEventListener('keydown', onGlobalKey);
  });

  // Tuning constants for the photo grid layout. These flow to
  // every chrome bar (page header, filters panel, saved-search
  // row) via --grid-width, so changing TILE here also resizes the
  // chrome on every page.
  //
  // Round-83: tile, gap, and max-cols now come from the user's
  // grid-size preference via GRID_SIZE_CONFIG. The numbers below
  // are derived so the ResizeObserver and any other reader see
  // the same values as PhotoGrid. Single source of truth.
  //
  // Large is 240px tiles at 6 cols, 14px gap → a 1460px content
  // width. The 384px source WebP is downscaled to 240 at dpr
  // 1.53 — essentially 1:1 device-pixel mapping, which is the
  // crispest the single-variant setup gets without going to
  // larger source files.
  const config = $derived(GRID_SIZE_CONFIG[$preferences.gridSize]);
  const TILE = $derived(config.tile);
  const GAP = $derived(config.gap);
  const MAX_COLS = $derived(config.maxCols);
  // Max row width = MAX_COLS * TILE + (MAX_COLS - 1) * GAP. Used
  // as `--grid-row-max-width` so the CSS grid's max-width clamp
  // matches the JS math — without it, auto-fit would let tracks
  // grow past the cap and the row would fill the viewport.
  const ROW_MAX_WIDTH = $derived(widthForColumns(MAX_COLS, config));

  // Compute and publish `--grid-width` on :root so any page
  // element (header bar, filters panel, etc.) can size to match
  // the rendered photo grid. Same math as PhotoGrid uses, but
  // evaluated at the layout level so pages WITHOUT PhotoGrid
  // mounted (e.g. / before any search runs, or a future page that
  // doesn't use PhotoGrid at all) still get the correct value.
  // PhotoGrid overwrites with its measured value once mounted —
  // the layout value is the seed/default.
  //
  // Math mirrors PhotoGrid's column calculation:
  //   N = min(MAX_COLS, floor((containerWidth + GAP) / (TILE + GAP)))
  //   gridWidth = N * TILE + (N - 1) * GAP
  //
  // The three new vars (--grid-tile-size, --grid-gutter-size,
  // --grid-row-max-width) come straight from the config so the
  // CSS can read them without needing its own preference
  // subscription. The ResizeObserver re-publishes them on every
  // container-width tick AND on every gridSize change.
  onMount(() => {
    if (typeof ResizeObserver === 'undefined') return;
    // Publish the size-driven CSS vars on mount and on every
    // gridSize change. The vars are derived from the current
    // preference, so changing the choice in Settings immediately
    // flows through to the CSS. The ResizeObserver below also
    // re-publishes them on width changes for symmetry, but the
    // var values don't depend on container width.
    const publishSizeVars = () => {
      document.documentElement.style.setProperty('--grid-tile-size', TILE + 'px');
      document.documentElement.style.setProperty('--grid-gutter-size', GAP + 'px');
      document.documentElement.style.setProperty('--grid-row-max-width', ROW_MAX_WIDTH + 'px');
    };
    publishSizeVars();
    const unsub = preferences.subscribe(() => publishSizeVars());

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const containerWidth = entry.contentRect.width;
        if (containerWidth <= 0) continue;
        // Re-publish size-driven vars on every width tick so a
        // gridSize change that happens to coincide with a width
        // change can't get out of sync.
        publishSizeVars();
        const cols = columnsForWidth(containerWidth, config);
        const gridWidth = widthForColumns(cols, config);
        document.documentElement.style.setProperty('--grid-width', gridWidth + 'px');
      }
    });
    const main = document.querySelector('.shell');
    if (main) ro.observe(main);
    return () => {
      ro.disconnect();
      unsub();
    };
  });

  // Round-6: View Transitions API crossfade between routes.
  // Wraps `goto()` and `<a>` navigations in
  // `document.startViewTransition` so the browser paints the
  // outgoing snapshot, runs the new route's render, then
  // crossfades between them. Falls back to plain navigation
  // on browsers without the API (Safari < 18.4, Firefox
  // < 137).
  //
  // The transition only fires for same-origin navigations —
  // external links and form submits skip it. The CSS rules
  // for ::view-transition-old(root) / ::view-transition-new(root)
  // in global.css drive the actual animation.
  onNavigate((navigation) => {
    if (
      typeof document === 'undefined' ||
      typeof document.startViewTransition !== 'function'
    ) {
      return; // Browser doesn't support View Transitions.
    }
    // Returning a Promise makes SvelteKit wait for it before
    // completing the navigation — this lets us wrap the
    // transition around the actual route swap.
    return new Promise((resolve) => {
      // Fire the transition. The browser snapshots the
      // current DOM, then swaps to the new route's DOM once
      // SvelteKit completes the navigation.
      document.startViewTransition!(async () => {
        resolve();
        // Wait for SvelteKit to finish the navigation
        // (it does this when our promise resolves).
        await navigation.complete;
      });
    });
  });
</script>

<div class="app-shell">
  <TopBar />
  <main class="shell">
    {@render children?.()}
  </main>
  <Dialog />
  <ScrollToTop />
</div>

<style>
  .app-shell {
    min-height: 100vh;
    position: relative;
    background: transparent; /* body supplies the mesh-gradient;
                               app-shell stays transparent so the
                               gradient shows through. Round-38. */
  }
  .shell,
  :global(.topbar) {
    position: relative;
    z-index: 2;
  }
  .shell {
    /* Round-1 polish: vertical rhythm owned here, not per-page.
       Padding + gap come from shell tokens so every route inherits
       the same spacing. Section gap (--shell-gap, 24px) controls
       the space between PageHeader and the content below it. */
    min-height: calc(100vh - var(--topbar-h));
    padding: var(--shell-pad-y) var(--shell-pad-x) var(--shell-pad-y);
    display: flex;
    flex-direction: column;
    gap: var(--shell-gap);
    /* Round‑36: raised from 1600 to 2400 so the grid can lay out
       6×384px tiles cleanly on 4K/2K monitors without big empty
       margins on either side. Content below 2400px stays
       viewport-natural. PhotoGrid's grid uses fixed 384px tiles
       (matches the 384px thumbnail source 1:1), so beyond this
       cap we'd just be adding whitespace — no benefit to raising
       further until the design supports 7+ columns. */
    max-width: 2200px;
    margin: 0 auto;
  }
  @media (max-width: 640px) {
    /* Mobile breakpoint: tighter shell padding because the 32/40
       budget at desktop sizes eats too much of a 375px viewport.
       Snap to var(--s-3) top, var(--s-2) sides, --shell-pad-y x2
       bottom to keep the same proportions on a small screen. */
    .shell {
      padding: var(--shell-pad-y) var(--s-2) var(--shell-pad-y);
    }
  }
</style>
