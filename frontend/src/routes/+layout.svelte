<script lang="ts">
  /**
   * Root layout — the base every route inherits.
   *
   * It carries only what is genuinely global: the stylesheets, the
   * view-transition wrapper, and the grid's geometry/motion
   * variables. The app chrome — top bar, page shell, dialogs,
   * scroll-to-top — lives in the (app) group layout instead, so a
   * route that breaks out of that group gets the bare root and
   * nothing else.
   *
   * The album presentation wall does exactly that: it resets to this
   * layout with `+page@.svelte`, so the photos are the only thing
   * on screen — no nav bar, no page shell, no floating controls.
   */
  import '../lib/styles/tokens.css';
  import '../lib/styles/global.css';
  import { onNavigate } from '$app/navigation';
  import { onMount } from 'svelte';
  import { preferences } from '$lib/stores/preferences';
  import { theme } from '$lib/theme';
  import {
    GRID_HOVER_ZOOM,
    GRID_MOTION_EASE,
    GRID_MOTION_MS,
    gridConfigFor,
    tileRadiusFor
  } from '$lib/grid/config';
  import { widthForColumns } from '$lib/grid/geometry';

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
  //
  // The theme is the second half of that decision. The dark theme's
  // grid has no gutter at all — its photos touch, like the
  // presentation wall's — so the gap and the tile radius are read
  // through the theme rather than off the size preset alone.
  const config = $derived(gridConfigFor($preferences.gridSize, $theme));
  // Max row width = MAX_COLS * TILE + (MAX_COLS - 1) * GAP. Used
  // as `--grid-row-max-width` so the CSS grid's max-width clamp
  // matches the JS math — without it, auto-fit would let tracks
  // grow past the cap and the row would fill the viewport.
  const ROW_MAX_WIDTH = $derived(widthForColumns(config.maxCols, config));

  // Publish the grid's own numbers on :root so any element can
  // size itself against the grid without subscribing to the
  // preference. The three size vars (--grid-tile-size,
  // --grid-gutter-size, --grid-row-max-width) come straight from
  // the config so the CSS can read them without needing its own
  // preference subscription. They change only with the grid-size
  // preference, so the subscription below covers every case.
  //
  // --grid-width (the measured content width) is NOT published
  // here: it is derived from the app shell's width, and the shell
  // only exists on app routes. The (app) layout owns it.
  onMount(() => {
    const publishSizeVars = () => {
      const root = document.documentElement.style;
      root.setProperty('--grid-tile-size', config.tile + 'px');
      root.setProperty('--grid-gutter-size', config.gap + 'px');
      root.setProperty('--grid-row-max-width', ROW_MAX_WIDTH + 'px');
      root.setProperty('--grid-hover-zoom', String(GRID_HOVER_ZOOM));
      root.setProperty('--grid-motion-ms', GRID_MOTION_MS + 'ms');
      root.setProperty('--grid-motion-ease', GRID_MOTION_EASE);
      // The tile radius is published so the tile and the loading
      // skeleton — two different components — cannot drift apart
      // in shape. The presentation wall overrides it on its own
      // grid wrapper to 0, so its photos touch pixel-to-pixel.
      // The dark theme's radius is 0 everywhere: sharp, no rounded
      // aspects at all.
      root.setProperty('--grid-tile-radius', tileRadiusFor($theme) + 'px');
    };
    publishSizeVars();
    const unsub = preferences.subscribe(() => publishSizeVars());
    // The theme is fixed per deployment — the marker is stamped
    // onto the shell — so this rarely fires. But it must fire at
    // all: the gutter and the radius are part of the theme, and a
    // change to the attribute has to reach the published vars.
    const unsubTheme = theme.subscribe(() => publishSizeVars());
    return () => {
      unsub();
      unsubTheme();
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
  // in global.css drive the actual crossfade.
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

{@render children?.()}
