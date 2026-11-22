<script lang="ts">
  /**
   * The app chrome — top bar, page shell, dialogs, scroll-to-top.
   *
   * Everything here is navigation and app furniture. It lives in a
   * route GROUP so that a route which breaks out of it (the album
   * presentation wall, via `+page@.svelte`) gets the bare root
   * layout instead: no nav bar, no page shell, no floating
   * controls. The presentation mode is the one surface that is
   * genuinely "on its own".
   *
   * The split is structural rather than a conditional: the chrome is
   * not "hidden" on the wall, it simply is not in the wall's layout
   * hierarchy, so nothing here can leak into it.
   */
  import AppChrome from '$lib/components/AppChrome.svelte';
  import { onMount } from 'svelte';
  import { preferences, layoutEnabled } from '$lib/stores/preferences';
  import { layoutGridConfig } from '$lib/grid/config';
  import {
    columnsForWidth,
    gridContainerWidth,
    widthForColumns
  } from '$lib/grid/geometry';
  import { JUSTIFIED_GRID_SIZE_CONFIG } from '$lib/grid/config';

  let { children } = $props();

  // Each layout has its own size ladder (see MASONRY_GRID_SIZE_CONFIG
  // and JUSTIFIED_GRID_SIZE_CONFIG). The published --grid-width must
  // come from the SAME ladder the grid renders with, or the chrome
  // would be sized for a different column count than the grid draws.
  const config = $derived(
    layoutGridConfig($preferences.gridSize, layoutEnabled($preferences))
  );

  // Round-8: Cmd/Ctrl+K focuses the search composer input.
  // The convention is shared with GitHub, Linear, Vercel, and
  // most search-first apps. Implementation: a single
  // document-level keydown listener installed on mount, scoped
  // to modifier+K. We avoid stealing the key when the user is
  // already typing in an input/textarea (the conventional case
  // where Cmd+K should pass through to the browser — though
  // most browsers have no default Cmd+K binding, some users
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
      // Select existing text so the user can retype to replace
      // it (matches GitHub/Linear behaviour).
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

  // Publish `--grid-width` on :root so any page element (header
  // bar, filters panel, etc.) can size itself to match the rendered
  // photo grid. Same math as PhotoGrid uses, but evaluated here so
  // pages WITHOUT PhotoGrid mounted (e.g. / before any search runs)
  // still get the correct value. PhotoGrid overwrites with its
  // measured value once mounted — the layout value is the seed.
  //
  // Math mirrors PhotoGrid's column calculation:
  //   N = min(MAX_COLS, floor(containerWidth / TILE))
  //   gridWidth = N * TILE
  //
  // The width comes from the app shell, which only exists on app
  // routes — so this measurement belongs here, not in the root
  // layout. It re-runs on every width tick AND on every grid-size
  // change, because the column count depends on both.
  onMount(() => {
    if (typeof ResizeObserver === 'undefined') return;
    const publishGridWidth = () => {
      const main = document.querySelector('.shell');
      if (!main) return;
      // The justified layout has NO column count: a row fills the
      // width it is given, so the grid's width IS the container's
      // content width, capped at the ladder's 1440px promise. The
      // other two layouts pack fixed-size columns, so their width is
      // a multiple of the tile size.
      //
      // The container width is measured the same way the fetch
      // layer measures it (grid/geometry.ts) — the CONTENT box, not
      // the border box — so the chrome lines up with the rows.
      if (layoutEnabled($preferences) === 'justified') {
        const contentWidth = gridContainerWidth();
        if (contentWidth <= 0) return;
        const cap = JUSTIFIED_GRID_SIZE_CONFIG[$preferences.gridSize].maxWidth;
        document.documentElement.style.setProperty(
          '--grid-width',
          Math.min(contentWidth, cap) + 'px'
        );
        return;
      }
      const containerWidth = main.getBoundingClientRect().width;
      if (containerWidth <= 0) return;
      const cols = columnsForWidth(containerWidth, config);
      const gridWidth = widthForColumns(cols, config);
      document.documentElement.style.setProperty('--grid-width', gridWidth + 'px');
    };
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width <= 0) continue;
        publishGridWidth();
      }
    });
    const main = document.querySelector('.shell');
    if (main) ro.observe(main);
    publishGridWidth();
    const unsub = preferences.subscribe(() => publishGridWidth());
    return () => {
      ro.disconnect();
      unsub();
      // The var describes the app shell's width, and the shell only
      // exists on app routes. Leaving it on :root after the shell is
      // gone would silently constrain the next route's grid to the
      // previous page's width — including the presentation wall,
      // which measures its own wrapper and must fill the viewport.
      document.documentElement.style.removeProperty('--grid-width');
    };
  });
</script>

<AppChrome {children} />
