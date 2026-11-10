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
  import TopBar from '$lib/components/TopBar.svelte';
  import ScrollToTop from '$lib/components/ScrollToTop.svelte';
  import Dialog from '$lib/components/Dialog.svelte';
  import { onMount } from 'svelte';
  import { preferences } from '$lib/stores/preferences';
  import { GRID_SIZE_CONFIG } from '$lib/grid/config';
  import { columnsForWidth, widthForColumns } from '$lib/grid/geometry';

  let { children } = $props();

  const config = $derived(GRID_SIZE_CONFIG[$preferences.gridSize]);

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
  //   N = min(MAX_COLS, floor((containerWidth + GAP) / (TILE + GAP)))
  //   gridWidth = N * TILE + (N - 1) * GAP
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
