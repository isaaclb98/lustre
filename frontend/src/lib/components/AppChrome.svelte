<script lang="ts">
  /**
   * AppChrome — the navigation and app furniture every app route
   * sits inside.
   *
   * The chrome lives in a component rather than in a layout so
   * that more than one surface can carry it. The app routes use
   * the (app) layout; the 404 page is outside that group — an
   * unmatched URL matches no route at all, so it renders at the
   * root — and it still needs the top bar, or a mistyped URL
   * strands the user on a page with no way back.
   *
   * The presentation wall deliberately carries NONE of this. It
   * resets to the root layout, which has no chrome at all, so the
   * wall's tree never contains the top bar, the shell or the
   * floating scroll button.
   */
  import TopBar from './TopBar.svelte';
  import ScrollToTop from './ScrollToTop.svelte';
  import Dialog from './Dialog.svelte';

  let { children }: { children?: import('svelte').Snippet } = $props();
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
