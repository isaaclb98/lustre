<script lang="ts">
  /**
   * PageHeader — the standard title bar used on the result-grid
   * pages (/random, /similar, /albums, /albums/favourites,
   * /albums/favourites, /albums/[id]).
   *
   * Single source of truth for the page-header pattern. Every
   * page passes a subtitle — title-only headers render ~23px
   * shorter and break the card rhythm across pages. Variants:
   *   - title + subtitle    — random, similar, home, settings
   *   - title + actions     — albums (+ New album button)
   *   - title + meta        — albums/favourites ("N photos")
   *   - title + all of the above — albums/[id] (zip download)
   *
   * Title can be a plain string OR a snippet (for cases that need
   * an inline icon, like the heart on /albums/favourites).
   *
   * Layout: title block on the left (h1 + optional subtitle/meta),
   * actions slot on the right. When actions is empty, the title
   * block fills the row. The header is wrapped in `.glass` for
   * the standard tinted-panel look matching the rest of the app.
   */
  import type { Snippet } from 'svelte';

  type Props = {
    /** Page title — string for the common case, snippet for icon+text. */
    title: string | Snippet;
    /** Optional one-line subtitle under the title. */
    subtitle?: string;
    /** Optional small meta line under the subtitle (e.g. "182 photos"). */
    meta?: string;
    /** Optional right-side action area (buttons, links). */
    actions?: Snippet;
  };

  let { title, subtitle, meta, actions }: Props = $props();
</script>

<section class="head glass">
  <div class="text">
    <h1>
      {#if typeof title === 'string'}{title}{:else}{@render title()}{/if}
    </h1>
    {#if subtitle}<p>{subtitle}</p>{/if}
    {#if meta}<p class="meta">{meta}</p>{/if}
  </div>
  {#if actions}
    <div class="actions">
      {@render actions()}
    </div>
  {/if}
</section>

<style>
  .head {
    /* Standard page-header geometry. Aligned to the photo grid
       below via --grid-width so the chrome edges line up. Round-1
       polish: tightened padding (was 22/26) and dropped top/bottom
       margins — the surrounding .shell owns vertical rhythm via
       flex gap, so the header itself should be flush with the
       shell padding.

       margin: 0 auto centers inside .shell so the header card
       aligns with TopBar.bar and PhotoGrid.grid (both also
       width: var(--grid-width) + margin: 0 auto). Without auto
       margins the .head sits flush-left of .shell — at wide
       viewports this puts the page-title card visibly to the
       left of the brand mark and tabs above it. */
    margin: 0 auto;
    padding: var(--page-header-pad);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--s-4);
    width: var(--grid-width, 100%);
    max-width: 100%;
  }
  .text {
    flex: 1;
    min-width: 0;
  }
  h1 {
    margin: 0;
    /* Round-1 polish: was --fs-2xl (28px), now --fs-xl (22px) —
       less demo-y, more "shipping product". */
    font-size: var(--fs-xl);
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--fg-1);
  }
  p {
    margin: var(--s-0) 0 0;
    color: var(--fg-2);
    /* Round-1 polish: subtitle one step smaller so the h1/subtitle
       hierarchy reads cleaner. */
    font-size: var(--fs-sm);
  }
  .meta {
    color: var(--fg-3);
    font-size: var(--fs-sm);
  }
  .actions {
    flex-shrink: 0;
  }
  @media (max-width: 640px) {
    /* Drop the --grid-width constraint on mobile — it derives
       from the photo-grid column math (1 column at 375px
       viewport = 240px), which makes the header card look
       pinched on phones. The grid below still aligns because
       the cards inside use their own 260px auto-fit track. */
    .head {
      width: 100%;
    }
    /* When the header card is full-width on mobile, the actions
       slot + title block compete for horizontal space and the
       subtitle wraps into 10+ narrow lines (see Albums mobile
       screenshot, 2026-09-29). Stack them vertically so the
       title takes the full width and the actions sit below. */
    .head {
      flex-direction: column;
      align-items: stretch;
      gap: var(--s-3);
    }
  }
</style>