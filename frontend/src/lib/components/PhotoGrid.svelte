<script lang="ts">
  /**
   * PhotoGrid — the single, canonical photo-tile grid.
   *
   * Round‑27: unified the old SearchGrid + ForYouRow into one
   * component. Used by every page that lays out photo tiles:
   *   - Home page search results
   *   - Home page "For you" row (via ForYouRow wrapper)
   *   - /random, /for-you, /search, /similar/[id], /albums/[id]
   *
   *   - Renders tiles via PhotoTile
   *   - Right-click on a tile opens the ImageContextMenu
   *   - Left-click opens the Lightbox (or runs onPhotoOpen when set)
   *   - Infinite scroll: when the sentinel near the bottom
   *     intersects the viewport, calls onLoadMore
   *   - Virtual scrolling: only renders visible rows for perf;
   *     no‑op when the item count fits in one viewport (the common
   *     case for the Home For-You row of 20 items).
   *
   * All callers share one outer padding token and one gutter token
   * (--s-4 and --grid-gutter), so the grid renders the same on every
   * page. No more divergent copy-paste.
   */
  import { onDestroy } from 'svelte';
  import type { SearchResult } from '$lib/api/endpoints';
  import { createWindowVirtualizer } from '@tanstack/svelte-virtual';
  import type { SvelteVirtualizer } from '@tanstack/svelte-virtual';
  import PhotoTile from './PhotoTile.svelte';
  import ImageContextMenu from './ImageContextMenu.svelte';
  import Lightbox from './Lightbox.svelte';
  import { preferences } from '$lib/stores/preferences';
  import { GRID_SIZE_CONFIG } from '$lib/grid/config';
  import { columnsForWidth } from '$lib/grid/geometry';

  // Use the generated SearchResult shape so `url`, `width`,
  // `height`, `is_disliked` etc. stay in lockstep with the
  // backend. Everything optional because some callers (ForYouRow)
  // pass a strict subset and that's fine.
  type Item = Partial<SearchResult> & { id: string };

  type Props = {
    items: Item[];
    loading?: boolean;
    hasMore?: boolean;
    onLoadMore?: (signal?: AbortSignal) => void;
    onToggleFavorite?: (id: string) => void;
    onDislike?: (id: string) => void;
    /**
     * Optional remove affordance shown as a small − button in the
     * top-right corner of every tile (revealed on hover, or always
     * on touch). Forwarded to PhotoTile. See PhotoTile's docs for
     * the rationale (curating sets: likes, dislikes, albums).
     */
    onRemove?: (id: string) => void;
    /** Tooltip / aria-label for the remove button. */
    removeLabel?: string;
    albums?: { id: number; name: string }[];
    /**
     * Optional override for the default left-click behaviour
     * (open the lightbox). The Similar page uses this to take the
     * user to the dedicated photo page instead of opening a
     * lightbox on top of the dedicated page. The handler receives
     * the clicked item so the caller can decide what to do.
     */
    onPhotoOpen?: (item: Item) => void;
    /**
     * Bindable. When the parent owns the lightbox state (e.g. the
     * ForYouRow wrapper), bind here so clicks reset the parent's
     * lightboxIndex instead of an internal one. If unbound, the
     * component manages its own lightboxIndex internally.
     */
    lightboxIndex?: number | null;
  };

  let {
    items,
    loading = false,
    hasMore = false,
    onLoadMore,
    onToggleFavorite,
    onDislike,
    onRemove,
    removeLabel,
    albums,
    onPhotoOpen,
    lightboxIndex = $bindable<number | null>(null)
  }: Props = $props();

  // Grid config
  //
  // All three numbers (tile, gap, maxCols) come from the user's
  // grid-size preference via GRID_SIZE_CONFIG. Single source of
  // truth — Settings page, the layout-level ResizeObserver, and
  // this component all read the same record so they cannot drift.
  //
  // The +layout.svelte onMount writes the active config to
  // `--grid-tile-size`, `--grid-gutter-size`, and
  // `--grid-row-max-width` on `:root`. The CSS uses those vars;
  // the virtualizer uses the JS derived values directly.
  const config = $derived(GRID_SIZE_CONFIG[$preferences.gridSize]);
  const TILE = $derived(config.tile);
  const GAP = $derived(config.gap);
  const MAX_COLS = $derived(config.maxCols);

  // Number of skeleton placeholder rows rendered while a fetch
  // is in flight. Sized to roughly one viewport at the typical
  // rowHeight. Any less and the placeholder band is shorter
  // than the visible area below the last real row, so the user
  // sees a gap between tiles and skeleton. Any more and the
  // skeleton forces the page to grow when the next page arrives,
  // causing a tiny layout jump.
  const SKELETON_ROWS = 3;
  // Estimated row height — used by the virtualizer before the
  // first measurement lands. The actual rowHeight is
  // tileSize + GAP, but the virtualizer only needs a sane
  // starting value; the ResizeObserver and the $effect below
  // correct to the real value within one frame. Round-36
  // commented 280 as the previous "approximate tile height +
  // gap" — derived now so a different gridSize (which changes
  // the row height) gets a sensible default.
  const ESTIMATED_ROW_HEIGHT = $derived(TILE + GAP);

  // Sensible floor before we know the wrapper width — used on the
  // very first render when both containerWidth and tileSize are 0.
  // Round-36: bumped from 5 to 6 — at the 2400px container cap the
  // new fixed 384px tiles give 6 cols. The ResizeObserver fires
  // within one frame and corrects to the real value before the
  // user notices. Round-83: now derived from MAX_COLS so a
  // smaller gridSize choice (max 10 cols) gets a proportionally
  // larger initial estimate instead of guessing 6.
  const ESTIMATED_COLUMNS = $derived(Math.min(MAX_COLS, 6));

  // State
  let contextMenu = $state<{ x: number; y: number; item: Item } | null>(null);
  // Grid wrapper ref (for width measurement only — not a scroll parent).
  // The body is the scroll context; the virtualizer watches window.
  let gridWrapper: HTMLDivElement | undefined = $state();
  let containerWidth = $state(0);
  // Round‑32: actual rendered tile width (read from DOM). The CSS
  // grid uses auto-fill so the number of columns changes with
  // viewport width — we measure the first tile directly. Default
  // to the previous JS estimate so the virtualizer has a sane
  // starting value before the first measurement lands.
  let renderedTileSize = $state(0);

  // tileSize drives the virtualizer's rowHeight (= tileSize + GAP).
  // The CSS grid decides the actual tile width (auto-fill means it
  // changes with viewport), so we read it directly off a rendered
  // `.grid-tile`. Until the first tile mounts we fall back to TILE
  // (the canonical CSS tile width) — the ResizeObserver fires within
  // one frame and the $effect below re-measures.
  let tileSize = $derived(renderedTileSize > 0 ? renderedTileSize : TILE);
  let rowHeight = $derived(tileSize + GAP);

  // Derive column count from the same math the CSS uses:
  //   containerWidth = COLUMNS * tileSize + (COLUMNS - 1) * GAP
  // Solving for COLUMNS:
  //   COLUMNS = (containerWidth + GAP) / (tileSize + GAP)
  // Falls back to ESTIMATED_COLUMNS on the first frame (before any
  // measurement). Floors to 1 if the wrapper is narrower than a
  // single min-width tile. Kept in sync with `.grid-row`'s
  // `auto-fill minmax(COLUMN_MIN, 1fr)` so the virtualizer's row
  // slicing matches the DOM's actual column count — otherwise the
  // last columns of every row stay empty.
  //
  // columnsForWidth() is the shared implementation the layout's
  // ResizeObserver and the fetch layer also call, so the row count
  // the grid renders, the width the chrome publishes, and the
  // number of photos a request asks for can never drift apart.
  let columns = $derived(
    tileSize > 0 && containerWidth > 0
      ? columnsForWidth(containerWidth, config)
      : ESTIMATED_COLUMNS
  );

  // Group items into rows
  let rows = $derived(
    Array.from({ length: Math.ceil(items.length / columns) }, (_, i) =>
      items.slice(i * columns, (i + 1) * columns)
    )
  );

  // Virtualizer — created once, scrolls with window. Created at
  // module scope so it isn't recreated on every `rows` change
  // (recreating resets scroll position and drops the new items).
  const virtualizerStore = createWindowVirtualizer({
    count: 0,
    estimateSize: () => ESTIMATED_ROW_HEIGHT,
    overscan: 5
  });

  // Stable reference to the virtualizer instance. Don't use $derived
  // here — setOptions forces a store update, which would re-trigger
  // the $effect below and call setOptions again, creating a loop.
  let theVirtualizer: SvelteVirtualizer<Window, Element> | undefined;
  virtualizerStore.subscribe((v) => {
    theVirtualizer = v;
  });

  // Push reactive count/rowHeight into the virtualizer when they change.
  // Note: do NOT read `theVirtualizer` here — only rows.length and
  // rowHeight are the reactive dependencies.
  $effect(() => {
    const n = rows.length;
    const h = rowHeight;
    theVirtualizer?.setOptions({
      count: n,
      estimateSize: () => h
    });
    // setOptions updates the estimate but doesn't always force the
    // virtualizer to recompute cached row offsets. Call measure()
    // explicitly so already-laid-out rows pick up the new rowHeight
    // — without this, the first paint uses the initial fallback
    // (280 + GAP) and rows visually touch each other on first load.
    theVirtualizer?.measure?.();
  });

  // Read virtualItems/totalSize from the store (re-runs when store updates).
  let virtualItems = $derived($virtualizerStore?.getVirtualItems() ?? []);
  let totalSize = $derived($virtualizerStore?.getTotalSize() ?? 0);

  // ResizeObserver for container width.
  //
  // Use $effect instead of onMount: in Svelte 5, bind:this on a
  // {#if}-gated element only populates the variable once the branch
  // renders, which happens AFTER onMount when items load async.
  // The $effect re-runs whenever gridWrapper flips undefined → bound.
  //
  // The second $effect (further down) re-measures the rendered tile
  // whenever items.length changes — the ResizeObserver fires for
  // the wrapper, but `.grid-tile` elements only exist once items
  // load, so we read the tile width on every items change too.
  let resizeObserver: ResizeObserver | null = null;

  $effect(() => {
    if (!gridWrapper) return;
    // Sync current width now that the wrapper is mounted.
    containerWidth = gridWrapper.getBoundingClientRect().width;
    if (resizeObserver) return;
    resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        containerWidth = entry.contentRect.width;
      }
    });
    resizeObserver.observe(gridWrapper);
  });

  // Actual rendered width of the .grid-row inside the wrapper.
  // Equals N×TILE + (N−1)×GAP where N is the column count. The
  // wrapper itself is wider than this (it includes the leftover
  // space when the container can't fit another 384px tile).
  //
  // Note: --grid-width is also computed at the layout level in
  // +layout.svelte so pages without PhotoGrid (e.g. / before
  // any search runs) still have the var set. PhotoGrid's
  // measurement is the source of truth when mounted; the layout
  // fallback handles the empty-items case.
  // Re-measure the rendered tile whenever items change, the
  // container width changes, OR the grid size preference
  // changes. The CSS grid uses auto-fill, so the number of
  // columns (and therefore the tile width) depends on
  // `containerWidth`. The ResizeObserver may fire before the
  // first row renders, so we read the tile on every items change
  // too. When the user changes grid size in Settings, the CSS
  // var update flows through the layout's onMount (which writes
  // the new tile size to :root), then the grid re-flows, then
  // we re-measure so the virtualizer's rowHeight matches the
  // new tile size.
  $effect(() => {
    // Track dependencies.
    void items.length;
    void containerWidth;
    void $preferences.gridSize;
    if (!gridWrapper) return;
    // Defer one tick so the DOM has rendered the new tiles.
    queueMicrotask(() => {
      const firstTile = gridWrapper!.querySelector('.grid-tile');
      if (firstTile) {
        const r = firstTile.getBoundingClientRect();
        if (r.width > 0) renderedTileSize = r.width;
      }
    });
  });

  onDestroy(() => {
    resizeObserver?.disconnect();
  });

  // Infinite scroll — pre-fetch ahead.
  //
  // The user-visible "batch just dropped in" feel comes from a
  // fetch that starts AFTER they reach the bottom. We instead
  // trigger when the rendered virtual range is within `PRE_FETCH_AHEAD`
  // viewports of the total — by the time the user scrolls there,
  // the next page is already in `items`. The existing bottom
  // sentinel stays as a backstop (handles the edge case where the
  // virtualizer doesn't render the trigger row because of its
  // overscan estimate).
  //
  // The AbortSignal lets the caller cancel an in-flight pre-fetch
  // if the user scrolls back fast — a stale page-2 landing on
  // top of items that have since grown would be jarring.
  const PRE_FETCH_AHEAD = 2; // viewports
  let sentinel: HTMLDivElement | undefined = $state();
  let observer: IntersectionObserver | null = null;
  let preFetchController: AbortController | null = null;

  $effect(() => {
    if (sentinel && hasMore && onLoadMore && !loading) {
      observer?.disconnect();
      observer = new IntersectionObserver(
        (entries) => {
          if (entries[0].isIntersecting) {
            preFetchController?.abort();
            preFetchController = new AbortController();
            onLoadMore?.(preFetchController.signal);
          }
        },
        { root: null, rootMargin: '800px 0px', threshold: 0 }
      );
      observer.observe(sentinel);
    }
    return () => observer?.disconnect();
  });

  // Virtualizer-driven pre-fetch trigger. Runs when virtualItems
  // or rowHeight changes; recomputes whether the rendered tail is
  // close enough to the end to warrant fetching the next page.
  // Skipped while already loading (avoids piling up fetches) and
  // skipped when there's no more data.
  $effect(() => {
    if (!hasMore || loading || !onLoadMore) return;
    if (typeof window === 'undefined') return; // SSR
    const totalRows = rows.length;
    if (totalRows === 0) return;
    const vis = virtualItems;
    if (vis.length === 0) return;
    const lastRenderedRow = vis[vis.length - 1].index;
    const rowsPerViewport = Math.max(
      1,
      Math.floor(window.innerHeight / rowHeight)
    );
    const triggerAt = totalRows - PRE_FETCH_AHEAD * rowsPerViewport;
    if (lastRenderedRow >= triggerAt) {
      preFetchController?.abort();
      preFetchController = new AbortController();
      onLoadMore?.(preFetchController.signal);
    }
  });

  // Tile interactions
  function openLightbox(itemIndex: number) {
    lightboxIndex = itemIndex;
  }

  function closeLightbox() {
    lightboxIndex = null;
  }

  // Round-37: removed the per-scroll and per-lightbox pageTint
  // effects. They were pushing the first-visible row's blurhash
  // (and the lightbox-active photo's URL) to a global backdrop
  // element on every IntersectionObserver tick, which caused two
  // visible problems on /random, /for-you, and any grid feed:
  //   - the backdrop colour shifted continuously as the user
  //     scrolled past photos (the anchor row changed, the wash
  //     followed);
  //   - the photo's saturated regions bled through the 90px blur
  //     as radiating colour bands.
  // Per-tile colour still comes from `.glass-tint::before`; the
  // page-level backdrop is now a flat dark base owned by +layout.
  // Per-panel ambient colour inside the Lightbox is owned by the
  // Lightbox component itself (the `.tint` rule), not by a
  // page-wide backdrop.

  function openContextMenu(item: Item, e: MouseEvent) {
    e.preventDefault();
    contextMenu = { x: e.clientX, y: e.clientY, item };
  }

  function closeContextMenu() {
    contextMenu = null;
  }
</script>

{#if loading && items.length === 0}
  <div class="empty loading">
    <div class="spinner" aria-label="Loading results"></div>
    <span>Searching...</span>
  </div>
{:else if items.length === 0 && !loading}
  <div class="empty">No results</div>
{:else}
  <div class="grid-wrapper" data-grid-size={$preferences.gridSize} bind:this={gridWrapper}>
    <div
      class="grid-virtual"
      style="height: {totalSize}px; position: relative;"
    >
      {#each virtualItems as virtualRow, vRowIdx (virtualRow.key)}
        <div
          class="grid-row"
          style="position: absolute; top: 0; left: 0; width: 100%; height: {virtualRow.size}px; transform: translateY({virtualRow.start}px);"
        >
          {#each rows[virtualRow.index] as item, colIndex (item.id)}
            {@const itemIndex = virtualRow.index * columns + colIndex}
            <!-- Tier 1 of 3 (perf round 1): tag the first three tiles
                 of the first visible row as eager so the browser
                 starts fetching them with the rest of the HTML. The
                 virtualizer guarantees this row is in the viewport
                 (overscan ≥ 5 covers the first row even when scrolled). -->
            {@const eagerIndex = vRowIdx === 0 ? colIndex : null}
            <div class="grid-tile">
              <PhotoTile
                pointId={item.id}
                blurhash={item.blurhash}
                scoreStr={item.score_str}
                isFavorite={item.is_favorite}
                isDisliked={item.is_disliked}
                {eagerIndex}
                onOpen={onPhotoOpen ? () => onPhotoOpen(item) : () => openLightbox(itemIndex)}
                onContextMenu={(id, e) => openContextMenu(item, e)}
                onRemove={onRemove}
                removeLabel={removeLabel}
              />
            </div>
          {/each}
        </div>
      {/each}
    </div>

    {#if hasMore}
      <div class="sentinel" bind:this={sentinel}></div>
      {#if loading}
        <!--
          Skeleton placeholder rows. Sit in normal flow just below
          the virtualized grid so the user sees tile-shaped
          placeholders in the gap between when the pre-fetch
          fires and when the next page's data arrives. Without
          these, the user sees a partially-filled last row + a
          large expanse of empty space, then a sudden "dump" of
          new tiles — the "loads in at each scroll" perception.

          The number of rows (SKELETON_ROWS) is sized so the
          placeholder band is roughly one viewport tall — enough
          to cover the visible area below the last rendered row
          without being so tall it would force-scroll the page.

          Rows use the same `.grid-row` class so the column
          structure (auto-fill, 384px tracks, 4px gutter,
          centered) matches the real grid exactly. Tiles inherit
          `.grid-tile`'s aspect-ratio: 1 and become the shimmer
          surface via `.skeleton-tile`.
        -->
        <div class="skeleton-rows" aria-hidden="true">
          {#each Array(SKELETON_ROWS) as _, sr (sr)}
            <div class="grid-row skeleton-row">
              {#each Array(columns) as _, sc (sc)}
                <div class="grid-tile skeleton-tile"></div>
              {/each}
            </div>
          {/each}
        </div>
      {/if}
    {/if}
  </div>
{/if}

{#if lightboxIndex !== null && lightboxIndex < items.length}
  <Lightbox
    items={items.map((it) => ({
      id: it.id,
      blurhash: it.blurhash,
      isFavorite: it.is_favorite,
      isDisliked: it.is_disliked
    }))}
    index={lightboxIndex}
    onClose={closeLightbox}
    onToggleFavorite={onToggleFavorite}
    onDislike={onDislike}
    albums={albums}
    {hasMore}
    {onLoadMore}
  />
{/if}

{#if contextMenu}
  <ImageContextMenu
    x={contextMenu.x}
    y={contextMenu.y}
    pointId={contextMenu.item.id}
    path={contextMenu.item.path}
    isFavorite={contextMenu.item.is_favorite}
    albums={albums}
    onClose={closeContextMenu}
    onToggleFavorite={onToggleFavorite}
  />
{/if}

<style>
  .grid-wrapper {
    /* Body is the scroll context. The grid is in normal flow; only
       virtual rows the user are looking at are rendered.

       Width mirrors the PageHeader above (var(--grid-width) =
       cols * tileSize + gaps, capped). Without this constraint
       the wrapper fills the page container (2352px at 2510
       viewport) and the grid edges bleed 14px past the header on
       each side. */
    width: var(--grid-width, 100%);
    max-width: 100%;
    margin: 0 auto;
  }

  .grid-virtual {
    width: 100%;
  }

  .grid-row {
    display: grid;
    /* Round-36: fixed-tile-width grid (was auto-fill minmax(180, 1fr)
       → 8 cols at ~183px). The tile size comes from the user's
       grid-size preference (--grid-tile-size written by
       +layout.svelte's ResizeObserver from GRID_SIZE_CONFIG).
       auto-fit (not auto-fill) collapses empty tracks when the
       item count is below the column count, so sparse pages
       (For You with 2 photos) center the filled tiles rather
       than leaving an empty band on the right. Dense pages
       (Random with 40+ photos) hit auto-fill's full-track
       behavior because all tracks are occupied — same look as
       before. */
    grid-template-columns: repeat(auto-fit, var(--grid-tile-size, 240px));
    /* Center the row when the container is wider than the tile
       grid (the common case on viewports >1548px). Without this,
       CSS Grid's default `justify-content: start` packs cols to
       the left and leaves an empty band on the right — looks like
       a layout bug. Centering reads as deliberate. */
    justify-content: center;
    /* Gap follows the same preference as the tile size — see
       GRID_SIZE_CONFIG. Larger photos get a larger gap (the
       Round-38 design intent: 14px reads as 'personal photo
       library', tighter gaps read as utilitarian). The JS-side
       GAP constant in this file's <script> matches this value
       exactly so the virtualizer's rowHeight math is right. */
    gap: var(--grid-gutter-size, 14px);
    /* Cap the row at the max-cols width. Without this, on a 2400px
       viewport with the small setting (10 cols × 144 + 9 × 4 =
       1476px), the row would expand to fill the available width
       because the `auto-fit` tracks grow with `1fr`. The cap
       forces the row to stay at its natural width and the
       `justify-content: center` above centers it. */
    max-width: var(--grid-row-max-width, 1548px);
    /* No horizontal padding here — `.shell` already provides
       24px of side padding, and the row belongs to the wrapper
       which fills main's content area. Adding more here would
       double-inset the grid (48px total side padding) and make
       it look left-shifted on wide viewports where the wrapper
       is narrower than the viewport. */
  }

  .grid-tile {
    aspect-ratio: 1;
    min-width: 0;
    /* Round‑36: CSS containment isolates each tile's size, layout,
       paint, and style from the rest of the grid. With a wall
       of 100+ tiles on screen, this prevents the browser from
       re-laying-out the whole grid when one tile's contents
       change (e.g. blurhash decoded, hover chrome mounted).
       No visual cost; small paint-side memory bump. */
    contain: size layout paint style;
  }

  .empty {
    /* Width mirrors the PageHeader above (var(--grid-width) =
       cols * tileSize + gaps, capped). Without this the
       placeholder fills the page container (2352px at 2510
       viewport) and its edges bleed 14px past the header on
       each side. */
    width: var(--grid-width, 100%);
    max-width: 100%;
    margin: 0 auto;
    padding: var(--s-6, 48px) var(--s-4, 24px);
    color: var(--fg-3);
    font-size: var(--fs-sm);
  }

  .sentinel {
    padding: var(--s-4, 24px);
    text-align: center;
  }

  .loading {
    color: var(--fg-3);
    font-size: var(--fs-sm);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--s-3, 12px);
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--glass-edge);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* Skeleton placeholder rows shown while the next page is
     fetching. Sits in normal flow below the virtualized grid so
     the visual height of the grid-wrapper grows during the fetch
     gap. Tiles inherit .grid-tile's aspect-ratio: 1 and pick up
     the .grid-row column structure (auto-fill 384px, 4px gap,
     centered), so the placeholder band aligns pixel-for-pixel
     with the real tile grid above it.

     The shimmer is a diagonal gradient sweep across each tile.
     1.5s duration is slow enough to feel like a deliberate
     affordance, fast enough that a sub-second fetch doesn't
     show the animation in a way that suggests slowness. */
  .skeleton-rows {
    padding-top: 4px;
  }
  .skeleton-row {
    /* Inherits .grid-row's grid-template-columns, gap, and
       justify-content. No additional sizing needed. The
       margin-bottom matches the row gap so the shimmer
       band has the same vertical rhythm as the real grid. */
    margin-bottom: var(--grid-gutter-size, 14px);
    pointer-events: none;
  }
  .skeleton-tile {
    /* Inherits aspect-ratio: 1 from .grid-tile. */
    background: var(--skeleton-base);
    border-radius: var(--r-md, 8px);
    overflow: hidden;
    position: relative;
  }
  .skeleton-tile::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(
      100deg,
      var(--skeleton-track) 20%,
      var(--skeleton-peak) 50%,
      var(--skeleton-track) 80%
    );
    background-size: 200% 100%;
    animation: skeleton-shimmer 1.5s infinite linear;
  }
  @keyframes skeleton-shimmer {
    0% {
      background-position: 100% 0;
    }
    100% {
      background-position: -100% 0;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .skeleton-tile::after {
      animation: none;
    }
  }

  /* No dim-on-hover effect (Round‑36 removed).
     The row-level dim felt like a UI interruption on the dense
     6×4K grid — non-hovered tiles already have their own hover
     state (scale + border), so dimming siblings added noise
     without conveying useful info. */
</style>
