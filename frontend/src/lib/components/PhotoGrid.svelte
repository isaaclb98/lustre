<script lang="ts">
  /**
   * PhotoGrid — the single, canonical photo-tile grid.
   *
   * Round‑27: unified the old SearchGrid into one
   * component. Used by every page that lays out photo tiles:
   *   - Home page search results
   *   - Home page result rows
   *   - /random, /search, /similar/[id], /albums/[id]
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
  import { preferences, layoutEnabled } from '$lib/stores/preferences';
  import { thumbnailSize } from '$lib/layout';
  import {
    gridConfigFor,
    justifiedConfigFor,
    type GridSizeConfig
  } from '$lib/grid/config';
  import { columnsForWidth, wallColumnsForWidth } from '$lib/grid/geometry';
  import { isClamped, layoutMasonry } from '$lib/grid/masonry';
  import { layoutJustified } from '$lib/grid/justified';

  // Use the generated SearchResult shape so `url`, `width`,
  // `height`, `is_disliked` etc. stay in lockstep with the
  // backend. Everything optional because some callers
  // pass a strict subset and that's fine.
  type Item = Partial<SearchResult> & { id: string };

  type Props = {
    items: Item[];
    loading?: boolean;
    hasMore?: boolean;
    onLoadMore?: (signal?: AbortSignal) => void;
    onToggleFavorite?: (id: string) => void;
    /**
     * Optional remove affordance shown as a small − button in the
     * top-right corner of every tile (revealed on hover, or always
     * on touch). Forwarded to PhotoTile. See PhotoTile's docs for
     * the rationale (curating sets: likes, dislikes, albums).
     */
    onRemove?: (id: string) => void;
    /** Tooltip / aria-label for the remove button. */
    removeLabel?: string;
    /**
     * Vote affordance — the Explore page's per-tile up/down pair.
     * See PhotoTile's `onVote` for the semantics.
     */
    onVote?: (id: string, direction: 'up' | 'down') => void;
    /** Photo ids that were up-voted — highlights their up chip. */
    upVoted?: Set<string>;
    /** Photo ids that were down-voted — highlights their down chip. */
    downVoted?: Set<string>;
    /**
     * Label on the up-vote chip. Forwarded to PhotoTile; defaults
     * to "Add to album".
     */
    voteUpLabel?: string;
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
     * wrapper), bind here so clicks reset the parent's
     * lightboxIndex instead of an internal one. If unbound, the
     * component manages its own lightboxIndex internally.
     */
    lightboxIndex?: number | null;
    /**
     * Optional override for the grid's geometry (tile size, gap,
     * column cap). When absent the grid uses the user's grid-size
     * preference — the app-wide behaviour every route relies on.
     *
     * The presentation wall passes one to get square tiles with no
     * gutter. It has to be a prop rather than a CSS-only change
     * because the gap is not just CSS: the virtualizer's row height
     * is tileSize + GAP and the column count comes from
     * columnsForWidth(), both read from the same config. If the CSS
     * rendered a zero-gap grid while the JS still assumed a 14px
     * one, every row would be 14px taller than it renders and the
     * scroll would drift.
     */
    tileConfig?: GridSizeConfig;
    /**
     * Render the tiles as a continuous wall: square corners and no
     * gutter. Forwarded to PhotoTile, which drops the tile border
     * and moves the state rings to a negative-offset outline so the
     * photos touch pixel-to-pixel.
     */
    wall?: boolean;
    /**
     * A read-only surface: the tiles are pictures, not targets.
     *
     * Suppresses everything the grid would otherwise put on a
     * tile — the lightbox, the right-click context menu — so a
     * surface that must have no controls and no way to mutate
     * anything cannot leak one in through the shared grid
     * component.
     */
    readOnly?: boolean;
  };

  let {
    items,
    loading = false,
    hasMore = false,
    onLoadMore,
    onToggleFavorite,
    onRemove,
    removeLabel,
    onVote,
    upVoted,
    downVoted,
    voteUpLabel,
    albums,
    onPhotoOpen,
    lightboxIndex = $bindable<number | null>(null),
    tileConfig,
    wall = false,
    readOnly = false
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
  //
  // The grid has no gutter — its photos touch, like the
  // presentation wall's do —
  // so the gap the virtualizer assumes has to be the gap the CSS
  // renders, or every row would be the wrong height and the scroll
  // would drift.
  // Masonry has its own size ladder (MASONRY_GRID_SIZE_CONFIG): a
  // masonry tile is not cropped to a square, so the same width
  // shows a smaller photo and the ladder starts one rung higher.
  //
  // A caller that passes its own tileConfig — the presentation wall
  // — keeps that config entirely: the wall is a square, gapless
  // field by design, so it renders from the square ladder whatever
  // the deployment's layout is.
  //
  // The wall is exempt from the self-laid-out layouts for the same
  // reason it is exempt from them as a LAYOUT: it is a square field
  // by design.
  const mode = $derived(layoutEnabled($preferences));
  const masonry = $derived(mode === 'masonry' && !wall);
  const justified = $derived(mode === 'justified' && !wall);

  const config = $derived(
    tileConfig ?? gridConfigFor($preferences.gridSize, masonry)
  );

  // The justified layout's own ladder (JUSTIFIED_GRID_SIZE_CONFIG): a
  // row HEIGHT instead of a tile WIDTH, because every photo in a row
  // has its own width. The row WIDTH is the grid's own width, capped
  // at the ladder's promise — the same 1440px the other two ladders
  // keep, so the grid does not sprawl across a wide monitor.
  //
  // The ladder's numbers are the same as the masonry's (360 / 240 /
  // 180), which is what makes the two layouts comparable: the same
  // density of photos, a different arrangement.
  const justConfig = $derived(justifiedConfigFor($preferences.gridSize));
  const JUST_ROW_HEIGHT = $derived(justConfig.rowHeight);
  const JUST_MAX_WIDTH = $derived(justConfig.maxWidth);

  // The size the CSS and the skeleton size themselves against.
  //
  // For the uniform grid and the masonry that is a tile WIDTH; for
  // the justified rows it is the row HEIGHT, which is the same
  // quantity: how big a photo appears.
  const TILE = $derived(justified ? JUST_ROW_HEIGHT : config.tile);
  const MAX_COLS = $derived(justified ? justConfig.cols : config.maxCols);

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
  const ESTIMATED_ROW_HEIGHT = $derived(TILE);

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
  let rowHeight = $derived(tileSize);

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
  //
  // The wall asks a different question — how many columns divide
  // the viewport evenly — so its tracks stretch to fill the screen
  // instead of leaving a band of background beside them. The count
  // is published as --grid-cols and the CSS uses it directly, so
  // the tracks the browser lays out and the rows the virtualizer
  // slices are the same rows by construction.
  let columns = $derived(
    justified
      ? // The justified layout has no columns: a row holds however
        // many photos fit its height, and that count depends on the
        // photos' own ratios. The ladder's figure is the AVERAGE —
        // it is what the request size and the loading skeleton
        // need, and nothing else.
        justConfig.cols
      : tileSize > 0 && containerWidth > 0
        ? wall
          ? Math.min(
              wallColumnsForWidth(containerWidth, config),
              // The wall fills the screen with the photos it has.
              // An album with fewer photos than the viewport has
              // columns would leave empty tracks at the right, and a
              // strip of page background beside the photos is
              // exactly what the wall is not allowed to have.
              Math.max(1, items.length)
            )
          : columnsForWidth(containerWidth, config)
        : ESTIMATED_COLUMNS
  );

  // Group items into rows
  let rows = $derived(
    Array.from({ length: Math.ceil(items.length / columns) }, (_, i) =>
      items.slice(i * columns, (i + 1) * columns)
    )
  );

  // ----- The self-laid-out layouts -----
  //
  // The uniform grid slices items into rows of a FIXED height and
  // lets the virtualizer render the rows in view. The other two
  // layouts cannot be expressed that way — every tile has its own
  // size — so the layout is computed here and the visible slice is
  // a filter over the computed placements.
  //
  // The tile sizes are known before the images load: the API
  // returns each photo's own width and height, so a tile can be
  // sized in the same paint as the layout and nothing reflows when a
  // thumbnail arrives.
  //
  // The presentation wall is EXEMPT from both — it is a gapless
  // square field by design, so neither layout must touch it. The
  // wall's own tile config (square corners, no gutter) still
  // applies.
  const masonryLayout = $derived(
    masonry ? layoutMasonry(items, { tile: TILE, columns }) : null
  );

  // The justified rows — the layout Immich uses. Every ROW has one
  // height and the photos in it are scaled to fill the row's width
  // exactly, so the row's right edge is flush with the grid's.
  //
  // The row width is the grid's own width, capped at the ladder's
  // promise. The row HEIGHT is the target: the algorithm lets each
  // row's height stray from it within the tolerance, because the
  // photos' own ratios decide how tall a row of them has to be to
  // fill the width.
  //
  // containerWidth is the WRAPPER's width, which the layout sizes
  // from --grid-width. In the justified layout --grid-width is the
  // container's own width (a justified row fills whatever width it
  // is given), so the row width is the container width — capped at
  // the ladder's 1440px promise so the grid does not sprawl on a
  // wide monitor.
  const justRowWidth = $derived(
    containerWidth > 0
      ? Math.min(containerWidth, JUST_MAX_WIDTH)
      : JUST_MAX_WIDTH
  );
  const justifiedLayout = $derived(
    justified
      ? layoutJustified(items, {
          rowWidth: justRowWidth,
          rowHeight: JUST_ROW_HEIGHT
        })
      : null
  );

  // The visible band, in the grid's own coordinates. The grid's top
  // edge sits at -rect.top in viewport coordinates, so what the user
  // can see runs from -rect.top to -rect.top + the viewport height.
  const BAND_OVERSCAN = 600;
  let visibleBand = $state({ top: 0, bottom: 0 });

  function updateVisibleBand() {
    if (!gridWrapper) return;
    const rect = gridWrapper.getBoundingClientRect();
    const viewportHeight =
      typeof window === 'undefined' ? 0 : window.innerHeight;
    visibleBand = {
      top: -rect.top - BAND_OVERSCAN,
      bottom: -rect.top + viewportHeight + BAND_OVERSCAN
    };
  }

  // The items in the visible band, with their placements.
  //
  // The masonry's placements are NOT sorted by y — the
  // shortest-column packing puts a photo in whichever column is
  // lowest, so a later photo can sit higher than an earlier one.
  // The filter therefore scans every placement rather than stopping
  // at the first one below the band.
  const masonryItems = $derived.by(() => {
    const layout = masonryLayout;
    if (!layout) return [];
    const out: {
      item: Item;
      index: number;
      x: number;
      y: number;
      width: number;
      height: number;
      clamped: boolean;
    }[] = [];
    for (let i = 0; i < items.length; i++) {
      const placement = layout.placements[i];
      if (!placement) break;
      if (placement.y + placement.height < visibleBand.top) continue;
      if (placement.y > visibleBand.bottom) continue;
      out.push({
        item: items[i],
        index: i,
        ...placement,
        // A photo whose ratio is outside the layout's bounds is laid
        // out in a box that is not its shape, so the tile shows it
        // whole rather than cropping it.
        clamped: isClamped(items[i])
      });
    }
    return out;
  });

  // The justified rows in the visible band.
  //
  // The rows ARE sorted by their top edge, so the scan can stop at
  // the first row below the band — everything after it is lower
  // still. The masonry's filter cannot do that, which is why the
  // justified rows are cheaper to slice.
  const justifiedRows = $derived.by(() => {
    const layout = justifiedLayout;
    if (!layout) return [];
    const out: {
      top: number;
      height: number;
      width: number;
      start: number;
      tiles: {
        item: Item;
        index: number;
        x: number;
        width: number;
      }[];
    }[] = [];
    for (const row of layout.rows) {
      if (row.top > visibleBand.bottom) break;
      if (row.top + row.height < visibleBand.top) continue;
      const tiles: {
        item: Item;
        index: number;
        x: number;
        width: number;
      }[] = [];
      for (let i = row.start; i < row.start + row.count; i++) {
        const placement = layout.placements[i];
        if (!placement) break;
        tiles.push({
          item: items[i],
          index: i,
          x: placement.x,
          width: placement.width
        });
      }
      out.push({
        top: row.top,
        height: row.height,
        width: row.width,
        start: row.start,
        tiles
      });
    }
    return out;
  });

  // Keep the visible band in step with the scroll.
  //
  // The uniform grid gets this from the virtualizer, which watches
  // the window for itself. The self-laid-out layouts have no
  // virtualizer, so the grid listens — the same window scroll the
  // virtualizer would have used — and recomputes the band on the
  // next frame.
  $effect(() => {
    if (!masonry && !justified) return;
    void gridWrapper;
    let frame = 0;
    const onScroll = () => {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        updateVisibleBand();
      });
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    updateVisibleBand();
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      if (frame) cancelAnimationFrame(frame);
    };
  });

  // The three numbers the CSS grid renders from, re-published on
  // this grid's own wrapper so an override applies HERE and nowhere
  // else. The layout publishes the app-wide values on :root from
  // the user's preference; when a caller passes its own config
  // (the presentation wall), the row's gap, tile size and max width
  // must follow the OVERRIDE instead — otherwise the CSS would
  // render a zero-gap wall while the virtualizer still assumed the
  // preference's 14px, and every row would be the wrong height.
  //
  // max-width mirrors the layout's formula: MAX_COLS * TILE +
  // (MAX_COLS - 1) * GAP.
  //
  // --grid-cols is the COLUMN COUNT the virtualizer slices rows by.
  // The wall's CSS uses it directly (rather than letting the browser
  // count tracks itself) so the two can never disagree: if the CSS
  // rendered a different number of tracks than the JS assumed, the
  // last columns of every row would stay empty.
  // In the justified layout there is no fixed tile WIDTH — every
  // photo in a row has its own — so the size knob is the row
  // HEIGHT, and it is published under the same var the other two
  // layouts publish their tile width under. The two ladders use the
  // same numbers (360 / 240 / 180), so the layouts are directly
  // comparable: the same density, a different arrangement.
  let rowMaxWidth = $derived(justified ? JUST_MAX_WIDTH : MAX_COLS * TILE);
  let gridVars = $derived(
    `--grid-tile-size:${TILE}px;--grid-row-max-width:${rowMaxWidth}px;--grid-cols:${columns};` +
      // The wall's tiles are square. Published on the wrapper so the
      // tile and the loading skeleton — which read the var — both
      // pick it up, and only for this grid.
      //
      // The wall also drops the row's max-width cap. The cap exists
      // so the app grid keeps its natural width on a wide monitor
      // (and centres the leftover); a presentation has no such
      // business — the photos fill the screen.
      (wall ? '--grid-tile-radius:0px;--grid-row-max-width:none;' : '')
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
    // The self-laid-out layouts position the tiles themselves, so
    // the row virtualizer has nothing to slice — it renders nothing
    // and costs nothing. The justified rows cannot use it anyway:
    // every row has its own height, and the virtualizer assumes
    // they are all the same.
    const n = masonry || justified ? 0 : rows.length;
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
    if (masonry || justified) {
      // The self-laid-out layouts have no rows to count: the question
      // is how much of the laid-out grid is still below the fold.
      const layout = masonry ? masonryLayout : justifiedLayout;
      if (!layout) return;
      const rect = gridWrapper?.getBoundingClientRect();
      if (!rect) return;
      // How much of the laid-out grid is still below the fold.
      //
      // `rect.top` is VIEWPORT-relative and `layout.height` is the
      // grid's own height, so `rect.top + layout.height` is the
      // grid's bottom in viewport coordinates. The viewport's own
      // bottom is `window.innerHeight` in that same space — and the
      // scroll terms cancel, because the grid's document offset
      // (rect.top + scrollY) and the viewport's (scrollY +
      // innerHeight) both carry the same scrollY.
      //
      // Comparing against `scrollY + innerHeight` instead would mix
      // the two spaces: that understates the remaining gap by
      // exactly the scroll distance, so the further down the page
      // the user is, the sooner the next page is fetched. Measured
      // at the bottom of a page: 3282px of gap reported against
      // 5976px of actual gap — thirteen viewports of content left
      // while the grid thought there were two.
      const remaining = rect.top + layout.height - window.innerHeight;
      if (remaining < PRE_FETCH_AHEAD * window.innerHeight) {
        preFetchController?.abort();
        preFetchController = new AbortController();
        onLoadMore?.(preFetchController.signal);
      }
      return;
    }
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

  /**
   * What a tile click and a right-click do.
   *
   * On a read-only surface both are undefined: there is no
   * lightbox to open (the lightbox carries an action bar with
   * favourite / dislike / add-to-album) and no context menu (it
   * carries the same mutations). The tile renders as a plain
   * picture, so nothing on the surface can be changed.
   */
  function tileOpen(item: Item, itemIndex: number) {
    if (readOnly) return undefined;
    return onPhotoOpen ? () => onPhotoOpen(item) : () => openLightbox(itemIndex);
  }

  function tileContextMenu(item: Item) {
    if (readOnly) return undefined;
    return (_id: string, e: MouseEvent) => openContextMenu(item, e);
  }

  /**
   * The accessible name of a read-only tile.
   *
   * An interactive tile is named "Open photo" — the name of the
   * action. A read-only tile has no action, so the name has to
   * identify the PHOTO instead, or a screen reader announces a
   * grid of anonymous images. The file name is the only identity
   * the photo has.
   */
  function tileLabel(item: Item): string | undefined {
    if (!readOnly) return undefined;
    const name = item.path?.split('/').pop();
    return name || undefined;
  }

  // Round-37: removed the per-scroll and per-lightbox pageTint
  // effects. They were pushing the first-visible row's blurhash
  // (and the lightbox-active photo's URL) to a global backdrop
  // element on every IntersectionObserver tick, which caused two
  // visible problems on /random and any grid feed:
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
  <div
    class="grid-wrapper"
    class:wall
    data-grid-size={$preferences.gridSize}
    style={gridVars}
    bind:this={gridWrapper}
  >
    <div
      class="grid-virtual"
      style="height: {masonry
        ? (masonryLayout?.height ?? 0)
        : justified
          ? (justifiedLayout?.height ?? 0)
          : totalSize}px; position: relative;"
    >
      {#if justified}
        <!-- The justified rows — the layout Immich uses.
             Each row is one height and its photos are scaled to fill
             the row's width exactly, so the row's right edge is flush
             with the grid's. The grid positions the tiles, so a row
             is a plain box and the photos inside it are placed by
             their own x offsets.

             The row's WIDTH is the layout's own — a row that does not
             fill the grid's width (the last row usually) is left
             short rather than stretched, which is what Immich
             does. -->
        {#each justifiedRows as row (row.start)}
          <div
            class="justified-row"
            style="position: absolute; top: {row.top}px; left: 0; width: {row.width}px; height: {row.height}px;"
          >
            {#each row.tiles as tile, tileIdx (tile.item.id)}
              {@const eagerIndex = row === justifiedRows[0] && tileIdx < 3 ? tileIdx : null}
              <div
                class="grid-tile justified-tile"
                style="position: absolute; left: {tile.x}px; top: 0; width: {tile.width}px; height: 100%;"
              >
                <!-- The tile's own width: the row scales the photo to
                     the row's height and derives the width from the
                     photo's own ratio, so the tile IS the photo's
                     shape. `natural` tells the tile not to impose a
                     square of its own, and `renderWidth` tells it
                     how wide it actually is, so the browser fetches
                     the right asset for THIS photo.

                     `alwaysSrcset` gives EVERY tile the variant
                     ladder, not just the eager ones: a justified row
                     holds photos of every width, so one shared
                     `sizes` would be wrong for nearly all of them,
                     and a tile served the 640px canonical when it
                     needs 240px wastes five times the pixels. -->
                <PhotoTile
                  pointId={tile.item.id}
                  blurhash={tile.item.blurhash}
                  scoreStr={tile.item.score_str}
                  isFavorite={tile.item.is_favorite}
                  isDisliked={tile.item.is_disliked}
                  {eagerIndex}
                  onOpen={tileOpen(tile.item, tile.index)}
                  onContextMenu={tileContextMenu(tile.item)}
                  onRemove={onRemove}
                  removeLabel={removeLabel}
                  onVote={onVote}
                  votedUp={upVoted?.has(tile.item.id) ?? false}
                  votedDown={downVoted?.has(tile.item.id) ?? false}
                  voteUpLabel={voteUpLabel}
                  label={tileLabel(tile.item)}
                  natural
                  alwaysSrcset
                  thumbnailSize={$thumbnailSize}
                  tileWidth={TILE}
                  renderWidth={tile.width}
                  photoWidth={tile.item.width}
                  photoHeight={tile.item.height}
                  {readOnly}
                />
              </div>
            {/each}
          </div>
        {/each}
      {:else if masonry}
        {#each masonryItems as entry, mIdx (entry.item.id)}
          <div
            class="grid-tile masonry-tile"
            style="position: absolute; left: {entry.x}px; top: {entry.y}px; width: {entry.width}px; height: {entry.height}px;"
          >
            <!-- The masonry slice is already the visible one, so the
                 first three of them are the eager ones — the same
                 rule the row virtualizer applies to the first row. -->
            <PhotoTile
              pointId={entry.item.id}
              blurhash={entry.item.blurhash}
              scoreStr={entry.item.score_str}
              isFavorite={entry.item.is_favorite}
              isDisliked={entry.item.is_disliked}
              eagerIndex={mIdx < 3 ? mIdx : null}
              onOpen={tileOpen(entry.item, entry.index)}
              onContextMenu={tileContextMenu(entry.item)}
              onRemove={onRemove}
              removeLabel={removeLabel}
              onVote={onVote}
              votedUp={upVoted?.has(entry.item.id) ?? false}
              votedDown={downVoted?.has(entry.item.id) ?? false}
              voteUpLabel={voteUpLabel}
              label={tileLabel(entry.item)}
              natural
              clamped={entry.clamped}
              thumbnailSize={$thumbnailSize}
              tileWidth={TILE}
              photoWidth={entry.item.width}
              photoHeight={entry.item.height}
              {readOnly}
            />
          </div>
        {/each}
      {:else}
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
                onOpen={tileOpen(item, itemIndex)}
                onContextMenu={tileContextMenu(item)}
                onRemove={onRemove}
                removeLabel={removeLabel}
                onVote={onVote}
                votedUp={upVoted?.has(item.id) ?? false}
                votedDown={downVoted?.has(item.id) ?? false}
                voteUpLabel={voteUpLabel}
                label={tileLabel(item)}
                thumbnailSize={$thumbnailSize}
                tileWidth={TILE}
                photoWidth={item.width}
                photoHeight={item.height}
                {wall}
                {readOnly}
              />
            </div>
          {/each}
        </div>
      {/each}
      {/if}
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
      isFavorite: it.is_favorite
    }))}
    index={lightboxIndex}
    onClose={closeLightbox}
    onToggleFavorite={onToggleFavorite}
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
    gap: 0;
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

  /* The wall fills the screen.
     Two changes from the app grid:
       1. The track count is the published --grid-cols rather than
          the browser's own auto-fit. The wall's columns are
          computed to DIVIDE the viewport evenly (see
          wallColumnsForWidth), so the browser has to lay out
          exactly that many tracks — auto-fit would round them
          down to the tile size and leave a band of background at
          the edge.
       2. The tracks are 1fr, so they stretch to fill the width.
          With a fixed track size the leftover space stays empty
          and the background shows through beside the photos; a
          presentation with the page bleeding in at its edges is
          not a presentation.
     The tile's own aspect-ratio keeps every photo square
     whatever the track width works out to, and the row height
     comes from the measured tile width, so the virtualizer's
     geometry follows the same number. */
  .grid-wrapper.wall .grid-row {
    grid-template-columns: repeat(var(--grid-cols, 6), 1fr);
    justify-content: stretch;
  }
  .grid-wrapper.wall {
    /* The wall is the screen: no shell padding, no grid-width
       cap, and it spans the full viewport. */
    width: 100%;
    max-width: 100%;
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

  /* The self-laid-out tiles are sized by the layout, not by CSS: the
     grid publishes the photo's own shape as the tile's box, so the
     square the uniform grid imposes has to step aside. The photo
     inside is `object-fit: cover`, and because the box's ratio IS the
     photo's ratio, nothing is cropped.

     The masonry tile is a fixed WIDTH and a varying height; the
     justified tile is a fixed HEIGHT (the row's) and a varying
     width. */
  .grid-tile.masonry-tile,
  .grid-tile.justified-tile {
    aspect-ratio: auto;
  }

  /* A justified row is a positioned box, not a CSS grid: the layout
     decides where each photo sits, and the row's height is the
     layout's, not the CSS's. */
  .justified-row {
    position: absolute;
    left: 0;
    width: 100%;
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
    border: 3px solid var(--spinner-track);
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
    margin-bottom: 0;
    pointer-events: none;
  }
  .skeleton-tile {
    /* Inherits aspect-ratio: 1 from .grid-tile. */
    background: var(--skeleton-base);
    /* The same published radius the real tile reads, so the
       placeholder does not change shape when the photo arrives.
       The fallback is the tile's own radius (--r-1). */
    border-radius: var(--grid-tile-radius, var(--r-1));
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
