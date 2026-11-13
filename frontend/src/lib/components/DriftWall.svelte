<script lang="ts">
  /**
   * DriftWall — the album wall that moves itself.
   *
   * The static wall is a page you scroll: the rows are laid out in
   * document order and the window virtualizer renders the slice you
   * are looking at. Drift is the SAME wall with a TIME AXIS — the
   * rows slide upward at a constant speed, forever, and nothing
   * about the surface changes except where the rows are.
   *
   * Three things make it work, and all three live in the arithmetic
   * (grid/drift.ts) rather than in the markup:
   *
   *  - The photos are a RING. A plain list cannot loop: the last
   *    row would be half-empty and the photos would stop. The ring
   *    puts the album's end against its start inside a row, where
   *    the random order hides the meeting, so every row is full and
   *    the wall never ends.
   *
   *  - The rows are positioned from a ROW INDEX, not from a scroll
   *    position. The index grows forever; a row's content comes from
   *    the ring, so one index always shows the same photos and the
   *    rendered slice changes only when a whole row has drifted past
   *    the top edge.
   *
   *  - The motion is a transform on the track, not a scroll. Nothing
   *    scrolls: the wall moves itself, so the surface can be exactly
   *    one screen.
   *
   * The wall's appearance is NOT redefined here. The tiles are the
   * same PhotoTile in the same `wall` variant the static wall uses,
   * at the same column count (grid/geometry.ts) and the same gapless
   * geometry — this component only decides WHERE the rows are.
   *
   * The one control is the exception the moving surface has to
   * make: content that starts moving on its own has to be pausable
   * (WCAG 2.2.2, Pause/Stop/Hide). The static wall has no controls
   * because nothing on it moves; here a single pause button sits in
   * the corner, and `prefers-reduced-motion` starts the wall still.
   */
  import { onMount } from 'svelte';
  import PhotoTile from './PhotoTile.svelte';
  import Icon from './Icon.svelte';
  import { wallColumnsForWidth } from '$lib/grid/geometry';
  import type { GridSizeConfig } from '$lib/grid/config';
  import {
    DRIFT_OVERSCAN_ROWS,
    DRIFT_SPEED_PX_PER_SEC,
    driftRowItems,
    driftWindowRows
  } from '$lib/grid/drift';

  type Item = {
    id: string;
    path?: string;
    blurhash?: string | null;
    is_favorite?: boolean;
  };

  type Props = {
    /** The album's photos, in the order the wall shows them. */
    items: Item[];
    /** The wall's grid config — square tiles, no gutter. */
    tileConfig: GridSizeConfig;
  };

  let { items, tileConfig }: Props = $props();

  let surface: HTMLDivElement | undefined = $state();
  let surfaceWidth = $state(0);
  let surfaceHeight = $state(0);
  /** The measured width of a rendered tile — the wall's row height. */
  let measuredTile = $state(0);

  let paused = $state(false);
  /** How far the wall has drifted upward, in CSS pixels. */
  let scrollPx = $state(0);
  /** The absolute index of the first rendered row. */
  let firstRow = $state(-DRIFT_OVERSCAN_ROWS);
  /** One past the absolute index of the last rendered row. */
  let lastRow = $state(0);

  const columns = $derived(
    surfaceWidth > 0 ? wallColumnsForWidth(surfaceWidth, tileConfig) : 0
  );

  // The row height is the tile width: the wall's tiles are square and
  // touch, so one row is exactly one tile tall. It is MEASURED rather
  // than derived from the column math — the browser hands out
  // fractional track widths, and a row height that is a hair off the
  // tiles would open a seam between rows.
  const rowHeight = $derived(
    measuredTile > 0
      ? measuredTile
      : columns > 0 && surfaceWidth > 0
        ? surfaceWidth / columns
        : 0
  );

  // The rows on screen. Both bounds are integers, so they change
  // only when a whole row has drifted past the edge — not on every
  // frame. The continuous motion lives in the track's transform.
  const rows = $derived(
    columns > 0 && rowHeight > 0
      ? Array.from({ length: Math.max(0, lastRow - firstRow) }, (_, i) => {
          const index = firstRow + i;
          return { index, items: driftRowItems(items, columns, index) };
        })
      : []
  );

  // Where the track sits. A row's screen position is
  // `index * rowHeight - scrollPx`, expressed as the row's offset
  // inside the track plus the track's own shift. The two move in
  // opposite directions when the window advances, so the photos do
  // not jump when a row leaves the top — and the value stays
  // bounded however long the wall runs.
  const trackOffset = $derived(firstRow * rowHeight - scrollPx);

  const surfaceVars = $derived(
    `--grid-cols:${columns};--grid-tile-radius:0px;--grid-row-max-width:none;`
  );

  onMount(() => {
    let frame = 0;
    let last = 0;

    const step = (now: number) => {
      frame = requestAnimationFrame(step);
      const dt = last === 0 ? 0 : (now - last) / 1000;
      last = now;
      // A backgrounded tab hands back a huge dt on return, which
      // would teleport the wall. Cap it: the drift is a slow
      // breathing, not a jump.
      if (!paused) scrollPx += DRIFT_SPEED_PX_PER_SEC * Math.min(dt, 0.1);

      // The rendered window advances one row at a time. Kept in
      // state rather than derived from the pixel offset so the row
      // list is rebuilt only when the slice actually changes.
      if (rowHeight > 0) {
        const range = driftWindowRows(scrollPx, rowHeight, surfaceHeight);
        if (range.first !== firstRow || range.last !== lastRow) {
          firstRow = range.first;
          lastRow = range.last;
        }
      }
    };
    frame = requestAnimationFrame(step);

    // Reduced motion: the wall is still, and the pause control says
    // so. Turning the preference on mid-session parks it again; it
    // never starts on its own once the user has asked for stillness.
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (reduce.matches) paused = true;
    const onReduceChange = (e: MediaQueryListEvent) => {
      if (e.matches) paused = true;
    };
    reduce.addEventListener('change', onReduceChange);

    return () => {
      cancelAnimationFrame(frame);
      reduce.removeEventListener('change', onReduceChange);
    };
  });

  // The surface's size — the column count, the row height and the
  // visible window all follow it.
  $effect(() => {
    if (!surface) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        surfaceWidth = entry.contentRect.width;
        surfaceHeight = entry.contentRect.height;
      }
    });
    observer.observe(surface);
    return () => observer.disconnect();
  });

  // Measure a rendered tile. The wall's tiles are 1fr tracks, so the
  // first tile's width IS the row height. Re-measured whenever the
  // album or the surface changes, because the tracks re-flow.
  $effect(() => {
    void items.length;
    void surfaceWidth;
    void columns;
    const el = surface;
    if (!el) return;
    queueMicrotask(() => {
      const tile = el.querySelector('.grid-tile');
      if (!tile) return;
      const width = tile.getBoundingClientRect().width;
      if (width > 0) measuredTile = width;
    });
  });

  function onKeydown(e: KeyboardEvent) {
    if (e.key !== ' ' && e.code !== 'Space') return;
    // Let the button handle its own Space, and let any other control
    // keep its native behaviour — only the bare surface toggles.
    const target = e.target as HTMLElement | null;
    if (target?.closest('button, a, input, textarea, select, [contenteditable]')) return;
    e.preventDefault();
    paused = !paused;
  }

  /**
   * The accessible name of a read-only tile.
   *
   * A read-only tile has no action, so the name has to identify the
   * PHOTO instead — the file name is the only identity it has. The
   * static wall names its tiles the same way.
   */
  function tileLabel(item: Item): string | undefined {
    const name = item.path?.split('/').pop();
    return name || undefined;
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="drift" bind:this={surface} style={surfaceVars}>
  <div class="drift-track" style="transform: translate3d(0, {trackOffset}px, 0)">
    {#each rows as row (row.index)}
      <div
        class="grid-row"
        style="height: {rowHeight}px; transform: translateY({(row.index - firstRow) * rowHeight}px)"
      >
        {#each row.items as item, colIndex (colIndex)}
          <div class="grid-tile">
            <PhotoTile
              pointId={item.id}
              blurhash={item.blurhash}
              isFavorite={item.is_favorite}
              wall
              readOnly
              label={tileLabel(item)}
            />
          </div>
        {/each}
      </div>
    {/each}
  </div>

  <button
    type="button"
    class="drift-pause glass-strong"
    onclick={() => (paused = !paused)}
    aria-label={paused ? 'Resume the drift' : 'Pause the drift'}
    title={paused ? 'Resume the drift' : 'Pause the drift'}
  >
    <Icon name={paused ? 'play' : 'pause'} size={18} />
  </button>
</div>

<style>
  /* The surface: one screen, and nothing scrolls. The wall moves
     itself inside it. */
  .drift {
    position: absolute;
    inset: 0;
    overflow: hidden;
    background: var(--bg-0);
  }

  .drift-track {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    /* The track is transformed every frame — promote it so the
       drift is composited and never re-lays-out. */
    will-change: transform;
  }

  /* A row of the wall. The tracks are 1fr, so they divide the
     surface evenly and the photos touch pixel-to-pixel — the same
     gapless square field the static wall renders. */
  .grid-row {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    display: grid;
    grid-template-columns: repeat(var(--grid-cols, 6), 1fr);
    gap: 0;
  }

  .grid-tile {
    aspect-ratio: 1;
    min-width: 0;
    /* Round-36: the same containment the scrolling grid uses — the
       drift renders a bounded window of rows, and the containment
       stops one tile's contents from re-laying-out the rest. */
    contain: size layout paint style;
  }

  /* The one control the moving surface carries. It sits in the
     corner and is ALWAYS visible: a pause that only appears on
     hover is not a pause you can find. The treatment is the
     floating-button idiom (ScrollToTop) — same size, same radius,
     same lift — so the app teaches one gesture. */
  .drift-pause {
    position: absolute;
    right: 20px;
    bottom: 20px;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    background: var(--bg-1);
    border: 1px solid var(--glass-edge-strong);
    color: var(--fg-1);
    box-shadow: var(--shadow-2);
    z-index: 1;
    transition:
      background var(--t-fast),
      transform var(--t-fast);
  }
  .drift-pause:hover {
    background: var(--bg-2);
    transform: translateY(-2px);
  }
  .drift-pause:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
</style>
