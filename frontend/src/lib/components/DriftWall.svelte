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
   * at the same gapless geometry — this component only decides
   * WHERE the rows are and HOW BIG they are.
   *
   * The controls are the moving surface's own. Content that starts
   * moving on its own has to be pausable (WCAG 2.2.2), and once the
   * wall has controls it can have the ones the wall wants: how fast
   * it drifts and how many photos sit across it.
   *
   * The pause button stays visible ALWAYS — a pause you have to
   * discover by moving the mouse is not a pause you can find, and
   * it is the one control that has to be there for the moving
   * content to be lawful. The two adjustment controls fade in while
   * the pointer is moving and fade out once it rests, because a
   * panel of sliders parked over the photos is a panel you look
   * past. Both fade rules keep the controls up while the pointer
   * is on them or the keyboard is inside them.
   */
  import { onMount } from 'svelte';
  import PhotoTile from './PhotoTile.svelte';
  import Icon from './Icon.svelte';
  import { wallColumnsForWidth } from '$lib/grid/geometry';
  import type { GridSizeConfig } from '$lib/grid/config';
  import {
    DRIFT_MIN_COLUMNS,
    DRIFT_OVERSCAN_ROWS,
    DRIFT_SPEED_MAX_PX_PER_SEC,
    DRIFT_SPEED_MIN_PX_PER_SEC,
    DRIFT_SPEED_PX_PER_SEC,
    DRIFT_SPEED_STEP_PX_PER_SEC,
    clampDriftColumns,
    clampDriftSpeed,
    driftMaxColumns,
    driftRowItems,
    driftWindowRows,
    quantizePhotoWidth
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

  /** How fast the wall drifts, in px/s. */
  let speedPxPerSec = $state(DRIFT_SPEED_PX_PER_SEC);
  /**
   * The user's column count, or null while they have not chosen one.
   *
   * Null means the wall decides for itself — the same count the
   * static wall would render at this width. Keeping the untouched
   * state as null (rather than a number) is what lets the wall
   * follow the viewport until the user takes the wheel.
   */
  let columnOverride = $state<number | null>(null);

  /** Whether the adjustment controls are showing. */
  let controlsVisible = $state(false);
  /** Whether the pointer or the keyboard is on them, so they stay. */
  let controlsPinned = $state(false);

  /**
   * How long the pointer may rest before the adjustment controls
   * fade away.
   *
   * Long enough that a glance does not dismiss them, short enough
   * that the wall is bare again while you watch it.
   */
  const CONTROLS_IDLE_MS = 2200;

  let hideTimer: ReturnType<typeof setTimeout> | undefined;

  /**
   * The column count the wall would pick for itself.
   *
   * The static wall's count: the tile size the user chose, rounded
   * to the nearest whole number of columns across the surface.
   */
  const autoColumns = $derived(
    surfaceWidth > 0 ? wallColumnsForWidth(surfaceWidth, tileConfig) : 0
  );

  /** The column count actually in force. */
  const columns = $derived(columnOverride ?? autoColumns);

  /** The most columns the control may offer at this width. */
  const maxColumns = $derived(driftMaxColumns(surfaceWidth, autoColumns));

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

  /**
   * The width to fetch each photo at.
   *
   * The tile's own width, rounded up to the next step, and the
   * photo is fetched at that width instead of at the app's 384px
   * thumbnail. Zoomed out (many columns) the tiles are small and the
   * thumbnail is exactly right; zoomed in (few columns) the tile is
   * wider than any thumbnail and has to come from the original.
   *
   * The step matters: without it every column count would ask for a
   * different width, and each one would be a separate server-side
   * resize of the same photo. Quantising to a few steps keeps the
   * number of distinct resized images bounded.
   */
  const tileWidth = $derived(columns > 0 ? surfaceWidth / columns : 0);
  const photoWidth = $derived(quantizePhotoWidth(tileWidth));

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
      if (!paused) scrollPx += speedPxPerSec * Math.min(dt, 0.1);

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

    // The adjustment controls show themselves once on arrival, so
    // the wall's controls are discoverable before the user has
    // moved the pointer, and then they fade like any other time.
    revealControls();

    return () => {
      cancelAnimationFrame(frame);
      reduce.removeEventListener('change', onReduceChange);
      if (hideTimer) clearTimeout(hideTimer);
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
  // album, the surface or the column count changes, because the
  // tracks re-flow.
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

  /**
   * Show the adjustment controls, and schedule their retreat.
   *
   * The controls answer to the pointer: they appear while the
   * pointer is moving over the wall and fade once the pointer
   * rests. A control that only appeared on hover would be one the
   * user could miss entirely, so they also show on arrival and
   * whenever the keyboard is on them.
   */
  function revealControls() {
    controlsVisible = true;
    if (hideTimer) clearTimeout(hideTimer);
    hideTimer = setTimeout(() => {
      if (!controlsPinned) controlsVisible = false;
    }, CONTROLS_IDLE_MS);
  }

  /**
   * The pointer or the keyboard is on the controls — they stay up.
   *
   * Pinning still SHOWS them: a slider the keyboard has landed on
   * must not be invisible, and the hide timer checks the pin before
   * it fades anything, so pinning and showing are the same act.
   */
  function pinControls(pinned: boolean) {
    controlsPinned = pinned;
    revealControls();
  }

  function onFocusIn(e: FocusEvent) {
    const panel = e.currentTarget as HTMLElement;
    if (panel.contains(e.relatedTarget as Node | null)) return;
    pinControls(true);
  }

  function onFocusOut(e: FocusEvent) {
    const panel = e.currentTarget as HTMLElement;
    if (panel.contains(e.relatedTarget as Node | null)) return;
    pinControls(false);
  }

  function onSpeedInput(e: Event) {
    const el = e.currentTarget as HTMLInputElement;
    speedPxPerSec = clampDriftSpeed(Number(el.value));
  }

  function onColumnsInput(e: Event) {
    const el = e.currentTarget as HTMLInputElement;
    columnOverride = clampDriftColumns(Number(el.value), maxColumns);
    // The tiles are about to be a different width, so the row
    // height has to be measured again — the old measurement belongs
    // to the old column count.
    measuredTile = 0;
  }

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

<svelte:window onkeydown={onKeydown} onpointermove={revealControls} />

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
              srcWidth={photoWidth}
              wall
              readOnly
              label={tileLabel(item)}
            />
          </div>
        {/each}
      </div>
    {/each}
  </div>

  <div
    class="drift-controls"
    role="group"
    aria-label="Drift controls"
    onmouseenter={() => pinControls(true)}
    onmouseleave={() => pinControls(false)}
    onfocusin={onFocusIn}
    onfocusout={onFocusOut}
  >
    <!-- The adjustment controls. They fade in while the pointer is
         moving over the wall and fade out once it rests; they stay
         up while the pointer or the keyboard is on them. -->
    <div class="drift-fields glass-strong" class:visible={controlsVisible}>
      <label class="drift-field">
        <span class="drift-label">Speed</span>
        <input
          class="drift-range"
          type="range"
          min={DRIFT_SPEED_MIN_PX_PER_SEC}
          max={DRIFT_SPEED_MAX_PX_PER_SEC}
          step={DRIFT_SPEED_STEP_PX_PER_SEC}
          value={speedPxPerSec}
          oninput={onSpeedInput}
          aria-label="Speed — how fast the wall drifts, in pixels per second"
        />
        <span class="drift-value">{speedPxPerSec} px/s</span>
      </label>

      <label class="drift-field">
        <span class="drift-label">Columns</span>
        <input
          class="drift-range"
          type="range"
          min={DRIFT_MIN_COLUMNS}
          max={Math.max(DRIFT_MIN_COLUMNS, maxColumns)}
          step="1"
          value={Math.max(DRIFT_MIN_COLUMNS, columns)}
          oninput={onColumnsInput}
          aria-label="Columns — how many photos fit across the wall"
          title="How many photos fit across the wall — fewer columns means a bigger, more zoomed-in photo"
        />
        <span class="drift-value">{Math.max(DRIFT_MIN_COLUMNS, columns)}</span>
      </label>
    </div>

    <!-- The pause. It is ALWAYS visible: it is the control that
         makes the moving content lawful (WCAG 2.2.2), and a pause
         you can only find by moving the mouse is not a pause you
         can find. -->
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
</div>

<style>
  /* The surface: one screen, and nothing scrolls. The wall moves
     itself inside it. */
  .drift {
    position: absolute;
    inset: 0;
    overflow: hidden;
    background: var(--bg-0);
    /* The wall's photos do NOT lean in on hover. The static wall
       keeps the app-wide hover zoom, but here the wall is MOVING:
       a photo that grows under the pointer is a photo that shifts
       against a field that is already sliding, and on a ring
       every tile is under the pointer at some point — the zoom
       would be a flicker, not a response. */
    --grid-hover-zoom: 1;
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

  /* The wall's controls, anchored to the bottom-right corner where
     the app's other floating control lives. The pause sits at the
     corner and the adjustments sit to its left, so the pause never
     moves when the adjustments appear. */
  .drift-controls {
    position: absolute;
    right: 20px;
    bottom: 20px;
    z-index: 2;
    display: flex;
    align-items: center;
    gap: var(--s-2);
  }

  /* The adjustment panel. It fades rather than pops: it arrives
     with the pointer and leaves with it, and it never takes the
     pointer's click while it is invisible. */
  .drift-fields {
    display: flex;
    align-items: center;
    gap: var(--s-3);
    max-width: calc(100vw - 140px);
    padding: var(--s-2) var(--s-3);
    border-radius: var(--r-3);
    opacity: 0;
    transform: translateX(8px);
    pointer-events: none;
    transition:
      opacity var(--t-fast) var(--ease-out),
      transform var(--t-fast) var(--ease-out);
  }
  .drift-fields.visible {
    opacity: 1;
    transform: none;
    pointer-events: auto;
  }

  /* The pause control — the floating-button treatment, transplanted
     so the app teaches one gesture. Its label flips between Pause
     and Resume. */
  .drift-pause {
    flex: none;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    color: var(--fg-1);
    transition:
      background var(--t-fast) var(--ease-out),
      transform var(--t-fast) var(--ease-out);
  }
  .drift-pause:hover {
    transform: translateY(-2px);
  }
  .drift-pause:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }

  /* One control: a label, the slider, and what it currently reads
     as. The label sits above the slider rather than beside it so
     the panel stays narrow enough to sit over a corner of the wall. */
  .drift-field {
    display: grid;
    grid-template-columns: auto 1fr;
    grid-template-areas:
      'label value'
      'range range';
    align-items: center;
    column-gap: var(--s-2);
    row-gap: 2px;
    cursor: pointer;
  }
  .drift-label {
    grid-area: label;
    font-size: var(--fs-xs);
    color: var(--fg-2);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .drift-value {
    grid-area: value;
    font-size: var(--fs-xs);
    color: var(--fg-1);
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .drift-range {
    grid-area: range;
    width: 110px;
    accent-color: var(--accent);
    cursor: pointer;
  }
  .drift-range:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
</style>
