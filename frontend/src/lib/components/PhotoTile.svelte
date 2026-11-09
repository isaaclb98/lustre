<script lang="ts">
  /**
   * One tile in the grid. Renders:
   *   - blurhash as a tinted placeholder (decoded async, cheap)
   *   - the real photo on top, once loaded
   *   - a glass "hover" chrome with the score (top-left) and
   *     context-menu trigger (top-right). No buttons until hover.
   *   - a persistent visual state for liked (pink glow + badge)
   *     and disliked (cool-grey dim + slash) tiles so the user
   *     can see their feedback across all pages. (Round-6
   *     issue #3 — Like/Dislike should change colour across
   *     the board, not just in the lightbox.)
   *
   * Tiles are responsible for their own click + right-click
   * interactions. The grid above just lays them out.
   */
  import { onMount } from 'svelte';
  import { thumbUrl } from '../api/endpoints';
  import { blurhashToDataUrl } from './blurhash-bg';
  import Icon from './Icon.svelte';

  type Props = {
    pointId: string;
    blurhash?: string | null;
    /** Back-compat alias for `blurhash` — some callers pass the
     *  pre-computed blurhash data URL under this name. */
    dataUrl?: string | null;
    scoreStr?: string;
    isFavorite?: boolean;
    isDisliked?: boolean;
    contextMenuOpen?: boolean;
    /**
     * Tier 1 of 3 (perf round 1): first-screen tiles should bypass
     * the browser's lazy-loader so the initial grid paint isn't gated
     * on the lazy-load heuristic. 0-indexed position in the visible
     * items array; false/null → normal lazy behaviour. We pass this
     * down from PhotoGrid because PhotoTile doesn't know whether it's
     * in the first row of the visible virtualizer slice.
     */
    eagerIndex?: number | false | null;
    onOpen?: (id: string) => void;
    onContextMenu?: (id: string, e: MouseEvent) => void;
    /**
     * Optional remove affordance. When provided, a small circular
     * button appears at the top-right of the tile (replacing the
     * corner spot the persistent fav/dislike badges use, so the
     * remove badge takes precedence visually when both apply).
     * Used by the album detail page, the Likes grid, and the
     * Dislikes grid — i.e. anywhere the user is curating a set of
     * photos and removing one is the primary action.
     *
     * The remove button is invisible by default and fades in on
     * hover/focus so it doesn't visually clutter the tile grid;
     * on touch / coarse-pointer devices where there's no hover, it
     * stays visible (the typical mobile pattern: action chips are
     * always present). Uses data-no-open so the click doesn't open
     * the photo lightbox.
     */
    onRemove?: (id: string) => void;
    /** Visible label on the remove button + its title attribute. */
    removeLabel?: string;
    /**
     * Vote affordance — the Explore page's per-tile up/down pair.
     *
     * When set, two action chips appear over the tile: an up-vote
     * (joins the album) and a down-vote (a temporary negative for
     * the session's next round). The two are mutually exclusive in
     * the caller, so only one is ever highlighted.
     *
     * Separate from `onRemove`: removing a photo drops it from a
     * set permanently, whereas an Explore down-vote only steers the
     * current round.
     */
    onVote?: (id: string, direction: 'up' | 'down') => void;
    /** This photo was up-voted — highlights the up chip. */
    votedUp?: boolean;
    /** This photo was down-voted — highlights the down chip. */
    votedDown?: boolean;
    /**
     * Label on the up-vote chip. The default is "Add to album",
     * which is what an up-vote does on a user album; the built-in
     * Favourites album passes "Favourite this photo" because there
     * the up-vote marks the photo as a favourite.
     */
    voteUpLabel?: string;
  };
  let {
    pointId,
    blurhash,
    dataUrl: dataUrlProp,
    scoreStr,
    isFavorite,
    isDisliked,
    contextMenuOpen,
    eagerIndex,
    onOpen,
    onContextMenu,
    onRemove,
    removeLabel = 'Remove',
    onVote,
    votedUp = false,
    votedDown = false,
    voteUpLabel = 'Add to album'
  }: Props = $props();

  // Pre-computed data URL takes priority (caller already decoded
  // the blurhash), otherwise we resolve it lazily on mount.
  // `dataUrlProp` is read once on init; tiles don't swap
  // their blurhash prop after mount, so the initial-value
  // capture is the documented behaviour. (svelte-check
  // would prefer a $derived; the difference is
  // "react to prop changes" which we deliberately don't
  // do here.)
  // svelte-ignore state_referenced_locally
  let dataUrl = $state<string | null>(dataUrlProp ?? null);
  let loaded = $state(false);
  let imgEl: HTMLImageElement | undefined = $state();

  onMount(() => {
    if (!dataUrl && blurhash) {
      blurhashToDataUrl(blurhash).then((u) => {
        if (u) dataUrl = u;
      });
    }
  });

  function handleClick(e: MouseEvent) {
    if ((e.target as HTMLElement).closest('[data-no-open]')) return;
    e.preventDefault();
    onOpen?.(pointId);
  }
  function handleContext(e: MouseEvent) {
    e.preventDefault();
    onContextMenu?.(pointId, e);
  }
  // Best-effort tile-width hint for the first-screen srcset. We can't
  // measure the tile itself from here (no DOM ref on the wrapping
  // grid cell), so we approximate from window.innerWidth divided by
  // the auto-fill column target. Falls back to 384 on SSR / dev
  // edge cases. Cached per-tile via $derived so we only recompute
  // when the breakpoint flips.
  //
  // Post the model-variant migration plan: the only thumbnail variant
  // served is 384×384 (matches so400m model input resolution). So
  // `eagerSrcWidth` and `eagerSrcset` reflect that single asset —
  // no srcset ladder needed, just a single src.
  function tileSizeGuess(): number | undefined {
    if (typeof window === 'undefined') return undefined;
    const w = window.innerWidth;
    // Mirror PhotoGrid's `auto-fill minmax(180px, 1fr)` heuristic.
    const cols = Math.max(1, Math.floor((Math.min(w, 1600) + 20) / (180 + 20)));
    return Math.floor((Math.min(w, 1600) - (cols - 1) * 20) / cols);
  }
  // True for the first three tiles of the first visible row. These
  // are the tiles a user is staring at on first paint — we want them
  // marked eager + high-priority so the browser starts fetching them
  // alongside the HTML/CSS instead of waiting on its lazy heuristic.
  let isEager = $derived(eagerIndex === 0 || eagerIndex === 1 || eagerIndex === 2);
  // Rendered srcset width for the eager tiles. The single-variant
  // pipeline serves 384×384; the rounded-up × 2 calculation would
  // be 480, but the backend only generates 384 (no 480), so just
  // pin to 384.
  let eagerSrcWidth = $derived(384);
  let eagerSrcset = $derived(`${thumbUrl(pointId, 384)} 384w`);
  // Sizes attribute must reflect the actual rendered tile width
  // so the browser picks the right srcset entry. Without this,
  // `sizes` was hardcoded to "240px" on every viewport, so the
  // browser always picked the 240w variant — over-fetching on
  // mobile (a 120px-wide tile still pulled 240w of pixels) and
  // under-fetching on big monitors (a 480px-wide tile got only
  // 240w of pixels, blurry at 2x DPR).
  //
  // Post the model-variant migration plan, the only thumbnail
  // variant served is 384×384 — the srcset is a single 384w entry,
  // so the `sizes` attribute just needs to advertise the tile's
  // rendered CSS pixels at each breakpoint.
  let eagerSizes = $derived.by(() => {
    const w = tileSizeGuess() ?? 384;
    // Pick the smallest variant >= w so the browser has enough
    // pixels to render crisply at this DPR. With a single 384w
    // variant, this is just `w` for any w <= 384.
    const variant = w <= 120 ? 120 : w <= 180 ? 180 : w <= 240 ? 240 : 384;
    // Re-derive the breakpoint from `w`: below 600px viewport the
    // tiles are ~120-180px, above that they're ~240px, above
    // 1200px they grow further (capped at our 384 variant).
    return `(max-width: 600px) ${Math.min(variant, 120)}px, (max-width: 1200px) ${Math.min(variant, 180)}px, ${variant}px`;
  });
</script>

<a
  class="tile"
  class:loaded
  class:favorite={isFavorite}
  class:disliked={isDisliked}
  class:menu-open={contextMenuOpen}
  href={`/photo/${encodeURIComponent(pointId)}`}
  onclick={handleClick}
  oncontextmenu={handleContext}
  aria-label="Open photo"
>
  {#if dataUrl}
    <img class="ph" src={dataUrl} alt="" aria-hidden="true" />
  {:else}
    <div class="ph fallback" aria-hidden="true"></div>
  {/if}
  <img
    bind:this={imgEl}
    class="full"
    src={isEager ? thumbUrl(pointId, eagerSrcWidth) : thumbUrl(pointId, 384)}
    srcset={isEager ? eagerSrcset : undefined}
    sizes={isEager ? eagerSizes : undefined}
    alt=""
    loading={isEager ? 'eager' : 'lazy'}
    fetchpriority={eagerIndex === 0 ? 'high' : isEager ? 'auto' : 'low'}
    decoding="async"
    onload={() => (loaded = true)}
    onerror={() => (loaded = true)}
  />
  {#if scoreStr}
    <span class="score">{scoreStr}</span>
  {/if}
  {#if isFavorite}
    <span class="fav" data-no-open title="Like">
      <Icon name="heart-filled" size={14} />
    </span>
  {/if}
  {#if isDisliked}
    <span class="neg-badge" data-no-open title="Dislike">
      <Icon name="minus" size={14} />
    </span>
  {/if}
  {#if onRemove}
    <button
      type="button"
      class="remove-btn"
      data-no-open
      title={removeLabel}
      aria-label={removeLabel}
      onclick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        onRemove?.(pointId);
      }}
    >
      <Icon name="minus" size={16} />
    </button>
  {/if}
  {#if onVote}
    <!--
      Explore's vote pair. Sits at the BOTTOM of the tile, unlike
      the remove chip (top-right) and the score/fav badges (top) —
      the bottom edge is otherwise unused, so the two vote chips
      don't compete with the existing chrome for the same corner.
    -->
    <div class="vote-bar" data-no-open>
      <button
        type="button"
        class="vote-btn vote-up"
        class:voted={votedUp}
        data-no-open
        title={voteUpLabel}
        aria-label={voteUpLabel}
        aria-pressed={votedUp}
        onclick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onVote?.(pointId, 'up');
        }}
      >
        <Icon name="plus" size={16} />
      </button>
      <button
        type="button"
        class="vote-btn vote-down"
        class:voted={votedDown}
        data-no-open
        title="Reject — steers the next round away from this"
        aria-label="Reject this photo"
        aria-pressed={votedDown}
        onclick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onVote?.(pointId, 'down');
        }}
      >
        <Icon name="minus" size={16} />
      </button>
    </div>
  {/if}
</a>

<style>
  .tile {
    position: relative;
    display: block;
    aspect-ratio: 1 / 1;
    border-radius: var(--r-1);
    overflow: hidden;
    background: var(--bg-1);
    border: 1px solid var(--glass-edge);
    /* Round-38: lighter shadow on hover (was --shadow-glass-strong
       which was tuned for dark surfaces). --shadow-1 is the
       standard tile resting shadow. */
    box-shadow: var(--shadow-1);
    transition: transform var(--t-fast) var(--ease-out),
                border-color var(--t-fast) var(--ease-out),
                box-shadow var(--t-fast) var(--ease-out),
                opacity var(--t-fast) var(--ease-out);
  }
  .tile:hover {
    transform: translateY(-3px);
    border-color: var(--glass-edge-strong);
    box-shadow: var(--shadow-3);
  }
  .tile.menu-open {
    border-color: var(--accent);
    box-shadow: var(--shadow-glass-strong);
  }
  /* Liked — only the floating fav badge remains (Round‑36 dropped
     the persistent pink border + outer glow). The badge in the
     top-right is enough signal: "this photo is saved". The border
     was reading as an orange outline against the dark backdrop
     and competed with the image; without it the grid reads as
     a clean photo wall with discrete like markers. */
  /* Disliked — cool blue-grey dim + slightly washed-out image so
     the feedback state is visible without being loud. (Round-6
     issue #3 — applies to every page the tile appears on.) */
  .tile.disliked {
    /* Round-65: was #8ca0c8 - a custom blue-gray focus ring
       that did not match the rest of the app accent. Now
       uses --accent for the focus state so the ring is
       on-brand. */
    border-color: color-mix(in srgb, var(--accent) 45%, transparent);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 30%, transparent);
    opacity: 0.72;
  }
  .tile.disliked .full,
  .tile.disliked .ph {
    filter: saturate(0.55) brightness(0.92);
  }
  .tile.disliked:hover {
    opacity: 0.95;
  }
  /* A tile shouldn't be both liked AND disliked, but if a stale
     render shows both states, fade the favourite styling under
     the disliked styling so the "I hid this from recommendations"
     message wins. */
  .tile.disliked.favorite {
    /* Round-65: was rgba(140,160,200) — a custom blue-gray
       used only on the .disliked.favorite mixed state. Now
       derives from --accent so this rare state still reads
       as on-brand. */
    border-color: var(--mixed-tile-border);
    box-shadow: 0 0 0 1px var(--mixed-tile-ring);
    opacity: 0.72;
  }
  .ph, .full {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: opacity var(--t-med) var(--ease-out);
  }
  .ph {
    z-index: 0;
    filter: saturate(1.15) blur(0.5px);
  }
  .full {
    z-index: 1;
    opacity: 0;
  }
  .tile.loaded .full { opacity: 1; }
  .fallback {
    background: linear-gradient(
      135deg,
      rgba(var(--halo), 0.10),
      var(--empty-tile-accent)
    );
  }

  /* Tile chrome (score pill, fav heart, neg badge) sits above the
   * thumbnail layer: .ph is z:0, .full is z:1, so z-auto chrome
   * paints UNDER the thumbnail and only shows while the tile is
   * still in blurhash state. z-index: 2 matches .remove-btn (which
   * already documented this stacking requirement). */
  .score {
    position: absolute;
    top: 8px;
    left: 8px;
    padding: var(--s-0) var(--s-2);
    font-size: var(--fs-xs);
    color: var(--pill-fg);
    background: var(--pill-bg);
    border: 1px solid var(--pill-border);
    border-radius: var(--r-pill);
    backdrop-filter: var(--pill-blur);
    -webkit-backdrop-filter: var(--pill-blur);
    opacity: 0;
    z-index: 2;
    transition: opacity var(--t-fast) var(--ease-out);
  }
  .tile:hover .score,
  .tile.menu-open .score { opacity: 1; }

  .fav {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 24px; height: 24px;
    border-radius: 50%;
    background: var(--pill-bg);
    border: 1px solid var(--pill-border);
    color: var(--warn);
    font-size: 14px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: var(--pill-blur);
    -webkit-backdrop-filter: var(--pill-blur);
    z-index: 2;
  }
  .neg-badge {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 24px; height: 24px;
    border-radius: 50%;
    background: var(--pill-bg);
    border: 1px solid var(--pill-border);
    /* Round-64: was #c9d3e6 — a light blue-gray meant to read
       on the old dark pill background. White pills make this
       look washed-out, so use the negative token (red) so
       the dislike badge reads as a state, not as a neutral
       icon. Matches the heart-filled favorite (var(--warn)). */
    color: var(--negative);
    font-size: 16px;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: var(--pill-blur);
    -webkit-backdrop-filter: var(--pill-blur);
    z-index: 2;
  }
  /* Remove-from-set button — same corner slot as the persistent
   * fav/dislike badges, but interactive and only revealed on
   * hover so it doesn't add noise to the static grid. On
   * coarse-pointer (touch) devices it stays visible; the
   * (hover: none) media query handles that automatically.
   *
   * Click handler calls preventDefault + stopPropagation so the
   * surrounding <a class="tile"> doesn't navigate. data-no-open
   * is belt-and-braces for the same purpose.
   *
   * z-index: 2 (above .full which is z:1) so the button is hit-
   * testable even when the thumbnail fully covers the tile. */
  .remove-btn {
    position: absolute;
    top: 6px;
    right: 6px;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    /* Glass action chip — uses the light glass tier (small blur
       is sufficient for a 28px chip sitting on a photo) plus the
       layered shadow for lift. */
    background: var(--dialog-fill);
    border: 1px solid var(--glass-edge);
    color: var(--fg-1);
    font-size: 18px;
    font-weight: 500;
    line-height: 1;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: var(--glass-light);
    -webkit-backdrop-filter: var(--glass-light);
    box-shadow: var(--shadow-glass);
    opacity: 0;
    z-index: 2;
    transition:
      opacity var(--t-fast) var(--ease-out),
      background var(--t-fast) var(--ease-out),
      border-color var(--t-fast) var(--ease-out),
      transform var(--t-fast) var(--ease-out);
  }
  .tile:hover .remove-btn,
  .tile:focus-within .remove-btn,
  .remove-btn:focus-visible {
    opacity: 1;
  }
  .remove-btn:hover {
    background: color-mix(in srgb, var(--negative) 18%, transparent);
    border-color: color-mix(in srgb, var(--negative) 65%, transparent);
  }
  /* Explore's vote pair. Sits at the bottom edge, which the score
     pill and the fav/dislike badges (both top) leave free. The
     chips reuse the remove-btn glass treatment so they read as the
     same family of control. */
  .vote-bar {
    position: absolute;
    bottom: 6px;
    right: 6px;
    display: flex;
    gap: 4px;
    z-index: 2;
  }
  .vote-btn {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--dialog-fill);
    border: 1px solid var(--glass-edge);
    color: var(--fg-1);
    font-size: 18px;
    font-weight: 500;
    line-height: 1;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: var(--glass-light);
    -webkit-backdrop-filter: var(--glass-light);
    box-shadow: var(--shadow-glass);
    opacity: 0;
    transition:
      opacity var(--t-fast) var(--ease-out),
      background var(--t-fast) var(--ease-out),
      border-color var(--t-fast) var(--ease-out),
      transform var(--t-fast) var(--ease-out);
  }
  .tile:hover .vote-btn,
  .tile:focus-within .vote-btn,
  .vote-btn:focus-visible,
  .vote-btn.voted {
    opacity: 1;
  }
  .vote-up:hover,
  .vote-up.voted {
    background: color-mix(in srgb, var(--accent) 22%, transparent);
    border-color: color-mix(in srgb, var(--accent) 70%, transparent);
  }
  .vote-down:hover,
  .vote-down.voted {
    background: color-mix(in srgb, var(--negative) 18%, transparent);
    border-color: color-mix(in srgb, var(--negative) 65%, transparent);
  }
  .remove-btn:active {
    transform: scale(0.92);
  }
  /* Mobile / touch: hover doesn't exist, so the chip is always
   * present so the user can see the affordance. */
  @media (hover: none) {
    .remove-btn { opacity: 1; }
    .vote-btn { opacity: 1; }
  }
</style>
