<script lang="ts">
  /**
   * Lightbox — full-screen modal showing one image at a time.
   *   - ←/→ or A/D navigates prev/next
   *   - Esc or click outside closes
   *   - background blurs + tints from the current photo
   *   - "Most similar" navigates to /similar/{id} (closes itself)
   *
   * Slideshow: a Play/Pause button in the action bar starts
   * auto-advance from the currently-shown photo. The cadence
   * comes from the user's preference (see stores/preferences.ts,
   * surfaced on the Settings page) so it stays consistent across
   * sessions. Wrap-around is on while playing — Prev from the
   * first photo lands on the last — and off when paused so
   * manual nav respects the existing boundaries. Manual nav
   * (chevrons, ←/→, A/D) also pauses the timer, so a quick
   * back-step never has the photo "skip past" the user's click.
   * Press Space to toggle Play/Pause from anywhere in the
   * lightbox (unless a button is focused, in which case Space
   * activates that button naturally).
   *
   * Photo bytes come from /photo/{id}/raw?w=N — the server does a
   * Lanczos downsample and serves the cached JPEG. This avoids the
   * quality loss of letting the browser scale a 12 MP source down
   * to a 1408 px lightbox, and slashes bandwidth by ~10x.
   *
   * Caller provides the items array (with point IDs) and the
   * index of the currently shown item, plus a way to toggle
   * favourite.
   */
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { SvelteSet } from 'svelte/reactivity';
  import { photoUrl, addPhotoToAlbum, removePhotoFromAlbum, listAlbums, listAlbumsForFavorite } from '$lib/api/endpoints';
  import { useAlbumMembership } from '$lib/composables/useAlbumMembership';
  import { useSlideshow } from '$lib/composables/useSlideshow.svelte';
  import { useKeyboardNav } from '$lib/composables/useKeyboardNav';
  import { usePhotoPreloader } from '$lib/composables/usePhotoPreloader.svelte';
  import { useBlurhashTint } from '$lib/composables/useBlurhashTint.svelte';
  import { preferences } from '$lib/stores/preferences';
  import Icon from './Icon.svelte';
  import { blurhashToDataUrl } from './blurhash-bg';
  import ActionButton from './ActionButton.svelte';
  import {
    canAdvance,
    shouldRequestMore,
    stallForMore,
    resolveStall
  } from './lightbox-pagination';
  import Dropdown from './Dropdown.svelte';
  async function goSimilar(id: string) {
    onClose();
    await goto(resolve(`/similar/${encodeURIComponent(id)}`));
  }

  /**
   * Pick the right server-side resize width. We aim for 2x of the
   * rendered CSS width (retina) but cap at 1920 so 4K monitors
   * don't pull multi-MB files when 1920 px is enough visually.
   * Falls back to 1920 in SSR (window not available).
   */
  function lightboxWidth(): number {
    if (typeof window === 'undefined') return 1920;
    const cssWidth = window.innerWidth - 32;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    return Math.min(1920, Math.ceil(cssWidth * dpr));
  }

  type Item = {
    id: string;
    blurhash?: string | null;
    isFavorite?: boolean;
  };

  type Props = {
    items: Item[];
    index: number;
    onClose: () => void;
    onToggleFavorite?: (id: string) => void;
    /** User-created albums — passed down to the right-click
     *  "Add to album" submenu (round-5). */
    albums?: { id: number; name: string }[];
    /** Infinite navigation: when the parent's result set has more
     *  pages, arrowing forward near/at the loaded end requests the
     *  next page and continues instead of dead-ending. Both props
     *  are optional — without them the lightbox behaves exactly as
     *  before (bounded by `items`). */
    hasMore?: boolean;
    onLoadMore?: (signal?: AbortSignal) => void;
  };
  let {
    items,
    index,
    onClose,
    onToggleFavorite,
    albums,
    hasMore = false,
    onLoadMore
  }: Props = $props();

  // The lightbox manages its own cursor (`idx`) internally so
  // that prev/next + the keyboard composable can mutate it
  // without round-tripping through the parent. The `index`
  // prop is the initial position (parent decides which photo
  // to open first); the `index` prop is not expected to
  // change after mount, so capturing the initial value
  // here is the documented behaviour. (svelte-check would
  // prefer a $derived; that would re-anchor the cursor to
  // the prop every render, which is NOT what we want — the
  // user's left/right keys must not snap back to the
  // parent's `index` mid-browse.)
  // svelte-ignore state_referenced_locally
  let idx = $state(index);
  // Infinite-nav bookkeeping (see lightbox-pagination.ts):
  // `requestedAtLen` dedupes lookahead prefetch requests per loaded
  // length; `pendingStall` holds the length we stalled at when
  // `next()` was pressed exactly at the loaded end — when `items`
  // grows past it the pending press auto-advances.
  let requestedAtLen: number | null = null;
  let pendingStall = $state<number | null>(null);
  // Clamp `idx` to a valid range only when it's actually out of
  // bounds (e.g., items shrunk). Don't reset it on every items
  // update — earlier a `$effect(() => idx = clamp(index, ...))`
  // would fire mid-render with a transient items array and send
  // idx to 0, which made the Like click jump the user back to
  // the first photo (issue round-4 #1).
  $effect(() => {
    if (items.length > 0 && idx >= items.length) {
      idx = items.length - 1;
    }
  });

  // Infinite nav: when items grow past a pending stall (next() was
  // pressed at the loaded end and we asked for more), complete the
  // deferred press by auto-advancing. Deliberately does NOT trigger
  // lookahead prefetch — that lives in next() so End/Home/jump keys
  // don't silently grow the window out from under the user.
  $effect(() => {
    const r = resolveStall(pendingStall, items.length);
    if (r.advance) idx += 1;
    pendingStall = r.stall;
  });

  // Phase 6d: the blurhash decode + crossfade + freeze
  // policy are extracted to useBlurhashTint. The composable
  // owns the tint data URL, the crossfade gate (tintReady),
  // and the session-freeze flag (tintFrozen). The template
  // reads `tintLayer.tint` and `tintLayer.tintReady` (the
  // CSS class for the crossfade).
  const tintLayer = useBlurhashTint({
    getBlurhash: () => current()?.blurhash,
    decode: (blurhash, w, h) => blurhashToDataUrl(blurhash, w, h)
  });

  // Photo-load state for crossfade. The lightbox used to flash on
  // every navigation because the `<img>` was inside `{#key it.id}`,
  // which unmounted the old image and mounted a new one — the dark
  // backdrop showed through until the new fetch resolved. Now we
  // reuse the same `<img>` element across navigations (its `src`
  // updates reactively when `idx` changes) and gate its visibility
  // on `naturalWidth > 0` via a CSS transition. While the new src
  // is loading, the blurhash backdrop (`tint`) stays visible so
  // there's never a fully-blank frame.
  //
  // `photoReady` flips to false whenever the displayed src changes
  // and back to true when the next `load` event lands. We track the
  // src we last saw so a duplicate `load` event for the same photo
  // (browsers fire these on cache hits too) doesn't toggle state
  // twice.
  let photoReady = $state(false);
  let lastLoadedSrc = $state('');
  let photoSrc = $state('');

  // Adjacent-photo preloader. Hidden `<img>` elements whose `src`
  // is set to the next/previous photo's URL. The browser shares
  // the HTTP cache with the visible `<img>` and reuses the
  // decoded bitmap on navigation, so prev/next feel instant.
  //
  // Why a Set instead of just rendering all of `items.slice(idx-2,
  // idx+3)`: preloading the same URL twice would create two <img>
  // nodes holding the same data — wasteful when the user lingers
  // on a photo and `idx` stays put. The Set ensures we only mount
  // each preloader once; subsequent effect runs (caused by
  // blurhash decode, memberOf refresh, etc.) are no-ops.
  //
  // Distance: 1 behind, 2 ahead by default. The lightbox is
  // browsed forward, so the window leans ahead: warming two photos
  // forward means the SECOND press of a → run lands on an
  // already-cached image instead of starting a cold fetch. The 1s
  // slideshow preset only gives the user 1000 ms between ticks —
  // `playing` bumps that to ±3 so the auto-advance doesn't race
  // the network. Single-photo lightbox (`items.length <= 1`) skips
  // preloading entirely.
  //
  // Bandwidth respect: skip preloading when the user has
  // declared save-data via the Network Information API or the
  // `prefers-reduced-data` media query. Both are
  // feature-detected; old browsers that lack them get the
  // default preload window (a strict-power-user setting, but
  // that's the conservative default for a desktop-class
  // lightbox).
  /* Local "pressed" state for the Like / Dislike action buttons.
     Synced from the current photo only when the user navigates
     to a different photo — re-syncing after every server response
     would clobber the in‑flight toggle we just made. Pressing
     Like clears Dislike (and vice‑versa) so a photo can only be
     in one of those two states at a time. The parent (grid) still
     owns the API call. */
  let isFavorite = $state(false);
  let lastSyncedIdx = $state(-1);

  $effect(() => {
    if (idx === lastSyncedIdx) return;
    const it = items[idx];
    isFavorite = !!it?.isFavorite;
    lastSyncedIdx = idx;
  });

  // Phase 3 note: the Lightbox's toggle handlers (below) are
  // intentionally NOT migrated to usePhotoFeedback. The
  // composable assumes a stable photo id + a per-id state
  // record; the Lightbox has its own bespoke pattern — two
  // separate $state booleans synced from `items[idx]` via the
  // `lastSyncedIdx` $effect, and it fires a parent-owned
  // callback rather than calling the API itself. Forcing
  // the composable onto this shape would either lose the
  // idx-change sync semantics or add a wrapper indirection
  // with no real win. The handlers stay as-is.
  function toggleFavorite() {
    const it = current();
    if (!it) return;
    if (isFavorite) {
      isFavorite = false;
    } else {
      isFavorite = true;
    }
    onToggleFavorite?.(it.id);
  }

  // Round-6 — Add to album dropdown logic. Now lives in <Dropdown>;
  // this handler is just the API glue + lazy album loading.
  //
  // The Dropdown shows membership state via the `memberOf` Set,
  // and `onPick` carries the isMember boolean so we can decide
  // add-vs-remove. Both the photo page and the right-click menu
  // use the same pattern.
  async function ensureAlbumsLoaded() {
    if (albums && albums.length > 0) return;
    try {
      const res = (await listAlbums()) as { albums?: { id: number; name: string }[] };
      albums = res.albums ?? [];
    } catch {
      albums = [];
    }
  }

  // Membership set for the current photo: which albums it
  // already belongs to. Loaded lazily when the user opens the
  // dropdown; reloaded after every add/remove so the visual
  // indicator stays in sync with the server.
  // eslint-disable-next-line svelte/no-unnecessary-state-wrap -- reassigned on refresh; SvelteSet alone doesn't track the variable reference
  let memberOf: SvelteSet<number> = $state(new SvelteSet());

  async function refreshMembership() {
    const it = current();
    if (!it) return;
    try {
      const albumsForPhoto = await listAlbumsForFavorite(it.id);
      // Phase 4: useAlbumMembership.setAll replaces the
      // membership set in one place.
      albumMembership.setAll(albumsForPhoto.map((a) => a.id));
    } catch {
      // Leave the previous Set in place — a transient failure
      // shouldn't blank out the indicator.
    }
  }

  // Phase 4: useAlbumMembership consolidates the add/remove
  // photo-from-album mutation pattern (was duplicated
  // verbatim here and in routes/photo/[id]/+page.svelte).
  const albumMembership = useAlbumMembership({
    getMemberOf: () => memberOf,
    setMemberOf: (next) => {
      memberOf = next;
    },
    getPointId: () => current()?.id,
    onAdd: (albumId, pointId) => addPhotoToAlbum(albumId, pointId),
    onRemove: (albumId, pointId) => removePhotoFromAlbum(albumId, pointId)
  });

  async function toggleMembership(albumId: number, albumName: string) {
    await albumMembership.toggle(albumId, albumName);
  }

  function current(): Item | null {
    return items[idx] ?? null;
  }

  /* Slideshow state. The Play/Pause button in the action bar is
     always present when there's more than one photo to cycle
     through, and the timer starts idle — the user has to opt in
     by pressing Play. Auto-play on open would be jarring: the
     lightbox pops in, then 4 seconds later starts moving on its
     own with no visual indicator until the timer first fires.
     Explicit Play also sidesteps prefers-reduced-motion entirely
     (the user who cares about motion sensitivity just won't hit
     the button).

     Phase 6: the auto-advance timer + the items-length guard
     are extracted to useSlideshow. The composable owns
     `playing` + the setInterval effect; the Lightbox just
     reads `slideshow.playing` and calls `slideshow.togglePlay`
     from the action bar / keyboard handler. */
  const slideshow = useSlideshow({
    getItemsLength: () => items.length,
    getIdx: () => idx,
    setIdx: (next) => {
      idx = next;
    },
    getIntervalMs: () => $preferences.slideshowIntervalMs
  });

  // Phase 6b: useKeyboardNav consolidates the keydown
  // dispatch table (Escape, arrows, Home/End, PageUp/Down,
  // 'f' for like, Space for slideshow). The composable
  // owns the focus gates (button / text-input) and the
  // boundary clamps (idx ∈ [0, items.length-1]).
  const keyboard = useKeyboardNav({
    getItemsLength: () => items.length,
    getIdx: () => idx,
    setIdx: (next) => {
      idx = next;
    },
    onClose: () => onClose(),
    onPrev: () => prev(),
    onNext: () => next(),
    onToggleFavorite: () => toggleFavorite(),
    onToggleSlideshow: () => slideshow.togglePlay()
  });

  // Phase 6c: usePhotoPreloader owns the adjacent-photo
  // preload effect. The composable reads `idx` /
  // `slideshow.playing` / `items.length` / `photoSrc` and
  // owns the SvelteSet of preloaded URLs. The template
  // reads `preloader.preloadedSrcs` to render the hidden
  // <img> nodes.
  const preloader = usePhotoPreloader({
    getItemsLength: () => items.length,
    getCursor: () => idx,
    getIsPlaying: () => slideshow.playing,
    getPhotoSrc: () => photoSrc,
    getItemUrl: (id: string) => photoUrl(id, lightboxWidth()),
    getItemIdAt: (i: number) => items[i]?.id,
    getWidth: () => lightboxWidth()
  });

  // Re-run the photo-dependent effects whenever the current photo
  // changes (lightbox navigation, photo prop change). Photo src +
  // photoReady + membership refresh stay here; the tint is now
  // owned by useBlurhashTint (Phase 6d).
  $effect(() => {
    const it = current();
    // Compute the next src. `lightboxWidth()` is called fresh so
    // a window resize between navigations picks up the new width
    // without us needing a separate resize listener.
    const nextSrc = it ? photoUrl(it.id, lightboxWidth()) : '';
    if (nextSrc !== photoSrc) {
      photoSrc = nextSrc;
      // Mark the photo as not-ready until the new `<img>` fires
      // its `load` event. If the browser serves this from cache
      // the load fires synchronously-ish, but the $effect below
      // still sees it and flips photoReady back on.
      photoReady = false;
    }
    // Phase 6d: the tint decode / freeze / crossfade live in
    // useBlurhashTint. The composable's $effect tracks the
    // current photo's blurhash, so a no-photo / no-blurhash
    // situation automatically clears the tint and unfreezes
    // it. The membership refresh below is the only remaining
    // side effect for the photo-change case.
    // Refresh the membership indicator for the new photo. Done
    // in the same effect so a single `idx` change triggers both
    // (one round trip, no flicker between the two updates).
    void refreshMembership();
  });

  // Phase 6c: the 80-line preload $effect is now preloader
  // (usePhotoPreloader). The distance / wrap-around / data-
  // saver / eviction logic is all in the composable. The
  // composable's own $effect tracks the same deps (idx,
  // slideshow.playing, items.length, photoSrc) so no
  // placeholder effect is needed here.

  function prev() {
    // Snapshot the playing flag before mutating it. We want
    // wrap-around in two cases: (a) the slideshow is currently
    // running, and (b) the user just paused it via this very
    // click — they're still mid-slideshow mentally, so looping
    // to the previous photo is the intuitive continuation.
    // Outside that window, respect the pre-slideshow boundary
    // (prev disabled at the first photo).
    const wasPlaying = slideshow.playing;
    if (wasPlaying) slideshow.pause();
    if (!wasPlaying) {
      if (idx > 0) idx -= 1;
      return;
    }
    if (items.length === 0) return;
    idx = (idx - 1 + items.length) % items.length;
  }
  function next() {
    const wasPlaying = slideshow.playing;
    if (wasPlaying) slideshow.pause();
    if (!wasPlaying) {
      if (canAdvance(idx, items.length)) {
        idx += 1;
        // Lookahead prefetch: now that forward navigation moved us
        // near the loaded end, ask for the next page so continuous
        // arrowing rarely stalls. Only next() does this — jump keys
        // (End/Home/±10) intentionally don't grow the window.
        if (shouldRequestMore(idx, items.length, hasMore, requestedAtLen)) {
          requestedAtLen = items.length;
          onLoadMore?.();
        }
        return;
      }
      // At the loaded end: if the parent has more pages, stall and
      // request — the items-growth effect auto-advances when they
      // land, so the press is never lost.
      pendingStall = stallForMore(idx, items.length, hasMore);
      if (pendingStall !== null) {
        requestedAtLen = items.length;
        onLoadMore?.();
      }
      return;
    }
    if (items.length === 0) return;
    idx = (idx + 1) % items.length;
  }

  // Phase 6b: the 75-line onKey handler is now keyboard.handleKey.
  // The dispatch table, focus gates, and boundary clamps all live
  // in useKeyboardNav; see that module for the test coverage.
  onMount(() => {
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    // Hide the top tab bar while the lightbox is open — otherwise
    // its z-50 sticky strip bleeds through the semi-transparent
    // overlay.
    document.body.classList.add('lightbox-open');
    return () => {
      document.body.style.overflow = prevOverflow;
      document.body.classList.remove('lightbox-open');
    };
  });
</script>

<svelte:window onkeydown={keyboard.handleKey} />

<!-- svelte-ignore a11y_click_events_have_key_events -->
<div
  class="overlay"
  style={tintLayer.tint ? `--glass-tint: url(${tintLayer.tint})` : undefined}
  onclick={onClose}
  role="dialog"
  aria-modal="true"
  tabindex="-1"
>
  {#if tintLayer.tint}
    <!-- Crossfade the tint layer when its background-image swaps.
         The opacity transition runs on the same axis as the photo
         crossfade (both 150 ms), so the user perceives a single
         coordinated swap rather than two distinct layer changes.
         `.tint-ready` is set on the next animation frame after
         the bg-url changes — without the rAF gate, the browser
         would batch the style change and never run the transition
         (it has to first paint the old URL at full opacity, then
         paint the new one and animate down/up). -->
    <div
      class="tint"
      class:tint-ready={tintLayer.tintReady}
      style:background-image="var(--glass-tint)"
      aria-hidden="true"
    ></div>
  {/if}
  <!-- role="presentation" — click-swallowing layer;
       the real <button>s inside handle their own
       keydown. -->
  <div class="content" role="presentation" onclick={onClose} oncontextmenu={(e) => e.preventDefault()}>
    <button class="nav close" type="button" onclick={(e) => { e.stopPropagation(); onClose(); }} aria-label="Close">
      <Icon name="close" size={20} />
    </button>
    <button
      class="nav prev"
      type="button"
      onclick={(e) => { e.stopPropagation(); prev(); }}
      disabled={!slideshow.playing && idx === 0}
      aria-label="Previous"
    >
      <Icon name="chevron-left" size={22} />
    </button>
    {#if current()}
      <!-- Single reusable <img>; src updates reactively when idx
           changes. Crossfade via .photo { opacity } tied to
           photoReady so the new image only fades in once the
           browser has decoded it. No {#key} — tearing down the
           element on every nav caused the old flash because the
           browser would briefly render the dark backdrop until
           the new fetch resolved. -->
      <img
        class="photo"
        class:photo-ready={photoReady}
        src={photoSrc}
        alt=""
        onload={() => {
          // Guard against cache-hit replays for the same src.
          // The browser may also fire `load` after the src has
          // already changed to a different photo; we only flip
          // photoReady when the loaded src matches what we're
          // currently showing.
          if (photoSrc && photoSrc === lastLoadedSrc) return;
          lastLoadedSrc = photoSrc;
          photoReady = true;
        }}
      />
    {/if}
    <!-- Adjacent-photo preloaders. Hidden, off-screen, no
         interaction — they exist only so the browser's image
         cache + decoded bitmap are warm when the user
         navigates. `fetchpriority="low"` keeps them from
         competing with the visible <img>; `loading="eager"`
         overrides the default lazy-loading because we
         explicitly want the fetch now (not "when the
         IntersectionObserver notices this offscreen node").
         The Set is the source of truth — additions/removals
         drive mount/unmount. aria-hidden + tabindex=-1 so
         screen readers and keyboard focus skip them. -->
    {#each [...preloader.preloadedSrcs] as preloadSrc (preloadSrc)}
      <img
        class="preload"
        aria-hidden="true"
        tabindex="-1"
        alt=""
        src={preloadSrc}
        loading="eager"
        decoding="async"
        fetchpriority="low"
      />
    {/each}
    <button
      class="nav next"
      type="button"
      onclick={(e) => { e.stopPropagation(); next(); }}
      disabled={!slideshow.playing && !canAdvance(idx, items.length) && !(hasMore && items.length > 0)}
      aria-label="Next"
    >
      <Icon name="chevron-right" size={22} />
    </button>
  </div>

  <!-- role="presentation" — bottom bar that swallows
       clicks (so the bar tap doesn't close the lightbox).
       Real <button>s inside handle their own keydown. -->
  <div class="bar glass-strong" role="presentation" onclick={(e) => e.stopPropagation()} oncontextmenu={(e) => e.preventDefault()}>
      <span class="count">{idx + 1} / {items.length}</span>
      {#if items.length >= 2}
        <ActionButton
          onclick={slideshow.togglePlay}
          title={slideshow.playing ? 'Pause slideshow (Space)' : 'Play slideshow (Space)'}
          ariaPressed={slideshow.playing ? 'true' : 'false'}
        >
          <Icon name={slideshow.playing ? 'pause' : 'play'} size={14} />
        </ActionButton>
      {/if}
      <ActionButton
        onclick={toggleFavorite}
        title="Like (F)"
        ariaPressed={isFavorite ? 'true' : 'false'}
      >
        Like
      </ActionButton>
      <ActionButton
        onclick={() => current() && goSimilar(current()!.id)}
        title="Open the dedicated most-similar page for this photo"
      >
        Most similar
      </ActionButton>
      <Dropdown
        items={(albums ?? []).map((a) => ({ id: a.id, label: a.name }))}
        onPick={async (it, isMember) => {
          await toggleMembership(it.id as number, it.label);
          // `isMember` is now passed for callers that want to
          // branch on it, but toggleMembership already read the
          // canonical `memberOf` Set before the click so it
          // always flips the right way.
          void isMember;
        }}
        label="Add this photo to an album"
        align="up"
        emptyMessage="No albums yet — create one from the Albums page."
        memberOf={memberOf}
      >
        {#snippet trigger({ open, toggle })}
          <ActionButton
            onclick={async () => {
              if (!open) {
                await ensureAlbumsLoaded();
                await refreshMembership();
              }
              toggle();
            }}
            title="Add this photo to an album"
            ariaHaspopup="menu"
            ariaExpanded={open}
          >
            Add to album
          </ActionButton>
        {/snippet}
      </Dropdown>
      <ActionButton
        href={current() ? photoUrl(current()!.id) : '#'}
        target="_blank"
        rel="noopener"
      >Open raw</ActionButton>
    </div>
</div>

<style>
  .overlay {
    position: fixed;
    inset: 0;
    /* Above the top bar (z-50) and every other layer. */
    z-index: 500;
    /* Round-66: frosted glass over the page (was a dark
       rgba(0,0,0,0.45) scrim that read as "dark surface"
       against the rest of the light-themed app). The new
       value is the same glass vocabulary as the home page
       PromptChips card — semi-transparent white over a
       heavy backdrop blur. The page shows through as a
       soft pastel wash instead of being blacked out,
       matching the glass aesthetic Isaac called out. The
       photo cell below stays near-opaque so the actual
       image isn't tinted by the overlay. */
    /* Round-66d: outer overlay is transparent — the
       page/grid below shows through normally, no dimming.
       Isaac: 'outer part shouldn't be transparent. It should
       just be grid as normal.' The focus on the photo comes
       from the chrome (nav buttons, action bar) and the
       glass photo cell on top, not from dimming the page.
       No backdrop-filter either — there's nothing to blur. */
    background: transparent;
    /* Two stacked rows: the image region (content) and the action
       bar. The row-gap is the sole source of spacing between
       them — .content has no border or margin contributing extra
       space. */
    display: grid;
    grid-template-rows: 1fr auto;
    row-gap: var(--s-3);
    padding: var(--s-4);
    box-sizing: border-box;
    animation: fade var(--t-med) var(--ease-out);
  }
  /* When the lightbox is open, the top tab bar would otherwise
     bleed through the semi-transparent overlay (rgba 0.55). Hide
     it via a body class so the user isn't fighting two layers of
     nav at once. */
  :global(body.lightbox-open .topbar) {
    display: none;
  }
  @keyframes fade {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
.tint {
    position: absolute;
    /* inset: 0 — the previous `inset: -40px` made the tint div
       spill 40px outside the overlay on every side, and the
       60px blur on its background then bled back across the
       viewport edge as a visible colour ring. Containing it
       inside the overlay stops that ring without losing the
       tint (the .overlay's 40% dark scrim still lets it read
       through — what you see is the same colour, minus the
       harsh edge). */
    inset: 0;
    /* background-image is set via inline style from the JS blurhash
       decode; the rest of the shorthand stays here so the inline
       style only has to ship the data-URL, not repeat the layout
       commands. Default opacity 0 forces the layer to animate in
       when `.tint-ready` is applied (see the template note above
       for why the rAF gate matters). */
    background-repeat: no-repeat;
    background-position: center;
    background-size: cover;
    /* Round-66d: outer overlay is transparent, so the .tint
       layer (blurhash-tinted blur) no longer makes sense —
       there's no scrim to colour-wash. The layer is kept in
       the DOM (the JS still sets the blurhash background-image)
       but with opacity 0 so it's invisible. The page grid
       below shows through normally, exactly as Isaac asked. */
    opacity: 0;
    pointer-events: none;
    z-index: -1;
  }
  .tint.tint-ready {
    opacity: 0;
  }
  .content {
    /* Fill the first (1fr) row of the overlay grid. The grid
       already gives it the vertical space left after the action
       bar, so we just need width/height 100% inside that row. */
    position: relative;
    /* Include the 1px border inside the cell's width/height so
       100% on .photo never overflows the overlay. */
    box-sizing: border-box;
    width: 100%;
    height: 100%;
    display: grid;
    place-items: center;
    border-radius: var(--r-3);
    overflow: hidden;
    /* Round-66d: removed backdrop-filter. Isaac wanted the
       page grid to show through normally (no blur); the
       translucent white fill (0.4) keeps the inner cell
       as a glass surface but the underlying page stays
       sharp. */
    /* Round-70b: match the home page PromptChips glass
       vocabulary — since revised: the home page cards moved to
       the opaque .glass utility (Round-38 direction: frosted
       glass doesn't read on the light pastel backdrop), but the
       Lightbox deliberately keeps the translucent treatment.
       It floats over a full-bleed photo, where the blur+wash
       reads as frosted glass against real content rather than
       against a flat pastel wash. */
    background: var(--glass-1);
    backdrop-filter: var(--glass-medium);
    -webkit-backdrop-filter: var(--glass-medium);
    border: 1px solid var(--glass-edge);
  }
  .photo {
    display: block;
    /* Fill the entire photo cell above the action bar while
       preserving the image's natural aspect ratio. Absolute
       positioning + inset 0 is the most reliable way to size a
       replaced element (img) against a grid cell without being
       overridden by its intrinsic dimensions. object-fit: contain
       then scales the rendered image up or down to fit inside
       that box — small thumbnails get enlarged, large ones get
       shrunk, nothing crops and nothing overflows. */
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
    border-radius: var(--r-2);
    box-shadow: var(--shadow-3);
    /* Crossfade between photos. Default opacity 0 keeps the dark
       backdrop visible until the <img> fires its `load` event,
       at which point .photo-ready is applied and the transition
       fades the new image in over ~150 ms. The old `{#key}`
       approach unmounted and remounted the element on every nav,
       so the user briefly saw the bare backdrop — this is the
       fix. Tuned to 150 ms: long enough to mask the decode
       jank, short enough that fast slideshow ticks don't smear
       into a blur. The transition is opacity-only so the
       gallery geometry stays stable across swaps. */
    opacity: 0;
    transition: opacity 150ms var(--ease-out);
  }
  .photo.photo-ready {
    opacity: 1;
  }
  /* Adjacent-photo preloaders. Visually inert — the browser
     still renders the image into its image cache, but we hide
     the element from the user. Zero-size off-screen positioning
     is the lightest way to do this: `display: none` would
     cancel the fetch in some browsers, `visibility: hidden`
     leaves the decoded bitmap off the compositor. `position:
     fixed` with a far-offscreen top keeps them in the layout
     tree (so the browser considers them "visible enough" to
     fetch + decode) but out of any viewport. */
  .preload {
    position: fixed;
    top: -10000px;
    left: -10000px;
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
  }
  .nav {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: var(--glass-fill-light);
    border: 1px solid var(--glass-edge);
    color: var(--fg-1);
    backdrop-filter: var(--glass-medium);
    -webkit-backdrop-filter: var(--glass-medium);
    font-size: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transition: background var(--t-fast);
    z-index: 1;
  }
  .nav:hover {
    background: var(--accent-soft);
    color: var(--accent);
    border-color: var(--accent);
  }
  .nav:disabled { opacity: 0.3; cursor: not-allowed; }
  .prev { left: 12px; }
  .next { right: 12px; }
  .close {
    top: 12px;
    right: 12px;
    transform: none;
    font-size: 22px;
  }
  .bar {
    /* Grid item in the overlay's second row. justify-self + align-self
       centre the pill horizontally and vertically inside that row,
       so the action bar is always centred between the dialog's left
       and right edges regardless of viewport width. */
    justify-self: center;
    align-self: center;
    display: flex;
    gap: var(--s-3);
    align-items: center;
    padding: var(--s-1) var(--s-3);
    border-radius: var(--r-pill);
    box-shadow: var(--shadow-2);
  }
  .count {
    color: var(--fg-2);
    font-size: var(--fs-sm);
    padding: 0 var(--s-1);
    /* Same rationale as .action: never wrap "1" and "/ 20" onto
       separate lines when the bar is tight. */
    white-space: nowrap;
    flex-shrink: 0;
  }

</style>
