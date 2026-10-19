<script lang="ts">
  /**
   * For You — Gumbel-max sample from a Qdrant recommend() pool
   * of `pool_size` candidates ranked by relevance to the user's
   * taste (likes minus dislikes).
   *
   * The temperature slider controls the Gumbel noise scale:
   *   T=0   → deterministic top-K, the most taste-relevant
   *           photos in recommend() order. Curated.
   *   T=0.05 → mostly top-K, occasional surprise. Default.
   *   T=1.0 → near-uniform over the pool. Approaches /random
   *           behaviour.
   *
   * The pool slider controls how many recommend() candidates
   * the Gumbel draw is taken from. Larger pool = more variety
   * but slower first page.
   *
   * Round-91: the page no longer auto-fetches. The user has
   * to press the "Surprise me" button to fetch a deck. The
   * sliders only update pending state. The button label is
   * always "Surprise me" — no state suffix — clicking it
   * always commits the current slider values with a fresh
   * seed (a new random draw). Same text as the home page's
   * Surprise me button.
   *
   * The visual is one .card.glass (same as the home page's
   * Additional options panel) with the two sliders directly
   * inside, no card title, no description. The page header
   * says what the page is. The button is right-aligned
   * below the card, mirroring the home page's "Search"
   * button placement.
   *
   * Round-88. Replaces the previous uniform-shuffle-the-top-1%
   * implementation. The uniform shuffle destroyed the ranking
   * signal — the user could not tell the difference from
   * /random. Gumbel-max preserves the ranking as a
   * controllable knob.
   */
  import { forYouFeed, likePoint, unlikePoint, dislikePoint } from '$lib/api/endpoints';
  import { usePhotoFeedback } from '$lib/composables/usePhotoFeedback';
  import {
    POOL_STOPS_FOR_YOU,
    DEFAULT_POOL_INDEX_FOR_YOU,
    DEFAULT_TEMPERATURE,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX
  } from '$lib/api/limits';
  import { preferences, GRID_SIZE_CONFIG } from '$lib/stores/preferences';
  import { SvelteMap } from 'svelte/reactivity';
  import PhotoGrid from '$lib/components/PhotoGrid.svelte';
  import PageHeader from '$lib/components/PageHeader.svelte';
  import Button from '$lib/components/Button.svelte';

  type Item = {
    id: string;
    path?: string;
    score?: number;
    score_str?: string;
    blurhash?: string | null;
    is_favorite?: boolean;
    is_disliked?: boolean;
  };

  // Same per-grid-size scaling as home and /random: 4 rows ×
  // maxCols at every size, so the first page fills the viewport
  // regardless of which density the user picked.
  const PAGE = $derived(GRID_SIZE_CONFIG[$preferences.gridSize].pageSize);
  // Phase 1: temperature / pool bounds now live in
  // $lib/api/limits (POOL_STOPS_FOR_YOU, DEFAULT_POOL_INDEX_FOR_YOU,
  // TEMPERATURE_MIN, TEMPERATURE_MAX, DEFAULT_TEMPERATURE) so the
  // home page Surprise Me and the for-you Surprise Me read the
  // same source of truth. The local POOL_STOPS / DEFAULT_POOL_INDEX
  // / DEFAULT_TEMPERATURE / TEMPERATURE_* consts are gone.

  let items = $state<Item[]>([]);
  let loading = $state(false);
  let hasMore = $state(false);
  let nextPage = $state(0);
  // Sliders update these freely; the "Surprise me" button
  // is the only thing that commits them into a request.
  let temperature = $state(DEFAULT_TEMPERATURE);
  let poolSizeIndex = $state(DEFAULT_POOL_INDEX_FOR_YOU);
  let poolSize = $derived(POOL_STOPS_FOR_YOU[poolSizeIndex]);
  let sessionId = $state<string | null>(null);
  // The seed is set fresh on every "Surprise me" press.
  // Same seed within a deck → same first page; paginated
  // scroll walks the same deck forward. Different seed →
  // fresh deck.
  let mountSeed = $state(0);
  let errorMsg = $state<string | null>(null);
  // AbortController for the in-flight first-page request.
  // If the user clicks the button again while a request is
  // in flight, we cancel the previous one before firing the
  // new one. Without this, two requests can race and the
  // later one (or earlier, depending on response order) can
  // clobber the deck.
  let inFlight = $state<AbortController | null>(null);

  function rollSeed(): number {
    if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
      const buf = new Uint32Array(1);
      crypto.getRandomValues(buf);
      return buf[0];
    }
    return Math.floor(Math.random() * 0x100000000);
  }

  function onPoolIndexChange(idx: number) {
    const clamped = Math.max(0, Math.min(idx, POOL_STOPS_FOR_YOU.length - 1));
    if (clamped === poolSizeIndex) return;
    poolSizeIndex = clamped;
  }

  function onTemperatureChange(v: number) {
    if (v === temperature) return;
    temperature = v;
  }

  function formatPool(idx: number): string {
    return POOL_STOPS_FOR_YOU[idx].toLocaleString();
  }

  async function onSurprise() {
    // Cancel any in-flight request so the previous fetch
    // doesn't clobber the new deck.
    inFlight?.abort();
    const ctrl = new AbortController();
    inFlight = ctrl;
    mountSeed = rollSeed();
    sessionId = null;
    items = [];
    nextPage = 0;
    hasMore = false;
    errorMsg = null;
    loading = true;
    try {
      const res = await forYouFeed({
        limit: PAGE,
        offset: 0,
        temperature,
        pool_size: poolSize,
        seed: mountSeed,
        signal: ctrl.signal
      });
      items = (res?.results ?? []) as Item[];
      nextPage = 1;
      sessionId = res?.session_id ?? null;
      hasMore = !!res?.has_more && items.length > 0;
    } catch (e: unknown) {
      if (ctrl.signal.aborted) return;
      // 400 with "for-you requires at least one favourite"
      // is a friendly error, not a server problem. Surface
      // it inline.
      const eAny = e as { body?: { detail?: string }; message?: string };
      const detail = eAny?.body?.detail ?? eAny?.message ?? '';
      if (typeof detail === 'string' && detail.includes('favourite')) {
        errorMsg = 'Like a few photos first to seed the taste signal.';
      } else {
        errorMsg = 'Could not load recommendations. Please try again.';
      }
      hasMore = false;
    } finally {
      if (!ctrl.signal.aborted) loading = false;
      if (inFlight === ctrl) inFlight = null;
    }
  }

  async function loadMore(signal?: AbortSignal) {
    if (loading || !hasMore) return;
    loading = true;
    try {
      const res = await forYouFeed({
        limit: PAGE,
        offset: nextPage * PAGE,
        temperature,
        pool_size: poolSize,
        seed: mountSeed,
        session: sessionId ?? undefined,
        signal
      });
      const more = (res?.results ?? []) as Item[];
      items = [...items, ...more];
      nextPage += 1;
      sessionId = res?.session_id ?? sessionId;
      hasMore = !!res?.has_more && more.length > 0;
    } catch (_e) {
      if (signal?.aborted) return; // clean cancel from pre-fetch retrigger
      hasMore = false;
    } finally {
      loading = false;
    }
  }

  // Round-9 perf: O(1) item lookup + update via shadow Map.
  // eslint-disable-next-line svelte/no-unnecessary-state-wrap -- reassigned by $effect; SvelteMap alone doesn't track the variable reference
  let indexById: SvelteMap<string, number> = $state(new SvelteMap());

  $effect(() => {
    const m = new SvelteMap<string, number>();
    for (let i = 0; i < items.length; i++) m.set(items[i].id, i);
    indexById = m;
  });

  // Phase 3: usePhotoFeedback consolidates the like/unlike
  // mutation pattern. for-you's dislike is 1-way (set the
  // flag, no toggle, no rollback — the dislikes page is
  // the only place to un-dislike), so the composable only
  // handles the favourite toggle here.
  const feedback = usePhotoFeedback({
    get: (id) => {
      const idx = indexById.get(id);
      if (idx === undefined) return undefined;
      const it = items[idx];
      return { is_favorite: !!it.is_favorite, is_disliked: !!it.is_disliked };
    },
    set: (id, patch) => {
      const idx = indexById.get(id);
      if (idx === undefined) return;
      items[idx] = { ...items[idx], ...patch };
    },
    onLike: (id) => likePoint(id),
    onUnlike: (id) => unlikePoint(id)
  });

  function onToggleFavorite(id: string) {
    void feedback.toggleFavorite(id);
  }

  async function onDislike(id: string) {
    try {
      await dislikePoint(id);
    } catch {
      /* noop: dislike failure is non-critical */
    }
  }
</script>

<svelte:head>
  <title>For you · Lustre</title>
</svelte:head>

<PageHeader
  title="For you"
  subtitle="Photos matched to your taste, with a knob for how much variety."
/>

<!--
  Round-91: single .card.glass (same pattern as Settings) with
  the two sliders directly inside. No "Surprise me" card title,
  no description — the page header "For you" + subtitle already
  says what the page is. The "Surprise me" button below mirrors
  the home page's button label (text-only, no icon, no state
  suffix — clicking it always commits the current slider
  values with a fresh seed).
-->
<section class="card glass">
  <div class="slider-row">
    <span class="slider-label">
      Temperature: <strong>{temperature.toFixed(2)}</strong>
      <span class="hint">— 0 = top-K, 1 = roughly uniform</span>
    </span>
    <input
      type="range"
      min={TEMPERATURE_MIN}
      max={TEMPERATURE_MAX}
      step="0.01"
      value={temperature}
      oninput={(e) => onTemperatureChange(parseFloat((e.target as HTMLInputElement).value))}
      aria-label="Surprise temperature"
    />
  </div>
  <div class="slider-row">
    <span class="slider-label">
      Pool: <strong>{formatPool(poolSizeIndex)}</strong>
      <span class="hint">— candidates the Gumbel draw is taken from. "Max" is the Qdrant search ceiling (50k).</span>
    </span>
    <input
      type="range"
      min="0"
      max={POOL_STOPS_FOR_YOU.length - 1}
      step="1"
      value={poolSizeIndex}
      oninput={(e) => onPoolIndexChange(parseInt((e.target as HTMLInputElement).value, 10))}
      aria-label="Pool size"
    />
  </div>
</section>

<div class="actions">
  <Button
    variant="primary"
    size="md"
    onclick={onSurprise}
    disabled={loading}
    title="Roll a new deck with the current slider values"
  >
    Surprise me
  </Button>
</div>

{#if errorMsg}
  <div class="error-msg">{errorMsg}</div>
{/if}

{#if items.length > 0 || loading}
  <PhotoGrid
    {items}
    {hasMore}
    {loading}
    onLoadMore={loadMore}
    {onToggleFavorite}
    {onDislike}
  />
{/if}

<style>
  /* Round-91: the Surprise card is a single .card.glass with
     sliders directly inside — no nested .surprise-controls
     block, no orphan "Surprise me" label. The button sits
     below the card, right-aligned, mirroring the home page's
     "Search" button placement. */
  .card {
    /* Match the home page's AdditionalFilters width so the two
       pages feel like one app: the home page uses
       --grid-width as the max-width; settings uses the same.
       width: 100% makes the block-level card fill the available
       space up to --grid-width — without it the card shrinks
       to the slider's intrinsic width and ends up narrower
       than the actions row below it, which looks misaligned. */
    width: 100%;
    max-width: var(--grid-width, 1200px);
    margin: 0 auto var(--s-3);
    padding: var(--s-3);
    display: flex;
    flex-direction: column;
    gap: var(--s-3);
    box-sizing: border-box;
  }
  .slider-row {
    display: flex;
    flex-direction: column;
    gap: var(--s-1);
  }
  .slider-label {
    color: var(--fg-2);
    font-size: var(--fs-sm);
    display: flex;
    align-items: baseline;
    gap: 4px;
    flex-wrap: wrap;
  }
  .slider-label strong {
    color: var(--fg-1);
    font-weight: 600;
  }
  .slider-label .hint {
    color: var(--fg-2);
    opacity: 0.7;
  }
  .slider-row input[type='range'] {
    width: 100%;
    accent-color: var(--accent);
  }
  .actions {
    /* Match the card's width so the button is on the
       same right edge as the card's right edge. Without
       width:100% the flex container would shrink-to-fit
       the button and "justify-content: flex-end" would
       just hug the button to the left. */
    width: 100%;
    max-width: var(--grid-width, 1200px);
    margin: 0 auto var(--s-3);
    padding: 0 var(--s-3);
    display: flex;
    justify-content: flex-end;
  }
  .error-msg {
    max-width: var(--grid-width, 1200px);
    margin: 0 auto var(--s-3);
    padding: var(--s-2) var(--s-3);
    color: var(--accent);
    text-align: center;
  }
</style>
