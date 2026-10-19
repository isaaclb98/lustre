<script lang="ts">
  /**
   * Home page — single landing experience per the merge (round‑23).
   *
   * Combines what used to live on `/` and `/search`:
   *   - PromptChips at the top
   *   - Inline search results (paginated via loadMore on scroll)
   *   - For‑You row at the bottom (sample of recommendations)
   *
   * URL ⇄ state sync:
   *   - On mount, read URL params into local state, fire a search.
   *   - When state changes, write back to the URL via replaceState
   *     so reloads and share‑links restore the same query.
   *   - Search runs ONLY on explicit Search button clicks (issue #6);
   *     typing or toggling a select updates the URL but does not
   *     hit /api/search.
   *
   * PromptChips is a pure UI child — the page owns the truth.
   */
  import { page } from '$app/stores';
  import { onMount, tick } from 'svelte';
  import { browser } from '$app/environment';
  import { resolve } from '$app/paths';
  import PromptChips from '$lib/components/PromptChips.svelte';
  import SavedSearchesMenu from '$lib/components/SavedSearchesMenu.svelte';
  import AdditionalFilters from '$lib/components/AdditionalFilters.svelte';
  import PhotoGrid from '$lib/components/PhotoGrid.svelte';
  import {
    search,
    likePoint,
    unlikePoint,
    dislikePoint,
    getSystemStatus,
    getAdminIndexStatus
  } from '$lib/api/endpoints';
  import { usePhotoFeedback } from '$lib/composables/usePhotoFeedback';
  import { readUrl, writeUrl, pushUrlSearch } from '$lib/state/urlState';
  import type { SavedSearch } from '$lib/api/endpoints';
  import { POOL_STOPS_HOME, DEFAULT_POOL_INDEX_HOME } from '$lib/api/limits';
  import { preferences, GRID_SIZE_CONFIG } from '$lib/stores/preferences';
  import Button from '$lib/components/Button.svelte';
  import PageHeader from '$lib/components/PageHeader.svelte';

  type Item = {
    id: string;
    path?: string;
    score?: number;
    score_str?: string;
    blurhash?: string | null;
    is_favorite?: boolean;
    is_disliked?: boolean;
  };

  // Per-grid-size page size — drives both the initial search
  // limit and every loadMore() call. Scales with the grid so
  // every size fills ~4 rows on first load: Large 24 (6×4),
  // Medium 32 (8×4), Small 40 (10×4). Reads from the same
  // GRID_SIZE_CONFIG the virtualizer + CSS use, so a Settings
  // → Grid size change applies to page size too.
  //
  // GRID_PAGE_SIZE in $lib/api/limits is now a back-compat
  // export for callers that don't have a gridSize preference
  // (the For You page uses its own sampling math). Keep the
  // constant there as the documented default (24 = Large's
  // pageSize).
  const PAGE = $derived(GRID_SIZE_CONFIG[$preferences.gridSize].pageSize);

  // Composer state (hoisted from PromptChips).
  let positives = $state<string[]>([]);
  let negatives = $state<string[]>([]);
  let input = $state('');
  let mode = $state<'pos' | 'neg'>('pos');
  let filename = $state('');
  let collections = $state<string[]>([]);
  let filtersOpen = $state(false);

  // Results state.
  let items = $state<Item[]>([]);
  let loading = $state(false);
  let hasMore = $state(false);
  let offset = $state(0);
  let error = $state<string | null>(null);
  let ctrl: AbortController | null = null;
  let hasSearched = $state(false);
  // Round-31: show a one-time empty-index prompt when the user has
  // never indexed anything. Detected on mount via /api/admin/index/status;
  // disappears once any job has run successfully (last_run_at != null).
  let indexIsEmpty = $state(false);
  // Round‑29: when set, the page treats this as a "search by album
  // centroid" — the PromptChips is hidden and reload() hits the
  // centroid endpoint instead of /api/search. URL ?centroid=...
  // sets this on mount; writing back to URL is suppressed.
  let activeCentroid = $state<string | null>(null);

  // Round-78: Surprise Me pre-generated deck state. When
  // `surpriseOn` is true, reload() appends ?surprise=true plus
  // the three knobs (temperature, pool) to the request and stores
  // the response's session_id so loadMore() can walk the cached
  // deck without recomputing. A new search auto-rolls a fresh
  // deck server-side (the route generates a random seed when
  // the client omits surprise_seed); the session is invalidated
  // by reload() which clears surpriseSession.
  // Corpus size is loaded once on mount so the "All" label
  // shows the actual count. The route clamps the pool to the
  // corpus at request time too, so a stale UI value is
  // recoverable (the request will just use the lower of the two).
  let corpusSize = $state<number | null>(null);
  let surpriseOn = $state(false);
  let surpriseTemperature = $state(0.05);
  // Pool slider stops. The rightmost is the upper bound for
  // the route (currently 50k — see POOL_SIZE_MAX in
  // search/surprise_deck.py for the why). 50k is the practical
  // ceiling for a Qdrant HNSW `limit` on a 2M-point corpus:
  // beyond it, the server-side 30s timeout fires before the
  // search returns the candidate set. A Gumbel draw over 50k
  // already gives a wide distribution because the score
  // distribution has a long tail (~0.04-0.06 for the bulk of
  // the pool, vs ~0.16 for the top-200). The slider is
  // log-spaced so the wide range is reachable in 5 ticks.
  // Round-90: POOL_STOPS also lives in AdditionalFilters (the
  // Surprise section is rendered there now) so the slider
  // markup and the size-array stay in sync.
  // Phase 1: now imported from $lib/api/limits so the home page,
  // AdditionalFilters, and the for-you page all read the same
  // list of legal values. The local POOL_STOPS const is gone.
  let surprisePoolIndex = $state(DEFAULT_POOL_INDEX_HOME);
  let surprisePool = $derived(POOL_STOPS_HOME[surprisePoolIndex]);
  let surpriseSession = $state<string | null>(null);
  let surpriseTotal = $state<number | null>(null);

  function onPoolIndexChange(idx: number) {
    surprisePoolIndex = Math.max(0, Math.min(idx, POOL_STOPS_HOME.length - 1));
  }

  // Phase 7: bindToUrl-style helpers in $lib/state/urlState
  // replace the hand-rolled readFromUrl / writeToUrl pair.
  // The schema declares which fields the URL carries and
  // what kind (array vs string). Defaults are the empty
  // state — any field at its default is elided from the
  // URL.
  const urlSchema = {
    positives: { kind: 'array' as const, param: 'positives' },
    negatives: { kind: 'array' as const, param: 'negatives' },
    filename: { kind: 'string' as const, param: 'filename' },
    collections: { kind: 'array' as const, param: 'collection' }
  };
  const urlDefaults = {
    positives: [] as string[],
    negatives: [] as string[],
    filename: '',
    collections: [] as string[]
  };

  function readFromUrl() {
    const r = readUrl($page.url.search, urlSchema);
    positives = r.positives;
    negatives = r.negatives;
    filename = r.filename;
    collections = r.collections;
    // activeCentroid isn't part of the writable schema —
    // it's read once on mount but never written back
    // (writeToUrl bails out while it's set, see the
    // centroid-search comment below). Read it separately
    // so deep links like /?centroid=album_2 still work.
    activeCentroid = $page.url.searchParams.get('centroid');
    filtersOpen = !!filename || collections.length > 0;
    // Round-92: removed the `mode=sample` URL param. The
    // per-card "Surprise" button on /albums that wrote it was
    // removed because the in-panel Surprise toggle is now
    // strictly more capable (Gumbel-max + temperature + pool
    // sliders, not a per-refresh cluster re-roll). External
    // links with `?mode=sample` are silently treated as the
    // default centroid search — the `mode=` query param is
    // ignored here and never sent to the backend.
  }

  function writeToUrl() {
    if (!browser) return;
    // Round‑29: don't clobber the ?centroid= param while we're
    // running a centroid search — that would cause the URL to
    // flicker and could re-trigger reload via onMount.
    if (activeCentroid) return;
    const next = writeUrl(
      { positives, negatives, filename, collections },
      urlSchema,
      urlDefaults
    );
    pushUrlSearch($page.url.search, next);
  }

  function addPrompt(text: string, kind: 'pos' | 'neg') {
    if (kind === 'pos') {
      if (!positives.includes(text)) positives = [...positives, text];
    } else {
      if (!negatives.includes(text)) negatives = [...negatives, text];
    }
  }
  function removePositive(i: number) {
    positives = positives.filter((_, idx) => idx !== i);
  }
  function removeNegative(i: number) {
    negatives = negatives.filter((_, idx) => idx !== i);
  }
  function toggleCollection(name: string) {
    collections = collections.includes(name)
      ? collections.filter((c) => c !== name)
      : [...collections, name];
  }

  async function reload() {
    if (ctrl) ctrl.abort();
    ctrl = new AbortController();
    offset = 0;
    items = [];
    error = null;
    hasMore = false;
    surpriseSession = null;
    surpriseTotal = null;

    // Round‑29: centroid search activates on `?centroid=` even if
    // the composer is empty. Otherwise the user must fill a prompt
    // before the search runs (which doesn't make sense — the
    // centroid IS the query).
    //
    // A collections-only filter is also a valid trigger: the user
    // wants to scroll one specific library without typing any
    // semantic prompt.
    const hasCollectionFilter = collections.length > 0;
    const active =
      activeCentroid ||
      positives.length ||
      negatives.length ||
      filename.trim() ||
      hasCollectionFilter;
    if (!active) {
      loading = false;
      hasSearched = false;
      return;
    }
    hasSearched = true;
    loading = true;
    // New search: drop the cached session so the server's
    // auto-random seed rolls a fresh deck. Without this, a
    // search-then-search would hit the same cached deck
    // (because the cache key would re-derive to the same value
    // if all params were equal).
    surpriseSession = null;
    surpriseTotal = null;
    try {
      const res = await search(
        {
          positives,
          negatives,
          filename,
          limit: PAGE,
          offset: 0,
          centroid: activeCentroid ?? undefined,
          // Round-92: `centroidMode` is no longer forwarded.
          // The old "centroid" / "sample" discriminator (which
          // the per-card Surprise button on /albums used to
          // drive) is gone — the in-panel Surprise toggle is
          // the only randomisation UI now, and it composes
          // with `centroid` (not `sample`) on the backend.
          collections: collections.length ? collections : undefined,
          // Round-78: pass the surprise knobs through; the
          // backend caches the deck under a hash of these plus
          // the search params, so the same combo always returns
          // the same deck. We deliberately omit surpriseSeed —
          // the server auto-generates a fresh random seed on
          // every new search so a fresh query gives a fresh deck.
          surprise: surpriseOn,
          surpriseTemperature: surpriseOn ? surpriseTemperature : undefined,
          surprisePool: surpriseOn ? surprisePool : undefined,
          surpriseSession: surpriseOn ? surpriseSession ?? undefined : undefined,
        },
        ctrl.signal
      );
      items = (res?.results ?? []) as Item[];
      offset = items.length;
      hasMore = items.length >= PAGE && !!res?.has_more;
      // Capture the surprise session for the next page; without
      // this, loadMore() would either rebuild the deck (slow)
      // or send surprise_session="" and miss the cache (also
      // slow). Set on every response so the latest session id
      // is the one we walk.
      if (surpriseOn) {
        const sid = (res as { session_id?: string | null })?.session_id;
        if (sid) surpriseSession = sid;
        const total = (res as { session_total?: number | null })?.session_total;
        surpriseTotal = total ?? null;
      }
    } catch (e: unknown) {
      if ((e as { name?: string }).name === 'AbortError') return;
      error = (e as Error).message ?? 'search failed';
    } finally {
      loading = false;
    }
  }

  async function loadMore(signal?: AbortSignal) {
    if (loading || !hasMore) return;
    loading = true;
    try {
      const res = await search({
        positives, negatives, filename,
        limit: PAGE, offset,
        centroid: activeCentroid ?? undefined,
        // Round-92: `centroidMode` no longer forwarded (see
        // reload() for the full rationale).
        collections: collections.length ? collections : undefined,
        surprise: surpriseOn,
        surpriseTemperature: surpriseOn ? surpriseTemperature : undefined,
        surprisePool: surpriseOn ? surprisePool : undefined,
        surpriseSession: surpriseOn ? surpriseSession ?? undefined : undefined,
      }, signal);
      const more = (res?.results ?? []) as Item[];
      items = [...items, ...more];
      offset += more.length;
      hasMore = more.length >= PAGE && !!res?.has_more;
      // Refresh the session id in case the backend regenerated
      // it (e.g. a cache miss with a fresh key).
      if (surpriseOn) {
        const sid = (res as { session_id?: string | null })?.session_id;
        if (sid) surpriseSession = sid;
        const total = (res as { session_total?: number | null })?.session_total;
        surpriseTotal = total ?? null;
      }
    } catch (_e) {
      if (signal?.aborted) return; // clean cancel from pre-fetch retrigger
      hasMore = false;
    } finally {
      loading = false;
    }
  }

  // Phase 3: usePhotoFeedback consolidates the like/unlike
  // mutation pattern that was duplicated across home,
  // for-you, Lightbox, and the photo page. Home's dislike
  // remains its own 1-way function (the dislikes page is
  // the only place to un-dislike), so the composable only
  // handles the favourite toggle here.
  const feedback = usePhotoFeedback({
    get: (id) => {
      const it = items.find((x) => x.id === id);
      return it ? { is_favorite: !!it.is_favorite, is_disliked: !!it.is_disliked } : undefined;
    },
    set: (id, patch) => {
      items = items.map((x) => (x.id === id ? { ...x, ...patch } : x));
    },
    onLike: (id) => likePoint(id),
    onUnlike: (id) => unlikePoint(id)
  });

  async function onToggleFavorite(id: string) {
    await feedback.toggleFavorite(id);
  }

  async function onDislike(id: string) {
    try {
      await dislikePoint(id);
      items = items.map((x) =>
        x.id === id ? { ...x, is_disliked: true } : x
      );
    } catch {
      /* noop: dislike failure is non-critical */
    }
  }

  // Run on Search button (from composer) and on initial mount if
  // URL has params (so deep links like /?positives=cat or
  // /?centroid=album_2 still load).
  onMount(async () => {
    // Fetch the corpus size so the pool slider's "Max" label
    // can show "All (N)" when the corpus is smaller than the
    // 50k slider cap (a fresh install with 30k photos
    // shouldn't be limited to 30k of 50k for no reason).
    // Fire-and-forget — a stale or null corpusSize just means
    // the label shows "Max" without a number, not that the
    // request fails. The route clamps the pool to the corpus
    // at request time so any UI/server drift is handled
    // gracefully.
    // Phase 2: uses getSystemStatus() instead of a raw
    // fetch(). The endpoint wrapper handles credentials,
    // dev-mode Zod validation, and timeout/abort plumbing
    // uniformly with the rest of the app.
    void getSystemStatus()
      .then((d) => {
        if (d.qdrant_count > 0) corpusSize = d.qdrant_count;
      })
      .catch(() => {
        /* leave corpusSize null */
      });
    readFromUrl();
    await tick();
    if (
      activeCentroid ||
      positives.length ||
      negatives.length ||
      filename.trim()
    ) {
      await reload();
    }
    // Round-31: surface a one-time prompt when the library is
    // empty so first-time users discover Settings → Index.
    // Fire-and-forget — never blocks the search.
    //
    // "Empty" = the status endpoint reports `points_count === 0`.
    // We deliberately DO NOT key off `last_run_at`: a fresh dev
    // container that mounted pre-existing data has photos in the
    // cache but has never run the indexer, and shouldn't nag.
    // We also don't show the prompt when points_count is null
    // (unknown): false-negatives on this banner are much less
    // annoying than false-positives, so we assume "not empty"
    // when we can't tell.
    // Phase 2: getAdminIndexStatus() handles credentials and
    // dev-mode Zod validation. Schema is on Z.IndexerStatus.
    try {
      const body = await getAdminIndexStatus();
      if (body.points_count === 0) indexIsEmpty = true;
    } catch {
      // ignore — admin endpoint may be unavailable in tests
    }
  });

  // Mirror composer state into the URL (no auto‑reload; only
  // Search button clicks or picking a saved search fire reload).
  $effect(() => {
    void positives;
    void negatives;
    void filename;
    void input;
    if (!browser) return;
    writeToUrl();
  });
</script>

<svelte:head>
  <title>Lustre</title>
</svelte:head>

<section class="hero">
  {#if indexIsEmpty}
    <!-- Round-31: empty-library discoverability. Shows once on a
         fresh install where no job has ever run. The user dismisses
         it by going to Settings → Index (or by closing it manually). -->
    <div class="empty-prompt" role="status">
      <span>No photos indexed yet.</span>
      <a href={resolve('/settings')}>Go to Settings → Index</a>
      <button class="dismiss" onclick={() => (indexIsEmpty = false)} aria-label="Dismiss">×</button>
    </div>
  {/if}
  <!-- Round-56: every page's title block now lives in a
       <PageHeader> card for full-app consistency. The home was
       the last holdout — the h1 + .sub were plain centered text
       while /random, /for-you, /albums, /albums/[id], /albums/likes,
       /albums/dislikes, /settings, /similar all used PageHeader.
       Now both default and centroid modes of / use the same chrome
       (glass card, .head padding, --page-header-pad) as every
       other page.

       Centroid mode: title "Searching by album", subtitle shows
       the centroid info, action slot carries the "← Back to
       albums" link so the chrome and the nav share a single
       card.
       Default mode: title "Find photos by what they look like.",
       subtitle "Type what you remember. Save what you love." —
       no actions slot, no left-aligned CTA. The composer lives
       below the header. -->
  {#if activeCentroid}
    <PageHeader
      title="Searching by album"
      subtitle={surpriseOn
        ? `Surprise Me on ${activeCentroid} — Gumbel-sampling the top ${surprisePool} closest photos at temperature ${surpriseTemperature.toFixed(2)}.`
        : `Showing the photos closest to the average of ${activeCentroid}.`}
    />
  {:else}
    <PageHeader
      title="Find photos by what they look like."
      subtitle="Type what you remember. Save what you love."
    />
  {/if}
  {#if !activeCentroid}
    <PromptChips
      {positives}
      {negatives}
      {input}
      {mode}
      onInput={(v) => (input = v)}
      onMode={(m) => (mode = m)}
      onAdd={addPrompt}
      onRemovePositive={removePositive}
      onRemoveNegative={removeNegative}
    />
  {/if}

  <!-- Additional options panel: filename + collections chip
       filter + (most cases) the Surprise me controls. Round-90:
       Surprise me was moved from a separate inline panel +
       action-row button into this single panel. The toggle,
       temperature slider, and pool slider all live here. The
       action row only has Saved + Search.

       Round-91: Surprise me also works in centroid search
       (`?centroid=...`). The route runs the same Gumbel-max
       deck with the centroid vector as the query, so the
       panel is shown regardless of `activeCentroid`.

       Round-92: the per-card "Surprise" button on /albums
       (which set `?centroid=...&mode=sample` to use the old
       cluster re-roll) was removed because the in-panel
       Surprise toggle is strictly more capable. `mode=sample`
       is now an un-surfaced backend primitive — the panel's
       Surprise toggle is the only randomisation entry point
       on the home page. -->
  <AdditionalFilters
    open={filtersOpen}
    {filename}
    {collections}
    onToggle={() => (filtersOpen = !filtersOpen)}
    onFilename={(v) => (filename = v)}
    onToggleCollection={toggleCollection}
    showSurprise={true}
    {surpriseOn}
    {surpriseTemperature}
    {surprisePoolIndex}
    {surpriseTotal}
    {corpusSize}
    onSurpriseToggle={() => {
      surpriseOn = !surpriseOn;
      // Reset the session so the next reload rebuilds the
      // deck. The toggle is the user's signal that they want
      // Surprise to start fresh, not continue the previous
      // session.
      surpriseSession = null;
      surpriseTotal = null;
      // Auto-open the panel when turning Surprise on so the
      // sliders are visible without a second click. When
      // turning off, leave the panel state alone so the user
      // can re-enable without reopening.
      if (surpriseOn) filtersOpen = true;
    }}
    onSurpriseTemperature={(v) => (surpriseTemperature = v)}
    onSurprisePoolIndex={(v) => onPoolIndexChange(v)}
  />

  <!-- Search button + saved-searches menu. Pulled out of
       PromptChips so it sits AFTER the additional-options
       panel — visual order: search inputs → filename +
       collections → action buttons. In centroid mode the
       saved-searches menu doesn't apply (the centroid IS the
       query), but the Search button does: filter changes
       (filename, collections) stage into state and run only
       on this click — same explicit-run contract as the home
       search. -->
  {#if !activeCentroid}
    <div class="search-actions">
      <SavedSearchesMenu
        {positives}
        {negatives}
        onPick={(s: SavedSearch) => {
          // Populate the state but don't auto-run. The Search
          // button is now enabled (positives/negatives are set)
          // and the user presses it to actually execute — same
          // pattern as typing prompts fresh.
          positives = [...s.positives];
          negatives = [...s.negatives];
        }}
      />
      <Button
        variant="primary"
        size="lg"
        onclick={reload}
        disabled={!positives.length && !negatives.length && !filename.trim() && !collections.length || loading}
        title="Run search"
      >
        Search
      </Button>
    </div>
  {:else}
    <div class="search-actions">
      <Button
        variant="primary"
        size="lg"
        onclick={reload}
        disabled={loading}
        title="Re-run the album search with the current options"
      >
        Search
      </Button>
    </div>
  {/if}
</section>

<section class="results">
  {#if hasSearched}
    {#if error}
      <div class="state error">Couldn't load results: {error}</div>
    {:else}
      <PhotoGrid
        items={items}
        {loading}
        {hasMore}
        onLoadMore={loadMore}
        {onToggleFavorite}
        {onDislike}
      />
    {/if}
  {/if}
</section>

<style>
  .hero {
    /* Matches the .head width on the other grid pages (random,
       for-you, albums, similar, ...). All page-level "sections"
       span --grid-width so they line up visually. Round-1 polish:
       dropped horizontal padding (shell owns it via --shell-pad-x),
       trimmed top padding (was 40px → 8px) since the shell already
       provides --shell-pad-y + --shell-gap before this section.
       Round-56: dropped text-align: center — the home's title
       block is now a <PageHeader> card with default left-aligned
       text, matching every other page in the app. The composer,
       filters, and saved-searches below keep their own internal
       alignment via their own component CSS. */
    width: var(--grid-width, 100%);
    max-width: 1548px;
    margin: 0 auto;
    /* Round-Y: padding-top dropped to 0 — .shell already provides
       --shell-pad-y (16px) above the page, and random/for-you/
       albums/similar all rely on that single source (16px gap
       from topbar). The home page was the outlier at 24px
       (16px shell + 8px hero) until this edit. The padding-top
       comment above mentioned this was "trimmed since shell
       already provides" it — the trim just stopped one step
       short of zero. */
    padding: 0;
  }
  @media (max-width: 640px) {
    /* Drop the --grid-width constraint on mobile — at 375px
       viewport the grid math resolves to 1 column × 240px, which
       makes the entire hero (PageHeader + composer + filters +
       actions) collapse to 240px wide. The shell already has
       12px horizontal padding, so full-width here means the
       hero fills that inner area without leaving the
       centered-narrow look on phones. */
    .hero {
      width: 100%;
      max-width: 100%;
    }
  }

  /* Round-54: empty-index notification banner.
     Previously inline-flex content-sized pill — width depended on
     text length and didn't match the rest of the hero
     (h1 full-width, .sub max-width: 56ch). Now: block, centered
     in a fixed-width banner matching the subtitle column so the
     hero reads as three stacked, consistently-aligned rows
     (prompt banner, h1, sub). Border-radius drops from pill to
     var(--r-2) (12px) since it's no longer a pill. Padding
     matches --card-pad-tight (16 12) so it shares the chrome
     vocabulary with the photo sidebar and other tight panels. */
  .empty-prompt {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--s-2);
    padding: var(--s-2) var(--s-3);
    margin: 0 auto var(--s-3);
    max-width: 56ch;
    background: var(--accent-soft);
    border: 1px solid var(--accent);
    border-radius: var(--r-2);
    color: var(--fg-1);
    font-size: var(--fs-sm);
  }
  .empty-prompt a {
    color: var(--accent);
    text-decoration: none;
    font-weight: var(--fw-medium);
  }
  .empty-prompt a:hover {
    text-decoration: underline;
  }
  /* Dismiss button: was hardcoded font-size: 18px + padding: 0
     var(--s-0) (4px). Now uses the standard --fs-md size and
     --s-2 padding so the click target matches other small
     buttons in the app. Same color as muted text until hover. */
  .empty-prompt .dismiss {
    background: none;
    border: none;
    color: var(--fg-2);
    cursor: pointer;
    font-size: var(--fs-md);
    line-height: 1;
    padding: 0 var(--s-1);
  }
  .empty-prompt .dismiss:hover {
    color: var(--fg-1);
  }
  /* Saved-searches + Search button. Pulled out of PromptChips
     so the action row sits below the additional-options panel,
     not below the search inputs (matches the layout the user
     wants: inputs → filename + collections → actions). */
  .search-actions {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: var(--s-2);
  }
  /* Round-90: the Surprise me controls (toggle + sliders) moved
     from this page into the AdditionalFilters component. The
     visual styling lives there now. */
  /* Round-56: h1 + .sub removed — the title block is now in a
     <PageHeader> card with its own chrome. The .hero still owns
     the hero stack layout (gap between PageHeader, composer,
     AdditionalFilters, search-actions) but the title typography
     and chrome live in PageHeader. The `code` styling inside
     the centroid-mode subtitle is preserved on PageHeader's p
     via the global `code` rule in tokens. */
  /* Spacing inside the hero stack:
       composer (PromptChips + CollectionsChips)
         ↓ 14px
       filename / collections controls (AdditionalFilters)
         ↓ 10px
       search actions (Saved + Search)
     These gaps are deliberate — title→subtitle is 6px (they
     read as one block), subtitle→composer is 28px (separate
     chunk), and these middle gaps give each card its own
     breathing room instead of stacking them flush. */
  /* Round-67: hero stack rhythm — same --s-3 (16px) between
     every section so the stack reads as one consistent
     rhythm. Previously the three gaps were --s-3 → --s-2 → --s-1,
     a stair-step that read as inconsistent. The composer's
     top edge was missing its --s-3 entirely — it relied on the
     PageHeader's 20px bottom padding alone, which gave ~0px of
     perceived gap between header and search box while every
     other boundary had 16px. */
  .hero > :global(.prompt-chips) {
    margin-top: var(--s-3);
  }
  .hero > :global(.filters) {
    margin-top: var(--s-3);
  }
  .search-actions {
    margin-top: var(--s-3);
  }
  /* Round-X: section gap between .hero and .results is owned
     entirely by --shell-gap (16px) on .shell in +layout.svelte.
     Hero's padding-bottom was dropped to 0 in lockstep — together
     they replace the previous padding-bottom:24 + margin-top:16 +
     shell-gap:16 = 56px stack with the single 16px shell gap,
     matching how random/for-you/albums/similar handle the same
     header → grid boundary. */
  .results { margin-top: 0; }
</style>