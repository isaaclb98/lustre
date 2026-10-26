<script lang="ts">
  /**
   * Random — pick N random photos. Infinite scroll: keeps loading
   * more pages of random samples as the user scrolls. The dedupe
   * pass in loadMore stops the cycle once the dedupe filter
   * returns nothing new.
   *
   * No "Roll again" button — the page is meant to be browsed, not
   * re-rolled. No "End of results" UI either: when the library is
   * exhausted the scroll sentinel just stops firing.
   *
   * Like / Dislike / Most-similar toggles work in-place; the tile
   * stays in the grid either way.
   */
  import { onMount } from 'svelte';
  import { SvelteMap } from 'svelte/reactivity';
  import {
    random,
    likePoint,
    unlikePoint,
    dislikePoint
  } from '$lib/api/endpoints';
  import { preferences } from '$lib/stores/preferences';
  import { currentPageSize } from '$lib/grid/geometry';
  import PhotoGrid from '$lib/components/PhotoGrid.svelte';
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

  // Photos per request — derived from the column count the grid
  // renders at the current viewport (see $lib/grid/geometry) so a
  // page always fills whole rows. Measured per request, so it
  // follows the viewport and the Settings → Grid size choice.
  function pageLimit(): number {
    return currentPageSize($preferences.gridSize);
  }
  let items = $state<Item[]>([]);
  let loading = $state(false);
  let hasMore = $state(true);

  // Server-side session cursor: the first call creates a shuffled
  // deck and returns a session id; subsequent calls pass the same
  // session id with an incremented offset to walk forward. The
  // server guarantees no duplicates within a session, so the client
  // doesn't need any dedupe logic — just append.
  let sessionId = $state<string | null>(null);
  let nextOffset = $state(0);

  async function refresh() {
    loading = true;
    try {
      // New session, start from offset 0.
      const limit = pageLimit();
      const res = await random({ limit });
      items = (res?.results ?? []) as Item[];
      sessionId = res?.session_id ?? null;
      // The deck advances by the limit we asked for — the server
      // slices its shuffled id list by absolute position and drops
      // ids it can't materialise, so the delivered count can be
      // smaller than the limit. Tracking the limit keeps the next
      // request from re-walking the tail of the current slice.
      nextOffset = limit;
      // session_total is the deck length. has_more from the server
      // tells us whether there's anything left to fetch.
      hasMore = !!res?.has_more && items.length > 0;
    } catch {
      items = [];
      hasMore = false;
    } finally {
      loading = false;
    }
  }

  async function loadMore(signal?: AbortSignal) {
    if (loading || !hasMore || !sessionId) return;
    loading = true;
    try {
      const limit = pageLimit();
      const res = await random({
        session: sessionId,
        offset: nextOffset,
        limit,
        signal
      });
      const more = (res?.results ?? []) as Item[];
      items = [...items, ...more];
      nextOffset += limit;
      hasMore = !!res?.has_more && more.length > 0;
    } catch (_e) {
      if (signal?.aborted) return; // clean cancel from pre-fetch retrigger
      hasMore = false;
    } finally {
      loading = false;
    }
  }

  // Round-9 perf: O(1) item lookup + update via shadow Map.
  // Before this round, onToggleFavorite / onDislike used
  // `items.find(...)` + `items.map(...)`, both O(n). For
  // infinite-scrolled pages (1000+ items) every Like click
  // walked the whole array twice. The Map mirrors `items`
  // (id → index); updates stay O(1) as long as callers
  // don't reorder / splice the array (they don't — we only
  // push to the end or replace in-place).
  // eslint-disable-next-line svelte/no-unnecessary-state-wrap -- reassigned by $effect; SvelteMap alone doesn't track the variable reference
  let indexById: SvelteMap<string, number> = $state(new SvelteMap());

  // Append-only when `items` changes; rebuilds only when
  // the array reference changes (after refresh / loadMore).
  // The shadow Map is never mutated directly — always
  // rebuilt from `items` so it can't drift.
  $effect(() => {
    const map = new SvelteMap<string, number>();
    for (let i = 0; i < items.length; i++) {
      map.set(items[i].id, i);
    }
    indexById = map;
  });

  async function onToggleFavorite(id: string) {
    const idx = indexById.get(id);
    if (idx === undefined) return;
    const liked = items[idx]?.is_favorite ?? false;
    try {
      if (liked) await unlikePoint(id);
      else await likePoint(id);
      // O(1) replace — same array reference, one slot updated.
      // Svelte 5 reactivity triggers because the inner object
      // changes (we spread to a new object).
      const next = items.slice();
      next[idx] = { ...next[idx], is_favorite: !liked };
      items = next;
    } catch {
      /* noop: like/unlike failure leaves state unchanged */
    }
  }

  async function onDislike(id: string) {
    const idx = indexById.get(id);
    if (idx === undefined) return;
    try {
      await dislikePoint(id);
      const next = items.slice();
      next[idx] = { ...next[idx], is_disliked: true };
      items = next;
    } catch {
      /* noop: dislike failure leaves state unchanged */
    }
  }

  onMount(refresh);
</script>

<svelte:head>
  <title>Random · Lustre</title>
</svelte:head>

<PageHeader title="Random" subtitle="Random photos from your library. Scroll for more." />

<section class="grid-wrap">
  {#if items.length === 0 && !loading}
    <div class="state empty">No photos in your library yet — index your library from Settings.</div>
  {:else}
    <PhotoGrid
      {items}
      {loading}
      {hasMore}
      onLoadMore={loadMore}
      {onToggleFavorite}
      {onDislike}
    />
  {/if}
</section>

<style>
  /* Round-58: constrain the .grid-wrap to the 5-tile grid width
     so the section reads at the same width as PageHeader and
     the photo grid below it. Without this, the section fills
     the shell content area (1376px) and bleeds 60px past the
     chrome on each side. */
  .grid-wrap {
    width: var(--grid-width);
    max-width: 100%;
    margin: 0 auto;
  }
</style>
