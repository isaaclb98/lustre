<!--
  ForYouRow — small section on the Home page showing a random sample
  of 20 photos from the For-You recommendation pool.

  Round‑27: thin wrapper around the canonical PhotoGrid. All the
  actual grid markup, padding, gutter, virtualisation, right-click
  context menu, and lightbox wiring now lives in PhotoGrid so the
  For-You row renders the same as search results, /random,
  /similar, /albums, etc.

  This wrapper owns only the data fetch + the section header.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { get } from 'svelte/store';
  import { forYouFeed, likePoint, unlikePoint, dislikePoint, listAlbums } from '$lib/api/endpoints';
  import PhotoGrid from '$lib/components/PhotoGrid.svelte';
  import { preferences } from '$lib/stores/preferences';
  import { currentColumns } from '$lib/grid/geometry';

  type Tile = {
    id: string;
    blurhash?: string | null;
    score_str?: string;
    is_favorite?: boolean;
    is_disliked?: boolean;
  };

  let items = $state<Tile[]>([]);
  let loading = $state(true);
  let lightboxIndex = $state<number | null>(null);
  let albums = $state<{ id: number; name: string }[]>([]);

  async function load() {
    loading = true;
    try {
      const poolSize = 800;
      // Target exactly 3 rows of tiles. The column count comes
      // from the shared geometry helper — the same measurement the
      // grid itself renders from — so the sample size is always a
      // multiple of the columns that actually appear, and the last
      // row is full (no partial row of clustered tiles on the
      // left). Read the current preference value at the moment
      // load() runs — the next load() picks up a new value if the
      // user changed grid size in Settings between fetches.
      const cols = currentColumns(get(preferences).gridSize);
      const want = cols * 3;
      const res = await forYouFeed({ limit: poolSize });
      const pool = res?.results ?? [];
      // Pick `want` items uniformly without replacement, shuffle first.
      const shuffled = [...pool].sort(() => Math.random() - 0.5);
      items = shuffled.slice(0, want).map((it) => ({
        id: it.id,
        blurhash: it.blurhash ?? null,
        score_str:
          typeof it.score === 'number' ? it.score.toFixed(3) : '',
        is_favorite: !!it.is_favorite,
        is_disliked: !!it.is_disliked
      }));
    } catch (e) {
      console.error('ForYouRow: feed fetch failed', e);
    } finally {
      loading = false;
    }
  }

  async function loadAlbums() {
    try {
      const res = await listAlbums();
      albums = (res?.albums ?? []).map((a) => ({ id: a.id, name: a.name }));
    } catch {
      albums = [];
    }
  }

  async function onToggleFavorite(id: string) {
    const it = items.find((x) => x.id === id);
    const liked = it?.is_favorite ?? false;
    try {
      if (liked) await unlikePoint(id);
      else await likePoint(id);
      items = items.map((x) =>
        x.id === id ? { ...x, is_favorite: !liked } : x
      );
      // Round‑28: do NOT close the lightbox here. The user just
      // tapped Like on a photo they like; dismissing the lightbox
      // forces a context switch and breaks the flow when they're
      // paging through photos with arrow keys. The visual feedback
      // (filled heart) is enough.
    } catch {
      /* intentional: keep lightbox open even if like fails */
    }
  }

  async function onDislike(id: string) {
    try {
      await dislikePoint(id);
      items = items.map((x) =>
        x.id === id ? { ...x, is_disliked: true } : x
      );
      // Round‑28: same — keep the lightbox open. The user can close
      // it themselves or page to the next photo.
    } catch {
      /* intentional: keep lightbox open even if dislike fails */
    }
  }

  function onPhotoOpen(item: Tile) {
    lightboxIndex = items.findIndex((x) => x.id === item.id);
  }

  onMount(() => {
    load();
    loadAlbums();
  });
</script>

<section class="row-section">
  <header class="head">
    <h2>For you</h2>
    <a class="more" href="/for-you">See all →</a>
  </header>

  {#if loading && items.length === 0}
    <p class="state">Loading recommendations…</p>
  {:else if items.length === 0}
    <p class="state empty">No recommendations yet — like or dislike a few photos to seed the signal.</p>
  {:else}
    <PhotoGrid
      {items}
      {loading}
      hasMore={false}
      {albums}
      {onToggleFavorite}
      {onDislike}
      {onPhotoOpen}
      bind:lightboxIndex
    />
  {/if}
</section>

<style>
  .row-section {
    margin-top: var(--s-5);
    /* No horizontal padding — `.shell` already provides 24px
       of side padding for the whole page. Adding it here too
       double-insets this section (48px total) and makes it look
       narrower than the search grid above it. */
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 var(--s-0) var(--s-3);
  }
  .head h2 {
    font-size: var(--fs-xl);
    font-weight: 500;
    margin: 0;
    letter-spacing: var(--ls-tighter);
  }
  .more {
    color: var(--fg-2);
    text-decoration: none;
    font-size: var(--fs-sm);
    transition: color var(--t-fast);
  }
  .more:hover { color: var(--fg-1); }

  /* Loading / empty states use the shared .state classes in
     global.css; no page-level placeholder styles needed. */
</style>