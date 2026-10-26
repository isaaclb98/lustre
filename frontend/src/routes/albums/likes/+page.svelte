<script lang="ts">
  /**
   * Likes — built-in non-deletable album. Reads from /api/favorites
   * and renders them in the same grid the user-created albums use.
   *
   * "Unlike" goes through the grid's onRemove handler wired below.
   * Unlike drops the photo from the grid (optimistic, restored on
   * failure).
   *
   * Phase 5: pagination + remove action were extracted to
   * usePagedAlbumList. The page keeps the endpoint wiring,
   * the title/subtitle copy, and the empty-state message.
   */
  import { onMount } from 'svelte';
  import {
    listFavorites,
    unlikePoint
  } from '$lib/api/endpoints';
  import { usePagedAlbumList } from '$lib/composables/usePagedAlbumList.svelte';
  import { preferences } from '$lib/stores/preferences';
  import { currentPageSize } from '$lib/grid/geometry';
  import PhotoGrid from '$lib/components/PhotoGrid.svelte';
  import PageHeader from '$lib/components/PageHeader.svelte';

  type Item = {
    id: string;
    path?: string;
    blurhash?: string | null;
    is_favorite?: boolean;
  };

  // Photos per request — derived from the column count the grid
  // renders at the current viewport (see $lib/grid/geometry) so a
  // page always fills whole rows. Read on every fetch, so it
  // follows the viewport and the Settings → Grid size choice.
  function pageLimit(): number {
    return currentPageSize($preferences.gridSize);
  }

  // Phase 5: usePagedAlbumList owns the load/loadMore/remove
  // pattern. The useIndexMap flag enables an O(1) SvelteMap
  // shadow of items (worth it on Likes at 1000+ items).
  const list = usePagedAlbumList<Item>({
    pageSize: () => pageLimit(),
    fetchPage: (limit, offset, signal) =>
      // listFavorites returns the SearchResponse envelope when
      // called via as_results=1 (which listFavorites sets).
      // usePagedAlbumList only reads .results and .has_more
      // so the type widening is fine here.
      listFavorites(limit, offset, signal) as unknown as Promise<{
        results?: Item[];
        has_more?: boolean;
      }>,
    onRemove: (id) => unlikePoint(id),
    useIndexMap: true
  });

  onMount(() => {
    list.load();
  });

  function onToggleFavorite(id: string) {
    return list.remove(id);
  }
</script>

<svelte:head>
  <title>Likes · Lustre</title>
</svelte:head>

<PageHeader
  title="Likes"
  subtitle="Photos you've liked. Built-in, always here."
  meta="{list.items.length} photo{list.items.length === 1 ? '' : 's'}"
/>

{#if list.loading}
  <div class="state">Loading…</div>
{:else if list.error}
  <div class="state error">Couldn't load likes: {list.error}</div>
{:else if list.items.length === 0}
  <div class="state empty">No likes yet — tap the heart on any photo to save it here.</div>
{:else}
  <section>
    <PhotoGrid
      items={list.items}
      loading={list.loadingMore}
      hasMore={list.hasMore}
      onLoadMore={list.loadMore}
      {onToggleFavorite}
      onRemove={onToggleFavorite}
      removeLabel="Unlike"
    />
  </section>
{/if}

<style>
  /* Loading / empty / error states use the shared .state classes
     in global.css; no page-level placeholder styles needed. */
</style>
