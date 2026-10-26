<script lang="ts">
  /**
   * Dislikes — built-in non-deletable album. Reads from
   * /api/dislikes and renders them in the standard grid.
   *
   * "Remove dislike" goes through the grid's onRemove handler
   * wired below; that drops the photo out of the grid.
   *
   * Phase 5: pagination + remove action were extracted to
   * usePagedAlbumList (see Likes for the same pattern). The
   * Dislikes album is sparser in practice, so we don't enable
   * the O(1) SvelteMap index shadow.
   */
  import { onMount } from 'svelte';
  import {
    listDislikes,
    undislikePoint
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

  const list = usePagedAlbumList<Item>({
    pageSize: () => pageLimit(),
    fetchPage: (limit, offset, signal) =>
      listDislikes(limit, offset, signal) as unknown as Promise<{
        results?: Item[];
        has_more?: boolean;
      }>,
    onRemove: (id) => undislikePoint(id)
    // useIndexMap: false (default) — the dislikes list is
    // small enough that a linear filter is fine.
  });

  onMount(() => {
    list.load();
  });

  function onDislike(id: string) {
    return list.remove(id);
  }
</script>

<svelte:head>
  <title>Dislikes · Lustre</title>
</svelte:head>

<PageHeader
  title="Dislikes"
  subtitle="Photos you've marked as not interested. Built-in, always here."
  meta="{list.items.length} photo{list.items.length === 1 ? '' : 's'}"
/>

{#if list.loading}
  <div class="state">Loading…</div>
{:else if list.error}
  <div class="state error">Couldn't load dislikes: {list.error}</div>
{:else if list.items.length === 0}
  <div class="state empty">No dislikes yet — tap the minus on any photo to mark it.</div>
{:else}
  <section>
    <PhotoGrid
      items={list.items}
      loading={list.loadingMore}
      hasMore={list.hasMore}
      onLoadMore={list.loadMore}
      {onDislike}
      onRemove={onDislike}
      removeLabel="Remove dislike"
    />
  </section>
{/if}

<style>
  /* Loading / empty / error states use the shared .state classes
     in global.css; no page-level placeholder styles needed. */
</style>
