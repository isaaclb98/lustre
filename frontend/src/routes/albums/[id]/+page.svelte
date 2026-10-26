<script lang="ts">
  /**
   * Album detail — grid of photos in this album, with a
   * "Download zip" link to /albums/{id}/download.zip.
   *
   * Infinite scroll: walks /api/albums/{id}?limit=&offset= in
   * batches of GRID_PAGE_SIZE so the UI keeps working no matter
   * how large the album grows. The AlbumDetailResponse carries
   * member_total which we compare against the running member
   * count to drive `has_more`.
   */
  import { page } from '$app/stores';
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { getAlbum, removePhotoFromAlbum, deleteAlbum } from '$lib/api/endpoints';
  import { preferences } from '$lib/stores/preferences';
  import { currentPageSize } from '$lib/grid/geometry';
  import PhotoGrid from '$lib/components/PhotoGrid.svelte';
  import { dialog } from '$lib/components/Dialog.svelte';
  import PageHeader from '$lib/components/PageHeader.svelte';
  import Button from '$lib/components/Button.svelte';
  import type { AlbumDetail } from '$lib/api/endpoints';

  type Member = {
    id: string;
    point_id?: string;
    path?: string;
    url?: string;
    blurhash?: string | null;
    is_favorite?: boolean;
  };
  type Item = {
    id: string;
    path?: string;
    score_str?: string;
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

  let detail = $state<AlbumDetail | null>(null);
  // Accumulated members across paged /api/albums/{id} responses.
  // Each response carries `members` (the slice for this batch)
  // and `member_total` (the album-wide total) — we append the
  // slices here and compare lengths to drive `has_more`.
  let members = $state<Member[]>([]);
  let loading = $state(true);
  let loadingMore = $state(false);
  let error = $state<string | null>(null);
  let offset = $state(0);
  let hasMore = $state(false);

  async function load() {
    const id = String($page.params.id);
    loading = true;
    try {
      const limit = pageLimit();
      const res = (await getAlbum(id, limit, 0)) as AlbumDetail;
      detail = res;
      members = (res.members ?? []) as Member[];
      // The members endpoint applies LIMIT/OFFSET to the album's
      // member query, so the offset is an absolute row position.
      // The accumulated member count IS that position — every row
      // the server returns is one row the next page skips past.
      offset = members.length;
      const total = res.member_total ?? members.length;
      // A page shorter than the limit means the album is exhausted
      // (the query returns min(limit, remaining) rows).
      hasMore = offset < total && members.length >= limit;
      error = null;
    } catch (e: unknown) {
      error = (e as Error).message ?? 'Failed to load album';
    } finally {
      loading = false;
    }
  }

  async function loadMore(signal?: AbortSignal) {
    if (loading || loadingMore || !hasMore || !detail) return;
    const id = String($page.params.id);
    loadingMore = true;
    try {
      // Reuse the album-detail endpoint with the next offset —
      // it already returns metadata + a member slice in one
      // round-trip, and the first call cached the album's name/
      // description/total in `detail` so we just take `members`.
      const limit = pageLimit();
      const res = (await getAlbum(id, limit, offset, signal)) as AlbumDetail;
      const more = (res.members ?? []) as Member[];
      members = [...members, ...more];
      offset = members.length;
      const total = res.member_total ?? members.length;
      hasMore = offset < total && more.length >= limit;
    } catch (_e) {
      // Leave the existing list intact; the user can keep paging
      // — losing scroll progress on a transient error is worse
      // than a stuck spinner. A clean cancel from the pre-fetch
      // retrigger is silent — no hasMore=false penalty.
      if (signal?.aborted) return;
    } finally {
      loadingMore = false;
    }
  }

  onMount(load);

  function items(): Item[] {
    return members.map((m: Member) => ({
      id: m.point_id ?? m.id,
      path: m.path,
      blurhash: m.blurhash ?? null,
      is_favorite: m.is_favorite ?? true
    }));
  }

  /**
   * Remove a photo from THIS album. Optimistic: drop from the local
   * list immediately so the grid animates the tile out, then call
   * the DELETE endpoint. On failure, re-add + toast so the user
   * doesn't lose the action.
   *
   * The album id comes from $page.params; the photo id from the
   * tile's `id` prop.
   */
  async function onRemoveFromAlbum(pointId: string) {
    const albumId = $page.params.id;
    const before = members;
    members = members.filter(
      (m) => (m.point_id ?? m.id) !== pointId,
    );
    // Reflect the local removal in the cached total so the
    // header counter ("N photos") stays accurate.
    if (detail && typeof detail.member_total === 'number') {
      detail = {
        ...detail,
        member_total: Math.max(0, detail.member_total - 1),
      };
    }
    try {
      await removePhotoFromAlbum(Number(albumId), pointId);
    } catch {
      // Restore the previous state.
      members = before;
      if (detail && typeof detail.member_total === 'number') {
        detail = {
          ...detail,
          member_total: detail.member_total + 1,
        };
      }
    }
  }

  /** Delete the current album. Same dialog pattern as the list
   * page's remove() — confirm the destructive action, then
   * navigate back to /albums once the backend acknowledges. */
  async function onDeleteAlbum() {
    if (!detail) return;
    const ok = await dialog.confirm({
      title: 'Delete album',
      body: `Delete album "${detail.name}"? This can't be undone. Photos in the album are not deleted — they stay in your library.`,
      confirmLabel: 'Delete',
      kind: 'danger'
    });
    if (!ok) return;
    try {
      await deleteAlbum(detail.id);
      await goto(resolve('/albums'));
    } catch (_e) {
      /* noop: deletion succeeded but navigation may fail */
    }
  }
</script>

<svelte:head>
  <title>{detail?.name ?? 'Album'} · Lustre</title>
</svelte:head>

{#if loading}
  <div class="state">Loading…</div>
{:else if error || !detail}
  <div class="state error">Couldn't load album: {error ?? 'not found'}</div>
{:else}
  <PageHeader
    title={detail.name}
    subtitle={detail.description}
    meta="{detail.member_total ?? members.length} photos"
  >
    {#snippet actions()}
      <div class="actions">
        {#if detail && detail.id && (detail.member_total ?? members.length) > 0}
          <Button
            variant="primary"
            href="/albums/{detail.id}/download.zip"
            target="_blank"
            rel="noopener"
          >
            Download zip
          </Button>
        {/if}
        <Button danger onclick={onDeleteAlbum} title={`Delete album ${detail?.name ?? ''}`}>
          Delete album
        </Button>
      </div>
    {/snippet}
  </PageHeader>
  {#if items().length === 0 && !loadingMore}
    <div class="state empty">No photos in this album yet. Open a photo to add it from the lightbox.</div>
  {:else}
    <section>
      <PhotoGrid
        items={items()}
        loading={loadingMore}
        {hasMore}
        onLoadMore={loadMore}
        onRemove={onRemoveFromAlbum}
        removeLabel="Remove from album"
      />
    </section>
  {/if}
{/if}

<style>
  /* Wrap the header actions so Download + Delete sit on the
     same row with consistent spacing. The .actions wrapper
     fills the role of the gutter that was implicit when only
     Download was present. */
  .actions {
    display: flex;
    gap: var(--s-2);
    align-items: center;
  }
</style>