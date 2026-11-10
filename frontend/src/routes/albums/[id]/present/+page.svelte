<script lang="ts">
  /**
   * Presentation mode — the album as a gallery.
   *
   * A full-bleed wall of the album's photos, in a random order,
   * with no page chrome. It is the album read as a whole rather
   * than browsed a row at a time: the random order means the
   * album's variety is what you perceive, not the sequence the
   * photos were added in.
   *
   * Read-only by design — there is no remove-from-album affordance
   * here. The album page is one click away for curating.
   */
  import { page } from '$app/stores';
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { getAlbum } from '$lib/api/endpoints';
  import { preferences } from '$lib/stores/preferences';
  import { GRID_SIZE_CONFIG, wallConfig } from '$lib/grid/config';
  import { shuffled } from '$lib/grid/shuffle';
  import PhotoGrid from '$lib/components/PhotoGrid.svelte';
  import Button from '$lib/components/Button.svelte';
  import type { AlbumDetail } from '$lib/api/endpoints';

  type Member = {
    id: string;
    point_id?: string;
    path?: string;
    blurhash?: string | null;
    is_favorite?: boolean;
  };
  type Item = {
    id: string;
    path?: string;
    blurhash?: string | null;
    is_favorite?: boolean;
  };

  /**
   * The whole album in one request.
   *
   * The album detail page pages its members because it scrolls
   * forever. The wall cannot: a random order over a PREFIX of the
   * album is not a random order over the album, and the last page
   * would be the only unsorted part. So the wall asks for every
   * member at once.
   *
   * The endpoint has no upper cap on `limit` — user albums can grow
   * arbitrarily — and the largest album in the library is 224
   * photos, so this is a single request either way.
   */
  const ALL_MEMBERS_LIMIT = 1000;

  let detail = $state<AlbumDetail | null>(null);
  let members = $state<Member[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);

  // The order the wall renders. Re-rolled on every open and by the
  // "Shuffle again" button. The members themselves are kept
  // unshuffled so a re-shuffle does not need another fetch.
  let order = $state<number[]>([]);

  let items = $derived<Item[]>(
    order.map((i) => {
      const m = members[i];
      return {
        id: m.id,
        path: m.path,
        blurhash: m.blurhash ?? null,
        is_favorite: m.is_favorite
      };
    })
  );

  // The wall's geometry: the user's tile SIZE, with no gutter.
  // The gap is the whole point of the wall — the photos touch
  // pixel-to-pixel — but the size stays the user's choice so the
  // photos are as large as they asked for.
  let tileConfig = $derived(wallConfig(GRID_SIZE_CONFIG[$preferences.gridSize]));

  function reshuffle() {
    order = shuffled(members.map((_, i) => i));
  }

  async function load() {
    const id = String($page.params.id);
    loading = true;
    // Same guard as the album page: a non-numeric id names an album
    // that does not exist, and asking the API anyway returns HTTP
    // 422 with the raw status leaking into the page.
    if (!/^\d+$/.test(id) || Number(id) <= 0) {
      detail = null;
      members = [];
      error = 'Album not found';
      loading = false;
      return;
    }
    try {
      const res = (await getAlbum(id, ALL_MEMBERS_LIMIT, 0)) as AlbumDetail;
      detail = res;
      members = (res.members ?? []) as Member[];
      error = null;
      reshuffle();
    } catch (e: unknown) {
      error = (e as Error).message ?? 'Failed to load album';
    } finally {
      loading = false;
    }
  }

  onMount(load);
</script>

<svelte:head>
  <title>{detail?.name ?? 'Album'} · Lustre</title>
</svelte:head>

<div class="present">
  <!-- A minimal bar: the album name, and the way out. The wall is
       the content, so nothing else competes with it. -->
  <div class="bar">
    <Button variant="secondary" onclick={() => goto(resolve(`/albums/${$page.params.id}`))}>
      ← Back to album
    </Button>
    <h1>{detail?.name ?? 'Album'}</h1>
    <div class="bar-right">
      <span class="count">{members.length} photos</span>
      <Button variant="secondary" onclick={reshuffle} title="Show them in a different order">
        Shuffle again
      </Button>
    </div>
  </div>

  {#if loading}
    <div class="state">Loading…</div>
  {:else if error || !detail}
    <div class="state error">Couldn't load album: {error ?? 'not found'}</div>
  {:else if members.length === 0}
    <div class="state empty">No photos in this album yet.</div>
  {:else}
    <PhotoGrid
      items={items}
      {tileConfig}
      wall
    />
  {/if}
</div>

<style>
  .present {
    /* Full-bleed: the wall runs to the viewport edges, so the
       photos are the whole page. */
    margin: calc(-1 * var(--shell-pad-y)) calc(-1 * var(--shell-pad-x));
    padding: 0;
  }
  .bar {
    display: flex;
    align-items: center;
    gap: var(--s-3);
    padding: var(--s-3) var(--shell-pad-x);
    /* The bar floats over the wall's top edge rather than pushing
       the photos down — the wall is the content, and a solid bar
       across the top would crop the first row. */
    background: color-mix(in srgb, var(--bg-0) 85%, transparent);
    backdrop-filter: var(--pill-blur);
    -webkit-backdrop-filter: var(--pill-blur);
    border-bottom: 1px solid var(--glass-edge);
    position: sticky;
    top: 0;
    z-index: 3;
  }
  .bar h1 {
    margin: 0;
    font-size: var(--fs-5, 1.25rem);
    font-weight: 600;
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .bar-right {
    display: flex;
    align-items: center;
    gap: var(--s-3);
  }
  .count {
    color: var(--fg-2);
    font-size: var(--fs-2, 0.875rem);
    white-space: nowrap;
  }
  .state {
    padding: var(--s-6);
    text-align: center;
    color: var(--fg-2);
  }
  .state.error {
    color: var(--negative);
  }
</style>
