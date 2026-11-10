<script lang="ts">
  /**
   * Presentation mode — the album as a gallery, in the foreground.
   *
   * This route breaks out of the app's layout hierarchy: it resets
   * to the ROOT layout with `+page@.svelte`, so the app chrome (the
   * top bar, the page shell, the floating scroll-to-top button) is
   * not in its tree at all. Nothing is behind the wall and nothing
   * is around it — the photos are the whole screen.
   *
   * The surface has no controls either: no menu, no buttons, not
   * even the tile chrome the rest of the app shows. A presentation
   * is a thing you look at, and every control on it is a control
   * that can be pressed by accident while the album is on screen.
   * That also makes it read-only: the wall cannot favourite,
   * dislike or re-file a photo, because it has no affordance for it.
   *
   * A random order, re-rolled on every open. An album in the order
   * the photos were added reads as a sequence the user already
   * knows; a random one shows the album's variety.
   *
   * The way out is the Escape key and the browser's own back
   * button — the same gestures that leave any fullscreen surface.
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

  // The order the wall renders. Re-rolled on every open. The members
  // themselves are kept unshuffled so a re-shuffle does not need
  // another fetch.
  let order = $state<number[]>([]);

  let items = $derived<Item[]>(
    order.map((i) => {
      const m = members[i];
      return {
        id: m.point_id ?? m.id,
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
      order = shuffled(members.map((_, i) => i));
    } catch (e: unknown) {
      error = (e as Error).message ?? 'Failed to load album';
    } finally {
      loading = false;
    }
  }

  /**
   * Escape leaves the wall.
   *
   * The surface has no visible way out — no back button, no menu —
   * so the keyboard carries it.
   */
  function onKeydown(e: KeyboardEvent) {
    if (e.key !== 'Escape') return;
    void goto(resolve(`/albums/${$page.params.id}`));
  }

  onMount(() => {
    void load();
    window.addEventListener('keydown', onKeydown);
    return () => window.removeEventListener('keydown', onKeydown);
  });
</script>

<svelte:head>
  <title>{detail?.name ?? 'Album'} · Lustre</title>
</svelte:head>

<div class="present" data-photo-count={members.length}>
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
      readOnly
    />
  {/if}
</div>

<style>
  .present {
    /* The wall is the whole screen. There is no shell padding to
       cancel here (the route breaks out of the shell entirely) and
       no top bar to slide under — the photos start at the very top
       of the viewport and run to its edges. */
    min-height: 100vh;
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
