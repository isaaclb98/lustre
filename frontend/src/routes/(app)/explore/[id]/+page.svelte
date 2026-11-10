<script lang="ts">
  /**
   * Explore — a per-album feedback loop.
   *
   * The album's photos are the POSITIVES of a Qdrant `recommend()`
   * call; the photos the user down-votes are the NEGATIVES. Each
   * round re-queries with the accumulated down-votes, so the results
   * drift away from what was rejected while staying anchored to the
   * album.
   *
   * Two kinds of vote, deliberately different:
   *
   * - DOWN is temporary. It joins the session's negative set and
   *   never reaches the permanent dislikes table — "not this one
   *   right now" is not "never show me this again".
   * - UP is permanent. The photo joins the album, which makes it a
   *   positive for every later round and every later session.
   *
   * The route parameter is the album id, or the literal "favourites"
   * for the built-in album. The built-in album is not a row in the
   * albums table — its members are the favourited photos — so it
   * needs its own source key, and its up-vote means "favourite this"
   * rather than "add to the album".
   *
   * The server materialises a round once and slices it; the client
   * pages through the cached list rather than re-querying Qdrant,
   * because offset-paging a recommend query returns duplicate rows at
   * the page boundaries.
   */
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { resolve } from '$app/paths';
  import { SvelteSet } from 'svelte/reactivity';
  import {
    FAVOURITES_EXPLORE_SOURCE,
    albumExploreSource,
    exploreRound,
    exploreVote,
    getAlbum
  } from '$lib/api/endpoints';
  import { preferences } from '$lib/stores/preferences';
  import { currentPageSize } from '$lib/grid/geometry';
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

  // The route param is an album id, or "favourites" for the built-in
  // album. `albumId` is null for the built-in one, which has no row
  // in the albums table.
  const isFavourites = $derived($page.params.id === 'favourites');
  const albumId = $derived(isFavourites ? null : Number($page.params.id));
  // The source key the backend keys the round and its votes by.
  const source = $derived(
    isFavourites
      ? FAVOURITES_EXPLORE_SOURCE
      : albumExploreSource(Number($page.params.id))
  );

  let items = $state<Item[]>([]);
  let loading = $state(false);
  let hasMore = $state(false);
  let errorMsg = $state<string | null>(null);
  let albumName = $state('');
  let sessionId = $state<string | null>(null);
  // Position within the materialised round.
  let nextOffset = $state(0);
  let roundTotal = $state(0);
  // Photos the user has voted on in this session, so the buttons can
  // show the vote state. Up-votes persist in the album; down-votes
  // are session-scoped, so they live here.
  let upVoted = new SvelteSet<string>();
  let downVoted = new SvelteSet<string>();
  let voting = new SvelteSet<string>();

  // Photos per request — derived from the column count the grid
  // renders at the current viewport (see $lib/grid/geometry) so a
  // page always fills whole rows.
  function pageLimit(): number {
    return currentPageSize($preferences.gridSize);
  }

  async function loadAlbumName() {
    if (isFavourites) {
      albumName = 'Favourites';
      return;
    }
    try {
      const detail = (await getAlbum(albumId as number)) as { name?: string };
      albumName = detail?.name ?? '';
    } catch {
      /* non-fatal — the header falls back to the generic title */
    }
  }

  async function loadRound(offset: number, append: boolean, signal?: AbortSignal) {
    loading = true;
    try {
      const limit = pageLimit();
      const res = await exploreRound(
        source,
        {
          limit,
          offset,
          session: sessionId ?? undefined,
          signal
        },
        signal
      );
      const page_items = (res?.results ?? []) as Item[];
      items = append ? [...items, ...page_items] : page_items;
      // The round advances by the limit sent — ids deleted between
      // the round build and the page request are skipped
      // server-side, so the delivered count can be smaller than the
      // limit. Tracking the limit keeps the next request aligned.
      nextOffset = offset + limit;
      sessionId = res?.session_id ?? sessionId;
      roundTotal = res?.session_total ?? 0;
      hasMore = !!res?.has_more && page_items.length > 0;
      errorMsg = null;
    } catch (e: unknown) {
      const eAny = e as { body?: { detail?: string }; message?: string };
      const detail = eAny?.body?.detail ?? eAny?.message ?? '';
      if (typeof detail === 'string' && detail.includes('no photos')) {
        errorMsg =
          'No more photos to explore for this album. Try a different album.';
      } else {
        errorMsg = 'Could not load this round. Please try again.';
      }
      hasMore = false;
    } finally {
      loading = false;
    }
  }

  async function loadMore(signal?: AbortSignal) {
    if (loading || !hasMore) return;
    await loadRound(nextOffset, true, signal);
  }

  /**
   * Record a vote and start a fresh round.
   *
   * A vote changes the query — a down-vote adds a negative, an
   * up-vote adds a positive — so the round is rebuilt rather than
   * continued.
   */
  async function vote(id: string, direction: 'up' | 'down') {
    if (voting.has(id)) return;
    voting.add(id);
    // Optimistic — the button shows the vote immediately.
    const wasUp = upVoted.has(id);
    const wasDown = downVoted.has(id);
    if (direction === 'up') {
      upVoted.add(id);
      downVoted.delete(id);
    } else {
      downVoted.add(id);
      upVoted.delete(id);
    }
    try {
      await exploreVote(source, sessionId ?? undefined, id, direction);
      // Rebuild the round with the new signal.
      items = [];
      nextOffset = 0;
      await loadRound(0, false);
    } catch {
      // Rollback the optimistic state.
      if (wasUp) upVoted.add(id); else upVoted.delete(id);
      if (wasDown) downVoted.add(id); else downVoted.delete(id);
    } finally {
      voting.delete(id);
    }
  }

  onMount(async () => {
    await loadAlbumName();
    await loadRound(0, false);
  });

  function onVote(id: string, direction: 'up' | 'down') {
    void vote(id, direction);
  }
</script>

<svelte:head>
  <title>Explore{albumName ? ` · ${albumName}` : ''} · Lustre</title>
</svelte:head>

<PageHeader
  title={albumName ? `Explore ${albumName}` : 'Explore'}
  subtitle={isFavourites
    ? "Vote down the photos that don't match, and the next round leans away from them. Voting up favourites the photo."
    : "Vote down the photos that don't match, and the next round leans away from them. Voting up adds a photo to the album."}
/>

<div class="actions">
  <Button
    variant="secondary"
    size="md"
    href={resolve('/albums')}
    title="Back to albums"
  >
    All albums
  </Button>
  {#if roundTotal > 0}
    <span class="round-info">
      {#if upVoted.size > 0 || downVoted.size > 0}
        {upVoted.size} added to the album · {downVoted.size} rejected ·
      {/if}
      {roundTotal} in this round
    </span>
  {/if}
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
    onVote={onVote}
    {upVoted}
    {downVoted}
    voteUpLabel={isFavourites ? 'Favourite this photo' : 'Add to album'}
  />
{:else if !loading && !errorMsg}
  <div class="error-msg">
    {isFavourites
      ? 'No favourites to explore yet — like a photo and come back.'
      : 'This album has no photos to explore yet.'}
  </div>
{/if}

<style>
  .actions {
    width: 100%;
    max-width: var(--grid-width, 1200px);
    margin: 0 auto var(--s-3);
    padding: 0 var(--s-3);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--s-3);
  }
  .round-info {
    color: var(--fg-2);
    font-size: var(--fs-sm);
  }
  .error-msg {
    max-width: var(--grid-width, 1200px);
    margin: 0 auto var(--s-3);
    padding: var(--s-2) var(--s-3);
    color: var(--accent);
    text-align: center;
  }
</style>
