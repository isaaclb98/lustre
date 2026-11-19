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
   *
   * The round advances ONLY when the user asks for it. A vote is
   * recorded and the photos on screen stay exactly where they are —
   * the user can add three photos to the album and reject six, and
   * only then ask for the next round, which is built from all of it.
   * The floating button at the bottom of the page is that request.
   */
  import { onMount, tick } from 'svelte';
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
  import Icon from '$lib/components/Icon.svelte';

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
  // True while the next round is being built. The grid stays put
  // through it — the photos on screen do not change under the user.
  let advancing = $state(false);
  // The lightbox the grid opens over the photos. Bound so the
  // floating control can step aside while it is open.
  let lightboxIndex = $state<number | null>(null);

  // The votes recorded so far — the feedback the next round is
  // built from.
  const voteCount = $derived(upVoted.size + downVoted.size);

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
   * Record a vote.
   *
   * A vote changes the query — a down-vote adds a negative, an
   * up-vote adds a positive — but it does NOT change the photos on
   * screen. The round the user is looking at stays exactly where it
   * is; the vote is the feedback the NEXT round is built from, and
   * they ask for that round with the floating control.
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
      // The round is left alone. Nothing is re-fetched and nothing
      // on screen moves — the vote waits in the session until the
      // user asks for the next round.
    } catch {
      // Rollback the optimistic state.
      if (wasUp) upVoted.add(id); else upVoted.delete(id);
      if (wasDown) downVoted.add(id); else downVoted.delete(id);
    } finally {
      voting.delete(id);
    }
  }

  /**
   * Draw the next round.
   *
   * The votes recorded so far are the feedback the new round is
   * built from: the source's current membership as the positives
   * (an up-vote joins it, so it counts), the session's down-votes as
   * the negatives, and the photos the user judged as the excluded
   * set.
   *
   * The grid keeps the old photos on screen until the new round
   * arrives — the photos the user is looking at do not change under
   * them while they vote.
   */
  async function advanceRound() {
    if (advancing) return;
    advancing = true;
    try {
      const limit = pageLimit();
      const res = await exploreRound(source, {
        limit,
        offset: 0,
        session: sessionId ?? undefined,
        advance: true
      });
      const page_items = (res?.results ?? []) as Item[];
      items = page_items;
      // The round advances by the limit sent — ids deleted between
      // the round build and the page request are skipped
      // server-side, so the delivered count can be smaller than the
      // limit. Tracking the limit keeps the next request aligned.
      nextOffset = limit;
      sessionId = res?.session_id ?? sessionId;
      roundTotal = res?.session_total ?? 0;
      hasMore = !!res?.has_more && page_items.length > 0;
      errorMsg = null;
      // The lightbox holds an index into the grid that just went.
      lightboxIndex = null;
      // A new round starts at the top of the grid — the user asked
      // for a fresh batch, not the middle of one. The scroll lands
      // after the grid has re-rendered, so the new round's first
      // row is what is under the viewport.
      await tick();
      window.scrollTo({ top: 0 });
    } catch (e: unknown) {
      const eAny = e as { body?: { detail?: string }; message?: string };
      const detail = eAny?.body?.detail ?? eAny?.message ?? '';
      errorMsg =
        typeof detail === 'string' && detail.includes('no photos')
          ? 'No more photos to explore for this album. Try a different album.'
          : 'Could not load the next round. Please try again.';
    } finally {
      advancing = false;
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
    ? "Vote down the photos that don't match, then ask for the next round — it leans away from them. Voting up favourites the photo."
    : "Vote down the photos that don't match, then ask for the next round — it leans away from them. Voting up adds a photo to the album."}
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
    bind:lightboxIndex
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

<!-- The next round is the user's call, not the vote's. The control
     floats at the bottom of the page so it is reachable from
     anywhere in the grid, and it carries the count of votes that
     will shape the round it draws. It steps aside while the
     lightbox is open — the lightbox owns the bottom of the screen
     then. -->
{#if items.length > 0 && lightboxIndex === null}
  <div class="next-round">
    <Button
      variant="primary"
      size="md"
      onclick={advanceRound}
      disabled={advancing}
      title="Draw the next round from the votes you have cast"
    >
      {#if advancing}
        <span class="spin" aria-hidden="true">
          <Icon name="spinner" size={16} />
        </span>
      {:else}
        <Icon name="chevron-right" size={16} />
      {/if}
      Next round
      {#if voteCount > 0}
        <span class="count">{voteCount}</span>
      {/if}
    </Button>
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

  /* The next-round control. Bottom-centre, so it never collides
     with the scroll-to-top button in the bottom-right corner. */
  .next-round {
    position: fixed;
    left: 50%;
    bottom: var(--s-4);
    transform: translateX(-50%);
    z-index: 40;
    display: flex;
    justify-content: center;
  }

  /* The count of votes the next round will be built from — the
     user's up and down votes in this session. */
  .count {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.25em;
    height: 1.25em;
    margin-left: var(--s-1);
    padding: 0 0.35em;
    border-radius: var(--r-pill);
    background: var(--fg-on-accent);
    color: var(--accent-fill);
    font-size: var(--fs-xs);
    font-variant-numeric: tabular-nums;
  }

  /* The spinner glyph is static — it only reads as motion when
     something turns it. */
  .spin {
    display: inline-flex;
    animation: next-round-spin 0.8s linear infinite;
  }
  @keyframes next-round-spin {
    to {
      transform: rotate(360deg);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .spin {
      animation: none;
    }
  }
</style>
