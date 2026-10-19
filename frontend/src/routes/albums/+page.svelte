<script lang="ts">
  /**
   * Albums — list + create. Two non-optional system albums (Likes,
   * Dislikes) are rendered first and cannot be deleted or renamed;
   * user-created albums follow.
   *
   * Round‑29: every card has a "Search" button that navigates to
   * the home page with ?centroid=album:{id} (or ?centroid=likes /
   * ?centroid=dislikes for the built-ins). The home page renders
   * results inline using that album's centroid.
   *
   * Round‑29b: the Likes centroid is now registered under
   * `likes` (was `favourites`). The backend keeps `favourites`
   * as a back-compat alias so legacy URLs still resolve.
   *
   * Round‑92: the per-card "Surprise" button was removed. The
   * home page now exposes Surprise me (Gumbel-max deck with
   * temperature + pool sliders) on every centroid via the
   * Additional options panel, which is strictly more capable
   * than the old `mode=sample` cluster re-roll. Users who want
   * a randomized centroid search land on `/?centroid=...` and
   * toggle Surprise in the panel.
   */
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import {
    listAlbums,
    createAlbum,
    deleteAlbum,
    listFavorites,
    listDislikes,
    getFavoriteCount,
    getDislikeCount,
    thumbUrl
  } from '$lib/api/endpoints';
  import { dialog } from '$lib/components/Dialog.svelte';
  import PageHeader from '$lib/components/PageHeader.svelte';
  import Button from '$lib/components/Button.svelte';
  import type { AlbumSummary } from '$lib/api/endpoints';

  // Round‑29b: renamed from 'favourites' to 'likes' to match
  // the user-visible album label. The backend also registers
  // 'favourites' as a back-compat alias so legacy URLs still
  // resolve.
  const LIKES_CENTROID = 'likes';
  const DISLIKES_CENTROID = 'dislikes';

  async function searchByAlbum(centroidName: string) {
    await goto(resolve(`/?centroid=${encodeURIComponent(centroidName)}`));
  }

  function searchByUserAlbum(albumId: number) {
    // Round‑29 fix: backend registers user-album centroids under
    // `album:<id>` (colon, not underscore). Sending `album_<id>`
    // 404s. See `_album_centroid_name` in search/app.py.
    searchByAlbum(`album:${albumId}`);
  }

  let albums = $state<AlbumSummary[]>([]);
  let loading = $state(true);
  let likesCount = $state(0);
  let dislikesCount = $state(0);
  // First photo id in each built-in album — used as the card cover.
  // Empty string while loading or when the album is empty.
  let likesFirstId = $state('');
  let dislikesFirstId = $state('');

  async function refresh() {
    loading = true;
    try {
      const res = (await listAlbums()) as { albums: AlbumSummary[] };
      albums = res?.albums ?? [];
    } catch {
      /* noop: keep previous album list on transient failure */
    } finally {
      loading = false;
    }
  }

  async function refreshSystemCounts() {
    try {
      // Phase 2: getFavoriteCount / getDislikeCount wrap the
      // raw `/api/favorites?limit=1` and `/api/dislikes?limit=1`
      // endpoints with Zod validation. No need for a hand-rolled
      // extractCount shim once the response is typed.
      const [likes, dislikes] = await Promise.all([
        getFavoriteCount(),
        getDislikeCount()
      ]);
      likesCount = likes;
      dislikesCount = dislikes;
    } catch {
      // Counts are decorative; a 502 on count fetch shouldn't
      // blank the whole page.
    }
    // Cover photo for each built-in album. Fire-and-forget — the
    // card is already informative without a thumbnail.
    Promise.all([
      fetchCover(listFavorites).then((id) => (likesFirstId = id)),
      fetchCover(listDislikes).then((id) => (dislikesFirstId = id))
    ]).catch(() => {
      /* non-fatal */
    });
  }

  /** Fetch one SearchResult from the list endpoint and return its id.
   * With `as_results=1` (which is what listFavorites / listDislikes
   * set), the backend returns a SearchResponse envelope — the
   * array lives under `results`, not `favorites` / `items`. */
  async function fetchCover(
    list: (limit?: number, offset?: number) => Promise<unknown>
  ): Promise<string> {
    try {
      const res = (await list(1, 0)) as { results?: { id: string }[] };
      return res.results?.[0]?.id ?? '';
    } catch {
      return '';
    }
  }

  async function create() {
    const name = await dialog.prompt({
      title: 'Create album',
      label: 'Album name',
      confirmLabel: 'Create',
      defaultValue: ''
    });
    if (!name) return;
    try {
      await createAlbum({ name: name.trim() });
      await refresh();
    } catch {
      /* noop: dialog handles user feedback; refresh on next open */
    }
  }

  async function remove(id: number, name: string) {
    const ok = await dialog.confirm({
      title: 'Delete album',
      body: `Delete album "${name}"? This can't be undone.`,
      confirmLabel: 'Delete',
      kind: 'danger'
    });
    if (!ok) return;
    try {
      await deleteAlbum(id);
      albums = albums.filter((a) => a.id !== id);
    } catch {
      /* noop: keep current list if delete fails */
    }
  }

  onMount(() => {
    // Both fetch from independent endpoints; run them in parallel
    // so the page reaches a fully-populated state in max(t1, t2)
    // instead of t1 + t2. (Tier 1.5.)
    Promise.all([refresh(), refreshSystemCounts()]);
    // Round-37: removed pushRandomTint() — the page-level
    // backdrop is a flat dark base (see +layout.svelte), no per-
    // page tint fetch needed.
  });
</script>

<svelte:head>
  <title>Albums · Lustre</title>
</svelte:head>

<PageHeader
  title="Albums"
  subtitle="Like photos to keep them handy, build collections, group memories."
>
  {#snippet actions()}
    <Button variant="primary" onclick={create}>+ New album</Button>
  {/snippet}
</PageHeader>

<div class="grid" aria-label="Albums">
  <article class="card glass">
    {#if likesFirstId}
      <img class="cover" src={thumbUrl(likesFirstId)} alt="" loading="lazy" />
    {:else}
      <div class="cover cover-empty" aria-hidden="true"></div>
    {/if}
    <a class="title" href={resolve('/albums/likes')}>
      <span>Likes</span>
    </a>
    <!-- Round-61: removed the description ("Photos you've liked.
         Built-in, always here."). The title "Likes" already says
         what the album is; "Built-in, always here" was filler
         text that pushed the card taller and added 50px of
         whitespace below the cover when the album was empty. -->
    <footer>
      <span class="count">{likesCount} photo{likesCount === 1 ? '' : 's'}</span>
    </footer>
    <!-- Round‑29: search button on every album card. The Likes
         centroid is named "likes" on the backend (round‑29b);
         "favourites" is a back‑compat alias. Clicking either
         takes the user to the home page with results.
         Round‑92: removed the sibling "Surprise" button. The
         home page's Additional options panel now exposes
         Surprise me on every centroid (Gumbel-max deck with
         temperature + pool sliders) — strictly more capable
         than the old `mode=sample` cluster re-roll this
         button used to trigger. -->
    <div class="search-row">
      <Button
        variant="ghost"
        size="sm"
        class="search-btn"
        data={{ 'data-centroid': LIKES_CENTROID }}
        onclick={() => searchByAlbum(LIKES_CENTROID)}
        disabled={likesCount === 0}
        aria-label="Search by Likes centroid"
      >Search</Button>
    </div>
  </article>
  <article class="card glass">
    {#if dislikesFirstId}
      <img class="cover" src={thumbUrl(dislikesFirstId)} alt="" loading="lazy" />
    {:else}
      <div class="cover cover-empty" aria-hidden="true"></div>
    {/if}
    <a class="title" href={resolve('/albums/dislikes')}>
      <span>Dislikes</span>
    </a>
    <!-- Round-61: removed the description ("Photos you've
         disliked. Built-in, always here.") and the "built-in"
         pill. The title is self-explanatory. -->
    <footer>
      <span class="count">{dislikesCount} photo{dislikesCount === 1 ? '' : 's'}</span>
    </footer>
    <div class="search-row">
      <Button
        variant="ghost"
        size="sm"
        class="search-btn"
        data={{ 'data-centroid': DISLIKES_CENTROID }}
        onclick={() => searchByAlbum(DISLIKES_CENTROID)}
        disabled={dislikesCount === 0}
        aria-label="Search by Dislikes centroid"
      >Search</Button>
    </div>
  </article>

  {#if loading}
    <div class="state state-wide">Loading albums…</div>
  {:else if albums.length > 0}
    {#each albums as a (a.id)}
      <article class="card glass">
        {#if a.first_member_id}
          <img class="cover" src={thumbUrl(a.first_member_id)} alt="" loading="lazy" />
        {:else}
          <div class="cover cover-empty" aria-hidden="true"></div>
        {/if}
        <a class="title" href={resolve(`/albums/${a.id}`)}>{a.name}</a>
        {#if a.description}<p class="desc">{a.description}</p>{/if}
        <footer>
          <span class="count">{a.member_count ?? 0} photos</span>
          <Button
            variant="ghost"
            size="sm"
            danger
            onclick={() => remove(a.id, a.name)}
            aria-label="Delete {a.name}"
          >Delete</Button>
        </footer>
        <div class="search-row">
          <Button
            variant="ghost"
            size="sm"
            class="search-btn"
            data={{ 'data-centroid': `album:${a.id}` }}
            onclick={() => searchByUserAlbum(a.id)}
            disabled={(a.member_count ?? 0) === 0}
            aria-label="Search by {a.name} centroid"
          >Search</Button>
        </div>
      </article>
    {/each}
  {/if}
</div>

<style>
  /* Search button row at the bottom of each album card.
     Round‑92: the per-card "Surprise" button was removed
     (the home page now exposes Surprise me on every centroid
     via the Additional options panel). The row now holds a
     single Search button, which can stretch full-width
     instead of competing for space with a sibling. */
  .search-row {
    margin-top: auto;
    display: flex;
    gap: var(--s-1);
    align-self: stretch;
  }
  .search-row :global(.search-btn) {
    flex: 1 1 auto;
  }

  /* System albums (Likes, Dislikes) live in the same grid as the
     user-created albums below. They are pinned to the top of the
     flow by virtue of being declared first in the template; the
     `built-in` pill in the footer marks them visually. */
  /* Album cards (Likes/Dislikes + user albums) in a single auto-fit
     grid. auto-fit collapses empty tracks when item count is below
     the column count, so a sparse row (e.g. 2 user albums) centers
     the filled cards rather than leaving an empty band on the right.
     With 4+ items (the common case), all tracks fill and the grid
     reads as a continuous wall, same as auto-fill behavior. */
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, 260px);
    gap: var(--grid-gutter);
    margin: 0 auto var(--s-4);
    width: var(--grid-width, 100%);
    max-width: 100%;
    justify-content: center;
  }
  .card {
    /* No horizontal padding — the cover spans full width. Vertical
       padding only sits between the cover and the title. Round‑47:
       tightened from 16px to 12px (--s-2) — the cover-to-title
       gap felt airy when titles were short. Round‑61:
       bottom padding tightened from var(--s-3) (16) to
       var(--s-2) (8) to match the search-row margin-top
       above — same scale, balanced chrome. */
    padding: 0 0 var(--s-2);
    display: flex;
    flex-direction: column;
    /* Round-49: per-section rhythm replaces the uniform 8px
       gap. Cover→title (16px) gets the most breathing room
       because the cover is the focal point and the text needs
       a clear entry; title→desc (4px) sits tight as a single
       "title block"; desc→footer (12px) medium separation;
       footer→search (16px) more before the action buttons so
       the buttons read as actions, not as more metadata.
       The uniform 8px rhythm read as cramped and monotonous
       — different gaps create visual hierarchy between the
       five sections. Implemented as per-child margins; flex
       gap is 0 so the margins are the only thing applying. */
    gap: 0;
    overflow: hidden; /* rounded corners on the cover */
    transition: transform var(--t-fast), box-shadow var(--t-fast);
  }
  /* Round-49: per-section rhythm on .card's five children.
     Cover is full-bleed (no top padding on .card). Gaps:
       cover → title:    16px (var(--s-3))
       title → desc:     4px  (var(--s-0)) — tight, they're
                                   one "title block"
       desc → footer:    12px (var(--s-2))
       footer → search:  16px (var(--s-3)) — more space so
                                   the buttons read as actions
                                   not as more metadata
     Margins, not flex gap — gap is uniform across siblings. */
  /* Round-61: cover→title 16px (unchanged). Removed the
     title→desc 4px gap and desc→footer 12px gap because
     Likes/Dislikes no longer render a description. footer→search
     stays 16px. search-row gets margin-top: auto so it pins to
     the bottom of the card — any leftover vertical space in a
     row-stretched card lands between footer and search-row, not
     below the search-row. */
  .card .title { margin: var(--s-3) 0 0; }
  .card .desc { margin: var(--s-0) 0 var(--s-2); }
  .card footer { margin-top: 0; }
  .card .search-row { margin-top: auto; }
  .card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-2);
  }
  /* Cover thumbnail — first photo added to the album. Sits flush
     against the top of the card, full-bleed, fixed aspect ratio
     so the cards in a row all line up regardless of which photo
     the backend picked. */
  .cover {
    display: block;
    width: 100%;
    /* Round-61: 3/2 ratio instead of 4/3 — covers are
       shorter (173px at 260px wide vs 195px before) which
       leaves more room for the content section underneath.
       Pair with the smaller bottom padding below so cards
       feel balanced instead of cover-dominant. */
    aspect-ratio: 3 / 2;
    object-fit: cover;
    background: var(--glass-1);
    border-bottom: 1px solid var(--glass-edge);
  }
  /* Placeholder when the album has no members yet — keeps the
     card height the same as the others in its row. */
  .cover-empty {
    background: linear-gradient(
      135deg,
      var(--glass-1) 0%,
      var(--glass-2) 100%
    );
  }
  /* The card's horizontal padding sits between the full-bleed
     cover and the title/desc/footer. Round‑48: tightened to 20px
     (was 16) to match the --card-pad token used by Settings cards
     and the photo sidebar — same inner content edge across every
     panel in the app. */
  .card .title,
  .card .desc,
  .card footer,
  .card .search-row {
    margin-left: var(--card-pad-x, 20px);
    margin-right: var(--card-pad-x, 20px);
  }
  .title {
    font-size: var(--fs-lg);
    font-weight: var(--fw-medium);
    color: var(--fg-1);
    display: inline-flex;
    align-items: center;
    gap: var(--s-1);
  }
  .desc {
    margin: 0;
    color: var(--fg-2);
    font-size: var(--fs-sm);
  }
  footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: auto;
  }
  .count {
    color: var(--fg-3);
    font-size: var(--fs-sm);
  }
</style>
