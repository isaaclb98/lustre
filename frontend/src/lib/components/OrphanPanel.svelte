<script lang="ts">
  /**
   * Orphan review + confirmed purge.
   *
   * The orphan lifecycle is two-step. The indexer MARKS points whose
   * photo is gone from disk, and points belonging to a source that
   * was removed from the source list. A marked point is excluded
   * from every read path, so it stops being served immediately —
   * but its embedding is still in the collection.
   *
   * This panel is the second step: it shows how many orphans there
   * are and which photos they are, and lets an operator confirm the
   * purge that removes the embeddings for good.
   *
   * The split exists because the two halves have very different
   * consequences. Marking is reversible — a file that comes back, or
   * a source re-added to the source list, clears the marker and the
   * photo returns. Deleting is not. So the irreversible half waits
   * for a human to look at it.
   */
  import { getOrphans, purgeOrphans } from '$lib/api/endpoints';
  import { ApiError } from '$lib/api/client';
  import type { OrphanListResponse, OrphanPurgeResponse } from '$lib/api/endpoints';
  import Button from '$lib/components/Button.svelte';

  let orphans = $state<OrphanListResponse | null>(null);
  let loading = $state(true);
  let busy = $state(false);
  let errorMessage = $state<string | null>(null);
  let purgeResult = $state<OrphanPurgeResponse | null>(null);
  let confirming = $state(false);

  async function refresh() {
    try {
      orphans = await getOrphans();
      errorMessage = null;
    } catch (err) {
      errorMessage = describeError(err);
    } finally {
      loading = false;
    }
  }

  async function purge() {
    busy = true;
    errorMessage = null;
    try {
      purgeResult = await purgeOrphans();
      confirming = false;
      await refresh();
    } catch (err) {
      errorMessage = describeError(err);
    } finally {
      busy = false;
    }
  }

  function describeError(err: unknown): string {
    if (err instanceof ApiError) {
      const body = err.body as { detail?: unknown } | null;
      if (body && typeof body.detail === 'string') return body.detail;
      return err.message;
    }
    return err instanceof Error ? err.message : String(err);
  }

  $effect(() => {
    void refresh();
  });

  function sourceEntries(): [string, number][] {
    if (!orphans) return [];
    return Object.entries(orphans.by_source ?? {}).sort((a, b) => b[1] - a[1]);
  }

  function sample(): NonNullable<OrphanListResponse['sample']> {
    return orphans?.sample ?? [];
  }
</script>

<section class="card glass">
  <h2 class="card-title">Orphans</h2>
  <p class="card-desc">
    Photos the index has marked as unwanted: their file is gone from
    disk, or their source directory was removed from the library.
    Marked photos are already hidden from search — this panel is where
    you check the count and confirm that their embeddings are removed
    for good. Removal is irreversible, so it happens only when you
    ask for it.
  </p>

  {#if loading}
    <div class="state">Loading…</div>
  {:else if errorMessage}
    <div class="state error">{errorMessage}</div>
  {:else if orphans && orphans.count === 0}
    <div class="state">No orphans. The index matches the library.</div>
  {:else if orphans}
    <div class="counts">
      <span class="count-value">{orphans.count.toLocaleString()}</span>
      <span class="count-label">
        marked orphan{orphans.count === 1 ? '' : 's'}
      </span>
    </div>

    {#if sourceEntries().length > 0}
      <ul class="sources">
        {#each sourceEntries() as [name, count] (name)}
          <li>
            <span class="source-name">{name || '(no source)'}</span>
            <span class="source-count">{count.toLocaleString()}</span>
          </li>
        {/each}
      </ul>
    {/if}

    {#if sample().length > 0}
      <details class="sample-section">
        <summary>Sample ({sample().length})</summary>
        <ul class="sample">
          {#each sample() as item (item.id)}
            <li>
              <span class="sample-path">{item.path || '(no path)'}</span>
              {#if item.collection}
                <span class="sample-source">{item.collection}</span>
              {/if}
            </li>
          {/each}
        </ul>
      </details>
    {/if}

    {#if purgeResult}
      <div class="state">
        Removed {purgeResult.deleted.toLocaleString()} embedding{purgeResult.deleted === 1 ? '' : 's'}.
        {purgeResult.remaining.toLocaleString()} orphan{purgeResult.remaining === 1 ? '' : 's'} left.
      </div>
    {/if}

    <div class="actions">
      {#if confirming}
        <Button variant="secondary" onclick={() => (confirming = false)} disabled={busy}>
          Cancel
        </Button>
        <Button variant="primary" danger onclick={purge} disabled={busy}>
          {busy ? 'Removing…' : `Remove ${orphans.count.toLocaleString()} embedding${orphans.count === 1 ? '' : 's'}`}
        </Button>
      {:else}
        <Button variant="secondary" onclick={() => (confirming = true)} disabled={busy}>
          Remove orphans…
        </Button>
      {/if}
    </div>

    {#if confirming}
      <p class="confirm-note">
        This deletes the embeddings and the matching favourites,
        dislikes and album entries. It cannot be undone.
      </p>
    {/if}
  {/if}
</section>

<style>
  /* The settings page's card chrome is page-scoped, so this panel
     carries its own copy of the same vocabulary — same tokens,
     same shape. `.glass` and `.state` are global; the card
     furniture is not. */
  .card.glass {
    padding: var(--card-pad);
    display: flex;
    flex-direction: column;
    gap: var(--s-3);
  }

  .card-title {
    font-size: var(--fs-lg);
    font-weight: var(--fw-medium);
    margin: 0;
    color: var(--fg-1);
  }

  .card-desc {
    margin: 0;
    color: var(--fg-2);
    font-size: var(--fs-sm);
    line-height: 1.5;
  }

  .counts {
    display: flex;
    align-items: baseline;
    gap: var(--s-2);
  }

  .count-value {
    font-size: var(--fs-xl);
    font-weight: var(--fw-medium);
    color: var(--fg-1);
    font-variant-numeric: tabular-nums;
  }

  .count-label {
    color: var(--fg-2);
    font-size: var(--fs-sm);
  }

  .sources {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--s-1);
  }

  .sources li {
    display: flex;
    justify-content: space-between;
    gap: var(--s-3);
    padding: var(--s-1) var(--s-2);
    background: var(--glass-2);
    border-radius: var(--r-2);
    font-size: var(--fs-sm);
  }

  .source-name {
    color: var(--fg-1);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .source-count {
    color: var(--fg-2);
    font-variant-numeric: tabular-nums;
    flex-shrink: 0;
  }

  .sample-section summary {
    cursor: pointer;
    color: var(--fg-2);
    font-size: var(--fs-sm);
  }

  .sample {
    list-style: none;
    margin: var(--s-2) 0 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--s-1);
    max-height: 18rem;
    overflow-y: auto;
  }

  .sample li {
    display: flex;
    justify-content: space-between;
    gap: var(--s-3);
    font-size: var(--fs-sm);
    color: var(--fg-2);
  }

  .sample-path {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    direction: rtl;
    text-align: left;
  }

  .sample-source {
    flex-shrink: 0;
    color: var(--fg-3);
  }

  .actions {
    display: flex;
    gap: var(--s-2);
  }

  .confirm-note {
    margin: 0;
    color: var(--fg-2);
    font-size: var(--fs-sm);
    line-height: 1.5;
  }
</style>
