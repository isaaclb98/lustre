<script lang="ts">
  /**
   * CollectionsChips — chip-row filter for the indexer's
   * `collection` payload field (one value per `--source` passed
   * to the indexer).
   *
   * Pure UI:
   *   - Fetches `/api/collections` on mount (one round-trip per
   *     page load; not on the search hot path).
   *   - Parent owns the selected set + the toggle callback.
   *
   * Visual: a row of toggleable chips. Selected chips get a
   * filled background and a leading checkmark; unselected ones
   * are outlined. Multi-select: 0 selected means "search the
   * whole library" (matches the backend's empty-array semantics).
   *
   * The count badge on each chip comes from the same endpoint
   * (`/api/collections` returns `{name, count}` per value) so the
   * user can see at a glance how big each library is before
   * committing to one.
   */
  import { onMount } from 'svelte';
  import { listCollections } from '$lib/api/endpoints';

  type CollectionSummary = { name: string; count: number };

  type Props = {
    /** Currently selected collection names. Empty = no filter. */
    selected: string[];
    onToggle: (name: string) => void;
  };

  let { selected, onToggle }: Props = $props();

  let collections = $state<CollectionSummary[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);

  const selectedSet = $derived(new Set(selected));

  onMount(async () => {
    try {
      const res = await listCollections();
      collections = res?.collections ?? [];
    } catch (e) {
      error = (e as Error).message ?? 'Failed to load collections';
      collections = [];
    } finally {
      loading = false;
    }
  });
</script>

<div class="collections" aria-label="Filter by source library">
  {#if loading}
    <span class="muted">Loading libraries…</span>
  {:else if error}
    <span class="error">{error}</span>
  {:else if collections.length === 0}
    <span class="muted">No indexed libraries yet</span>
  {:else}
    {#each collections as c (c.name)}
      {@const isOn = selectedSet.has(c.name)}
      <button
        type="button"
        class="chip"
        class:on={isOn}
        aria-pressed={isOn}
        title={isOn
          ? `Remove "${c.name}" filter`
          : `Filter to "${c.name}" (${c.count.toLocaleString()} photos)`}
        onclick={() => onToggle(c.name)}
      >
        <span class="check" aria-hidden="true">{isOn ? '✓' : ''}</span>
        <span class="text">{c.name}</span>
        <span class="count">{c.count.toLocaleString()}</span>
      </button>
    {/each}
  {/if}
</div>

<style>
  .collections {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-1);
    align-items: center;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: var(--s-1);
    background: transparent;
    color: var(--fg-1);
    border: 1px solid var(--glass-edge);
    border-radius: var(--r-pill);
    padding: var(--s-0) var(--s-3) var(--s-0) var(--s-2);
    font-size: var(--fs-sm);
    line-height: 1;
    height: 28px;
    cursor: pointer;
    transition:
      background var(--t-fast) var(--ease-out),
      border-color var(--t-fast) var(--ease-out);
  }
  .chip:hover { background: var(--glass-1); }
  .chip.on {
    background: var(--accent-fill);
    color: var(--fg-on-accent);
    border-color: var(--accent-fill);
  }
  .chip.on:hover { background: var(--accent-fill-2); }
  .check {
    display: inline-flex;
    width: 14px;
    height: 14px;
    align-items: center;
    justify-content: center;
    font-size: var(--fs-xs);
    line-height: 1;
    font-weight: 500;
  }
  .count {
    font-size: var(--fs-xs);
    opacity: 0.65;
    font-variant-numeric: tabular-nums;
  }
  .chip.on .count { opacity: 0.85; }
  .muted { color: var(--fg-3); font-size: var(--fs-sm); }
  .error { color: var(--negative); font-size: var(--fs-sm); }
</style>
