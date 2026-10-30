<script lang="ts">
  /**
   * Settings page — hosts the Index button (the user-facing surface
   * of the in-app indexer). Round-35: only the incremental mode is
   * exposed to users. The rebuild-from-scratch mode still exists on
   * the backend (`POST /api/admin/index { mode: "rebuild" }`) for
   * operator use — it's destructive (wipes favourites + albums +
   * saved searches + vectors) so it shouldn't be one click away.
   *
   * While a job is running the page polls /api/admin/index/status
   * every second and shows live progress. The Cancel button sends
   * SIGTERM to the indexer subprocess; the page then returns to
   * idle.
   */
  import { onDestroy, onMount } from 'svelte';
  import { apiGet, apiPost, ApiError } from '$lib/api/client';
  import type { components } from '$lib/api/types.gen';
  import Button from '$lib/components/Button.svelte';
  import OrphanPanel from '$lib/components/OrphanPanel.svelte';
  import PageHeader from '$lib/components/PageHeader.svelte';
  import {
    preferences,
    SLIDESHOW_PRESETS,
    setSlideshowInterval,
    setGridSize
  } from '$lib/stores/preferences';
  import { type GridSize } from '$lib/grid/config';

  // Grid size — three user-facing labels, single knob. The
  // CSS/JS derive tile size, gap, and max columns from the
  // choice via GRID_SIZE_CONFIG ($lib/grid/config). Order
  // matters: rendered left-to-right in the settings row.
  const GRID_SIZE_PRESETS: { label: string; value: GridSize }[] = [
    { label: 'Large', value: 'large' },
    { label: 'Medium', value: 'medium' },
    { label: 'Small', value: 'small' }
  ];

  type IndexerStatusResponse = components['schemas']['IndexerStatusResponse'];
  type IndexerLogResponse = components['schemas']['IndexerLogResponse'];
  type State = IndexerStatusResponse['state'];

  let status = $state<IndexerStatusResponse | null>(null);
  let logText = $state<string>('');
  let loading = $state(true);
  let busy = $state(false);
  let errorMessage = $state<string | null>(null);

  let pollTimer: ReturnType<typeof setInterval> | null = null;

  async function refresh() {
    try {
      status = await apiGet<IndexerStatusResponse>('/api/admin/index/status');
      errorMessage = null;
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  async function refreshLog() {
    try {
      const resp = await apiGet<IndexerLogResponse>('/api/admin/index/log');
      logText = resp.lines.join('\n');
    } catch {
      // ignore log fetch errors — status is the source of truth
    }
  }

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      await refresh();
      await refreshLog();
      if (status && status.state !== 'running') {
        stopPolling();
      }
    }, 1000);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  onMount(async () => {
    await refresh();
    if (status?.state === 'running') startPolling();
  });

  onDestroy(stopPolling);

  async function startIndex() {
    // Round-35: only the incremental mode is exposed in the UI. The
    // backend still accepts `rebuild` for operator use via direct API
    // calls — see the page-level comment above.
    busy = true;
    errorMessage = null;
    try {
      status = await apiPost<IndexerStatusResponse>('/api/admin/index', {
        mode: 'incremental',
      });
      startPolling();
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        errorMessage = 'An indexer job is already running.';
      } else {
        errorMessage = err instanceof Error ? err.message : String(err);
      }
    } finally {
      busy = false;
    }
  }

  async function cancelIndex() {
    busy = true;
    errorMessage = null;
    try {
      status = await apiPost<IndexerStatusResponse>('/api/admin/index/cancel');
      await refresh();
      await refreshLog();
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
    } finally {
      busy = false;
    }
  }

  function isRunning(s: State | undefined): boolean {
    return s === 'running';
  }
</script>

<svelte:head>
  <title>Settings · Lustre</title>
</svelte:head>

<div class="settings-page">
  <PageHeader
    title="Settings"
    subtitle="Index the photo library, tune search and slideshow behaviour."
  />
  <section class="card glass">
    <h2 class="card-title">Index</h2>
    <p class="card-desc">
      Walk the photo library, embed every image, and write it to the
      search index. Embeddings run in the background as a child
      process; you can keep using the app while a job runs.
    </p>

    {#if loading}
      <div class="state">Loading…</div>
    {:else if status}
      <div class="status-row" data-state={status.state}>
        <span class="status-dot" aria-hidden="true"></span>
        <span class="status-label">
          {#if status.state === 'idle'}
            Idle{#if status.last_run_at} · last run {formatRelative(status.last_run_at)}{/if}
          {:else if status.state === 'running'}
            {#if status.phase === 'warming_up'}
              Warming up model ({status.progress.indexed} indexed so far)
            {:else if status.phase === 'scanning'}
              Scanning…
            {:else}
              Running · {status.progress.indexed} indexed, {status.progress.reembedded} re-embedded{#if status.progress.thumb_regens}, {status.progress.thumb_regens} thumbs regenerated{/if}
            {/if}
          {:else}
            Failed{status.last_error ? ` — ${status.last_error}` : ''}
          {/if}
        </span>
      </div>

      {#if errorMessage}
        <div class="state error">{errorMessage}</div>
      {/if}

      <div class="actions">
        {#if isRunning(status.state)}
          <Button onclick={cancelIndex} disabled={busy}>Cancel</Button>
        {:else}
          <Button onclick={startIndex} disabled={busy}>Index</Button>
        {/if}
      </div>

      {#if logText}
        <details class="log-section">
          <summary>Log</summary>
          <pre class="log">{logText}</pre>
        </details>
      {/if}
    {/if}
  </section>

  <OrphanPanel />

  <section class="card glass">
    <h2 class="card-title">Slideshow</h2>
    <p class="card-desc">
      How long each photo stays up during Lightbox auto-advance
      (press Play inside any lightbox with multiple images to start
      a slideshow). The setting applies the next time you press
      Play, even mid-playback.
    </p>
    <div
      class="preset-row"
      role="radiogroup"
      aria-label="Slideshow photo duration"
    >
      {#each SLIDESHOW_PRESETS as preset (preset.ms)}
        <button
          type="button"
          class="preset"
          role="radio"
          aria-checked={$preferences.slideshowIntervalMs === preset.ms}
          data-active={$preferences.slideshowIntervalMs === preset.ms}
          onclick={() => setSlideshowInterval(preset.ms)}
        >
          {preset.label}
        </button>
      {/each}
    </div>
  </section>

  <section class="card glass">
    <h2 class="card-title">Grid size</h2>
    <p class="card-desc">
      How dense the photo grid renders. Larger tiles are easier to
      scan one at a time; smaller tiles pack more photos per row
      and work better on wide monitors. Applies to every page that
      shows the photo grid (home, /random, /for-you, /albums). The
      number of photos loaded per page follows your window width, so
      every row comes out full — resize the window and the page size
      adjusts to match.
    </p>
    <div
      class="preset-row"
      role="radiogroup"
      aria-label="Grid size"
    >
      {#each GRID_SIZE_PRESETS as preset (preset.value)}
        <button
          type="button"
          class="preset"
          role="radio"
          aria-checked={$preferences.gridSize === preset.value}
          data-active={$preferences.gridSize === preset.value}
          onclick={() => setGridSize(preset.value)}
        >
          {preset.label}
        </button>
      {/each}
    </div>
  </section>
</div>

<script module lang="ts">
  function formatRelative(iso: string): string {
    const then = new Date(iso).getTime();
    const now = Date.now();
    const sec = Math.round((now - then) / 1000);
    if (sec < 60) return `${sec}s ago`;
    const min = Math.round(sec / 60);
    if (min < 60) return `${min}m ago`;
    const hr = Math.round(min / 60);
    if (hr < 24) return `${hr}h ago`;
    return `${Math.round(hr / 24)}d ago`;
  }
</script>

<style>
  /* Settings sits in the same column as every other page — the
     page header (PageHeader), the grid (Random / For you /
     Albums), and these form cards all share --grid-width
     (1216px at desktop). Previously this wrapper was
     unconstrained (1376px — full shell content), which made
     the cards wider than the rest of the app's content.
     Before that it was 720px centered, which made the cards
     narrower. Both wrong. Now: width: var(--grid-width) +
     margin: 0 auto matches the photo-grid width and the
     header chrome, like every other page in the app. */
  .settings-page {
    width: var(--grid-width);
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: var(--shell-gap);
  }
  @media (max-width: 640px) {
    /* Drop the --grid-width constraint on mobile — same reason as
       the home page's .hero override: at 375px the grid math
       resolves to 240px, making the entire settings layout
       (PageHeader + every card) collapse to 240px wide. */
    .settings-page { width: 100%; }
  }

  /* Cards use the shared .glass panel vocabulary (background,
     hairline border, lit-from-above highlight, soft shadow,
     rounded). No bespoke card chrome here. The flex layout
     + inter-child gap is page-local since not every card needs
     it (album cards stack title + desc + footer via the
     .card rule in the albums page). .glass does not carry
     its own padding — that's the page-local concern. */
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

  .status-row {
    display: flex;
    align-items: center;
    gap: var(--s-2);
    padding: var(--s-2) var(--s-3);
    background: var(--glass-2);
    border-radius: var(--r-2);
    font-size: var(--fs-sm);
    color: var(--fg-1);
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--fg-3);
    flex-shrink: 0;
  }

  .status-row[data-state="running"] .status-dot {
    background: var(--accent);
    animation: pulse 1.5s ease-in-out infinite;
  }
  .status-row[data-state="failed"] .status-dot {
    background: var(--danger);
  }
  .status-row[data-state="idle"] .status-dot {
    background: var(--success);
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .actions {
    display: flex;
    gap: var(--s-2);
  }

  .log-section {
    margin-top: var(--s-2);
  }

  .log-section summary {
    cursor: pointer;
    color: var(--fg-2);
    font-size: var(--fs-sm);
    user-select: none;
  }

  .log {
    margin: var(--s-2) 0 0 0;
    padding: var(--s-2);
    background: var(--bg-1);
    border: 1px solid var(--glass-edge);
    border-radius: var(--r-2);
    color: var(--fg-2);
    font-size: var(--fs-xs);
    max-height: 240px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* Slideshow preset row. Five pill buttons in a single horizontal
     line; the active preset swaps to a darker glass-2 fill so
     the current duration is unambiguous at a glance. Uses
     aria-checked + data-active so the same visual state is
     reachable from both the accessibility tree (screen readers
     announce the selected radio) and pure CSS (no JS toggle
     bookkeeping). */
  .preset-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2);
  }
  .preset {
    appearance: none;
    background: var(--glass-1);
    color: var(--fg-1);
    border: 1px solid var(--glass-edge);
    border-radius: var(--r-pill);
    padding: 0 var(--s-3);
    height: 32px;
    font: inherit;
    font-size: var(--fs-sm);
    cursor: pointer;
    transition:
      background var(--t-fast),
      border-color var(--t-fast),
      color var(--t-fast);
  }
  /* Round-68: active preset now uses the same chrome as the
     TopBar active tab (accent-soft fill + accent border + accent
     text). The previous version used a hollow accent border on
     a transparent fill, which Isaac read as 'inconsistent
     outline' — it looked like an outlined button, not a
     selected state.

     --glass-3 and --fg-0 were undefined tokens (the rule
     resolved to initial values, so the active button was
     hollow). Replaced with --glass-2 (closest existing tier)
     and --fg-1 (standard primary text). The data-active state
     overrides both anyway. */
  .preset:hover {
    background: var(--glass-2);
    color: var(--fg-1);
  }
  .preset:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: var(--r-pill);
  }
  .preset[data-active='true'] {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--accent);
  }
</style>
