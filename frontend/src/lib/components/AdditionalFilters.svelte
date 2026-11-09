<script lang="ts">
  /**
   * AdditionalFilters — the collapsible panel below the search
   * controls. Has a header with title + chevron, body has the
   * filename input, the collections chip filter, and (on the
   * home page) the Surprise me controls.
   *
   * Round-90: the Surprise me controls (toggle + temperature +
   * pool) moved from a separate inline panel into this panel.
   * They used to be a standalone "Surprise me" button in the
   * action row plus a separate sliders block below, which read
   * as two disconnected surfaces. Now everything is one place.
   *
   * Collections: optional. Pass `collections` + `onToggleCollection`
   * to surface the chip-row filter inside this panel.
   *
   * Surprise: optional. Pass `showSurprise=true` plus the
   * surprise state and setters to render the section. Used on
   * the home page only — other pages don't need it (centroid
   * search goes through a different route, and
   * its own complete UI).
   */
  import CollectionsChips from './CollectionsChips.svelte';
  import { POOL_STOPS_HOME, POOL_SIZE_MAX } from '$lib/api/limits';

  type Props = {
    open: boolean;
    filename: string;
    onToggle: () => void;
    onFilename: (v: string) => void;
    collections?: string[];
    onToggleCollection?: (name: string) => void;
    // Surprise me section. When `showSurprise` is false the
    // section is not rendered (used on the centroid search
    // route, where Surprise is a no-op).
    showSurprise?: boolean;
    surpriseOn?: boolean;
    surpriseTemperature?: number;
    surprisePoolIndex?: number;
    surpriseTotal?: number | null;
    corpusSize?: number | null;
    onSurpriseToggle?: () => void;
    onSurpriseTemperature?: (v: number) => void;
    onSurprisePoolIndex?: (v: number) => void;
  };
  let {
    open,
    filename,
    onToggle,
    onFilename,
    collections = [],
    onToggleCollection,
    showSurprise = false,
    surpriseOn = false,
    surpriseTemperature = 0.05,
    surprisePoolIndex = 0,
    surpriseTotal = null,
    corpusSize = null,
    onSurpriseToggle,
    onSurpriseTemperature,
    onSurprisePoolIndex
  }: Props = $props();

  // Surprise me pool slider stops. Imported from $lib/api/limits
  // so the home page and AdditionalFilters
  // all read the same list of legal values.
  // Phase 1: replaced the local POOL_STOPS const with
  // POOL_STOPS_HOME. The rightmost is the upper bound for the
  // route (mirrors POOL_SIZE_MAX in search/surprise_deck.py).

  // Format the pool size for the slider label. The rightmost
  // position resolves to the corpus size if it's smaller than
  // the slider max (a fresh install), otherwise "Max".
  function formatPool(idx: number, total: number | null): string {
    const size = POOL_STOPS_HOME[idx] ?? POOL_STOPS_HOME[POOL_STOPS_HOME.length - 1];
    if (size >= POOL_SIZE_MAX) {
      return total !== null && total < POOL_SIZE_MAX
        ? `All (${total.toLocaleString()})`
        : 'Max';
    }
    return size.toLocaleString();
  }
</script>

<section class="filters glass" class:open>
  <button
    class="head"
    type="button"
    onclick={onToggle}
    aria-expanded={open}
  >
    <span>Additional options</span>
    <span class="chev" aria-hidden="true">{open ? '▴' : '▾'}</span>
  </button>
  {#if open}
    <div class="body">
      <label class="field">
        <span class="lab">Filename contains</span>
        <input
          type="text"
          placeholder="e.g. IMG_2024"
          value={filename}
          oninput={(e) => onFilename((e.target as HTMLInputElement).value)}
          autocomplete="off"
          spellcheck="false"
        />
      </label>
      {#if onToggleCollection}
        <div class="collections-row">
          <span class="lab">Limit to library</span>
          <CollectionsChips
            selected={collections}
            onToggle={onToggleCollection}
          />
        </div>
      {/if}
      {#if showSurprise && onSurpriseToggle}
        <div class="surprise-row">
          <div class="surprise-head">
            <span class="lab">Surprise me</span>
            <button
              type="button"
              class="surprise-toggle"
              class:on={surpriseOn}
              onclick={onSurpriseToggle}
              aria-pressed={surpriseOn}
              title={surpriseOn
                ? 'Surprise Me is on — click to turn off (returns to top-K)'
                : 'Turn on Surprise Me (random deck from the top candidates)'}
            >
              <span class="surprise-toggle-track" aria-hidden="true">
                <span class="surprise-toggle-thumb"></span>
              </span>
              <span class="surprise-toggle-label">
                {surpriseOn ? 'On' : 'Off'}
              </span>
            </button>
          </div>
          {#if surpriseOn}
            <div class="surprise-controls">
              <label class="surprise-slider">
                <span class="surprise-slider-label">
                  Temperature: <strong>{surpriseTemperature.toFixed(2)}</strong>
                  <span class="hint">— 0 = top-K, 1 = roughly uniform</span>
                </span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={surpriseTemperature}
                  oninput={(e) =>
                    onSurpriseTemperature?.(
                      parseFloat((e.target as HTMLInputElement).value)
                    )}
                  aria-label="Surprise temperature"
                />
              </label>
              <label class="surprise-slider">
                <span class="surprise-slider-label">
                  Pool: <strong>{formatPool(surprisePoolIndex, corpusSize)}</strong>
                  <span class="hint">— candidates the Gumbel draw is taken from. "Max" is the Qdrant search ceiling (50k).</span>
                </span>
                <input
                  type="range"
                  min="0"
                  max={POOL_STOPS_HOME.length - 1}
                  step="1"
                  value={surprisePoolIndex}
                  oninput={(e) =>
                    onSurprisePoolIndex?.(
                      parseInt((e.target as HTMLInputElement).value, 10)
                    )}
                  aria-label="Surprise pool size"
                />
              </label>
              {#if surpriseTotal !== null}
                <div class="surprise-status">
                  Deck: {surpriseTotal.toLocaleString()} photos
                </div>
              {/if}
            </div>
          {/if}
        </div>
      {/if}
    </div>
  {/if}
</section>

<style>
  .filters {
    overflow: hidden;
    padding: 0;
    /* Width matches the PromptChips card width above us
       (which fills .hero minus its 16px horizontal padding).
       The hero itself is --grid-width on the home page, matching
       the .head width on the other grid pages, but inside the
       hero the composer card has internal padding so its chrome
       is slightly narrower — we match that chrome edge so the
       two cards visually line up. */
    width: 100%;
    margin: 0 auto;
    /* The hero is text-align: center; override so this panel's
       labels (Filename, Surprise me, Limit to library) read as
       a form, not as centered prose. */
    text-align: left;
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    padding: var(--s-2) var(--s-3);
    color: var(--fg-1);
    font-weight: 500;
  }
  .chev {
    color: var(--fg-2);
    font-size: 14px;
  }
  .body {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: var(--s-3);
    padding: var(--s-0) var(--s-3) var(--s-3);
    border-top: 1px solid var(--glass-edge);
  }
  /* Inline label + control pattern. The label
     sits on the left at a fixed width so "Filename" and "Surprise
     me" align at the same x, and the input/control fills the
     remaining row width via flex: 1. Without this the fields
     stack label-on-top-of-control which forces the control to
     fill the full column width (235px) and look oversized for
     short values like "5,000" or "Off". */
  .field {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }
  .field .lab {
    color: var(--fg-2);
    font-size: var(--fs-sm);
    /* Wider than the label column (96px) because "Filename contains" and
     "Surprise me" don't fit in 96px without wrapping. The +24px
     keeps all labels on one line while still aligning the
     dropdowns at a consistent x. */
    width: 120px;
    flex-shrink: 0;
  }
  .field input[type='text'] {
    background: var(--bg-1);
    border: 1px solid var(--glass-edge);
    border-radius: var(--r-pill);
    padding: 0 var(--s-2);
    height: 36px;
    color: var(--fg-1);
    transition: border-color var(--t-fast);
    /* Fill the grid cell so every column reads the same width.
       Without this, native <select> sizes to its content (often
       narrower than the field's allocated column), making the
       dropdown look pinched next to a wider text input. */
    width: 100%;
    box-sizing: border-box;
  }
  .field input[type='text']:focus {
    border-color: var(--accent);
  }
  /* The collections chip-row spans the full grid width — it's a
     row of toggleable chips, not a single labeled input. */
  .collections-row {
    grid-column: 1 / -1;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .collections-row .lab {
    color: var(--fg-2);
    font-size: var(--fs-sm);
  }
  /* Round-90: the Surprise me section. Spans the full grid
     width, contains a header row (label + toggle) and a
     controls block (sliders) that only renders when Surprise
     is on. Visually consistent with .collections-row above. */
  .surprise-row {
    grid-column: 1 / -1;
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
  }
  .surprise-head {
    display: flex;
    align-items: center;
    gap: var(--s-2);
  }
  .surprise-head .lab {
    color: var(--fg-2);
    font-size: var(--fs-sm);
    width: 120px;
    flex-shrink: 0;
  }
  /* Toggle switch styled as a pill: a rounded track with a
     circular thumb that slides when the .on class is set. The
     track is wider than the thumb so there's visible movement
     on toggle. */
  .surprise-toggle {
    display: inline-flex;
    align-items: center;
    gap: var(--s-2);
    background: transparent;
    border: 0;
    padding: 0;
    cursor: pointer;
    color: var(--fg-2);
    font-size: var(--fs-sm);
  }
  .surprise-toggle-track {
    position: relative;
    width: 36px;
    height: 20px;
    border-radius: 999px;
    background: var(--bg-1, rgba(0, 0, 0, 0.06));
    border: 1px solid var(--glass-edge);
    transition: background var(--t-fast);
  }
  .surprise-toggle-thumb {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--fg-2);
    transition: transform var(--t-fast), background var(--t-fast);
  }
  .surprise-toggle.on .surprise-toggle-track {
    background: var(--accent);
    border-color: var(--accent);
  }
  .surprise-toggle.on .surprise-toggle-thumb {
    transform: translateX(16px);
    background: var(--bg-1, #fff);
  }
  .surprise-toggle.on .surprise-toggle-label {
    color: var(--accent);
    font-weight: 500;
  }
  .surprise-toggle-label {
    color: var(--fg-2);
    font-size: var(--fs-sm);
  }
  .surprise-controls {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    padding: var(--s-2) var(--s-3);
    border: 1px solid var(--glass-edge);
    border-radius: var(--r-md, 8px);
    background: var(--bg-1, rgba(0, 0, 0, 0.02));
  }
  .surprise-slider {
    display: flex;
    flex-direction: column;
    gap: var(--s-1);
  }
  .surprise-slider-label {
    color: var(--fg-2);
    font-size: var(--fs-sm);
    display: flex;
    align-items: baseline;
    gap: 4px;
    flex-wrap: wrap;
  }
  .surprise-slider-label strong {
    color: var(--fg-1);
    font-weight: 600;
  }
  .surprise-slider-label .hint {
    color: var(--fg-2);
    opacity: 0.7;
  }
  .surprise-slider input[type='range'] {
    width: 100%;
    accent-color: var(--accent);
  }
  .surprise-status {
    color: var(--fg-2);
    font-size: var(--fs-sm);
    text-align: right;
  }
  input[type='range'] {
    flex: 1;
    accent-color: var(--accent);
  }
</style>
