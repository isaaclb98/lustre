<script lang="ts">
  /**
   * PromptChips — positive/negative prompt row. Negative chips
   * render darker (per spec). The +/− toggle flips which kind
   * the next chip will be. Pressing Enter commits the current
   * input as a chip and clears it (per spec) but does NOT
   * fire a search — the user explicitly clicks the Search
   * button (issue #6 from the bug list).
   *
   * Pure UI: state lives in the parent (the search page). The
   * component just renders and signals via callbacks.
   */
  import Chip from './Chip.svelte';

  type Mode = 'pos' | 'neg';
  type Props = {
    positives: string[];
    negatives: string[];
    input: string;
    mode: Mode;
    onInput: (v: string) => void;
    onAdd: (text: string, mode: Mode) => void;
    onRemovePositive: (i: number) => void;
    onRemoveNegative: (i: number) => void;
    onMode: (m: Mode) => void;
  };
  let {
    positives,
    negatives,
    input,
    mode,
    onInput,
    onAdd,
    onRemovePositive,
    onRemoveNegative,
    onMode
  }: Props = $props();

  function commit() {
    const trimmed = (input ?? '').trim();
    if (!trimmed) return;
    onAdd(trimmed, mode);
    onInput(''); // clear on commit per spec
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      // Only commit the chip. Do NOT fire a search — the user
      // explicitly clicks the Search button.
      e.preventDefault();
      commit();
    } else if (
      e.key === 'Backspace' &&
      !input &&
      mode === 'pos' &&
      positives.length
    ) {
      onRemovePositive(positives.length - 1);
    }
  }
</script>

<!-- The card chrome comes from the shared .glass utility —
     same mechanism as PageHeader and AdditionalFilters. Local
     CSS keeps only layout (padding/gap), no surface. -->
<div class="prompt-chips glass">
  {#if positives.length > 0 || negatives.length > 0}
    <div class="row">
      {#each positives as p, i (i + '|' + p)}
        <Chip text={p} onRemove={() => onRemovePositive(i)} />
      {/each}
      {#each negatives as n, i (i + '|' + n)}
        <Chip text={n} negative onRemove={() => onRemoveNegative(i)} />
      {/each}
    </div>
  {/if}
  <div class="composer">
    <div class="toggle" role="tablist" aria-label="Prompt polarity">
      <button
        type="button"
        role="tab"
        class:active={mode === 'pos'}
        class="seg"
        onclick={() => onMode('pos')}
        aria-selected={mode === 'pos'}
      >+</button>
      <button
        type="button"
        role="tab"
        class:active={mode === 'neg'}
        class="seg"
        onclick={() => onMode('neg')}
        aria-selected={mode === 'neg'}
      >−</button>
    </div>
    <input
      class="composer-input"
      type="text"
      placeholder={mode === 'pos' ? 'Add a positive prompt…' : 'Add a negative prompt…'}
      value={input ?? ''}
      oninput={(e) => onInput((e.target as HTMLInputElement).value)}
      onkeydown={handleKeydown}
      autocomplete="off"
      spellcheck="false"
      aria-label="Add prompt"
    />
    <button
      type="button"
      class="add"
      onclick={commit}
      disabled={!(input ?? '').trim()}
      title="Add prompt"
    >+</button>
  </div>
</div>

<style>
  /* Surface (background/border/radius/shadow) comes from the
     shared .glass utility — same mechanism as PageHeader and
     AdditionalFilters. Only layout lives here. The old local
     surface used a translucent glass-1 fill + backdrop-filter;
     .glass is the canonical opaque panel (Round-38: frosted
     glass doesn't read on the light pastel backdrop). */
  .prompt-chips {
    padding: var(--s-3) var(--s-3) var(--s-2);
  }
  .row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-1);
    min-height: var(--btn-h-sm);
    /* Optical: 10px leaves room for the second line of a wrapped
       prompt without making the row appear empty. */
    padding-bottom: 10px;
  }
  /* No .row:empty guard needed: the {#if positives.length || ...}
     in the template keeps the row out of the DOM entirely when
     no chips are present. Previously this rule was here but
     never matched because {#each} leaves comment nodes inside
     the row, which broke :empty. */
  .composer {
    display: flex;
    align-items: center;
    gap: 8px;
    /* Round-38: white pill on light theme. Was rgba(14,15,20,0.45)
       dark fill — would now be a dark bar on the light backdrop. */
    background: var(--bg-1);
    border: 1px solid var(--edge-control-color);
    border-radius: var(--r-pill);
    padding: var(--s-0) var(--s-1) var(--s-0) var(--s-0);
    /* Single-stop subtle shadow. --shadow-1 is two-stop and
       would add an extra 1px halo that this element never
       had. The literal value matches --shadow-1's first
       stop exactly so it stays in lockstep if --shadow-1
       ever changes. */
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .toggle {
    display: inline-flex;
    background: var(--glass-1);
    border-radius: var(--r-pill);
    border: 1px solid var(--edge-control-color);
    overflow: hidden;
  }
  .seg {
    width: 32px;
    height: 32px;
    color: var(--fg-3);
    background: transparent;
    border-radius: 0;
    font-weight: 500;
    font-size: var(--fs-md);
  }
  /* Round-69: segmented control active state matches
     other ON-state chrome in the app (TopBar tab, Settings
     preset, ActionButton aria-pressed): accent-soft fill +
     accent text. Was solid accent fill which read as a
     primary action not a selected segment. */
  .seg.active {
    background: var(--accent-soft);
    color: var(--accent);
  }
  .composer-input {
    flex: 1;
    height: 32px;
    color: var(--fg-1);
    font-size: var(--fs-md);
  }
  .composer-input::placeholder { color: var(--fg-3); }
  .add {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--glass-2);
    color: var(--fg-1);
    border: 1px solid var(--glass-edge-strong);
    font-size: 18px;
    transition: background var(--t-fast);
  }
  .add:hover { background: var(--accent-soft); color: var(--fg-1); }
  .add:disabled { opacity: 0.35; pointer-events: none; }
</style>
