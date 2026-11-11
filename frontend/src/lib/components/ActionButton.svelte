<script lang="ts">
  import type { Snippet } from 'svelte';

  type Props = {
    children: Snippet;
    href?: string;
    onclick?: (event: MouseEvent) => void;
    disabled?: boolean;
    title?: string;
    ariaPressed?: boolean | 'true' | 'false';
    ariaExpanded?: boolean;
    ariaHaspopup?: 'menu' | 'dialog' | 'listbox' | 'tree' | 'grid' | 'false';
    target?: string;
    rel?: string;
  };

  let {
    children,
    href,
    onclick,
    disabled = false,
    title,
    ariaPressed,
    ariaExpanded,
    ariaHaspopup,
    target,
    rel
  }: Props = $props();

  /* The component does not manage pressed state itself — the
     parent owns it and signals it via `ariaPressed`. CSS reacts
     to `aria-pressed='true'` to render a permanent darker fill. */
  function pressedAttr(
    v: Props['ariaPressed']
  ): 'true' | 'false' | undefined {
    if (v === true || v === 'true') return 'true';
    if (v === false || v === 'false') return 'false';
    return undefined;
  }

  function handleClick(event: MouseEvent) {
    if (disabled) return;
    if (onclick) onclick(event);
  }
</script>

{#if href}
  <!-- svelte-ignore a11y_role_supports_aria_props_implicit -->
  <a
    class="action"
    aria-pressed={pressedAttr(ariaPressed)}
    {href}
    {target}
    {rel}
    {title}
    aria-expanded={ariaExpanded}
    aria-haspopup={ariaHaspopup}
    {onclick}
  >
    {@render children()}
  </a>
{:else}
  <button
    type="button"
    class="action"
    aria-pressed={pressedAttr(ariaPressed)}
    {disabled}
    {title}
    aria-expanded={ariaExpanded}
    aria-haspopup={ariaHaspopup}
    onclick={handleClick}
  >
    {@render children()}
  </button>
{/if}

<style>
  .action {
    height: 32px;
    padding: 0 14px;
    border-radius: var(--r-pill);
    background: var(--glass-1);
    color: var(--fg-1);
    border: 1px solid var(--edge-control-color);
    transition:
      background var(--t-fast),
      border-color var(--t-fast),
      color var(--t-fast);
    text-decoration: none;
    font-size: var(--fs-sm);
    font-weight: 500;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    /* Keep every label on one line. In the lightbox bar,
       "Most similar" / "Add to album" / "Open raw" wrap to two
       lines if the container is tight, producing uneven button
       heights. Each label stays a single line; if the bar ever
       grows too wide the container scrolls horizontally
       instead. (Original fix at 2ce16b9.) */
    white-space: nowrap;
    flex-shrink: 0;
  }
  .action:hover { background: var(--glass-2); }
  .action:disabled { cursor: not-allowed; opacity: 0.5; }
  /* Round-69: permanent ON state for toggle actions (Like,
     Dislike). Same chrome as TopBar active tab + Settings
     preset - accent-soft fill + accent border. Was a
     dialog-fill (white) + glass-edge-strong which read as
     just "thicker border" rather than "selected". */
  .action[aria-pressed='true'] {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--accent);
  }
</style>
