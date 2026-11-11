<script module lang="ts">
  /**
   * Tiny chip primitive used inside the prompt-chips list.
   * Has variants: positive (default) and negative (darker, red
   * soft accent). Each chip has a removal × button.
   */
</script>

<script lang="ts">
  import Icon from './Icon.svelte';

  type Props = {
    text: string;
    negative?: boolean;
    onRemove?: () => void;
    title?: string;
  };
  let { text, negative = false, onRemove, title }: Props = $props();
</script>

<span class="chip {negative ? 'neg' : 'pos'}" {title}>
  {#if negative}<span class="minus" aria-hidden="true"><Icon name="minus" size={12} /></span>{/if}
  <span class="text">{text}</span>
  {#if onRemove}
    <button class="x" onclick={onRemove} aria-label="Remove {text}">
      <Icon name="close" size={12} />
    </button>
  {/if}
</span>

<style>
  .chip {
    display: inline-flex;
    align-items: center;
    gap: var(--s-1);
    background: var(--glass-1);
    color: var(--fg-1);
    border: 1px solid var(--edge-control-color);
    border-radius: var(--r-pill);
    background-image: var(--control-film);
    backdrop-filter: var(--control-blur);
    -webkit-backdrop-filter: var(--control-blur);
    box-shadow: var(--control-texture);
    padding: var(--s-0) var(--s-2) var(--s-0) var(--s-3);
    font-size: var(--fs-sm);
    line-height: 1;
    height: 28px;
    transition: background var(--t-fast) var(--ease-out);
  }
  .chip.neg {
    color: var(--fg-2);
    background: var(--negative-soft);
    border-color: color-mix(in srgb, var(--negative) 30%, transparent);
  }
  .chip:hover { background-color: var(--glass-2); }
  .chip.neg:hover { background: color-mix(in srgb, var(--negative) 30%, transparent); }
  .minus {
    color: var(--negative);
    font-weight: 500;
  }
  .text { white-space: nowrap; }
  .x {
    width: 18px;
    height: 18px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: var(--fg-3);
    border-radius: 50%;
    font-size: 14px;
    line-height: 1;
    transition: background var(--t-fast) var(--ease-out), color var(--t-fast);
  }
  .chip.neg .x { color: var(--negative); }
  .x:hover { background: var(--glass-2); color: var(--fg-1); }
</style>
