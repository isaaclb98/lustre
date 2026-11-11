<script lang="ts">
  /**
   * Keyboard shortcut display primitive. Renders a single key
   * or a +separated combo (e.g. "⌘ K", "Esc", "↑↓") in a
   * small monospace-style chip that visually matches the rest
   * of the glass surface.
   *
   *   <Kbd>⌘</Kbd> + <Kbd>K</Kbd>
   *
   * Use in tooltips, command-palette hints, search-bar placeholders,
   * lightbox shortcut overlays, etc.
   */
  import type { Snippet } from 'svelte';
  type Props = {
    children: Snippet;
    title?: string;
  };
  let { children, title }: Props = $props();
</script>

<kbd class="kbd" {title}>
  {@render children()}
</kbd>

<style>
  .kbd {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.4em;
    height: 1.5em;
    padding: 0 0.4em;
    background: var(--glass-1);
    border: 1px solid var(--edge-control-color);
    border-radius: var(--r-1);
    color: var(--fg-2);
    font-family: var(--font-mono-stack);
    font-size: 0.85em;
    line-height: 1;
    /* Mono-ish character look without forcing a different font
       family — Inter's tabular numerals + the slight letter-spacing
       shift below make ⌘ / arrows read distinctly. */
    font-feature-settings: "tnum";
    letter-spacing: var(--ls-wider);
    box-shadow: 0 1px 0 var(--glass-edge-strong) inset;
  }
</style>
