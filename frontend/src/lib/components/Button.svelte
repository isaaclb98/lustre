<script lang="ts">
  /**
   * Single button primitive. Used for every interactive surface
   * that triggers an action. Three intents: primary, secondary,
   * ghost. The `additional` slot lets us drop in extras like a
   * label or icon.
   */
  type Variant = 'primary' | 'secondary' | 'ghost' | 'icon';
  type Size = 'sm' | 'md' | 'lg';
  type Props = {
    variant?: Variant;
    size?: Size;
    href?: string;
    /** Anchor-only passthrough. The album zip download needs
     *  target="_blank" rel="noopener" so the browser treats it
     *  as a file download instead of an SPA navigation. */
    target?: string;
    rel?: string;
    type?: 'button' | 'submit';
    disabled?: boolean;
    title?: string;
    onclick?: (e: MouseEvent) => void;
    children?: import('svelte').Snippet;
    /** Marker for the Modal primitive: when true, this button
     *  receives focus on dialog open. The Modal reads the
     *  attribute to find the initial-focus target rather than
     *  relying on DOM order, so destructive confirms can sit
     *  on the right without stealing focus from Cancel. */
    initialFocus?: boolean;
    /** Destructive-intent modifier. Composes with any variant:
     *  the idle state stays quiet (fg-3 text) so destructive
     *  actions don't compete for attention, and hover reveals
     *  the negative palette. Replaces the per-page bespoke
     *  `.del` buttons (albums list, album detail, saved
     *  searches) that each reimplemented this pattern. */
    danger?: boolean;
    /** ARIA attribute marking the button as opening a popup
     *  menu (e.g. the Dropdown trigger in /settings). The full
     *  ARIA enum is `menu | listbox | tree | grid | dialog`;
     *  we narrow to the two we actually use. */
    'aria-haspopup'?: 'menu' | 'listbox' | 'dialog';
    /** Expanded-state passthrough for dropdown triggers. The
     *  Dropdown primitive delegates ARIA semantics to its
     *  trigger snippet, so Button needs to carry the state
     *  rather than hardcoding it. */
    'aria-expanded'?: boolean;
    /** Pressed-state passthrough for toggle buttons (e.g. the
     *  Surprise Me on/off button on the home page). Lets screen
     *  readers announce the toggle state without forcing the
     *  consumer to drop down to a raw <button>. */
    'aria-pressed'?: boolean;
    /** Accessible name for icon-only triggers. */
    'aria-label'?: string;
    /** Extra class merged onto the button — for consumer-side
     *  hooks (E2E selectors, one-off layout tweaks like
     *  stretch-to-fill) that aren't a variant concern. */
    class?: string;
    /** Data-attribute passthrough (e.g. data-centroid on the
     *  album search buttons, consumed by E2E selectors). */
    data?: Record<string, string | undefined>;
  };
  let {
    variant = 'secondary',
    size = 'md',
    href,
    target,
    rel,
    type = 'button',
    disabled = false,
    title,
    onclick,
    initialFocus,
    danger = false,
    'aria-haspopup': ariaHaspopup,
    'aria-expanded': ariaExpanded,
    'aria-pressed': ariaPressed,
    'aria-label': ariaLabel,
    class: extraClass,
    data,
    children
  }: Props = $props();
</script>

{#if href}
  <!-- svelte-ignore a11y_role_supports_aria_props_implicit -->
  <a
    class="btn {variant} {size} {extraClass ?? ''}"
    class:danger
    {...(data ?? {})}
    {href}
    {target}
    {rel}
    aria-disabled={disabled ? 'true' : undefined}
    aria-pressed={ariaPressed}
    {title}
  >
    {#if children}{@render children()}{/if}
  </a>
{:else}
  <button
    class="btn {variant} {size} {extraClass ?? ''}"
    class:danger
    data-initial-focus={initialFocus ? '' : undefined}
    {...(data ?? {})}
    {type}
    {disabled}
    {title}
    {onclick}
    aria-haspopup={ariaHaspopup}
    aria-expanded={ariaExpanded}
    aria-pressed={ariaPressed}
    aria-label={ariaLabel}
  >
    {#if children}{@render children()}{/if}
  </button>
{/if}

<style>
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--s-1);
    border-radius: var(--r-pill);
    font-weight: var(--fw-medium);
    letter-spacing: var(--ls-default);
    transition: background var(--t-fast) var(--ease-out),
                transform var(--t-fast) var(--ease-out),
                border-color var(--t-fast) var(--ease-out),
                color var(--t-fast) var(--ease-out);
    border: 1px solid transparent;
    user-select: none;
    text-decoration: none;
    white-space: nowrap;
  }
  .btn:disabled, .btn[aria-disabled='true'] {
    /* NOT opacity. Fading a button dissolves it — the fill and the
       label fade together, and on the pale peach fill the whole
       button vanished into the page. A disabled control keeps its
       shape and reads as inert through a NEUTRAL fill and a muted
       label instead, which is what the text inputs already do. */
    opacity: 1;
    color: var(--disabled-fg);
    cursor: not-allowed;
    pointer-events: none;
  }
  .btn.primary:disabled, .btn.primary[aria-disabled='true'],
  .btn.secondary:disabled, .btn.secondary[aria-disabled='true'] {
    background: var(--disabled-bg);
  }

  /* sizes */
  .sm { height: 30px; padding: 0 12px; font-size: var(--fs-sm); }
  .md { height: 38px; padding: 0 16px; font-size: var(--fs-md); }
  /* The .lg size carries the cyan halo shadow — used only on the
     home page's primary Search button. The halo was previously
     hardcoded inside the home page; promoting it to the
     component makes the design intent visible in one place. */
  .lg {
    height: 46px;
    padding: 0 24px;
    font-size: var(--fs-lg);
    box-shadow: var(--shadow-button);
  }
  .lg:disabled,
  .lg[aria-disabled='true'] { box-shadow: none; }

  /* variants */
  .primary {
    background: var(--accent-fill);
    color: var(--fg-on-accent);
  }
  .primary:hover { background: var(--accent-fill-2); }

  .secondary {
    background: var(--glass-raised);
    color: var(--fg-1);
    border-color: var(--edge-strong-color);
    backdrop-filter: var(--glass-medium);
    -webkit-backdrop-filter: var(--glass-medium);
  }
  .secondary:hover {
    /* focus feedback: brighter glass + accent border hint */
    background: color-mix(in srgb, var(--fg-1) 14%, transparent);
    border-color: var(--accent-soft);
  }

  .ghost {
    background: transparent;
    color: var(--fg-2);
    border-color: var(--edge-control-color);
  }
  .ghost:hover {
    background: var(--glass-1);
    color: var(--fg-1);
  }

  .icon {
    width: 38px;
    height: 38px;
    padding: 0;
    background: transparent;
    color: var(--fg-2);
    border-radius: 50%;
  }
  .icon:hover { background: var(--glass-2); color: var(--fg-1); }
  .icon.sm { width: 30px; height: 30px; }
  .icon.lg { width: 46px; height: 46px; }

  .btn:active { transform: translateY(1px); }

  /* danger modifier — composes with any variant via
     :is() so one rule covers primary/secondary/ghost.
     Idle state goes quiet (fg-3) regardless of variant;
     hover reveals the negative palette. Declared after
     the variant rules and using :is() so it wins over
     .primary/.secondary/.ghost background+color without
     !important. */
  :is(.primary, .secondary, .ghost, .icon).danger {
    background: transparent;
    color: var(--fg-3);
    border-color: var(--edge-control-color);
  }
  :is(.primary, .secondary, .ghost, .icon).danger:hover {
    background: var(--negative-soft);
    color: var(--negative);
    border-color: var(--negative);
  }
</style>
