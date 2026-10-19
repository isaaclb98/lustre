<script lang="ts">
  /**
   * TopBar with tabs. Sticky on top, glass surface, single logo
   * + a horizontal row of tabs. No floating pill — sits as a
   * real document-flow strip with whitespace separating it from
   * content.
   *
   * Round‑29c: the Home tab uses `location.assign('/')` instead
   * of an anchor `href="/"` so clicking it from
   * `/?positives=…&diversity=…` actually navigates back to a
   * clean home (clears query string, resets composer state).
   *
   * A plain `<a href="/">` on the same pathname is a no-op for
   * the browser + SvelteKit — the URL doesn't change because
   * the path is already `/`, so the user stays on the search
   * results page. `goto('/')` from `$app/navigation` has the
   * same problem: it skips navigation when the URL is identical
   * to the current page.
   *
   * `location.assign('/')` always triggers a full page load,
   * which is exactly what we want for "reset to default home".
   */
  import { page } from '$app/stores';
  import Icon from './Icon.svelte';

  type Tab = { href: string; label: string; reset?: boolean };
  const tabs: Tab[] = [
    { href: '/', label: 'Home', reset: true },
    { href: '/random', label: 'Random' },
    { href: '/for-you', label: 'For you' },
    { href: '/albums', label: 'Albums' },
    { href: '/settings', label: 'Settings' }
  ];

  let currentPath = $derived($page.url.pathname);
  /* Round-60: detail pages (/photo/[id], /similar/[id]) don't
     match any tab href in the list, so before this rule the
     isActive() check would return false for every tab and the
     topbar read as 'no current location' on photo pages —
     Isaac called this 'stale'. Detail views are an extension
     of photo browsing, which lives under the Random tab, so
     we treat /photo and /similar paths as active-Random. The
     album detail page (/albums/[id]) keeps matching its own
     Albums href via the default startsWith check below. */
  function isActive(href: string, path: string): boolean {
    if (href === '/') return path === '/';
    if (path.startsWith('/photo/') || path.startsWith('/similar/')) {
      return href === '/random';
    }
    return path === href || path.startsWith(href + '/');
  }
  function onTabClick(t: Tab, e: MouseEvent) {
    // Modifier‑click (cmd / ctrl / shift) should still open in
    // a new tab / window — let the browser handle it.
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) {
      return;
    }
    // "Reset" tabs (Home) always do a full page load so the URL
    // and the composer state get cleared. Other tabs use plain
    // anchor navigation.
    if (t.reset) {
      e.preventDefault();
      location.assign(t.href);
    }
  }
</script>

<header class="topbar">
  <div class="bar">
    <a class="brand" href="/" aria-label="Home">
      <Icon name="sparkle" size={22} />
      <span class="brand-text">Lustre</span>
    </a>
    <nav class="tabs" aria-label="Main">
      {#each tabs as t (t.href)}
        <a
          class="tab"
          class:active={isActive(t.href, currentPath)}
          href={t.href}
          onclick={(e) => onTabClick(t, e)}
          aria-current={isActive(t.href, currentPath) ? 'page' : undefined}
        >{t.label}</a>
      {/each}
    </nav>
  </div>
</header>

<style>
  /* Solid tinted header. Round-38 switched from dark glass to
     light Immich-style. The topbar is a translucent white pill
     with hairline border — it sits on the mesh-gradient
     backdrop, lets the colour wash show through, and uses a
     border (not a shadow) to separate from content below. */
  .topbar {
    position: sticky;
    top: 0;
    z-index: 100;
    background-color: var(--glass-fill-light);
    border-bottom: 1px solid var(--glass-edge);
    /* Box-shadow gives the topbar the impression of a surface
       floating just above the page — without the backdrop-filter
       frost, this is what sells the "layer above content" read. */
    box-shadow: var(--shadow-glass);
  }
  .bar {
    height: var(--topbar-h);
    display: flex;
    align-items: center;
    justify-content: center;
    /* Brand + tabs together as one unit, centered in the bar.
       Round-52: justify-content: center puts the entire
       brand-and-tabs cluster at the dead-center of the bar.
       Brand sits immediately to the left of the tabs nav
       (gap: 24px), and the whole cluster is centered as one
       block. Symmetric whitespace on both sides. */
    gap: var(--s-4);
    padding: 0 var(--shell-pad-x);
    /* Match the page header card width via --grid-width so the
       brand mark and tabs stay vertically aligned with the random
       / for-you / albums header below. The hardcoded 1600px that
       used to live here drifted away from --grid-width at every
       zoom level (bar widened to 2400px at 150%, header only to
       2190px), pulling the nav tabs noticeably rightward of the
       page chrome they were supposed to align with. */
    width: var(--grid-width, 100%);
    max-width: 100%;
    margin: 0 auto;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: var(--s-1);
    color: var(--fg-1);
    font-weight: 500;
    letter-spacing: var(--ls-wider);
  }
  .brand:hover { color: var(--fg-1); }
  /* The brand mark is an SVG Icon — no CSS needed here. The
   * Icon's stroke uses currentColor so the surrounding .brand
   * color drives it; alignment with the text comes from the
   * Icon's own viewBox + the .brand flex layout. */
  .tabs {
    display: flex;
    gap: var(--s-1);
    align-items: center;
  }
  .tab {
    display: inline-flex;
    align-items: center;
    height: 34px;
    padding: 0 14px;
    border-radius: var(--r-pill);
    color: var(--fg-2);
    font-weight: 500;
    transition: background var(--t-fast) var(--ease-out),
                color var(--t-fast) var(--ease-out);
  }
  .tab:hover {
    background: var(--glass-1);
    color: var(--fg-1);
  }
  /* Active tab: soft blue wash + accent text. This is the
     Immich-style "you are here" affordance — not a solid blue
     pill, but a tinted backdrop with the accent colour text on
     top. Reads as "selected" without competing with the page
     content. */
  .tab.active {
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid transparent;
  }
  @media (max-width: 640px) {
    /* Hide the brand mark entirely below 640px so the 5 tabs
       fit without horizontal overflow at 375px (5 tabs + brand
       icon ≈ 397px > 375px viewport). At <480px we further
       shrink padding and font-size so the tabs themselves
       breathe inside the fixed-width bar. */
    .bar {
      /* Drop the --grid-width constraint on mobile — it derived
         from the photo-grid column math, but on small viewports
         the bar should fill the viewport so the tabs sit on the
         full topbar background rather than appearing centered
         inside a narrower inner bar. */
      width: 100%;
      padding: 0 var(--s-2);
      gap: var(--s-1);
    }
    .brand { display: none; }
    .brand-text { display: none; }
    .tab {
      padding: 0 10px;
      /* Prevent "For you" from wrapping to two lines at 375px —
         the tab's natural width at 15px font is wider than the
         tab wants to be when text wraps. */
      white-space: nowrap;
    }
  }
  @media (max-width: 480px) {
    /* At 375px viewport, 5 tabs at full font + padding still
       overflow (362 + 16 gap + 24 padding = 402 vs 375 viewport).
       Drop the font from --fs-md (15) to --fs-sm (13) so all
       five tabs fit inside the bar with normal padding. */
    .tab {
      padding: 0 8px;
      font-size: var(--fs-sm);
    }
  }
</style>
