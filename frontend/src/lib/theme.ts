/**
 * Theme selection — which palette and grid geometry the app renders.
 *
 * The frontend is ONE static build shared by every deployment, so the
 * theme cannot be baked into the bundle: switching themes must not
 * require a rebuild. The backend stamps the active theme onto the
 * shell it serves as a `data-theme` attribute on <html> (see
 * search/theme.py) and this module reads it back.
 *
 * One value drives three things, and they must never disagree:
 *   - the palette (CSS: `html[data-theme='dark']` in theme-dark.css),
 *   - the grid geometry (gap and tile radius — grid/config.ts),
 *   - the CSS custom properties the root layout publishes.
 *
 * The palette and the geometry are two halves of one decision, so
 * both read this module rather than each deciding for themselves.
 */
import { readable } from 'svelte/store';

/** The theme the app renders in. */
export type Theme = 'light' | 'dark';

/** The attribute the backend stamps the theme onto. */
export const THEME_ATTRIBUTE = 'data-theme';

/** The theme used when nothing says otherwise. */
export const DEFAULT_THEME: Theme = 'light';

/**
 * Read a theme from an attribute value. Anything that is not
 * explicitly 'dark' is 'light' — an unreadable marker must not
 * produce an undefined theme.
 */
export function themeFromAttribute(value: string | null | undefined): Theme {
  return value === 'dark' ? 'dark' : DEFAULT_THEME;
}

/**
 * The theme the current document was served with.
 *
 * Read from <html data-theme="...">. Outside the browser there is no
 * marker, so the default applies.
 */
export function readTheme(): Theme {
  if (typeof document === 'undefined') return DEFAULT_THEME;
  return themeFromAttribute(document.documentElement.getAttribute(THEME_ATTRIBUTE));
}

/**
 * The theme in force right now, read straight from the document.
 *
 * For code that runs outside a component (the fetch layer's column
 * math, which has no reactive context) — it reads the same marker
 * the store below watches, so the two can never disagree.
 */
export function currentTheme(): Theme {
  return readTheme();
}

/**
 * The active theme.
 *
 * A store rather than a constant because the marker is a DOM
 * attribute: a deployment stamps it before the page loads, but
 * anything that changes it afterwards (a test, a browser console)
 * has to reach the components too. The theme is not a user-facing
 * setting — there is no in-app switch — so this exists for
 * correctness, not for a control.
 */
export const theme = readable<Theme>(readTheme(), (set) => {
  if (typeof document === 'undefined' || typeof MutationObserver === 'undefined') {
    return;
  }
  const root = document.documentElement;
  const observer = new MutationObserver(() => set(readTheme()));
  observer.observe(root, {
    attributes: true,
    attributeFilter: [THEME_ATTRIBUTE]
  });
  return () => observer.disconnect();
});
