/**
 * Layout mode — the photo grid's arrangement, chosen at runtime.
 *
 * The frontend is ONE static build shared by every deployment, so the
 * grid's layout cannot be baked into the bundle: a deployment picks
 * between the uniform square grid and Pinterest-style masonry, and
 * switching must not require a rebuild.
 *
 * The backend stamps the deployment's default onto the shell it
 * serves, as a `data-masonry` attribute on <html> (see
 * search/layout.py), and this module reads it back — the same
 * contract the theme uses.
 *
 * This is the DEFAULT, not a lock. The user can override it in the
 * browser (the Settings page), and their choice is remembered there.
 */
import { readable } from 'svelte/store';

/** The attribute the backend stamps the layout mode onto. */
export const MASONRY_ATTRIBUTE = 'data-masonry';

/**
 * Read the masonry mode from an attribute value.
 *
 * Only an explicit "1" or "true" turns masonry on — a missing or
 * unreadable marker means the uniform grid, which is what the app
 * has always rendered.
 */
export function masonryFromAttribute(value: string | null | undefined): boolean {
  return value === '1' || value === 'true';
}

/**
 * The masonry mode the current document was served with.
 *
 * Read from <html data-masonry="...">. Outside the browser there is
 * no marker, so the default applies.
 */
export function readMasonryDefault(): boolean {
  if (typeof document === 'undefined') return false;
  return masonryFromAttribute(
    document.documentElement.getAttribute(MASONRY_ATTRIBUTE)
  );
}

/**
 * The deployment's masonry default, read once at module load.
 *
 * The shell's attribute is in the HTML the browser parsed before any
 * script ran, so it is available here — and it cannot change under
 * us, because a deployment change means a new shell.
 */
export const masonryDefault = readable(readMasonryDefault());
