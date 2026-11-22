/**
 * User preferences — persisted to localStorage so they survive
 * across sessions on the same browser. Per-device by design
 * (no backend round-trip; this is purely a UX knob).
 *
 * Currently two knobs:
 *   - slideshowIntervalMs: how long each photo stays up during
 *     Lightbox auto-advance (Play button in the action bar).
 *     Default 3000 (3s).
 *   - gridSize: how dense the photo grid renders. Single setting
 *     drives tile size, gap, and max columns. See $lib/grid/config
 *     for the per-size numbers.
 *     Default 'large' (preserves the Round-55 visual).
 *
 * Add new knobs by extending `Preferences` + `defaults` and
 * providing a typed setter. Subscribers are notified on every
 * update; the Svelte auto-subscription syntax (`$preferences`)
 * gives components a reactive read.
 *
 * localStorage is best-effort: private-browsing mode, disabled
 * storage, quota exceeded, jsdom in Vitest, etc. all result in
 * `localStorage === undefined` and we silently fall back to
 * in-memory defaults so the UI keeps working. The persistence
 * is a nice-to-have, never a hard requirement.
 */
import { get, writable } from 'svelte/store';
import { SLIDESHOW_PRESETS_MS, DEFAULT_SLIDESHOW_MS } from '$lib/api/limits';
import { DEFAULT_GRID_SIZE, type GridLayout, type GridSize } from '$lib/grid/config';
import { readLayoutDefault } from '$lib/layout';

const STORAGE_KEY = 'lustre:preferences';

export type SlideshowPreset = (typeof SLIDESHOW_PRESETS_MS)[number];

// Phase 1: SLIDESHOW_PRESETS_MS is now the source of truth in
// $lib/api/limits (mirrors the backend's slideshow cadence).
// This module keeps the labelled UI version, derived from the
// numeric preset list, so the Settings page can show "1s / 3s
// / 5s / 10s / 20s / 30s" without duplicating the ms values.
export const SLIDESHOW_PRESETS: { label: string; ms: SlideshowPreset }[] =
  SLIDESHOW_PRESETS_MS.map((ms) => ({
    label: ms < 1000 ? `${ms}ms` : `${Math.round(ms / 1000)}s`,
    ms
  }));

// Re-exported for callers (e.g. preferences.test.ts) that
// historically read DEFAULT_SLIDESHOW_MS from this module.
export { DEFAULT_SLIDESHOW_MS };

// The grid-size numbers (tile, gap, max columns, rows per page)
// live in $lib/grid/config — they are static layout constants, not
// user state, and both the CSS/JS renderer and the fetch layer
// read them from there. This module owns only the user's CHOICE
// of size; the mapping from choice to numbers lives in one place
// so the Settings page, the layout-level ResizeObserver, the
// virtualizer, and the request sizing cannot drift apart.

type Preferences = {
  slideshowIntervalMs: number;
  gridSize: GridSize;
  /**
   * The user's layout choice, or null for "no choice".
   *
   * Null is not the same as 'grid': the deployment picks a default
   * (LUSTRE_LAYOUT, stamped onto the shell), and the user can
   * override it. A stored mode means the user chose; null means
   * "follow the deployment". Storing the resolved default instead
   * would freeze the deployment's choice into this browser the first
   * time any other setting changed.
   */
  layout: GridLayout | null;
};

const defaults: Preferences = {
  slideshowIntervalMs: DEFAULT_SLIDESHOW_MS,
  gridSize: DEFAULT_GRID_SIZE,
  layout: null
};

const LAYOUTS: GridLayout[] = ['grid', 'masonry', 'justified'];

function load(): Preferences {
  if (typeof localStorage === 'undefined') return { ...defaults };
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...defaults };
    const parsed = JSON.parse(raw) as Partial<Preferences> & {
      masonry?: boolean | null;
    };
    // Validate the slideshow knob.
    const candidate = parsed.slideshowIntervalMs;
    const slideshowValid =
      typeof candidate === 'number' &&
      Number.isFinite(candidate) &&
      candidate > 0;
    // Validate the gridSize knob — must be one of the three
    // known values. A user-tampered or schema-drifted value
    // falls back to the default rather than rendering the
    // grid with NaN dimensions.
    const gridCandidate = parsed.gridSize;
    const gridSize: GridSize =
      gridCandidate === 'large' ||
      gridCandidate === 'medium' ||
      gridCandidate === 'small'
        ? gridCandidate
        : DEFAULT_GRID_SIZE;
    // Validate the layout knob — one of the three known modes.
    // A user-tampered or schema-drifted value falls back to
    // following the deployment rather than rendering the grid with
    // an unknown arrangement.
    //
    // A browser that stored the OLD shape has `masonry: boolean`
    // instead of `layout`. Migrate it: true became 'masonry', false
    // became 'grid', and null stayed null (follow the deployment).
    const layoutCandidate = parsed.layout;
    const legacyMasonry = parsed.masonry;
    let layout: GridLayout | null;
    if (layoutCandidate !== undefined && layoutCandidate !== null) {
      layout = LAYOUTS.includes(layoutCandidate as GridLayout)
        ? (layoutCandidate as GridLayout)
        : null;
    } else if (legacyMasonry !== undefined) {
      layout =
        legacyMasonry === true ? 'masonry' : legacyMasonry === false ? 'grid' : null;
    } else {
      layout = null;
    }
    return {
      slideshowIntervalMs: slideshowValid ? candidate : defaults.slideshowIntervalMs,
      gridSize,
      layout
    };
  } catch {
    return { ...defaults };
  }
}

function save(prefs: Preferences) {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    // localStorage unavailable (private mode, quota) — in-memory
    // value still works for the current session.
  }
}

export const preferences = writable<Preferences>(load());

// Subscribe once at module load: every set call goes through
// localStorage automatically. The store's own initial value is
// either the loaded-from-storage value or the defaults, so the
// first subscriber sees the correct state on the same tick.
preferences.subscribe(save);

export function setSlideshowInterval(ms: number) {
  // Clamp to a sane range: 500ms–60s. Outside that we just keep
  // the current value rather than writing nonsense; the Settings
  // page only offers presets so this is a defence-in-depth check
  // for any future programmatic caller.
  if (!Number.isFinite(ms) || ms < 500 || ms > 60_000) return;
  preferences.update((p) => ({ ...p, slideshowIntervalMs: ms }));
}

export function setGridSize(size: GridSize) {
  if (size !== 'large' && size !== 'medium' && size !== 'small') return;
  preferences.update((p) => ({ ...p, gridSize: size }));
}

/**
 * Choose the grid's layout.
 *
 * `null` hands the choice back to the deployment's default, which
 * is what "follow the deployment" means in the Settings page.
 */
export function setLayout(layout: GridLayout | null) {
  if (layout !== null && !LAYOUTS.includes(layout)) return;
  preferences.update((p) => ({ ...p, layout }));
}

/**
 * The layout the grid should render right now.
 *
 * The user's explicit choice wins; otherwise the deployment's
 * default applies.
 */
export function layoutEnabled(prefs: Preferences): GridLayout {
  return prefs.layout ?? readLayoutDefault();
}

/**
 * The layout mode right now, for code that runs outside a
 * component.
 *
 * The fetch layer's column math (grid/geometry.ts) has no reactive
 * context, exactly like the theme read there — it measures the DOM
 * and reads the store at call time. The store's current value
 * carries the user's Grid/Masonry/Justified choice, so this is the
 * same answer the components get, just read imperatively.
 */
export function currentLayout(): GridLayout {
  return layoutEnabled(get(preferences));
}
