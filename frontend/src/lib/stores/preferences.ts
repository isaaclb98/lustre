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
import { writable } from 'svelte/store';
import { SLIDESHOW_PRESETS_MS, DEFAULT_SLIDESHOW_MS } from '$lib/api/limits';
import { DEFAULT_GRID_SIZE, type GridSize } from '$lib/grid/config';

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
};

const defaults: Preferences = {
  slideshowIntervalMs: DEFAULT_SLIDESHOW_MS,
  gridSize: DEFAULT_GRID_SIZE
};

function load(): Preferences {
  if (typeof localStorage === 'undefined') return { ...defaults };
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...defaults };
    const parsed = JSON.parse(raw) as Partial<Preferences>;
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
    return {
      slideshowIntervalMs: slideshowValid ? candidate : defaults.slideshowIntervalMs,
      gridSize
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
