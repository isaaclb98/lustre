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
 *     drives tile size, gap, and max columns. See
 *     GRID_SIZE_CONFIG below for the per-size numbers.
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

// Grid size — single user-facing knob that drives three
// derived CSS/JS values. The mapping is in one place so the
// Settings page, the layout-level ResizeObserver, and the
// virtualizer all read the same numbers.
//
// Tile size: 240/192/144 px. Large matches the Round-55
// default; medium and small pack progressively more photos
// per row.
//
// Gap: 14/8/4 px. Larger photos get more breathing room
// (preserves the Round-38 design intent — 14px reads as
// "personal photo library", tighter gaps read as utilitarian).
// Smaller photos need a tighter gap to read as a continuous
// wall instead of fragmenting into a grid of small squares.
//
// Max cols: 6/8/10. The CSS grid uses `auto-fit` with the
// tile size as the minmax floor, so a narrow viewport shows
// fewer cols; the max-cols cap kicks in only on wide monitors.
// This is the per-size density promise: "Large is 6 cols
// wide, Small is 10 cols wide" regardless of viewport.
export type GridSize = 'large' | 'medium' | 'small';

export interface GridSizeConfig {
  tile: number;
  gap: number;
  maxCols: number;
  // First-page result count for infinite-scrolled grids
  // (home search, /random). Scales with the grid so every
  // size fills ~4 rows on initial load, matching the Large
  // preset's historical "fills the viewport" target. Sized
  // as a multiple of maxCols so the last row is always full
  // — no awkward trailing partial row.
  //
  // 4 rows × N cols is the design intent. Large stays at 24
  // (no behavior change for users who never touch the
  // setting), Medium gets 32 (8×4), Small gets 40 (10×4).
  pageSize: number;
}

export const GRID_SIZE_CONFIG: Record<GridSize, GridSizeConfig> = {
  large: { tile: 240, gap: 14, maxCols: 6, pageSize: 24 },
  medium: { tile: 192, gap: 8, maxCols: 8, pageSize: 32 },
  small: { tile: 144, gap: 4, maxCols: 10, pageSize: 40 }
};

export const DEFAULT_GRID_SIZE: GridSize = 'large';

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
