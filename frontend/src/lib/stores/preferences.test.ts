/**
 * tests/preferences.test.ts — unit tests for the user preferences
 * store. Exercises the validation guard on `setSlideshowInterval`
 * (clamping nonsense values so the slideshow never auto-advances
 * every 0ms from a corrupted write) and the load/save round-trip
 * with a stub localStorage (Vitest's jsdom env doesn't expose
 * one by default in v2.x).
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { get } from 'svelte/store';

// jsdom under Vitest 2.x doesn't expose `localStorage` on the
// global. The store reads it as a top-level global, so we have
// to stub one before importing the module under test.
class FakeStorage {
  private store = new Map<string, string>();
  get length() {
    return this.store.size;
  }
  clear() {
    this.store.clear();
  }
  getItem(key: string) {
    return this.store.get(key) ?? null;
  }
  key(i: number) {
    return Array.from(this.store.keys())[i] ?? null;
  }
  removeItem(key: string) {
    this.store.delete(key);
  }
  setItem(key: string, value: string) {
    this.store.set(key, value);
  }
}

beforeEach(() => {
  vi.resetModules();
  globalThis.localStorage = new FakeStorage() as unknown as Storage;
});

// Import after the stub is in place. vi re-evaluates modules per
// test thanks to vi.resetModules(), so the store re-initialises
// against the (new) fresh storage each time.
async function loadStore() {
  return import('./preferences');
}

describe('preferences store', () => {
  it('starts at the default cadence when localStorage is empty', async () => {
    const { preferences: store, DEFAULT_SLIDESHOW_MS } = await loadStore();
    expect(get(store).slideshowIntervalMs).toBe(DEFAULT_SLIDESHOW_MS);
    expect(DEFAULT_SLIDESHOW_MS).toBe(3000);
  });

  it('persists updates to localStorage on every set', async () => {
    const { preferences: store, setSlideshowInterval } = await loadStore();
    setSlideshowInterval(10_000);
    const raw = localStorage.getItem('lustre:preferences');
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw ?? '{}');
    expect(parsed.slideshowIntervalMs).toBe(10_000);
    // The store itself reflects the same value.
    expect(get(store).slideshowIntervalMs).toBe(10_000);
  });

  it('rejects out-of-range values without mutating the store', async () => {
    const { preferences: store, setSlideshowInterval } = await loadStore();
    setSlideshowInterval(5000);
    expect(get(store).slideshowIntervalMs).toBe(5000);

    // Zero — would make the slideshow tick every millisecond and
    // freeze the browser. Reject.
    setSlideshowInterval(0);
    expect(get(store).slideshowIntervalMs).toBe(5000);

    // Negative — same.
    setSlideshowInterval(-1000);
    expect(get(store).slideshowIntervalMs).toBe(5000);

    // Too long — over a minute per photo would feel broken.
    setSlideshowInterval(120_000);
    expect(get(store).slideshowIntervalMs).toBe(5000);

    // NaN — invalid.
    setSlideshowInterval(Number.NaN);
    expect(get(store).slideshowIntervalMs).toBe(5000);
  });

  it('exposes every preset as a sane integer ms value', async () => {
    const { SLIDESHOW_PRESETS } = await loadStore();
    // Sanity: the UI renders the presets as chips on the
    // Settings page. Every chip must correspond to a valid
    // integer millisecond value inside the clamp range — the
    // sliders/list shouldn't offer anything that setSlideshowInterval
    // would silently drop.
    expect(SLIDESHOW_PRESETS.length).toBeGreaterThan(0);
    for (const preset of SLIDESHOW_PRESETS) {
      expect(preset.ms).toBeTypeOf('number');
      expect(Number.isInteger(preset.ms)).toBe(true);
      expect(preset.ms).toBeGreaterThanOrEqual(500);
      expect(preset.ms).toBeLessThanOrEqual(60_000);
      expect(preset.label).toBeTruthy();
    }
  });
});

describe('grid size', () => {
  it('starts at the default (Large) when localStorage is empty', async () => {
    const { preferences: store } = await loadStore();
    const { DEFAULT_GRID_SIZE } = await import('../grid/config');
    expect(get(store).gridSize).toBe('large');
    expect(DEFAULT_GRID_SIZE).toBe('large');
  });

  it('persists grid size changes to localStorage', async () => {
    const { preferences: store, setGridSize } = await loadStore();
    setGridSize('small');
    const raw = localStorage.getItem('lustre:preferences');
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw ?? '{}');
    expect(parsed.gridSize).toBe('small');
    expect(get(store).gridSize).toBe('small');
  });

  it('rejects invalid grid size values', async () => {
    const { preferences: store, setGridSize } = await loadStore();
    setGridSize('medium');
    expect(get(store).gridSize).toBe('medium');

    // String outside the allowed union. The setGridSize guard
    // is a runtime check, not a type-level one — the test
    // exercises the runtime path by passing a typed cast
    // that bypasses the union.
    setGridSize('huge' as unknown as 'large' | 'medium' | 'small');
    expect(get(store).gridSize).toBe('medium');

    // Empty string.
    setGridSize('' as unknown as 'large' | 'medium' | 'small');
    expect(get(store).gridSize).toBe('medium');
  });

  it('falls back to default when localStorage has a tampered grid size', async () => {
    localStorage.setItem(
      'lustre:preferences',
      JSON.stringify({ gridSize: 'huge' })
    );
    const { preferences: store } = await loadStore();
    const { DEFAULT_GRID_SIZE } = await import('../grid/config');
    expect(get(store).gridSize).toBe(DEFAULT_GRID_SIZE);
  });

  it('loads a previously-saved grid size', async () => {
    localStorage.setItem(
      'lustre:preferences',
      JSON.stringify({ gridSize: 'medium' })
    );
    const { preferences: store } = await loadStore();
    expect(get(store).gridSize).toBe('medium');
  });
});

describe('GRID_SIZE_CONFIG', () => {
  it('exposes one entry per GridSize with sane numeric values', async () => {
    const { GRID_SIZE_CONFIG, DEFAULT_GRID_SIZE } = await import('../grid/config');
    // Every key in the config must be one of the three valid
    // GridSize values.
    const configKeys = Object.keys(GRID_SIZE_CONFIG).sort();
    expect(configKeys).toEqual(['large', 'medium', 'small']);
    // The default must be one of the configured sizes.
    expect(configKeys).toContain(DEFAULT_GRID_SIZE);
    // Sanity-check the per-size numbers: tile + gap + maxCols
    // are all positive integers, and the column density scales
    // monotonically with the tile (smaller tile → more cols).
    const sizes = ['large', 'medium', 'small'] as const;
    for (const s of sizes) {
      const c = GRID_SIZE_CONFIG[s];
      expect(c.tile).toBeGreaterThan(0);
      expect(c.gap).toBeGreaterThan(0);
      expect(c.maxCols).toBeGreaterThan(0);
      expect(Number.isInteger(c.tile)).toBe(true);
      expect(Number.isInteger(c.gap)).toBe(true);
      expect(Number.isInteger(c.maxCols)).toBe(true);
    }
    const large = GRID_SIZE_CONFIG.large;
    const medium = GRID_SIZE_CONFIG.medium;
    const small = GRID_SIZE_CONFIG.small;
    expect(large.tile).toBeGreaterThan(medium.tile);
    expect(medium.tile).toBeGreaterThan(small.tile);
    expect(large.maxCols).toBeLessThan(medium.maxCols);
    expect(medium.maxCols).toBeLessThan(small.maxCols);
  });

  it('scales the base page size with maxCols so every size fills ~4 rows on first load', async () => {
    const { GRID_SIZE_CONFIG, basePageSize } = await import('../grid/config');
    // Design intent: base page count = 4 rows × maxCols. At the
    // preset's maximum column count that is already a whole
    // number of rows; narrower viewports round the count up to
    // the next whole number of rows at the columns they render
    // (see $lib/grid/geometry). Pinned numbers, not just a
    // ratio, so a future tweak that breaks the "no partial
    // trailing row" promise will fail this test.
    expect(basePageSize(GRID_SIZE_CONFIG.large)).toBe(24);
    expect(basePageSize(GRID_SIZE_CONFIG.medium)).toBe(32);
    expect(basePageSize(GRID_SIZE_CONFIG.small)).toBe(40);

    // Cross-check: every base page size is a multiple of maxCols
    // and corresponds to 4 full rows. If a future preset adds a
    // 5th col-count, this loop will catch the off-by-one.
    for (const s of ['large', 'medium', 'small'] as const) {
      const c = GRID_SIZE_CONFIG[s];
      const base = basePageSize(c);
      expect(base % c.maxCols).toBe(0);
      expect(base / c.maxCols).toBe(4);
    }
  });
});