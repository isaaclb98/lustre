/**
 * Unit tests for lib/api/limits.ts — the centralised constants.
 *
 * These are "values, not behaviour" — the assertions document
 * the exact list of pool stops, default indices, and slideshow
 * presets the app uses. A drift here (someone tightening
 * POOL_STOPS or shifting a default index) should fail this
 * test, forcing the change to be intentional.
 */
import { describe, it, expect } from 'vitest';
import {
  GRID_PAGE_SIZE,
  POOL_SIZE_MIN,
  POOL_SIZE_MAX,
  TEMPERATURE_MIN,
  TEMPERATURE_MAX,
  POOL_STOPS_HOME,
  POOL_STOPS_FOR_YOU,
  DEFAULT_POOL_INDEX_HOME,
  DEFAULT_POOL_INDEX_FOR_YOU,
  DEFAULT_TEMPERATURE,
  SLIDESHOW_PRESETS_MS,
  DEFAULT_SLIDESHOW_MS
} from './limits';

describe('GRID_PAGE_SIZE', () => {
  it('is 24 (the fallback default for API calls that omit a limit)', () => {
    expect(GRID_PAGE_SIZE).toBe(24);
  });
});

describe('POOL_SIZE bounds', () => {
  it('POOL_SIZE_MIN and MAX are the Qdrant HNSW ceiling + lower bound', () => {
    expect(POOL_SIZE_MIN).toBe(200);
    expect(POOL_SIZE_MAX).toBe(50_000);
  });
  it('every HOME stop is within bounds', () => {
    for (const s of POOL_STOPS_HOME) {
      expect(s).toBeGreaterThanOrEqual(POOL_SIZE_MIN);
      expect(s).toBeLessThanOrEqual(POOL_SIZE_MAX);
    }
  });
  it('every FOR_YOU stop is within bounds', () => {
    for (const s of POOL_STOPS_FOR_YOU) {
      expect(s).toBeGreaterThanOrEqual(POOL_SIZE_MIN);
      expect(s).toBeLessThanOrEqual(POOL_SIZE_MAX);
    }
  });
});

describe('TEMPERATURE bounds', () => {
  it('matches the Gumbel-max API contract (0 = top-K, 1 = uniform)', () => {
    expect(TEMPERATURE_MIN).toBe(0);
    expect(TEMPERATURE_MAX).toBe(1);
    expect(DEFAULT_TEMPERATURE).toBeGreaterThanOrEqual(TEMPERATURE_MIN);
    expect(DEFAULT_TEMPERATURE).toBeLessThanOrEqual(TEMPERATURE_MAX);
  });
});

describe('POOL_STOPS_HOME', () => {
  it('is the documented 5-stop log-spaced array', () => {
    expect(POOL_STOPS_HOME).toEqual([200, 1_000, 5_000, 20_000, 50_000]);
  });
  it('the rightmost stop equals POOL_SIZE_MAX', () => {
    expect(POOL_STOPS_HOME[POOL_STOPS_HOME.length - 1]).toBe(POOL_SIZE_MAX);
  });
  it('DEFAULT_POOL_INDEX_HOME resolves to 5_000 (index 2)', () => {
    expect(POOL_STOPS_HOME[DEFAULT_POOL_INDEX_HOME]).toBe(5_000);
  });
});

describe('POOL_STOPS_FOR_YOU', () => {
  it('is the documented 6-stop log-spaced array (one finer step than home)', () => {
    expect(POOL_STOPS_FOR_YOU).toEqual([200, 500, 1_000, 2_000, 10_000, 50_000]);
  });
  it('the rightmost stop equals POOL_SIZE_MAX', () => {
    expect(POOL_STOPS_FOR_YOU[POOL_STOPS_FOR_YOU.length - 1]).toBe(POOL_SIZE_MAX);
  });
  it('DEFAULT_POOL_INDEX_FOR_YOU resolves to 2_000 (index 3)', () => {
    expect(POOL_STOPS_FOR_YOU[DEFAULT_POOL_INDEX_FOR_YOU]).toBe(2_000);
  });
});

describe('SLIDESHOW_PRESETS_MS', () => {
  it('is the documented 6-preset array', () => {
    expect(SLIDESHOW_PRESETS_MS).toEqual([1000, 3000, 5000, 10_000, 20_000, 30_000]);
  });
  it('DEFAULT_SLIDESHOW_MS is in the preset list', () => {
    expect(SLIDESHOW_PRESETS_MS).toContain(DEFAULT_SLIDESHOW_MS);
  });
  it('DEFAULT_SLIDESHOW_MS is 3000 (3 seconds)', () => {
    expect(DEFAULT_SLIDESHOW_MS).toBe(3000);
  });
});
