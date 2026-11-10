import { describe, it, expect } from 'vitest';
import { mulberry32, shuffled, shuffledWithSeed } from './shuffle';

describe('shuffled', () => {
  it('returns a new array and leaves the input untouched', () => {
    const input = [1, 2, 3, 4, 5];
    const out = shuffled(input);
    expect(out).not.toBe(input);
    expect(input).toEqual([1, 2, 3, 4, 5]);
    expect([...out].sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5]);
  });

  it('is a permutation — every element appears exactly once', () => {
    const input = Array.from({ length: 50 }, (_, i) => i);
    const out = shuffled(input);
    expect(out).toHaveLength(50);
    expect(new Set(out).size).toBe(50);
  });

  it('handles empty and single-element inputs', () => {
    expect(shuffled([])).toEqual([]);
    expect(shuffled([42])).toEqual([42]);
  });

  it('actually reorders — an all-same-order result would be a bug', () => {
    const input = Array.from({ length: 30 }, (_, i) => i);
    const out = shuffled(input);
    expect(out).not.toEqual(input);
  });

  it('produces a uniform distribution over positions', () => {
    // A biased shuffle shows up as one element being far more
    // likely to land in some positions than others. Over many
    // shuffles each element should visit each position roughly
    // equally often.
    const n = 6;
    const input = Array.from({ length: n }, (_, i) => i);
    const counts = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    const runs = 6000;
    for (let s = 0; s < runs; s++) {
      const out = shuffled(input);
      out.forEach((value, pos) => {
        counts[value][pos]++;
      });
    }
    const expected = runs / n;
    for (const row of counts) {
      for (const c of row) {
        // A uniform shuffle puts each element in each position
        // expected times; allow a generous band so the test is not
        // flaky while still catching a real skew.
        expect(Math.abs(c - expected)).toBeLessThan(expected * 0.12);
      }
    }
  });
});

describe('shuffledWithSeed', () => {
  it('is reproducible for the same seed', () => {
    const input = Array.from({ length: 40 }, (_, i) => i);
    expect(shuffledWithSeed(input, 1234)).toEqual(shuffledWithSeed(input, 1234));
  });

  it('differs for different seeds', () => {
    const input = Array.from({ length: 40 }, (_, i) => i);
    expect(shuffledWithSeed(input, 1)).not.toEqual(shuffledWithSeed(input, 2));
  });

  it('is still a permutation', () => {
    const input = Array.from({ length: 40 }, (_, i) => i);
    const out = shuffledWithSeed(input, 99);
    expect(new Set(out).size).toBe(40);
  });
});

describe('mulberry32', () => {
  it('returns values in [0, 1)', () => {
    const rng = mulberry32(7);
    for (let i = 0; i < 1000; i++) {
      const v = rng();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('is reproducible', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });

  it('produces different streams for different seeds', () => {
    const a = mulberry32(1);
    const b = mulberry32(2);
    expect(a()).not.toBe(b());
  });
});
