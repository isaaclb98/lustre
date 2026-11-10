/**
 * Deterministic-by-seed shuffling for the presentation wall.
 *
 * The wall shows an album's photos in a random order, re-rolled
 * every time it is opened. The order is the presentation's whole
 * point: an album in its natural order reads as a sequence the user
 * already knows, while a random one shows the album's variety.
 *
 * Two properties matter and both are easy to get wrong:
 *
 * 1. **Unbiased.** `array.sort(() => Math.random() - 0.5)` is NOT a
 *    uniform shuffle — the comparator is inconsistent (it does not
 *    define a total order), and the resulting distribution over
 *    permutations is measurably skewed. Fisher-Yates with a proper
 *    random index in range is uniform.
 *
 * 2. **Reproducible from a seed.** A seed lets the same album
 *    produce the same order twice, which is what makes a shared
 *    link reproduce what the sender saw. The seed is a plain
 *    integer, so it can travel in a URL.
 */

/**
 * A small, fast, seedable PRNG (mulberry32). Chosen over a full
 * PRNG library because the only requirement is reproducibility for
 * a given seed — not cryptographic quality or statistical
 * sophistication.
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function next(): number {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Fisher-Yates shuffle. Returns a NEW array; the input is not
 * mutated. With `rng` omitted the shuffle is seeded from
 * `Math.random()`, so two calls differ.
 */
export function shuffled<T>(items: readonly T[], rng: () => number = Math.random): T[] {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}

/**
 * Shuffle with an explicit integer seed. The same seed and the same
 * input always produce the same output — the property a shared link
 * depends on.
 */
export function shuffledWithSeed<T>(items: readonly T[], seed: number): T[] {
  return shuffled(items, mulberry32(seed));
}
