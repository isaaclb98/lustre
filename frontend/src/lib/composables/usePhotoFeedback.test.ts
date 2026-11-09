/**
 * Unit tests for the usePhotoFeedback composable.
 *
 * Covers:
 *   - optimistic flip (state updates before the API call).
 *   - mutual exclusion (liking clears dislike, disliking
 *     clears like).
 *   - rollback on API failure.
 *   - per-id in-flight lock (rapid double-tap is a no-op).
 *   - mutualExclusion:false preserves the opposite flag
 *     (some pages want this for the favourites flow).
 *   - missing id is a no-op (not a throw).
 */
import { describe, it, expect, vi } from 'vitest';
import { usePhotoFeedback, type FeedbackState } from './usePhotoFeedback';

type Row = FeedbackState & { id: string };

function makeStore(initial: Row[]) {
  const map = new Map<string, FeedbackState>();
  for (const r of initial) map.set(r.id, { is_favorite: r.is_favorite, is_disliked: r.is_disliked });
  return {
    get: (id: string) => map.get(id),
    set: (id: string, patch: Partial<FeedbackState>) => {
      const cur = map.get(id);
      if (!cur) return;
      map.set(id, { ...cur, ...patch });
    },
    rows: () => Array.from(map.values()),
    _map: map
  };
}

describe('usePhotoFeedback', () => {
  it('flips is_favorite on like', async () => {
    const store = makeStore([{ id: 'a', is_favorite: false, is_disliked: false }]);
    const onLike = vi.fn().mockResolvedValue(undefined);
    const fb = usePhotoFeedback({ ...store, onLike, onUnlike: vi.fn() });
    await fb.toggleFavorite('a');
    expect(store.get('a')?.is_favorite).toBe(true);
    expect(onLike).toHaveBeenCalledOnce();
  });

  it('flips back to false on unlike', async () => {
    const store = makeStore([{ id: 'a', is_favorite: true, is_disliked: false }]);
    const onUnlike = vi.fn().mockResolvedValue(undefined);
    const fb = usePhotoFeedback({ ...store, onLike: vi.fn(), onUnlike });
    await fb.toggleFavorite('a');
    expect(store.get('a')?.is_favorite).toBe(false);
    expect(onUnlike).toHaveBeenCalledOnce();
  });

  it('mutual exclusion: liking clears dislike', async () => {
    const store = makeStore([{ id: 'a', is_favorite: false, is_disliked: true }]);
    const fb = usePhotoFeedback({
      ...store,
      onLike: vi.fn().mockResolvedValue(undefined),
      onUnlike: vi.fn(),
      onUndislike: vi.fn().mockResolvedValue(undefined)
    });
    await fb.toggleFavorite('a');
    expect(store.get('a')?.is_favorite).toBe(true);
    expect(store.get('a')?.is_disliked).toBe(false);
  });

  it('mutual exclusion: disliking clears like', async () => {
    const store = makeStore([{ id: 'a', is_favorite: true, is_disliked: false }]);
    const fb = usePhotoFeedback({
      ...store,
      onDislike: vi.fn().mockResolvedValue(undefined),
      onUndislike: vi.fn(),
      onUnlike: vi.fn().mockResolvedValue(undefined)
    });
    await fb.toggleDislike('a');
    expect(store.get('a')?.is_favorite).toBe(false);
    expect(store.get('a')?.is_disliked).toBe(true);
  });

  it('mutualExclusion:false preserves the opposite flag', async () => {
    const store = makeStore([{ id: 'a', is_favorite: true, is_disliked: false }]);
    const fb = usePhotoFeedback({
      ...store,
      onDislike: vi.fn().mockResolvedValue(undefined),
      onUndislike: vi.fn(),
      mutualExclusion: false
    });
    await fb.toggleDislike('a');
    expect(store.get('a')?.is_favorite).toBe(true);
    expect(store.get('a')?.is_disliked).toBe(true);
  });

  it('rolls back on API failure', async () => {
    const store = makeStore([{ id: 'a', is_favorite: false, is_disliked: false }]);
    const onLike = vi.fn().mockRejectedValue(new Error('boom'));
    const fb = usePhotoFeedback({ ...store, onLike, onUnlike: vi.fn() });
    await fb.toggleFavorite('a');
    expect(store.get('a')?.is_favorite).toBe(false);
  });

  it('in-flight lock: rapid double-tap on the same id is a no-op', async () => {
    const store = makeStore([{ id: 'a', is_favorite: false, is_disliked: false }]);
    let resolve!: (v: undefined) => void;
    const onLike = vi.fn().mockImplementation(
      () => new Promise<undefined>((r) => (resolve = r))
    );
    const fb = usePhotoFeedback({ ...store, onLike, onUnlike: vi.fn() });
    const p1 = fb.toggleFavorite('a');
    const p2 = fb.toggleFavorite('a');
    expect(fb.isInFlight('a')).toBe(true);
    expect(onLike).toHaveBeenCalledOnce();
    resolve(undefined);
    await p1;
    await p2;
    expect(fb.isInFlight('a')).toBe(false);
  });

  it('unknown id is a silent no-op', async () => {
    const store = makeStore([{ id: 'a', is_favorite: false, is_disliked: false }]);
    const onLike = vi.fn();
    const fb = usePhotoFeedback({ ...store, onLike, onUnlike: vi.fn() });
    await fb.toggleFavorite('does-not-exist');
    expect(onLike).not.toHaveBeenCalled();
  });
});
