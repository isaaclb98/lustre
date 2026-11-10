/**
 * usePhotoFeedback — consolidate the like/dislike mutation
 * pattern that was duplicated across Lightbox.svelte,
 * routes/(app)/+page.svelte, routes/(app)/albums/favourites/+page.svelte,
 * and routes/(app)/photo/[id]/+page.svelte.
 *
 * The shared shape:
 *   1. Optimistic local update (flip the flag, clear the
 *      opposite flag for mutual exclusion).
 *   2. Fire the API call.
 *   3. Rollback on failure.
 *   4. Per-id in-flight tracking so a rapid double-tap
 *      doesn't race itself.
 *
 * The composable is intentionally callback-driven rather
 * than owning a $state map. The four call sites have four
 * different state shapes (single photo object, list of
 * items, lightbox local booleans, photo page photo ref)
 * and the lighter a composable is on the data model,
 * the easier it is to migrate a new call site.
 *
 * Usage:
 *   const feedback = usePhotoFeedback({
 *     get:  (id) => items.find((x) => x.id === id),
 *     set:  (id, patch) => items = items.map(x => x.id === id ? { ...x, ...patch } : x),
 *     onLike:     (id) => likePoint(id),
 *     onUnlike:   (id) => unlikePoint(id),
 *     onDislike:  (id) => dislikePoint(id),
 *     onUndislike:(id) => undislikePoint(id),
 *   });
 *   // <button onclick={() => feedback.toggleFavorite(id)}>like</button>
 */
export type FeedbackState = {
  is_favorite: boolean;
  is_disliked: boolean;
};

export type FeedbackOptions = {
  /** Read the current state for a given photo id. */
  get: (id: string) => FeedbackState | undefined | null;
  /** Apply a partial state update for a given photo id. */
  set: (id: string, patch: Partial<FeedbackState>) => void;
  /** API calls. The composable never imports the endpoints
   * directly — each call site wires the ones it actually uses. */
  onLike?: (id: string) => Promise<unknown>;
  onUnlike?: (id: string) => Promise<unknown>;
  onDislike?: (id: string) => Promise<unknown>;
  onUndislike?: (id: string) => Promise<unknown>;
  /** If false, the opposite flag is NOT cleared when the
   * user toggles one side. Defaults to true (the historical
   * app-wide behaviour: a photo is liked OR disliked, never
   * both). The Lightbox and photo/[id] both rely on this. */
  mutualExclusion?: boolean;
};

export type FeedbackController = {
  /** Flip favourite state. No-op if id unknown or in flight. */
  toggleFavorite: (id: string) => Promise<void>;
  /** Flip disliked state. No-op if id unknown or in flight. */
  toggleDislike: (id: string) => Promise<void>;
  /** Per-id lock query. Use to disable the button while
   * the API call is in flight. */
  isInFlight: (id: string) => boolean;
};

export function usePhotoFeedback(opts: FeedbackOptions): FeedbackController {
  const mutualExclusion = opts.mutualExclusion ?? true;
  const inFlight = new Set<string>();

  async function toggle(
    id: string,
    field: keyof FeedbackState,
    onSet: FeedbackOptions['onLike'],
    onUnset: FeedbackOptions['onUnlike']
  ) {
    if (inFlight.has(id)) return;
    const current = opts.get(id);
    if (!current) return;
    const was = !!current[field];
    const opposite: keyof FeedbackState =
      field === 'is_favorite' ? 'is_disliked' : 'is_favorite';
    inFlight.add(id);
    const patch: Partial<FeedbackState> = { [field]: !was };
    if (mutualExclusion) {
      patch[opposite] = false;
    }
    opts.set(id, patch);
    try {
      if (was) {
        if (onUnset) await onUnset(id);
      } else {
        if (onSet) await onSet(id);
      }
    } catch {
      // Rollback to the original state.
      opts.set(id, { [field]: was, [opposite]: current[opposite] });
    } finally {
      inFlight.delete(id);
    }
  }

  return {
    toggleFavorite: (id) => toggle(id, 'is_favorite', opts.onLike, opts.onUnlike),
    toggleDislike: (id) => toggle(id, 'is_disliked', opts.onDislike, opts.onUndislike),
    isInFlight: (id) => inFlight.has(id)
  };
}
