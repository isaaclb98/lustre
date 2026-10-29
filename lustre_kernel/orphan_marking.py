"""
lustre_kernel.orphan_marking

Marking points as orphans instead of deleting them.

Two things can make an indexed point unwanted:

1. **The file is gone.** The photo was deleted from the NAS. The
   indexer's prune used to delete these points outright.

2. **The source was retired.** The directory was removed from the
   configured source list (`INDEXER_SOURCES`), so it is no longer
   part of the library — but the files may still be sitting on disk.
   Nothing in the filesystem distinguishes this case from a live
   source; only the configured source list does.

Both cases mean the same thing for the index: this point should not
be served any more.

**Marking is not deleting.** A marked point keeps its embedding and
stays in the collection, but it is excluded from every read path, so
it stops being served immediately. The embedding is only removed when
an operator confirms the purge. That is the two-step:

    mark  ->  verify the count  ->  confirm  ->  delete

The marker is the `orphan_since` payload field — a unix timestamp
recorded when the point was marked. A float payload index on it makes
"is this an orphan?" a cheap server-side range query, and makes
"everything that is not an orphan" a `must_not` on the same range.
"""

from __future__ import annotations

import time
from typing import Any

from lustre_kernel.payload_schema import FIELD_ORPHAN_SINCE

__all__ = [
    "FIELD_ORPHAN_SINCE",
    "is_orphan_payload",
    "mark_orphans",
    "orphan_filter",
    "orphan_since_filter",
]


def is_orphan_payload(payload: dict[str, Any] | None) -> bool:
    """True if the payload carries an orphan marker.

    The marker is a timestamp. A `None` value means "not an orphan" —
    Qdrant keeps the key with a null value when it is unset that way,
    and a null is not a number, so it does not match the orphan
    range.
    """
    if not payload:
        return False
    value = payload.get(FIELD_ORPHAN_SINCE)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def orphan_filter() -> Any:
    """A Qdrant filter selecting ONLY the marked (orphan) points."""
    from qdrant_client.http import models as qmodels

    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key=FIELD_ORPHAN_SINCE,
                range=qmodels.Range(gte=0),
            ),
        ],
    )


def orphan_since_filter() -> Any:
    """A Qdrant filter EXCLUDING every marked (orphan) point.

    This is what every read path adds: `must_not` on the marker
    range. Points that never carried the marker, and points whose
    marker was explicitly cleared to null, are both unaffected —
    neither matches the range.
    """
    from qdrant_client.http import models as qmodels

    return qmodels.Filter(
        must_not=[
            qmodels.FieldCondition(
                key=FIELD_ORPHAN_SINCE,
                range=qmodels.Range(gte=0),
            ),
        ],
    )


def mark_orphans(
    client: Any,
    collection: str,
    point_ids: list[str],
    *,
    when: float | None = None,
) -> int:
    """Mark the given points as orphans.

    Marks are idempotent: a point already carrying a marker keeps its
    ORIGINAL timestamp, so "how long has this been an orphan?" stays
    meaningful across repeated index runs.

    Returns the number of points marked (i.e. newly marked).

    `set_payload` merges into the existing payload rather than
    replacing it, so the point keeps its `path`, `collection` and
    everything else — which matters because those are what the
    verification endpoint shows the operator and what the purge needs
    to clean up the matching user-state rows.
    """
    if not point_ids:
        return 0


    timestamp = time.time() if when is None else when

    # Which of these are already marked? Fetching the payload of each
    # candidate is one round trip per point, so batch it and only
    # fetch the ones we might actually need to change.
    already: set[str] = set()
    for i in range(0, len(point_ids), 256):
        chunk = point_ids[i:i + 256]
        try:
            existing = client.retrieve(
                collection_name=collection,
                ids=chunk,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:  # noqa: BLE001 — treat an unreadable point as unmarked
            existing = []
        for p in existing:
            if is_orphan_payload(p.payload or {}):
                already.add(str(p.id))

    to_mark = [pid for pid in point_ids if pid not in already]
    if not to_mark:
        return 0

    for i in range(0, len(to_mark), 256):
        chunk = to_mark[i:i + 256]
        try:
            client.set_payload(
                collection_name=collection,
                payload={FIELD_ORPHAN_SINCE: timestamp},
                points=chunk,
            )
        except Exception as exc:
            # A failed mark leaves the point visible. That is the safe
            # direction: the point keeps being served (stale) rather
            # than disappearing from a half-applied marking.
            raise RuntimeError(
                f"failed to mark {len(chunk)} point(s) as orphans: {exc}"
            ) from exc
    return len(to_mark)
