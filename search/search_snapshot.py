"""
search/search_snapshot.py — stable pagination for /api/search.

WHY THIS EXISTS
---------------
Qdrant offset pagination is unsound on HNSW. To return a page at
`offset` the engine must internally rank `offset + limit` candidates,
and the graph-search breadth scales with that depth. Two pages of the
same query are therefore ranked at *different depths* and disagree
near their boundary:

    offset=240 limit=24   ranks at depth 264
    offset=288 limit=24   ranks at depth 312
    -> 20 of 24 ids overlap

Measured on prod (2,052,334 points, Qdrant 1.19.0), a page walk to
offset 480 returned 78 duplicate ids across 504 results — with the
indexer completely idle. This is not collection mutation; it is
depth drift. Each request is individually correct, the pages simply
do not align.

THE FIX
-------
Rank once, at one fixed depth, and slice every page out of that
frozen list. Offset then indexes a stable snapshot instead of a
live HNSW walk, so pages are disjoint by construction.

Depth is grown incrementally rather than fixed up front, because a
single deep fetch is expensive on page 1 (depth 5000 = 179ms warm,
depth 10000 = 473ms) while a band fetch stays cheap and its cost
grows only with the exclusion set:

    excluded 0      ->  9.1ms
    excluded 1000   -> 12.0ms
    excluded 5000   -> 20.8ms
    excluded 10000  -> 33.1ms   (361KB request body)

A band of 500 is amortised over ~20 pages of 24, so even at 10k
deep a page costs ~1.6ms of band fetch plus ~2.2ms to hydrate
payloads. Page 1 stays ~10ms at any collection size.

Exclusion is a server-side `must_not HasIdCondition`, verified to
return zero overlap even with 10000 ids excluded.

DELIBERATELY NOT the diversity cache
------------------------------------
`DiversityResultCache` stores a `DiversityStats` alongside its hits
because it caches an MMR re-ranking. Plain search does no re-ranking,
so reusing that type would mean fabricating stats and recoupling two
independent features. This is a separate, smaller cache holding only
an ordered `(id, score)` list.

WHAT IS AND ISN'T GUARANTEED
----------------------------
Guaranteed: zero overlap between pages of one query.
Approximate: ordering *across* a band boundary. Band 2 is "the top
500 of what remains", not a perfect continuation of one global
ranking — HNSW cannot guarantee that. Given the flat score tail
(rank 2000 = 0.7798, rank 10000 = 0.7481) the reordering is
invisible.

Staleness: a snapshot is frozen for `ttl_seconds`. Photos indexed
mid-session don't appear until the entry expires. That is the price
of stability — a frozen page order and a live collection cannot both
hold. Same trade-off the diversity path already makes.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from search.qdrant_client import SearchHit

logger = logging.getLogger(__name__)


@dataclass
class SearchSnapshot:
    """One query's frozen ranking.

    Stores ids + scores only — not payloads. At the 20k cap that is
    ~800KB per entry (~50MB across 64 entries). Storing payloads
    instead would be ~4MB/entry -> ~256MB, for no benefit: payloads
    are hydrated per page via retrieve_batch, which also keeps
    favourite/dislike flags live instead of frozen.
    """

    ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    # True once a band fetch came back short, i.e. Qdrant has no more
    # candidates for this query. Until then the snapshot can grow.
    exhausted: bool = False
    created_at: float = field(default_factory=time.monotonic)

    # Guards extension so two concurrent page requests for the same
    # query don't fetch the same band twice. Held across a network
    # call (~10-35ms warm, up to ~1s cold at depth) — that is the point:
    # a racer waits for the in-flight band instead of issuing a
    # duplicate one.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # True while a background prefetch owns the next band. Guarded by
    # its own lock so the check-and-set never blocks on `lock` (which
    # may be held across a long fetch).
    prefetch_inflight: bool = False
    _flag_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __len__(self) -> int:
        return len(self.ids)

    def claim_prefetch(self) -> bool:
        """Atomically claim the right to prefetch one band.

        Returns False if a prefetch is already in flight, so at most
        one background band fetch runs per snapshot.
        """
        with self._flag_lock:
            if self.prefetch_inflight:
                return False
            self.prefetch_inflight = True
            return True

    def release_prefetch(self) -> None:
        with self._flag_lock:
            self.prefetch_inflight = False

    def extend(self, hits: list[SearchHit], band_size: int) -> None:
        """Append a band of hits, dropping any id already frozen.

        The exclusion filter should already have removed them
        server-side; the local check is a cheap second line of
        defence so a duplicate can never enter the snapshot even if
        the filter is dropped (as it silently is if the wrong REST
        field name is used — that cost an hour of debugging).
        """
        known = set(self.ids)
        added = 0
        for hit in hits:
            hid = str(hit.id)
            if hid in known:
                continue
            known.add(hid)
            self.ids.append(hid)
            self.scores.append(float(hit.score))
            added += 1
        # A short band means Qdrant ran out of candidates.
        if added < band_size:
            self.exhausted = True

    def page(self, offset: int, limit: int) -> list[tuple[str, float]]:
        """Slice `limit` (id, score) pairs starting at `offset`."""
        # strict=True: ids and scores are appended together in extend(),
        # so equal length is an invariant worth asserting rather than
        # silently truncating on a bug.
        return list(zip(
            self.ids[offset:offset + limit],
            self.scores[offset:offset + limit],
            strict=True,
        ))


class SearchSnapshotCache:
    """TTL + LRU cache of frozen rankings, keyed by query identity.

    Same shape as DiversityResultCache but holds SearchSnapshot and no
    re-ranking metadata. Two independent caches, two independent TTL
    knobs, neither feature's semantics leaking into the other.
    """

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 64):
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._entries: dict[str, SearchSnapshot] = {}
        # Guards dict mutation only — never held across a network call.
        self._lock = threading.Lock()

    def _expired(self, snap: SearchSnapshot) -> bool:
        if self.ttl_seconds == 0:
            return True
        return time.monotonic() - snap.created_at >= self.ttl_seconds

    def get_or_create(self, key: str) -> SearchSnapshot:
        """Return the live snapshot for `key`, creating an empty one.

        Callers must hold the returned snapshot's own lock while
        extending it. Creating under the dict lock keeps two racing
        first-page requests from building separate snapshots.
        """
        with self._lock:
            snap = self._entries.get(key)
            if snap is not None and not self._expired(snap):
                # Refresh LRU position on touch.
                self._entries.pop(key, None)
                self._entries[key] = snap
                return snap
            snap = SearchSnapshot()
            self._entries[key] = snap
            self._evict_locked()
            return snap

    def get(self, key: str) -> SearchSnapshot | None:
        """Peek without creating — used by tests and diagnostics."""
        with self._lock:
            snap = self._entries.get(key)
            if snap is None:
                return None
            if self._expired(snap):
                self._entries.pop(key, None)
                return None
            return snap

    def _evict_locked(self) -> None:
        while len(self._entries) > self.max_entries:
            self._entries.pop(next(iter(self._entries)))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def snapshot_key(
    *,
    collection: str,
    vector: list[float],
    collections: list[str] | None,
    allowed_ids: list[str] | None,
    favorite_ids: set[str] | None,
    band_size: int,
    path_filter: Any | None = None,
) -> str:
    """Hash the inputs that determine a ranking.

    Deliberately excludes `offset` and `limit` — those select a page
    *within* the snapshot, so including them would fragment one
    ranking into one snapshot per page and defeat the purpose.

    `band_size` is included because it affects the ranking: a larger
    band ranks deeper per fetch, so two configurations must not share
    a snapshot.
    """
    vector_digest = hashlib.sha256(
        repr(tuple(round(float(v), 8) for v in vector)).encode("ascii")
    ).hexdigest()[:20]
    return "|".join((
        "snap-v1",
        collection,
        str(band_size),
        vector_digest,
        _digest(collections),
        _digest(allowed_ids),
        _digest(sorted(favorite_ids) if favorite_ids else None),
        _digest_filter(path_filter),
    ))


def _digest_filter(path_filter: Any | None) -> str:
    """Stable digest of a Qdrant filter condition.

    The condition is a Pydantic model, so its canonical JSON form is
    stable across requests that build an equivalent filter. None maps
    to the same sentinel as `_digest(None)`, so a request without a
    filename filter never collides with one that has it.
    """
    if path_filter is None:
        return _digest(None)
    try:
        canonical = path_filter.model_dump_json(exclude_none=True)
    except AttributeError:
        canonical = repr(path_filter)
    return hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest()[:20]


def _digest(values: Iterable[str] | None) -> str:
    """Stable digest of a list/set/None of strings.

    Sorted + UTF-8 with replacement, so equivalent inputs in any
    order hash identically. None maps to a sentinel so "no filter" is
    itself part of the key (otherwise an unfiltered query would
    collide with a filtered one).
    """
    digest = hashlib.sha256()
    if values is None:
        digest.update(b"<none>\0")
    else:
        for value in sorted(str(v) for v in values):
            digest.update(value.encode("utf-8", "replace"))
            digest.update(b"\0")
    return digest.hexdigest()[:20]
