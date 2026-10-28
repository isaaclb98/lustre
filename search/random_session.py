"""
search/random_session.py — per-process store of /api/random shuffled decks.

Extracted from `search/routers/random.py` in round-93 PR 2
(`refactor/plan-002-module-layout`). The split keeps the router file
HTTP-only and puts the stateful session bookkeeping in a module
named for what it stores.

The store is deliberately process-local: a /random shuffle is cheap
to rebuild (an O(N) Qdrant scroll), so the operational cost of
sticking it in a shared store would outweigh the benefit. A
load balancer sticky-session becomes necessary if you ever share
this across processes — and that's the opposite of what /random
wants, because the user expects a fresh shuffle per call when
they ask for one.

Two non-obvious shapes this module owns:

1. **Splitting lookup from creation.** `get(session_id)` returns
   the live session if any (and clears expired ones); `put_new(ids)`
   materializes a new one. Splitting these lets the caller run
   the Qdrant materialize off the event loop
   (`asyncio.to_thread(...)`) without weaving sync/async into the
   store.

2. **TTL by `time.monotonic`.** The session expires after
   `RANDOM_SESSION_TTL_S` of wall-clock-less time. `monotonic` is
   the right clock for this — wall-clock jumps (NTP, daylight
   savings) can't accidentally drop a session or extend its life.
"""

from __future__ import annotations

import math
import secrets
import time
from collections.abc import Sequence
from typing import Any

# Default session lifetime. After this many seconds the shuffled
# deck is discarded and the next call gets a fresh shuffle.
RANDOM_SESSION_TTL_S = 30 * 60


class StridedDeck(Sequence[str]):
    """A random order over an id list, without materialising it.

    A random start plus a stride coprime with the list length is a
    permutation of the range: every index is visited exactly once, so
    the deck keeps the property the /random route promises — every
    photo served exactly once per session, forward only, no repeats.

    Materialising that permutation cost a pointer array per session:
    18.5 MB for the 2.2M-point production corpus, held for the
    session's 30-minute TTL, with no cap on the number of sessions.
    This view costs two integers.

    The underlying id list is in Qdrant's scroll order, which is
    point-id order — the ids are uuid5 hashes of the path, so the
    list is already uncorrelated with the photos. Any stride over it
    is therefore an uncorrelated order, which is why a stride is
    enough and a full shuffle is not needed.
    """

    __slots__ = ("_ids", "_n", "_start", "_stride")

    def __init__(self, ids: Sequence[str], start: int, stride: int) -> None:
        self._ids = ids
        self._start = start
        self._stride = stride
        self._n = len(ids)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            lo, hi, _ = index.indices(self._n)
            if hi <= lo:
                return []
            n, start, stride = self._n, self._start, self._stride
            pos = (start + stride * lo) % n
            ids = self._ids
            return [ids[(pos + stride * j) % n] for j in range(hi - lo)]
        i = index if index >= 0 else index + self._n
        if i < 0 or i >= self._n:
            raise IndexError(index)
        return self._ids[(self._start + self._stride * i) % self._n]


def random_stride(length: int, rng: Any) -> int:
    """A random stride coprime with `length`.

    Coprimality is what makes the stride a permutation: with
    `gcd(stride, length) == 1` the walk visits every index exactly
    once before it returns to the start.
    """
    if length <= 2:
        return 1
    while True:
        s = int(rng.integers(1, length))
        if math.gcd(s, length) == 1:
            return s


class RandomSession:
    """One shuffled walk through the library.

    Named without a leading underscore because it's part of the
    module's public surface — `search/routers/random.py` and
    tests construct and inspect these directly.
    """

    __slots__ = ("created_at", "ids", "ttl_s")

    def __init__(self, ids: Sequence[str], ttl_s: float):
        self.ids = ids
        self.created_at = time.monotonic()
        self.ttl_s = ttl_s

    def is_alive(self) -> bool:
        return (time.monotonic() - self.created_at) < self.ttl_s


class RandomSessionStore:
    """Process-local map of session_id → shuffled deck.

    Deliberately NOT shared across processes — see module
    docstring for why.
    """

    def __init__(self, ttl_s: float = RANDOM_SESSION_TTL_S):
        self._sessions: dict[str, RandomSession] = {}
        self._ttl_s = ttl_s

    def get(
        self,
        session_id: str | None,
    ) -> tuple[str, RandomSession] | None:
        """Look up a live session by id. Returns None if the id is
        missing, or if the session has expired (in which case the
        entry is removed).
        """
        if not session_id:
            return None
        sess = self._sessions.get(session_id)
        if sess is None:
            return None
        if not sess.is_alive():
            del self._sessions[session_id]
            return None
        return session_id, sess

    def put_new(self, ids: list[str]) -> tuple[str, RandomSession]:
        """Materialize a fresh session with a new id and the given
        shuffled deck. The caller is responsible for having run
        the materialize (possibly in a thread).
        """
        new_id = secrets.token_urlsafe(16)
        sess = RandomSession(ids, self._ttl_s)
        self._sessions[new_id] = sess
        return new_id, sess

    def clear(self) -> None:
        """Drop all sessions. Used by tests."""
        self._sessions.clear()

    def __len__(self) -> int:
        return len(self._sessions)


__all__ = [
    "RANDOM_SESSION_TTL_S",
    "RandomSession",
    "RandomSessionStore",
    "StridedDeck",
    "random_stride",
]
