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

import secrets
import time
import uuid
from collections.abc import Callable
from typing import Any

# Default session lifetime. After this many seconds the shuffled
# deck is discarded and the next call gets a fresh shuffle.
RANDOM_SESSION_TTL_S = 30 * 60


# How many ids one scroll reads. The scroll's cost is a fixed ~25ms
# seek plus ~0.01ms per id, so a batch amortises the seek over many
# pages: 1,000 ids is one ~10ms scroll that serves 41 pages of 24.
CURSOR_BUFFER_SIZE = 1000


class CursorDeck:
    """A per-session walk over the corpus's point ids, read on demand.

    The deck is a cursor, not a list. Qdrant's REST scroll takes its
    `offset` as a point-id cursor, and it accepts ANY uuid — even
    one that does not exist — starting from the first point whose
    id sorts after it. So a session can begin anywhere in the id
    space with a bare `uuid4()`, and read forward from there.

    The ids are uuid5 hashes of the photo's path, so the id order is
    already uncorrelated with the photos: walking it from a random
    point yields a random-looking sample of the corpus.

    This replaces a full id scroll — 25s over 2.2M points, 200 MB
    of cached uuid strings, and a scroll's worth of page-cache
    traffic evicting Qdrant's hot collection from the cache. A
    session here costs one uuid and a small buffer.

    The session covers the ids from its start to the end of the
    collection — a random half of the corpus on average, and never
    the same half twice in a row. Within a session no id repeats.
    """

    __slots__ = (
        "_buffer",
        "_buffer_start",
        "_cursor",
        "_exhausted",
        "_fetch",
        "_position",
        "_total",
        "_wrapped",
    )

    def __init__(
        self,
        fetch: Callable[[Any, int], tuple[list[str], Any]],
        start: str | None = None,
        total: int = 0,
    ) -> None:
        """`fetch(cursor, limit)` returns `(ids, next_cursor)`.

        `start` is the uuid the walk begins from. None means the
        beginning of the collection.

        `total` is how many ids the walk will produce — the size of
        the corpus, or of the filtered subset. The walk reads to the
        end of the collection and then continues from the beginning,
        so it covers every id exactly once whatever point it started
        at.
        """
        self._fetch = fetch
        self._cursor: Any = start
        self._buffer: list[str] = []
        self._buffer_start = 0
        self._position = 0
        self._exhausted = False
        self._wrapped = False
        self._total = total

    @property
    def total(self) -> int:
        """How many ids the walk will produce."""
        return self._total

    @property
    def position(self) -> int:
        """How many ids this deck has produced."""
        return self._position

    @property
    def exhausted(self) -> bool:
        """True once the walk has produced every id there is."""
        return self._exhausted or self._position >= self._total

    def take(self, offset: int, count: int) -> list[str]:
        """The `count` ids at absolute `offset` in the walk.

        The offset is absolute so the client's paging contract is
        unchanged: the walk's order is the order the ids are served
        in, whether they come from the tail of the collection or,
        after the wrap, from its head.

        A page is filled even when the walk starts near the end of
        the collection: the walk continues (wrapping if it has to)
        until the page is full or the corpus is exhausted.
        """
        if count <= 0:
            return []
        out: list[str] = []
        pos = offset
        while len(out) < count:
            if not (
                self._buffer_start
                <= pos
                < self._buffer_start + len(self._buffer)
            ):
                if not self._advance():
                    break
                continue
            i = pos - self._buffer_start
            chunk = self._buffer[i : i + (count - len(out))]
            out.extend(chunk)
            pos += len(chunk)
        return out

    def _advance(self) -> bool:
        """Read the next batch of ids from the cursor."""
        if self._exhausted:
            return False
        # The walk produces exactly `total` ids — the corpus (or the
        # filtered subset) holds that many. Stopping there is what
        # keeps the wrap from serving the same ids twice.
        remaining = self._total - self._position
        if remaining <= 0:
            self._exhausted = True
            return False
        limit = min(CURSOR_BUFFER_SIZE, remaining)
        ids, next_cursor = self._fetch(self._cursor, limit)
        if not ids and not self._wrapped:
            # The walk has reached the end of the collection. It
            # started at a random point, so it has covered the ids
            # from there to the end; continuing from the beginning
            # covers the rest, and every id exactly once.
            self._wrapped = True
            self._cursor = None
            ids, next_cursor = self._fetch(None, limit)
        if not ids:
            self._exhausted = True
            self._cursor = None
            return False
        # The walk must not produce more ids than the corpus holds.
        ids = ids[:remaining]
        self._buffer = ids
        self._buffer_start = self._position
        self._position += len(ids)
        self._cursor = next_cursor
        return True


def random_start_uuid() -> str:
    """A random point in the corpus's id space to start a walk from.

    The ids are uuid5 hashes, so a random uuid lands at a random
    place among them. `uuid4` is used rather than `uuid5` of
    something: only the ordering matters, and a v4 uuid is the
    cheapest random one.
    """
    return str(uuid.uuid4())


class RandomSession:
    """One shuffled walk through the library.

    Named without a leading underscore because it's part of the
    module's public surface — `search/routers/random.py` and
    tests construct and inspect these directly.
    """

    __slots__ = ("created_at", "deck", "ttl_s")

    def __init__(self, deck: CursorDeck, ttl_s: float):
        self.deck = deck
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

    def put_new(self, deck: CursorDeck) -> tuple[str, RandomSession]:
        """Materialize a fresh session with a new id and the given
        deck. The caller is responsible for having built the deck
        (possibly in a thread).
        """
        new_id = secrets.token_urlsafe(16)
        sess = RandomSession(deck, self._ttl_s)
        self._sessions[new_id] = sess
        return new_id, sess

    def clear(self) -> None:
        """Drop all sessions. Used by tests."""
        self._sessions.clear()

    def __len__(self) -> int:
        return len(self._sessions)


__all__ = [
    "CURSOR_BUFFER_SIZE",
    "RANDOM_SESSION_TTL_S",
    "CursorDeck",
    "RandomSession",
    "RandomSessionStore",
    "random_start_uuid",
]
