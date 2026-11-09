"""
search/explore.py — Explore session state.

Explore is a per-album feedback loop. The album's own photos are the
POSITIVES of a Qdrant `recommend()` call; the photos the user
down-votes during the session are the NEGATIVES. Each round re-issues
the recommend with the accumulated downvotes, so the results drift
away from what the user rejected while staying anchored to the album.

Three pieces of state live here, all scoped to one session:

- `downvotes` — the ids the user rejected. These are the negatives
  for the next round. They are TEMPORARY: they never reach the
  `dislikes` table, because that table feeds the permanent For You
  taste signal and an exploratory veto is not the same thing as
  "never show me this again".
- `seen` — every id served so far. Excluded from later rounds so a
  photo never comes back.
- `round_ids` — the ids of the round currently being served, plus
  `round_scores`, the score each one earned in that round's
  `recommend()` call. The client pages through the id list; the
  server slices it. Re-querying Qdrant with an `offset` instead
  would return duplicate rows at the page boundaries (HNSW depth
  drift), so the round is materialised once and sliced locally.

  The scores are kept alongside the ids because the page's payload
  fetch goes through `retrieve_batch`, which carries no score — the
  round's own score is the only source of the similarity the tile
  displays.

Session state is process-local, like `/random`'s shuffle store and the
For You deck cache: a round is cheap to rebuild (one recommend call),
so cross-process state would buy sticky sessions on a load balancer,
which is the opposite of what an exploratory feed wants.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

# How many photos one Explore round materialises. 240 is 10 screens
# at the default 24-per-screen grid, and it sits at the edge of the
# tight relevance band — measured on a 2,976-photo album, only 126
# photos sit within 0.01 of the top score and 1,561 within 0.02, so
# beyond ~240 the round starts serving photos the model scores
# materially lower.
EXPLORE_ROUND_SIZE = 240

# Session lifetime. Same 30 minutes as the `/random` shuffle store
# and the For You deck cache.
EXPLORE_SESSION_TTL_S = 30 * 60


@dataclass
class ExploreSession:
    """One Explore session against one album.

    `downvotes` and `seen` are sets for membership tests; both are
    ordered lists on the wire (the negatives list goes to Qdrant, and
    its order determines the seed order Qdrant averages), so the
    downvotes are kept in an ordered list instead.
    """

    album_id: int
    downvotes: list[str] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    round_ids: list[str] = field(default_factory=list)
    round_scores: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    ttl_s: float = EXPLORE_SESSION_TTL_S

    def is_alive(self) -> bool:
        """True while the session is within its TTL."""
        return (time.monotonic() - self.created_at) < self.ttl_s

    def add_downvote(self, point_id: str) -> bool:
        """Record a down-vote. Returns True if it was new.

        A down-voted photo is by definition one the user has already
        seen, so the seen-set is NOT consulted here — only the
        down-vote list itself, which keeps the negative list free of
        duplicates.
        """
        if point_id in self.downvotes:
            return False
        self.downvotes.append(point_id)
        return True

    def mark_seen(self, point_ids: list[str]) -> None:
        """Record ids as served, so later rounds skip them."""
        self.seen.update(point_ids)

    def set_round(
        self, ids: list[str], scores: dict[str, float]
    ) -> None:
        """Materialise the round: its ids in order, and their scores.

        The scores come from the round's own `recommend()` call.
        They are kept because the page's payload fetch goes through
        `retrieve_batch`, which returns no score — without this map
        every tile would display a similarity of 0.
        """
        self.round_ids = list(ids)
        self.round_scores = dict(scores)

    def clear_round(self) -> None:
        """Invalidate the round so the next request re-queries Qdrant."""
        self.round_ids = []
        self.round_scores = {}

    def score_for(self, point_id: str) -> float:
        """The score this photo earned in the current round."""
        return self.round_scores.get(point_id, 0.0)


class ExploreSessionStore:
    """Process-local map of session id → Explore session.

    Deliberately NOT shared across processes — see the module
    docstring for why.
    """

    def __init__(self, ttl_s: float = EXPLORE_SESSION_TTL_S):
        self._sessions: dict[str, ExploreSession] = {}
        self._ttl_s = ttl_s

    def get(self, session_id: str | None) -> tuple[str, ExploreSession] | None:
        """Look up a live session by id.

        Returns None if the id is missing or the session has expired
        (in which case the entry is removed).
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

    def put_new(self, album_id: int) -> tuple[str, ExploreSession]:
        """Create a session for `album_id` with a fresh id."""
        new_id = secrets.token_urlsafe(16)
        sess = ExploreSession(album_id=album_id, ttl_s=self._ttl_s)
        self._sessions[new_id] = sess
        return new_id, sess

    def clear(self) -> None:
        """Drop all sessions. Used by tests."""
        self._sessions.clear()

    def __len__(self) -> int:
        return len(self._sessions)


__all__ = [
    "EXPLORE_ROUND_SIZE",
    "EXPLORE_SESSION_TTL_S",
    "ExploreSession",
    "ExploreSessionStore",
]
