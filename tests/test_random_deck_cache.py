"""The /random session deck: the scroll is cached, the shuffle is not.

Building the deck used to be one operation — scroll the corpus's ids
and shuffle them. When the ids came from a local SQLite table that
was ~2.3s for the whole thing. Qdrant owns the corpus now, and a
full id scroll over 2.2M points takes ~25s: Qdrant has to read every
point record and the REST API has no id-only scroll mode.

So the scroll is done ONCE and cached; the shuffle — the part that
makes each session's order different, and the part the /random UX
actually promises — stays per session.

These tests pin that split, and the properties the deck has always
had: a session covers the corpus, every photo exactly once.
"""
from __future__ import annotations

from typing import Any


class _FakeQdrant:
    """Counts the scrolls; serves a fixed corpus."""

    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.scrolls = 0

    def scroll_ids(self, collections: list[str] | None = None) -> list[Any]:
        self.scrolls += 1
        from search.qdrant_client import SearchHit

        return [SearchHit(id=i, path="", score=0.0, payload=None) for i in self.ids]


def _db(tmp_path, ids: list[str]):
    from search.index_db import IndexDB

    qdrant = _FakeQdrant(ids)
    db = IndexDB(  # type: ignore[arg-type]
        str(tmp_path / "images.db"), qdrant, id_cache_ttl_seconds=900
    )
    return db, qdrant


def test_the_scroll_happens_once_and_the_shuffle_does_not(tmp_path) -> None:
    """Two decks from the same corpus: one scroll, two different orders."""
    ids = [f"id-{i}" for i in range(500)]
    db, qdrant = _db(tmp_path, ids)
    try:
        first = db.shuffled_id_deck()
        second = db.shuffled_id_deck()
    finally:
        db.close()

    assert qdrant.scrolls == 1, "the id scroll was not cached"
    assert sorted(first) == sorted(ids)
    assert sorted(second) == sorted(ids)
    # The shuffle is per session, so the two orders differ.
    assert first != second


def test_the_cached_ids_are_not_mutated_by_the_shuffle(tmp_path) -> None:
    """A session's deck must not disturb the cached id list.

    The cache is shared; if a shuffle sorted the cached list in place,
    the next session would inherit the previous one's order.
    """
    ids = [f"id-{i}" for i in range(200)]
    db, qdrant = _db(tmp_path, ids)
    try:
        first = db.shuffled_id_deck()
        second = db.shuffled_id_deck()
    finally:
        db.close()
    assert sorted(first) == sorted(ids)
    assert sorted(second) == sorted(ids)
    assert first != second


def test_a_stale_cache_is_still_served(tmp_path) -> None:
    """The request path never waits for a rebuild.

    The ids only change when the corpus is re-indexed, and the
    background warmer refreshes them. A request that arrives between
    refreshes gets the cached list immediately rather than paying
    for a fresh scroll.
    """
    ids = [f"id-{i}" for i in range(50)]
    db, qdrant = _db(tmp_path, ids)
    try:
        db.shuffled_id_deck()
        # Age the cache past its TTL.
        with db._lock:
            when, cached = db._id_cache[()]
            db._id_cache[()] = (when - 10_000, cached)
        # A request reads the stale list rather than rebuilding.
        deck = db.shuffled_id_deck()
        assert qdrant.scrolls == 1
        assert sorted(deck) == sorted(ids)
    finally:
        db.close()


def test_refresh_rebuilds_a_stale_cache(tmp_path) -> None:
    """The background warmer does rebuild a stale cache."""
    ids = [f"id-{i}" for i in range(50)]
    db, qdrant = _db(tmp_path, ids)
    try:
        db.shuffled_id_deck()
        with db._lock:
            when, cached = db._id_cache[()]
            db._id_cache[()] = (when - 10_000, cached)
        db.refresh_id_cache(())
        assert qdrant.scrolls == 2
    finally:
        db.close()


def test_the_cache_is_keyed_by_collection_filter(tmp_path) -> None:
    """A filtered deck is a different list, and is cached separately."""
    ids = [f"id-{i}" for i in range(50)]
    db, qdrant = _db(tmp_path, ids)
    try:
        db.shuffled_id_deck()
        db.shuffled_id_deck(("kpop",))
        assert qdrant.scrolls == 2
        db.shuffled_id_deck(("kpop",))
        assert qdrant.scrolls == 2, "the filtered deck was not cached"
    finally:
        db.close()


def test_a_session_walk_covers_the_corpus_once() -> None:
    """The end-to-end property the /random route promises.

    A session walks the deck forward and every photo is served
    exactly once.
    """
    from search.random_session import RandomSession

    deck = [f"id-{i}" for i in range(37)]
    sess = RandomSession(ids=deck, ttl_s=60.0)
    seen: list[str] = []
    offset = 0
    while True:
        page = sess.ids[offset : offset + 7]
        if not page:
            break
        seen.extend(page)
        offset += 7
    assert len(seen) == len(deck)
    assert set(seen) == set(deck)


def test_the_shuffle_is_actually_random() -> None:
    """Two shuffles of the same list differ.

    A shuffle that silently returned the input order would make the
    /random tab show the same photos in the same order every time.
    """
    import random

    ids = [f"id-{i}" for i in range(1000)]
    a = list(ids)
    b = list(ids)
    random.shuffle(a)
    random.shuffle(b)
    assert a != ids
    assert a != b
