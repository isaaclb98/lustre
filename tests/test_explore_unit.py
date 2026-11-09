"""Unit tests for the Explore session store.

The store holds the three pieces of per-session state: the
down-votes (which become the negatives of the next round), the
seen-set (which keeps a photo from being served twice), and the
materialised round (which the client pages through).

No Qdrant involved — the store is pure bookkeeping.
"""

from __future__ import annotations

import time

from search.explore import (
    EXPLORE_ROUND_SIZE,
    EXPLORE_SESSION_TTL_S,
    ExploreSession,
    ExploreSessionStore,
)


def test_round_size_is_240():
    assert EXPLORE_ROUND_SIZE == 240


def test_put_new_creates_a_session_with_an_id():
    store = ExploreSessionStore()
    session_id, sess = store.put_new(album_id=7)
    assert session_id
    assert sess.album_id == 7
    assert sess.downvotes == []
    assert sess.seen == set()
    assert sess.round_ids == []


def test_get_returns_the_session():
    store = ExploreSessionStore()
    session_id, _ = store.put_new(album_id=7)
    found = store.get(session_id)
    assert found is not None
    assert found[0] == session_id
    assert found[1].album_id == 7


def test_get_returns_none_for_an_unknown_id():
    store = ExploreSessionStore()
    assert store.get("nope") is None
    assert store.get(None) is None
    assert store.get("") is None


def test_an_expired_session_is_dropped():
    store = ExploreSessionStore(ttl_s=0.0)
    session_id, _ = store.put_new(album_id=7)
    time.sleep(0.001)
    assert store.get(session_id) is None
    # The expired entry is evicted, not just hidden.
    assert len(store) == 0


def test_default_ttl_is_thirty_minutes():
    assert EXPLORE_SESSION_TTL_S == 30 * 60


def test_add_downvote_records_once():
    sess = ExploreSession(album_id=1)
    assert sess.add_downvote("a") is True
    assert sess.add_downvote("a") is False
    assert sess.downvotes == ["a"]


def test_add_downvote_keeps_insertion_order():
    """The negative list order is the seed order Qdrant averages."""
    sess = ExploreSession(album_id=1)
    sess.add_downvote("a")
    sess.add_downvote("b")
    sess.add_downvote("c")
    assert sess.downvotes == ["a", "b", "c"]


def test_a_seen_photo_can_still_be_downvoted():
    """A down-voted photo is by definition one the user has seen.

    The seen-set must not block the vote — otherwise no photo could
    ever be down-voted, since every down-vote targets a photo that
    was already served.
    """
    sess = ExploreSession(album_id=1)
    sess.mark_seen(["a"])
    assert sess.add_downvote("a") is True
    assert sess.downvotes == ["a"]
    # A second down-vote of the same photo is a no-op.
    assert sess.add_downvote("a") is False
    assert sess.downvotes == ["a"]


def test_mark_seen_accumulates():
    sess = ExploreSession(album_id=1)
    sess.mark_seen(["a", "b"])
    sess.mark_seen(["b", "c"])
    assert sess.seen == {"a", "b", "c"}


def test_clear_drops_every_session():
    store = ExploreSessionStore()
    store.put_new(album_id=1)
    store.put_new(album_id=2)
    assert len(store) == 2
    store.clear()
    assert len(store) == 0
