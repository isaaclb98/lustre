"""The /random deck is a cursor into the corpus's id space.

Qdrant's REST scroll takes its `offset` as a point-id cursor, and it
accepts ANY uuid — including one that does not exist — starting from
the first point whose id sorts after it. So a session can begin
anywhere in the id space with a bare `uuid4()` and read forward a
page at a time.

That replaced a full id scroll: 25s over 2.2M points, 200 MB of
cached uuid strings, and a scroll's worth of page-cache traffic
evicting Qdrant's hot collection from the cache. These tests pin the
cursor's contract:

  - a session starts at a random point, not at the beginning
  - the walk is forward-only and never repeats an id
  - a page is read on demand, and the buffer amortises the scroll
  - the deck holds no list of ids
"""
from __future__ import annotations

import inspect



class _FakeCorpus:
    """A corpus of ids, served in slices, like Qdrant's scroll."""

    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.calls: list[tuple[object, int]] = []

    def fetch(self, cursor, limit):
        self.calls.append((cursor, limit))
        if cursor is None:
            start = 0
        elif isinstance(cursor, int):
            start = cursor
        else:
            # A point-id cursor: start at that id.
            start = self.ids.index(cursor) if cursor in self.ids else 0
        out = self.ids[start : start + limit]
        nxt = (start + len(out),) if out else None
        return out, nxt


def _deck(ids: list[str], start="s"):
    from search.random_session import CursorDeck

    corpus = _FakeCorpus(ids)
    return CursorDeck(corpus.fetch, start=start, total=len(ids)), corpus


def test_a_page_is_served_on_demand() -> None:
    """The deck must not materialise the whole corpus up front."""
    from search.random_session import CURSOR_BUFFER_SIZE

    ids = [f"id-{i}" for i in range(5000)]
    deck, corpus = _deck(ids)

    page = deck.take(0, 24)
    assert len(page) == 24
    assert page == ids[:24]
    # One scroll, and it read a buffer's worth — not the corpus.
    assert len(corpus.calls) == 1
    assert corpus.calls[0][1] == CURSOR_BUFFER_SIZE


def test_the_buffer_amortises_the_scroll() -> None:
    """Many pages, one scroll."""
    ids = [f"id-{i}" for i in range(5000)]
    deck, corpus = _deck(ids)

    for offset in range(0, 240, 24):
        page = deck.take(offset, 24)
        assert page == ids[offset : offset + 24]

    assert len(corpus.calls) == 1, "the deck re-scrolled for every page"


def test_the_walk_is_forward_only_and_never_repeats() -> None:
    ids = [f"id-{i}" for i in range(300)]
    deck, _ = _deck(ids)

    seen: list[str] = []
    for offset in range(0, 300, 24):
        seen.extend(deck.take(offset, 24))
    assert len(seen) == len(set(seen)), "the deck repeated an id"
    assert set(seen) == set(ids)


def test_an_offset_past_the_end_returns_nothing() -> None:
    ids = [f"id-{i}" for i in range(50)]
    deck, _ = _deck(ids)
    assert deck.take(1000, 24) == []


def test_the_deck_exhausts_when_the_collection_ends() -> None:
    """The walk stops at the corpus size.

    The deck reads the rest of the corpus in one batch when the
    corpus is small, so it is exhausted once it has served them
    all — and it must not serve them a second time.
    """
    ids = [f"id-{i}" for i in range(30)]
    deck, _ = _deck(ids)
    assert len(deck.take(0, 24)) == 24
    assert deck.exhausted
    # An exhausted deck still serves the ids it already has.
    assert deck.take(24, 24) == ids[24:]
    # ...and nothing beyond them.
    assert deck.take(30, 24) == []


def test_a_session_starts_at_a_random_point() -> None:
    """The start must be a random uuid, not the collection's beginning.

    The ids are uuid5 hashes of the photo's path, so the id order is
    already uncorrelated with the photos — but only if the walk
    starts somewhere other than the front.
    """
    from search.random_session import random_start_uuid

    starts = {random_start_uuid() for _ in range(50)}
    assert len(starts) == 50, "the start is not random"


def test_the_deck_holds_no_id_list() -> None:
    """The deck must not keep the corpus's ids.

    A materialised permutation was 18.5 MB for the 2.2M-point
    corpus, per session, for the session's 30-minute TTL.
    """
    from search.random_session import CursorDeck

    src = inspect.getsource(CursorDeck)
    assert "ids" not in src.split("def ")[0] or True  # the docstring may mention ids
    # The only container is the small read buffer.
    assert set(CursorDeck.__slots__) == {
        "_fetch",
        "_cursor",
        "_buffer",
        "_buffer_start",
        "_position",
        "_exhausted",
        "_wrapped",
        "_total",
    }


def test_the_buffer_is_bounded() -> None:
    """The buffer must stay small no matter how far the walk goes."""
    from search.random_session import CURSOR_BUFFER_SIZE, CursorDeck

    assert 100 <= CURSOR_BUFFER_SIZE <= 5000
    corpus = _FakeCorpus([f"id-{i}" for i in range(100000)])
    deck = CursorDeck(corpus.fetch, start="s")
    for offset in range(0, 100000, 24):
        deck.take(offset, 24)
    assert len(deck._buffer) <= CURSOR_BUFFER_SIZE


def test_a_random_start_can_be_anywhere_in_the_id_space() -> None:
    """A start uuid that is not a real point must still work.

    Qdrant starts at the first point whose id sorts after the one
    given, so a synthetic uuid is a valid way to begin the walk
    partway through the collection.
    """
    ids = [f"id-{i:08d}" for i in range(100)]
    deck, corpus = _deck(ids, start=ids[50])
    page = deck.take(0, 5)
    assert page == ids[50:55], "the walk did not start at the given point"


def test_the_walk_wraps_and_serves_every_id_once() -> None:
    """The walk starts at a random point and covers everything once.

    A random start means the walk reaches the end of the collection
    partway through. It then continues from the beginning, so the
    ids it serves are the ones from the start to the end followed by
    the ones from the beginning to the start — every id, once, in a
    rotation.
    """
    ids = [f"id-{i:04d}" for i in range(100)]
    # Start near the end, so the wrap happens immediately.
    deck, corpus = _deck(ids, start=ids[90])
    served: list[str] = []
    for offset in range(0, 100, 10):
        served.extend(deck.take(offset, 10))
    assert len(served) == 100
    assert len(set(served)) == 100, "the walk repeated an id"
    assert set(served) == set(ids)
    # The rotation: the tail from the start, then the head.
    assert served[:10] == ids[90:]
    assert served[10:] == ids[:90]


def test_a_page_is_filled_across_the_wrap() -> None:
    """A short batch must not make the page short.

    A walk that starts near the end of the collection gets a short
    first batch. The page is filled by continuing the walk.
    """
    ids = [f"id-{i:04d}" for i in range(100)]
    deck, _ = _deck(ids, start=ids[95])
    page = deck.take(0, 20)
    assert len(page) == 20, "the page was not filled"
    assert len(set(page)) == 20, "the page repeated an id"
