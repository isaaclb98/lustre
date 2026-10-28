"""
tests/test_routers_random.py — random router contract.

Pins the random router's contract: factory returns an APIRouter
with the documented endpoint, response shape, error handling.

The router's flow:
  1. Look up or create a session (shuffled deck) in the store.
  2. Slice [offset, offset+limit) from the deck.
  3. Resolve deck ids to full rows via index_db.rows_by_ids.
  4. Return SearchResponse with session_id, session_total,
     offset, limit, has_more.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build(index_db, cfg):
    from search.routers.random import build_random_router
    router = build_random_router(index_db=index_db, cfg=cfg)
    app = FastAPI()
    app.include_router(router)
    return app


def _fake_cfg():
    c = MagicMock()
    c.top_k_default = 35
    c.default_view = "grid"
    c.web_ui_url = "http://localhost:5173"
    return c


def _row(id: str, **kwargs):
    base = {
        "id": id,
        "path": f"/photos/{id}.jpg",
        "shard": "",
        "collection": "",
        "mtime": None,
        "size": None,
        "indexed_at": None,
        "is_favorite": 0,
        "favorited_at": None,
        "is_disliked": 0,
        "width": None,
        "height": None,
        "blurhash": None,
    }
    base.update(kwargs)
    return base


def _simple_deck(ids: list[str]):
    """A deck that serves `ids` in order, one page at a time."""
    from search.random_session import CursorDeck

    state = {"pos": 0}

    def fetch(cursor, limit):
        out = ids[state["pos"] : state["pos"] + limit]
        state["pos"] += len(out)
        return out, ("cursor", state["pos"])

    return CursorDeck(fetch, start="start", total=len(ids))


def _mock_index_db(deck_ids: list[str], rows_by_id: dict[str, dict] | None = None):
    """Mock IndexDB that hands out a cursor deck of `deck_ids` and
    looks up rows from rows_by_ids."""
    from search.random_session import CursorDeck

    index_db = MagicMock()
    index_db.count_images.return_value = len(deck_ids)

    def make_deck(collections=()):
        """A deck over `deck_ids` that serves them in order."""
        state = {"pos": 0}

        def fetch(cursor, limit):
            # The cursor is an opaque position; the fake ignores it
            # and serves the next slice.
            ids = deck_ids[state["pos"] : state["pos"] + limit]
            state["pos"] += len(ids)
            return ids, ("cursor", state["pos"])

        return CursorDeck(fetch, start="start", total=len(deck_ids))

    index_db.new_cursor_deck.side_effect = make_deck
    if rows_by_id is None:
        rows_by_id = {i: _row(i) for i in deck_ids}
    index_db.rows_by_ids.side_effect = lambda ids: [rows_by_id[i] for i in ids if i in rows_by_id]
    return index_db


# ----- Response shape -----


def test_random_returns_results_shape():
    """Response has session_id, session_total, offset, limit, has_more,
    and per-result fields the grid renders."""
    index_db = _mock_index_db(
        ["abc", "def"],
        {
            "abc": _row("abc", is_favorite=0, width=800, height=600,
                       blurhash="LKO2?U%2Tw=w]~RBVZRi};RPxuwH"),
            "def": _row("def", is_favorite=1, width=1024, height=768,
                       blurhash=None),
        },
    )
    app = _build(index_db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/random?limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["limit"] == 2
    assert data["has_more"] is False  # we asked for all 2
    assert len(data["results"]) == 2
    assert data["session_id"]
    assert data["session_total"] == 2
    assert data["offset"] == 0
    r0 = data["results"][0]
    assert r0["id"] == "abc"
    assert r0["is_favorite"] is False
    assert r0["is_disliked"] is False
    assert r0["width"] == 800
    r1 = data["results"][1]
    assert r1["id"] == "def"
    assert r1["is_favorite"] is True
    assert r1["blurhash"] is None


def test_random_marks_has_more_false_when_underfilled():
    """When the requested page fills exactly what's left of the deck,
    has_more is False."""
    # Deck of 1, asking for 5 → 1 result, has_more False.
    index_db = _mock_index_db(["only"])
    app = _build(index_db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/random?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["has_more"] is False
    assert data["session_total"] == 1


def test_random_marks_has_more_true_when_more_remains():
    """Deck of 10, asking for 2 → has_more True."""
    deck = [f"id-{i}" for i in range(10)]
    index_db = _mock_index_db(deck)
    app = _build(index_db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/random?limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_more"] is True
    assert data["session_total"] == 10


def test_random_session_walks_forward():
    """Same session_id + offset returns the next chunk of the deck."""
    deck = [f"id-{i}" for i in range(5)]
    index_db = _mock_index_db(deck)
    app = _build(index_db, _fake_cfg())
    with TestClient(app) as client:
        r1 = client.get("/api/random?limit=2")
        sid = r1.json()["session_id"]
        r2 = client.get(f"/api/random?session={sid}&offset=2&limit=2")
    assert r2.status_code == 200
    data = r2.json()
    # The second batch should be ids 2 and 3 from the deck.
    assert [r["id"] for r in data["results"]] == ["id-2", "id-3"]
    assert data["offset"] == 2


def test_random_session_offset_past_end_clamps():
    """Asking for offset > deck length returns empty results, clamped offset."""
    deck = [f"id-{i}" for i in range(3)]
    index_db = _mock_index_db(deck)
    app = _build(index_db, _fake_cfg())
    with TestClient(app) as client:
        r1 = client.get("/api/random?limit=3")
        sid = r1.json()["session_id"]
        r2 = client.get(f"/api/random?session={sid}&offset=100&limit=5")
    data = r2.json()
    assert data["results"] == []
    assert data["has_more"] is False
    assert data["offset"] == 3  # clamped to deck length


def test_random_limit_above_max_returns_400():
    index_db = MagicMock()
    app = _build(index_db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/random?limit=99999")
    assert resp.status_code == 400


def test_random_negative_offset_returns_400():
    index_db = MagicMock()
    app = _build(index_db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/random?offset=-1")
    assert resp.status_code == 400


def test_random_returns_500_on_internal_error():
    """When the deck cannot be created, the route returns a 500."""
    index_db = MagicMock()
    index_db.new_cursor_deck.side_effect = RuntimeError("db locked")
    app = _build(index_db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/random?limit=5")
    assert resp.status_code == 500


def test_random_default_view_is_grid():
    index_db = _mock_index_db(["a"])
    cfg = _fake_cfg()
    cfg.default_view = "grid"
    app = _build(index_db, cfg)
    with TestClient(app) as client:
        resp = client.get("/api/random?limit=1")
    assert resp.json()["view"] == "grid"


def test_random_invalid_view_falls_back_to_grid():
    """Unknown view param is coerced to 'grid'."""
    index_db = _mock_index_db(["a"])
    app = _build(index_db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/random?limit=1&view=bogus")
    assert resp.json()["view"] == "grid"


def test_random_collection_filter_restricts_deck():
    """The collection filter is passed through to the deck."""
    index_db = MagicMock()
    index_db.new_cursor_deck.return_value = _simple_deck(["a", "b"])
    index_db.rows_by_ids.return_value = [_row("a"), _row("b")]
    app = _build(index_db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/random?limit=10&collections=kpop")
    assert resp.status_code == 200
    # The deck was filtered to the kpop collection.
    args, _ = index_db.new_cursor_deck.call_args
    assert args[0] == ("kpop",)


def test_random_results_in_deck_order():
    """Results come back in the same order as the deck (not the
    arbitrary order from the IN(...) query)."""
    deck = ["c", "a", "b", "e", "d"]  # shuffled order
    index_db = _mock_index_db(deck)
    app = _build(index_db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/random?limit=10")
    data = resp.json()
    # Order matches the deck, not a sorted or random IN-query order.
    assert [r["id"] for r in data["results"]] == deck