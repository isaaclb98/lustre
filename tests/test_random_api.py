"""Tests for the /random page and /api/random endpoint."""
from __future__ import annotations

import pytest

from indexer import upsert
from indexer.upsert import VECTOR_DIM


# Reuse the search-side fixture but seed the photos directly into
# QDRANT — the collection owns the corpus, so the random page reads
# the deck's ids from there. The `images` table was a derived mirror
# of exactly this and is no longer the source.
@pytest.fixture
def random_app(tmp_path, monkeypatch):
    import uuid

    from search.app import create_app
    from search.config import Config

    cfg = Config(
        qdrant_url="memory://",
        qdrant_collection="images_test_random",
        qdrant_api_key=None,
        model_name="mock",
        model_revision="",
        device="cpu",
        top_k_default=35,
        top_k_max=200,
        query_timeout_ms=2000,
        nas_images_base=str(tmp_path),
        path_prefix="",
        web_ui_url="http://localhost:8000",
        log_level="WARNING",
        index_db_path=str(tmp_path / "images.db"),
        test_mode=True,
    )

    # Lazy liveness check (added in fix/dual-store-cleanup) calls
    # Path(path).exists() on every row the random / album routes
    # return. The fixture seeds paths like /photos/a.jpg directly via
    # SQL — they're not real files on disk, so the liveness check
    # correctly filters every row out. Mock the helper to return True
    # so the seeded rows survive.
    #
    # The fix for the UNC-payload bug (2026-08-15) calls resolve_local
    # before _is_path_alive. The fixture's seeded paths don't match
    # the production prefix, so resolve_local would return None and
    # drop every row. Mock it to a no-op identity so the seeded paths
    # round-trip through the filter.
    from pathlib import Path as _Path

    from search import app as _app_mod
    monkeypatch.setattr(
        _app_mod,
        "resolve_local",
        lambda path, *a, **kw: _Path(path) if path else None,
    )
    monkeypatch.setattr(_app_mod, "_is_path_alive", lambda path: True)

    # Seed QDRANT — the collection owns the corpus, so the random
    # page reads the deck's ids from there. The `images` table was
    # a derived mirror of exactly this and is no longer the source.
    #
    # The ids are short strings ('a', 'b'), not UUIDs, so they are
    # used verbatim — Qdrant accepts any non-empty string id.
    # Qdrant point ids must be UUIDs (or unsigned ints). The tests
    # assert on short ids ('a', 'b', ...), so each is mapped to a
    # deterministic UUID — and the payload's `id` field carries the
    # SAME value, exactly as the indexer writes it in production
    # (the point id and the payload id are the same uuid5 of the
    # path). The mapping is exposed so the assertions can map back.
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    client = QdrantClient(location=":memory:")
    upsert.ensure_collection(client, cfg.qdrant_collection, dim=VECTOR_DIM)

    short_to_uuid = {
        short: str(uuid.uuid5(uuid.NAMESPACE_OID, f"lustre-test-{short}"))
        for short in ("a", "b", "c", "d")
    }
    client.upsert(
        collection_name=cfg.qdrant_collection,
        points=[
            qmodels.PointStruct(
                id=short_to_uuid[short],
                vector=[0.0] * VECTOR_DIM,
                payload={
                    # Same value as the point id, as in production.
                    "id": short_to_uuid[short],
                    "path": path,
                    "collection": collection,
                    "width": width,
                    "height": height,
                    "shard": "",
                    "mtime": 100,
                    "size": 200,
                    "indexed_at": "2026-01-01T00:00:00+00:00",
                    "blurhash": "",
                },
            )
            for short, path, collection, width, height in (
                ("a", "/photos/a.jpg", "kpop", 640, 480),
                ("b", "/photos/b.jpg", "kpop", 800, 600),
                ("c", "/photos/c.jpg", "portrait", 400, 400),
                ("d", "/photos/d.jpg", "portrait", 500, 700),
            )
        ],
        wait=True,
    )

    from search.qdrant_client import QdrantSearch
    qdrant = QdrantSearch(client=client, collection=cfg.qdrant_collection, timeout_ms=2000)

    from fastapi.testclient import TestClient
    app = create_app(cfg=cfg, qdrant=qdrant)
    tc = TestClient(app)
    # The id mapping rides on the TestClient, so the tests can map
    # the API's UUID ids back to the short ids they assert on.
    tc.short_to_uuid = short_to_uuid
    return tc



def _short_ids(tc, ids):
    """Map the API's point ids back to the short ids the tests use.

    The fixture seeds Qdrant with UUID point ids (Qdrant requires
    them) and the payload's `id` field carries the same value, as in
    production. The tests assert on short ids, so this maps back.
    """
    mapping = getattr(tc, "short_to_uuid", {})
    reverse = {v: k for k, v in mapping.items()}
    return {reverse.get(i, i) for i in ids}


def test_api_random_returns_rows(random_app):
    resp = random_app.get("/api/random?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["view"] == "grid"  # DEFAULT_VIEW
    assert data["query"] == ""
    assert data["centroid"] is None
    assert data["limit"] == 10
    # All four seeded rows are present (no collection filter).
    ids = _short_ids(random_app, {r["id"] for r in data["results"]})
    assert ids == {"a", "b", "c", "d"}


def test_api_random_default_limit_is_35(random_app):
    response = random_app.get("/api/random")
    assert response.status_code == 200
    assert response.json()["limit"] == 35




def test_api_random_filters_by_collection(random_app):
    resp = random_app.get("/api/random?limit=10&collections=kpop")
    assert resp.status_code == 200
    ids = _short_ids(random_app, {r["id"] for r in resp.json()["results"]})
    assert ids == {"a", "b"}


def test_api_random_multi_collection_filter(random_app):
    resp = random_app.get(
        "/api/random?limit=10&collections=kpop&collections=portrait"
    )
    assert resp.status_code == 200
    ids = _short_ids(random_app, {r["id"] for r in resp.json()["results"]})
    assert ids == {"a", "b", "c", "d"}


def test_api_random_unknown_collection_returns_empty(random_app):
    resp = random_app.get("/api/random?limit=10&collections=nonexistent")
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_api_random_has_more_true_when_page_filled(random_app):
    """When the request returns exactly `limit` rows, has_more is
    True so the client keeps the sentinel alive and loads more on
    scroll.
    """
    resp = random_app.get("/api/random?limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    assert data["has_more"] is True


def test_api_random_has_more_false_when_collection_exhausted(random_app):
    """When the request returns fewer than `limit` rows, has_more is
    False — the collection (under filter) fits in this page, no
    point keeping the sentinel around.
    """
    # kpop has 2 seeded rows; limit=10 overshoots.
    resp = random_app.get("/api/random?limit=10&collections=kpop")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    assert data["has_more"] is False


def test_api_random_view_feed_echoed(random_app):
    resp = random_app.get("/api/random?limit=10&view=feed")
    assert resp.status_code == 200
    assert resp.json()["view"] == "feed"


def test_api_random_view_invalid_falls_back_to_grid(random_app):
    resp = random_app.get("/api/random?limit=10&view=blah")
    assert resp.status_code == 200
    assert resp.json()["view"] == "grid"


def test_api_random_limit_out_of_range(random_app):
    resp = random_app.get("/api/random?limit=0")
    assert resp.status_code == 400
    resp = random_app.get("/api/random?limit=99999")
    assert resp.status_code == 400


def test_api_random_results_have_basic_fields(random_app):
    """Each random result carries the fields the grid renderer reads:
    id, path, url, score, score_str, is_favorite.
    """
    resp = random_app.get("/api/random?limit=10")
    assert resp.status_code == 200
    for r in resp.json()["results"]:
        assert r["id"]
        assert r["path"]
        assert r["url"]
        assert r["score"] == 0.0
        assert r["score_str"] == ""


# ---------------------------------------------------------------------------
# Session cursor behavior — the new /random shape that lets the client
# walk through the entire library in random order without duplicates.
# ---------------------------------------------------------------------------


def test_session_first_call_returns_session_id_and_total(random_app):
    """First call materializes a shuffled deck and returns its id."""
    resp = random_app.get("/api/random?limit=2")
    data = resp.json()
    assert "session_id" in data
    assert data["session_id"]
    # token_urlsafe(16) produces ~22 chars of base64-ish text.
    assert len(data["session_id"]) >= 16
    # The fixture seeds 4 photos, so session_total is 4.
    assert data["session_total"] == 4
    assert data["offset"] == 0
    assert data["has_more"] is True


def test_session_walks_forward_no_duplicates(random_app):
    """A session walks through the deck in order. Each offset returns
    the next chunk; no photo appears twice within a session.

    The whole point of the session-cursor shape: you can scroll
    through 100% of the library without seeing duplicates.
    """
    r1 = random_app.get("/api/random?limit=2")
    sid = r1.json()["session_id"]
    seen = _short_ids(random_app, {r["id"] for r in r1.json()["results"]})

    # Walk the rest of the deck in chunks of 2.
    for offset in (2, 4):  # we have 4 photos total
        r = random_app.get(
            f"/api/random?session={sid}&offset={offset}&limit=2"
        )
        for photo in _short_ids(random_app, {p["id"] for p in r.json()["results"]}):
            assert photo not in seen, (
                f"Duplicate {photo} at offset {offset}"
            )
            seen.add(photo)

    # We saw all 4 distinct photos.
    assert seen == {"a", "b", "c", "d"}


def test_session_full_walk_returns_everything_once(random_app):
    """One session with a large limit covers the whole library,
    no duplicates, has_more=False at the end.
    """
    r = random_app.get("/api/random?limit=10")  # fixture has 4
    data = r.json()
    ids = [photo["id"] for photo in data["results"]]
    assert sorted(_short_ids(random_app, ids)) == ["a", "b", "c", "d"]
    assert len(ids) == len(set(ids))  # no dupes
    assert data["has_more"] is False


def test_session_has_more_false_when_page_fills_remaining(random_app):
    """When the requested page exactly fills what's left of the
    deck, has_more must be False."""
    # Fixture has 4 photos. Ask for all 4 in one call.
    r = random_app.get("/api/random?limit=4")
    data = r.json()
    assert len(data["results"]) == 4
    assert data["has_more"] is False


def test_session_has_more_true_when_more_remains(random_app):
    """When more photos remain after this page, has_more=True."""
    # 4 photos total; ask for 2 → 2 remain.
    r1 = random_app.get("/api/random?limit=2")
    assert r1.json()["has_more"] is True


def test_session_offset_past_end_returns_empty(random_app):
    """An offset >= session_total returns empty results with
    has_more=False. The client can use this to detect end-of-session
    without a separate 'end' signal."""
    r1 = random_app.get("/api/random?limit=2")
    sid = r1.json()["session_id"]
    r = random_app.get(f"/api/random?session={sid}&offset=100&limit=5")
    data = r.json()
    assert data["results"] == []
    assert data["has_more"] is False
    assert data["offset"] == 4  # clamped to session_total


def test_session_unknown_id_starts_fresh(random_app):
    """An invalid session id is treated as 'no session' — the server
    creates a new one and returns its id. This way clients that
    restart (e.g. cleared cookies) don't crash on a stale id.
    """
    r = random_app.get("/api/random?session=bogus-id&limit=2")
    data = r.json()
    assert data["session_id"]
    assert data["session_id"] != "bogus-id"
    assert data["session_total"] == 4


def test_two_sessions_dont_share_state(random_app):
    """Two clients with separate sessions walk independently. Each
    gets its own shuffled deck. The decks are independent — one
    client's progress doesn't affect the other's.
    """
    r1 = random_app.get("/api/random?limit=2")
    r2 = random_app.get("/api/random?limit=2")
    sid1 = r1.json()["session_id"]
    sid2 = r2.json()["session_id"]
    assert sid1 != sid2

    # Both got a full page (limit=2, total=4 → has_more True).
    assert len(r1.json()["results"]) == 2
    assert len(r2.json()["results"]) == 2

    # Walking session 1 to the end doesn't affect session 2.
    random_app.get(f"/api/random?session={sid1}&offset=2&limit=2")
    # Session 2 still has photos left.
    r2_again = random_app.get(
        f"/api/random?session={sid2}&offset=2&limit=2"
    )
    assert r2_again.json()["has_more"] is False  # session 2 also done


def test_session_with_collection_filter(random_app):
    """A session restricted to one collections only walks that
    collection's photos."""
    r = random_app.get(
        "/api/random?limit=10&collections=portrait"
    )
    data = r.json()
    ids = _short_ids(random_app, {photo["id"] for photo in data["results"]})
    # 'portrait' has c, d in the fixture.
    assert ids == {"c", "d"}
    assert data["session_total"] == 2
    assert data["has_more"] is False


def test_session_deterministic_within_session(random_app):
    """The deck order is fixed for a session. Two reads at the same
    offset return the same photo.
    """
    r1 = random_app.get("/api/random?limit=2")
    sid = r1.json()["session_id"]
    ids1 = [p["id"] for p in r1.json()["results"]]

    r2 = random_app.get(f"/api/random?session={sid}&offset=0&limit=2")
    ids2 = [p["id"] for p in r2.json()["results"]]
    assert ids1 == ids2


def test_offset_validation(random_app):
    """Invalid offsets return an error. FastAPI returns 422 for
    type-coercion failures (offset=abc); our manual range check
    returns 400 (offset=-1). Both are valid error responses.
    """
    r1 = random_app.get("/api/random?limit=2")
    sid = r1.json()["session_id"]
    # Non-integer → FastAPI's built-in validation (422).
    assert random_app.get(f"/api/random?session={sid}&offset=abc&limit=2").status_code == 422
    # Negative → our manual check (400).
    assert random_app.get(f"/api/random?session={sid}&offset=-1&limit=2").status_code == 400


def test_session_expiry_yields_new_session(random_app):
    """A session that has expired (TTL elapsed) is replaced with a
    fresh one. The user gets a new shuffled deck rather than seeing
    a 404 or stale data.

    TTL expiry goes through the same code path as 'unknown session
    id' (both fall into `store.get()` returning None), so the
    'unknown id starts fresh' test above already pins the behavior
    for the user-visible contract. This test documents the TTL
    behavior at the unit level — it constructs a store directly
    to avoid coupling to the router's internal state.
    """
    import time as _time
    from search.random_session import RandomSession, RandomSessionStore

    from search.random_session import CursorDeck

    store = RandomSessionStore()

    def deck():
        return CursorDeck(lambda cursor, limit: ([], None), start="s")

    # Seed a session that's already past its TTL.
    expired = RandomSession(deck=deck(), ttl_s=60.0)
    expired.created_at = _time.monotonic() - 120.0  # 2 minutes ago, TTL is 60s
    store._sessions["stale-id"] = expired

    # get() should reject the expired session and return None.
    assert store.get("stale-id") is None
    # And the entry should have been cleaned up.
    assert "stale-id" not in store._sessions

    # put_new after expiry works as expected.
    sid, sess = store.put_new(deck())
    assert sid != "stale-id"
    assert sess.is_alive()





