"""
tests/test_filename_filter.py

Tests for the `?filename=` path-substring filter:

  1. IndexDB.path_token_ids() — FTS5 query correctness, including
     token-substring, prefix (`cha*`), case-insensitivity,
     empty/whitespace handling, suffix-rejection, multi-token rejection,
     and the trigger-sync on INSERT/UPDATE/DELETE.

  2. qdrant_client.search() — the `allowed_ids` argument ANDs a
     `HasId` condition into the existing `must` filter (it doesn't
     replace the collections filter).

  3. Cardinality guard — when the matching set covers > 50% of the
     cache, the `HasId` filter is dropped server-side (logged) and
     the full result set comes back.

  4. End-to-end /api/search — the full round-trip from URL param
     through the FTS5 lookup, the Qdrant filter, and the response
     JSON. Includes `q` + `filename` interaction, empty pattern,
     invalid pattern (400), pagination (forwarding the filter on
     `loadMorePage` is exercised in the JS mirror), and the
     multi-prompt case.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from indexer import upsert
from indexer.upsert import VECTOR_DIM
from search import app as app_mod
from search.config import Config
from lustre_kernel.registry import MockEmbedder; _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text

# ---------------- Shared test data ----------------

# Five distinct image ids, paths, and vectors. The mock embedder
# returns the same vector for the same query string, so we pick
# query strings to give each image a different ranking. This is the
# same trick the existing tests use (see CAT_ID/DOG_ID/CAR_ID
# below).
CHAEWON_ID = "11111111-1111-1111-1111-111111111111"
KAZUHA_ID = "22222222-2222-2222-2222-222222222222"
YUJIN_ID = "33333333-3333-3333-3333-333333333333"
ANOTHER_ID = "44444444-4444-4444-4444-444444444444"
GENERIC_ID = "55555555-5555-5555-5555-555555555555"

# Paths are deliberately chosen so the FTS5 tokeniser splits them
# into useful tokens. Tokens split on `/`, `-`, `_`, `.` so
# `/photos/kpop/chaewon/2024.jpg` tokenises to
# `[photos, kpop, chaewon, 2024, jpg]`.
CHAEWON_PATH = "/photos/kpop/chaewon/2024.jpg"
KAZUHA_PATH = "/photos/kpop/kazuha/2024.jpg"
YUJIN_PATH = "/photos/kpop/yujin/2024.jpg"
ANOTHER_PATH = "/photos/portrait/winter/2023.jpg"
GENERIC_PATH = "/archive/2020/old.jpg"


@pytest.fixture
def app_with_filename_paths(qdrant_in_memory, nas_base, monkeypatch):
    """
    FastAPI app wired to:
      - in-memory Qdrant with 5 distinct points
      - paths designed to exercise FTS5 token matching
      - mock text encoder
      - nas_base configured so /photo/{id}/raw works
    """
    cfg = Config(
        qdrant_url="memory://",
        qdrant_collection=qdrant_in_memory.collection,
        qdrant_api_key=None,
        model_name="mock",
        model_revision="",
        device="cpu",
        top_k_default=50,
        top_k_max=200,
        query_timeout_ms=2000,
        nas_images_base=str(nas_base),
        path_prefix="",
        web_ui_url="http://localhost:8000",
        log_level="WARNING",
        test_mode=True,
    )

    client = qdrant_in_memory.client
    upsert.ensure_collection(client, qdrant_in_memory.collection, dim=VECTOR_DIM)

    items = [
        (CHAEWON_ID, _mock_embed("chaewon"),
         {"id": CHAEWON_ID, "path": str(nas_base / "chaewon_2024.jpg"), "collection": "kpop", "indexed_at": "2026-01-01T00:00:00Z"}),
        (KAZUHA_ID, _mock_embed("kazuha"),
         {"id": KAZUHA_ID, "path": str(nas_base / "kazuha_2024.jpg"), "collection": "kpop", "indexed_at": "2026-01-01T00:00:00Z"}),
        (YUJIN_ID, _mock_embed("yujin"),
         {"id": YUJIN_ID, "path": str(nas_base / "yujin_2024.jpg"), "collection": "kpop", "indexed_at": "2026-01-01T00:00:00Z"}),
        (ANOTHER_ID, _mock_embed("winter"),
         {"id": ANOTHER_ID, "path": str(nas_base / "winter_portrait.jpg"), "collection": "portrait", "indexed_at": "2026-01-01T00:00:00Z"}),
        (GENERIC_ID, _mock_embed("generic"),
         {"id": GENERIC_ID, "path": str(nas_base / "random.jpg"), "collection": "general", "indexed_at": "2026-01-01T00:00:00Z"}),
    ]
    upsert.upsert_batch(client, qdrant_in_memory.collection, items, wait=True)

    # Save matching PNGs so /photo/.../raw works for the IDs we'll
    # actually surface in test results.
    for _pid, _vec, payload in items:
        Image.new("RGB", (16, 16), (0, 0, 0)).save(
            nas_base / Path(payload["path"]).name
        )

    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)

    with TestClient(app) as tc:
        yield tc

    app_mod.reset_for_tests()


# ============================================================
#  IndexDB.path_token_ids() — FTS5 query correctness
# ============================================================


def test_path_token_ids_token_substring_match():
    """A bare token matches paths where the token appears."""

    qdrant = _FakeQdrant(
        [
            {"id": "a", "path": CHAEWON_PATH},
            {"id": "b", "path": KAZUHA_PATH},
            {"id": "c", "path": YUJIN_PATH},
        ]
    )
    db = _make_db(qdrant)
    try:
        ids = sorted(db.path_token_ids("chaewon"))
        assert ids == ["a"]
        ids = sorted(db.path_token_ids("kpop"))
        assert ids == ["a", "b", "c"]
    finally:
        db.close()


def test_path_token_ids_prefix_match():
    """A trailing `*` matches any token starting with the body."""

    qdrant = _FakeQdrant(
        [
            {"id": "a", "path": CHAEWON_PATH},
            {"id": "b", "path": KAZUHA_PATH},
            {"id": "c", "path": "/photos/kpop/chaewon_variant/2025.jpg"},
        ]
    )
    db = _make_db(qdrant)
    try:
        ids = sorted(db.path_token_ids("cha*"))
        assert ids == ["a", "c"]
    finally:
        db.close()


def test_path_token_ids_case_insensitive():
    """FTS5 with unicode61 is case-insensitive by default."""

    qdrant = _FakeQdrant([{"id": "a", "path": "/photos/Kpop/Chaewon/2024.JPG"}])
    db = _make_db(qdrant)
    try:
        # Path tokens are lowercased by the tokeniser, so any
        # casing of the query body matches.
        assert db.path_token_ids("chaewon") == ["a"]
        assert db.path_token_ids("CHAEWON") == ["a"]
        assert db.path_token_ids("Chaewon*") == ["a"]
    finally:
        db.close()


def test_path_token_ids_empty_returns_none():
    """Empty / whitespace-only patterns return None (skip filter)."""

    qdrant = _FakeQdrant([{"id": "a", "path": CHAEWON_PATH}])
    db = _make_db(qdrant)
    try:
        assert db.path_token_ids("") is None
        assert db.path_token_ids("   ") is None
        assert db.path_token_ids("\t\n") is None
    finally:
        db.close()


def test_path_token_ids_suffix_match_raises():
    """`*foo` suffix matching is not supported by FTS5."""

    qdrant = _FakeQdrant([{"id": "a", "path": CHAEWON_PATH}])
    db = _make_db(qdrant)
    try:
        with pytest.raises(ValueError, match="trailing"):
            db.path_token_ids("*ewon")
        with pytest.raises(ValueError, match="trailing"):
            db.path_token_ids("*.jpg")
    finally:
        db.close()


def test_path_token_ids_multi_token_raises():
    """Multiple tokens in one query are rejected — callers should
    pick one token at a time."""

    qdrant = _FakeQdrant([{"id": "a", "path": CHAEWON_PATH}])
    db = _make_db(qdrant)
    try:
        with pytest.raises(ValueError, match="single token"):
            db.path_token_ids("chaewon won")
    finally:
        db.close()


def test_path_token_ids_fts5_operator_raises():
    """FTS5 special characters are rejected before reaching FTS5."""

    qdrant = _FakeQdrant([{"id": "a", "path": CHAEWON_PATH}])
    db = _make_db(qdrant)
    try:
        # Quotes, parens, colons, +/- are FTS5 syntax we forbid.
        for bad in ['foo"bar', "foo(bar)", "foo:bar", "foo+bar", "foo-bar"]:
            with pytest.raises(ValueError):
                db.path_token_ids(bad)
    finally:
        db.close()


def test_path_token_ids_trigger_sync_on_insert():
    """Adding a new image via INSERT makes it searchable via FTS5
    without a manual rebuild."""

    qdrant = _FakeQdrant([{"id": "a", "path": CHAEWON_PATH}])
    db = _make_db(qdrant)
    try:
        assert "b" not in db.path_token_ids("kazuha")
        # Insert directly via the connection. The trigger should
        # populate images_fts.
        with db._lock:
            db._conn.execute(
                "INSERT INTO images(id, path) VALUES (?, ?)",
                ("b", KAZUHA_PATH),
            )
            db._conn.commit()
        assert sorted(db.path_token_ids("kazuha")) == ["b"]
    finally:
        db.close()


def test_path_token_ids_trigger_sync_on_update():
    """Updating an image's path updates the FTS index."""

    qdrant = _FakeQdrant([{"id": "a", "path": CHAEWON_PATH}])
    db = _make_db(qdrant)
    try:
        assert db.path_token_ids("kazuha") == []
        with db._lock:
            db._conn.execute(
                "UPDATE images SET path = ? WHERE id = ?",
                (KAZUHA_PATH, "a"),
            )
            db._conn.commit()
        # The old token is gone, the new one is present.
        assert db.path_token_ids("chaewon") == []
        assert db.path_token_ids("kazuha") == ["a"]
    finally:
        db.close()


def test_path_token_ids_trigger_sync_on_delete():
    """Deleting an image removes it from the FTS index."""

    qdrant = _FakeQdrant([{"id": "a", "path": CHAEWON_PATH}])
    db = _make_db(qdrant)
    try:
        assert db.path_token_ids("chaewon") == ["a"]
        with db._lock:
            db._conn.execute("DELETE FROM images WHERE id = ?", ("a",))
            db._conn.commit()
        assert db.path_token_ids("chaewon") == []
    finally:
        db.close()
def test_api_search_filename_filter_restricts_results(app_with_filename_paths):
    """`?filename=chaewon` restricts results to the matching ids only."""
    resp = app_with_filename_paths.get("/api/search?q=chaewon&filename=chaewon")
    assert resp.status_code == 200
    data = resp.json()
    ids = [r["id"] for r in data["results"]]
    # The mock embedder returns the same vector for "chaewon" as
    # the CHAEWON_ID point — so without the filter, that's the top
    # result. With the filename filter, only the chaewon image
    # qualifies. Verify it IS present (the filter doesn't drop the
    # genuine match) AND nothing else snuck in.
    assert CHAEWON_ID in ids
    # The filter is AND'd into the existing qdrant filter; since
    # the other points have distinct embeddings (mock_embed returns
    # a different vector per query string), the cosine ranking
    # wouldn't pick them anyway. But verify by collection count:
    # the matching set is {CHAEWON_ID}, so results must be a subset.
    assert all(r["id"] == CHAEWON_ID for r in data["results"])


def test_api_search_filename_filter_excludes_unmatched(app_with_filename_paths):
    """With `?filename=chaewon`, a query for `kazuha` (a different
    vector) still only returns the chaewon image — the filename
    filter is the binding constraint, not the text query."""
    resp = app_with_filename_paths.get("/api/search?q=kazuha&filename=chaewon")
    assert resp.status_code == 200
    data = resp.json()
    # The chaewon file is the only thing the filter permits.
    ids = [r["id"] for r in data["results"]]
    assert ids == [CHAEWON_ID]


def test_api_search_no_filename_returns_full_results(app_with_filename_paths):
    """No `?filename=` → the filter is skipped; ranking is purely by
    semantic similarity."""
    resp = app_with_filename_paths.get("/api/search?q=chaewon")
    assert resp.status_code == 200
    data = resp.json()
    # The mock embedder ranks chaewon first.
    assert data["results"][0]["id"] == CHAEWON_ID


def test_api_search_empty_filename_treated_as_no_filter(app_with_filename_paths):
    """`?filename=` (empty) is the same as not sending it at all."""
    resp = app_with_filename_paths.get("/api/search?q=chaewon&filename=")
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"][0]["id"] == CHAEWON_ID


def test_api_search_filename_prefix_match(app_with_filename_paths):
    """`?filename=cha*` matches any path token starting with `cha`."""
    resp = app_with_filename_paths.get("/api/search?q=chaewon&filename=cha*")
    assert resp.status_code == 200
    data = resp.json()
    # Only CHAEWON_ID has a `cha*` token (path: chaewon_2024.jpg).
    assert all(r["id"] == CHAEWON_ID for r in data["results"])


def test_api_search_filename_invalid_pattern_returns_400(app_with_filename_paths):
    """A leading `*` (suffix-match) is rejected with a 400."""
    resp = app_with_filename_paths.get("/api/search?q=chaewon&filename=*ewon")
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "bad_request"
    assert "filename" in body["detail"].lower()


def test_api_search_filename_invalid_fts_operator_returns_400(app_with_filename_paths):
    """Multi-token / FTS5 operators are rejected with a 400."""
    resp = app_with_filename_paths.get("/api/search?q=chaewon&filename=chaewon%20won")
    assert resp.status_code == 400


def test_api_search_filename_no_match_returns_empty_results(app_with_filename_paths):
    """A pattern that matches no image returns an empty result list
    (NOT a 400). The user typed something valid that just doesn't
    exist in the cache."""
    resp = app_with_filename_paths.get("/api/search?q=chaewon&filename=nonsense")
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []


def test_api_search_filename_only_mode(app_with_filename_paths):
    """Filename-only mode (no text query, no prompts) works. The
    server resolves this to a zero-vector query with the HasId
    filter applied. Browses the matching files deterministically
    by id."""
    resp = app_with_filename_paths.get("/api/search?filename=chaewon")
    assert resp.status_code == 200
    data = resp.json()
    # Only the chaewon image is in the matching set.
    assert [r["id"] for r in data["results"]] == [CHAEWON_ID]


def test_api_search_filename_applied_to_all_prompts(app_with_filename_paths):
    """Multi-prompt search honours the same filename filter on every
    prompt's result. With the mock embedder, the first prompt's
    vector dominates the ranking; what matters here is that the
    filter doesn't get dropped between prompts."""
    resp = app_with_filename_paths.get(
        "/api/search",
        params=[
            ("positives", "chaewon"),
            ("positives", "kazuha"),
            ("filename", "chaewon"),
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    # All results must be from the filename filter's matching set.
    assert all(r["id"] == CHAEWON_ID for r in data["results"])


def test_api_search_filename_combined_with_collection(app_with_filename_paths):
    """Filename AND a collection filter. Both must be satisfied."""
    # The chaewon image is in the `kpop` collection.
    resp = app_with_filename_paths.get(
        "/api/search?q=chaewon&filename=chaewon&collection=kpop"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert [r["id"] for r in data["results"]] == [CHAEWON_ID]
    # The chaewon image is NOT in the portrait collection.
    resp = app_with_filename_paths.get(
        "/api/search?q=chaewon&filename=chaewon&collection=portrait"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []


def test_api_search_filename_combined_with_favorites(app_with_filename_paths):
    """Filename AND favourites. Both must be satisfied — a photo
    that's not in the favourites set is filtered out even if its
    path matches."""
    # Mark CHAEWON_ID as favourite.
    resp = app_with_filename_paths.post(f"/api/favorites/{CHAEWON_ID}")
    assert resp.status_code == 200

    # With favourites=true, only favourited photos qualify.
    resp = app_with_filename_paths.get(
        "/api/search?q=chaewon&filename=chaewon&favorites=true"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert [r["id"] for r in data["results"]] == [CHAEWON_ID]

    # And without the filename filter, favourites still only
    # returns the favourited one.
    resp = app_with_filename_paths.get(
        "/api/search?q=chaewon&favorites=true"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert [r["id"] for r in data["results"]] == [CHAEWON_ID]


def test_api_search_filename_cardinality_guard_skips_filter(caplog):
    """When the matching set covers > 50% of the cache, the HasId
    filter is dropped (logged at INFO). The search then ranks the
    full collection by semantic similarity. This is the load-bearing
    optimisation for very-broad patterns."""
    from indexer import upsert
    from indexer.upsert import VECTOR_DIM
    from search import app as app_mod
    from search.config import Config
    from lustre_kernel.registry import MockEmbedder; _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text

    # Build a fresh in-memory Qdrant with 10 points, all with paths
    # matching the same single token. This triggers the guard
    # (10/10 = 100% > 50%). The mock embedder ranks each point by
    # its specific query string.
    cfg = Config(
        qdrant_url="memory://",
        qdrant_collection="cardinality_test",
        qdrant_api_key=None,
        model_name="mock",
        model_revision="",
        device="cpu",
        top_k_default=50,
        top_k_max=200,
        query_timeout_ms=2000,
        nas_images_base=str(Path(tempfile.gettempdir()).resolve()),
        path_prefix="",
        web_ui_url="http://localhost:8000",
        log_level="INFO",
        test_mode=True,
    )

    from qdrant_client import QdrantClient

    client = QdrantClient(location=":memory:")
    collection = "cardinality_test"
    upsert.ensure_collection(client, collection, dim=VECTOR_DIM)

    items = []
    for i in range(10):
        # Use valid UUID-format strings (Qdrant requires them).
        import uuid as _uuid
        pid = str(_uuid.UUID(int=i + 1, version=4))
        vec = _mock_embed(f"query{i}")
        items.append((pid, vec, {
            "id": pid,
            "path": f"/photos/kpop/section_{i}/chaewon.jpg",
            "collection": "kpop",
            "indexed_at": "2026-01-01",
        }))
    upsert.upsert_batch(client, collection, items, wait=True)

    from search.qdrant_client import QdrantSearch
    qdrant = QdrantSearch(client=client, collection=collection, timeout_ms=2000)

    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant)
    with TestClient(app) as tc, caplog.at_level(logging.INFO):
        # `chaewon` matches ALL 10 images (100% > 50%). The
        # guard should kick in and log the skip.
        resp = tc.get("/api/search?q=query0&filename=chaewon")
        assert resp.status_code == 200
        data = resp.json()
        # All 10 results should come back (semantic ranking of
        # the full collection, no HasId filter applied).
        assert len(data["results"]) == 10
        # The skip was logged.
        assert any(
            "cardinality guard" in record.message
            for record in caplog.records
        )
    app_mod.reset_for_tests()


# ============================================================
#  /api/search — qdrant_client allowed_ids path
# ============================================================


def test_qdrant_search_allowed_ids_is_anded_with_collections(app_with_filename_paths):
    """The `allowed_ids` filter is AND'd with the collections filter,
    not OR'd. With a non-overlapping collection, both filters
    together yield zero results."""
    # ANOTHER_ID is in the portrait collection, but with filename=kpop
    # only kpop images match. ANDing both filters returns only kpop
    # matches — which is empty here (no kpop-named file exists).
    resp = app_with_filename_paths.get(
        "/api/search?q=winter&filename=kpop&collection=portrait"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []


# ============================================================
#  /api/search — photo back-link round-trip
# ============================================================




# ============================================================
#  /api/centroids/{name}/search — filename filter applies
# ============================================================


def test_centroid_search_filename_filter_restricts_results(app_with_centroids):
    """The /api/centroids/{name}/search endpoint also honours the
    filename filter. The filter shape is identical to /api/search
    — the same `_resolve_filename_filter` helper resolves it."""
    # The centroid fixture has cat/dog/car in the `general`
    # collection. The mock embedder's vectors all look like
    # `cat`/`dog`/`car` so without a filter the ranking is by
    # exact match. With `?filename=dog` we expect only the dog.
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search?filename=dog"
    )
    assert resp.status_code == 200
    data = resp.json()
    # The dog photo is the only one with a path token matching
    # 'dog' (path: .../dog.jpg).
    assert [r["id"] for r in data["results"]] == [CENTROID_DOG_ID]


# ---------------- Helpers ----------------


class _FakeQdrant:
    """Minimal stand-in for IndexDB's `qdrant_client` collaborator.
    Used by the IndexDB tests so we don't drag the full
    in-memory Qdrant fixture in.
    """

    def __init__(self, rows):
        self.rows = list(rows)

    def scroll_all(self, batch_size=1000):
        yield [
            {"id": r["id"], "payload": {
                "id": r["id"], "path": r["path"],
                "shard": "", "mtime": 100, "size": 200,
                "indexed_at": "2026-01-01T00:00:00+00:00",
            }}
            for r in self.rows
        ]


def _make_db(qdrant, tmp_path_factory=None):
    """Build an IndexDB with the standard test columns. The db path
    is auto-generated per-test."""
    import tempfile

    from search.index_db import IndexDB

    with tempfile.NamedTemporaryFile(
        suffix=".db", delete=False
    ) as handle:
        db_path = handle.name
    db = IndexDB(db_path, qdrant, refresh_interval_seconds=3600)
    db.init_from_qdrant()
    return db


# Re-import the constants from the shared centroid fixture so we
# don't redeclare them.
from _centroid_fixture import (
    CENTROID_DOG_ID,
    WUXIA_CENTROID,
)
