"""
tests/test_surprise_api.py

Layer 2 — /api/search?surprise=true integration tests. Mirrors the
shape of tests/test_search_api.py (in-memory Qdrant + mock text
encoder) but uses a larger corpus so the Surprise Me deck isn't
degenerate.

Each test gets a fresh `app_with_qdrant_big` fixture with 50 random
points; that gives the Gumbel draw enough population to be a real
sampling operation (vs. 3 cats-and-dogs which is too small to
exercise the distribution).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from indexer import upsert
from indexer.upsert import VECTOR_DIM
from search import app as app_mod
from search.config import Config

from lustre_kernel.registry import MockEmbedder


# ---------------- fixtures ----------------


@pytest.fixture
def app_with_qdrant_big(qdrant_in_memory, nas_base, monkeypatch):
    """
    Like `app_with_qdrant` from tests/test_search_api.py, but with
    50 points so the Surprise Me Gumbel draw has a non-trivial
    population. All vectors are drawn from the same mock encoder
    so the ranking is well-defined (cosine sim over 1152-dim
    random-but-deterministic vectors).
    """
    cfg = Config(
        qdrant_url="memory://",
        qdrant_collection=qdrant_in_memory.collection,
        qdrant_api_key=None,
        model_name="mock",
        model_revision="",
        device="cpu",
        top_k_default=35,
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

    embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text

    n = 50
    # Use 50 distinct mock text labels so the embeddings are
    # diverse. The first label ("cat") is also the query text we
    # use in the tests, so at least one point has a high cosine
    # similarity to the query.
    labels = ["cat"] + [f"image_{i:03d}" for i in range(1, n)]
    items = []
    for i, label in enumerate(labels):
        point_id = f"{i:08x}-0000-0000-0000-000000000000"
        path = str(nas_base / f"img_{i:03d}.jpg")
        items.append((
            point_id,
            embed(label),
            {
                "id": point_id,
                "path": path,
                "collection": "general",
                "indexed_at": "2026-01-01T00:00:00Z",
            },
        ))
    upsert.upsert_batch(client, qdrant_in_memory.collection, items, wait=True)
    # Save a tiny PNG for every point so the index-db path can
    # resolve them later.
    for i in range(n):
        Image.new("RGB", (16, 16), (i * 5 % 256, 0, 0)).save(
            nas_base / f"img_{i:03d}.jpg"
        )

    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)
    with TestClient(app) as test_client:
        yield test_client
    app_mod.reset_for_tests()


# ---------------- Surprise Me: shape + ETag ----------------


def test_surprise_returns_session_id_and_total(app_with_qdrant_big):
    resp = app_with_qdrant_big.get("/api/search?q=cat&surprise=true&limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["surprise"] is True
    assert body["session_id"]
    assert len(body["session_id"]) == 16
    assert body["session_total"] == 50  # capped by 50-point corpus
    assert len(body["results"]) == 10
    assert body["has_more"] is True


def test_surprise_offset_walks_cached_deck(app_with_qdrant_big):
    # First page: build the deck.
    first = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=10&offset=0"
    ).json()
    session_id = first["session_id"]
    first_ids = [r["id"] for r in first["results"]]
    # Second page: pass session back, offset 10.
    second = app_with_qdrant_big.get(
        f"/api/search?q=cat&surprise=true&limit=10&offset=10&surprise_session={session_id}"
    ).json()
    second_ids = [r["id"] for r in second["results"]]
    # Same session, same query, same seed → same deck → no overlap.
    assert set(first_ids).isdisjoint(set(second_ids))
    assert second["session_id"] == session_id


def test_surprise_same_seed_same_deck(app_with_qdrant_big):
    # Two requests with identical params return identical decks.
    a = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=20&surprise_seed=42"
    ).json()
    b = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=20&surprise_seed=42"
    ).json()
    a_ids = [r["id"] for r in a["results"]]
    b_ids = [r["id"] for r in b["results"]]
    assert a_ids == b_ids


def test_surprise_no_seed_gives_different_decks(app_with_qdrant_big):
    # When the client doesn't pass surprise_seed, the server picks
    # a fresh random one per request, so the same query twice
    # returns two different decks. (The previous default of seed=1
    # made this test fail.)
    a = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=20"
    ).json()
    b = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=20"
    ).json()
    a_ids = [r["id"] for r in a["results"]]
    b_ids = [r["id"] for r in b["results"]]
    assert a["session_id"] != b["session_id"]
    assert a_ids != b_ids


def test_surprise_returns_real_scores(app_with_qdrant_big):
    # Bug fix: previously the score=0.0 on every result because
    # retrieve_batch doesn't populate scores. The route now stores
    # the raw cosine in the deck cache and hydrates the hit.score
    # from there.
    body = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=10&surprise_seed=1"
    ).json()
    for r in body["results"]:
        # Cosine sim is in [-1, 1]; SigLIP2 can produce negative
        # similarities for unrelated points.
        assert -1.0 <= r["score"] <= 1.0
        # score_str mirrors score rounded to 3dp.
        assert r["score_str"] == f"{r['score']:.3f}"
    # At least one result should have a non-trivial score
    # (the corpus is "cat" + 49 random labels, so most have ~0
    # similarity but at least the "cat" point should be > 0.1).
    assert any(r["score"] > 0.05 for r in body["results"])


def test_surprise_scores_persist_across_pagination(app_with_qdrant_big):
    # First page: build the deck.
    first = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=10&offset=0&surprise_seed=7"
    ).json()
    session_id = first["session_id"]
    first_scores = [r["score"] for r in first["results"]]
    # Second page: same session, different offset.
    second = app_with_qdrant_big.get(
        f"/api/search?q=cat&surprise=true&limit=10&offset=10"
        f"&surprise_session={session_id}"
    ).json()
    second_scores = [r["score"] for r in second["results"]]
    # All scores should be real, including on the cached page
    # (no 0.0).
    for s in first_scores + second_scores:
        # Cosine sim is in [-1, 1]; SigLIP2 can produce negative
        # similarities for unrelated points.
        assert -1.0 <= s <= 1.0
    assert any(s > 0.05 for s in first_scores + second_scores)


def test_surprise_different_seed_different_deck(app_with_qdrant_big):
    a = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=20&surprise_seed=1"
    ).json()
    b = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=20&surprise_seed=2"
    ).json()
    a_ids = {r["id"] for r in a["results"]}
    b_ids = {r["id"] for r in b["results"]}
    # Two independent Gumbel draws from the same 50-point pool
    # should differ; with 20 picks out of 50 the chance of
    # identical draws is ~0.
    assert a_ids != b_ids


def test_surprise_different_temperature_different_deck(app_with_qdrant_big):
    a = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=20&surprise_temperature=0.01"
    ).json()
    b = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=20&surprise_temperature=0.5"
    ).json()
    a_ids = {r["id"] for r in a["results"]}
    b_ids = {r["id"] for r in b["results"]}
    # Different temperature → different softmax distribution →
    # different top-20 sample. Not guaranteed disjoint (small
    # overlap is possible), but the multisets should not match.
    assert sorted(a_ids) != sorted(b_ids)


def test_surprise_zero_temperature_is_top_k(app_with_qdrant_big):
    # T=0 → deterministic top-K. The same query twice with T=0
    # returns the same set of ids in the same order. Different
    # seeds have no effect at T=0 (the Gumbel noise is multiplied
    # by T).
    a = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=10&surprise_temperature=0&surprise_seed=1"
    ).json()
    b = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=10&surprise_temperature=0&surprise_seed=2"
    ).json()
    assert [r["id"] for r in a["results"]] == [r["id"] for r in b["results"]]


# ---------------- Surprise Me: validation ----------------


def test_surprise_rejects_out_of_range_temperature(app_with_qdrant_big):
    # Below TEMPERATURE_MIN (0.0) is rejected.
    resp = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&surprise_temperature=-0.01"
    )
    assert resp.status_code == 400
    assert "surprise_temperature" in resp.json()["detail"]
    # Above TEMPERATURE_MAX (1.0) is rejected.
    resp = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&surprise_temperature=1.5"
    )
    assert resp.status_code == 400
    assert "surprise_temperature" in resp.json()["detail"]


def test_surprise_accepts_zero_temperature(app_with_qdrant_big):
    # T=0 is the deterministic "top-K" knob. Allowed by the
    # route so the user can dial randomness all the way to zero.
    resp = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&surprise_temperature=0&limit=5"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 5


def test_surprise_rejects_out_of_range_pool(app_with_qdrant_big):
    resp = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&surprise_pool=100"
    )
    assert resp.status_code == 400
    assert "surprise_pool" in resp.json()["detail"]
    # POOL_SIZE_MAX is now 50_000 — anything above is rejected.
    # The Qdrant server-side HNSW search crosses its 30s
    # timeout for pool sizes much above this on a 2M-point
    # corpus, so we cap the slider at 50k.
    resp = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&surprise_pool=75000"
    )
    assert resp.status_code == 400
    assert "surprise_pool" in resp.json()["detail"]


def test_surprise_still_rejects_diversity(app_with_qdrant_big):
    # Existing invariant: surprise + diverse is a 400.
    resp = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&diversity=balanced"
    )
    assert resp.status_code == 400
    assert "Diversity" in resp.json()["detail"]


def test_surprise_etag_is_stable_for_identical_queries(app_with_qdrant_big):
    # Same query+seed+offset+limit → same ETag. The route
    # doesn't itself return 304 server-side; the 304 is browser
    # cache behaviour (Cache-Control: private, max-age=10). The
    # server's job is to hand out a stable ETag so the browser
    # can do the conditional GET. This mirrors the existing
    # test_api_search_diversity_is_stable_across_pages pattern.
    url = "/api/search?q=cat&surprise=true&limit=10&surprise_seed=7"
    first = app_with_qdrant_big.get(url)
    second = app_with_qdrant_big.get(url)
    assert first.headers["ETag"] == second.headers["ETag"]


def test_surprise_etag_differs_when_seed_differs(app_with_qdrant_big):
    a = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=10&surprise_seed=1"
    )
    b = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=10&surprise_seed=2"
    )
    assert a.headers["ETag"] != b.headers["ETag"]


def test_surprise_etag_off_path_is_unaffected(app_with_qdrant_big):
    # Non-surprise requests must still hash the same as before this
    # feature landed (so the existing browser-cache 304 fast-path
    # keeps working).
    with_surprise_off_a = app_with_qdrant_big.get("/api/search?q=cat&limit=10")
    with_surprise_off_b = app_with_qdrant_big.get("/api/search?q=cat&limit=10")
    assert with_surprise_off_a.headers["ETag"] == with_surprise_off_b.headers["ETag"]
    # Confirm the new "off" sentinel didn't drift: a request with
    # the sentinel baked in (surprise=false explicitly) collides
    # with a request that doesn't set the param.
    with_surprise_explicit_off = app_with_qdrant_big.get(
        "/api/search?q=cat&limit=10&surprise=false"
    )
    assert with_surprise_explicit_off.headers["ETag"] == with_surprise_off_a.headers["ETag"]


def test_surprise_composes_with_favorites_filter(app_with_qdrant_big):
    # Mark one photo as a favorite, then ask for favorites +
    # surprise. The Surprise Me deck should be drawn from the
    # favourite-only pool, not the full corpus.
    from search.config import Config as _Cfg
    _Cfg(
        qdrant_url="memory://",
        qdrant_collection="ignored",
        qdrant_api_key=None,
        model_name="mock",
        model_revision="",
        device="cpu",
        top_k_default=35,
        top_k_max=200,
        query_timeout_ms=2000,
        nas_images_base="/tmp/ignored",  # noqa: S108
        path_prefix="",
        web_ui_url="http://localhost:8000",
        log_level="WARNING",
        test_mode=True,
    )
    # Use the same IndexDB the app wired up so the favorite
    # write hits the store the search reads from.
    db = app_with_qdrant_big.app.state.index_db
    # No cache to sync: mark_favorite checks the photo's existence
    # against Qdrant directly, and the id below is a real point in
    # the fixture collection.
    fav_id = "00000005-0000-0000-0000-000000000000"  # a real id in the fixture
    db.mark_favorite(fav_id)
    try:
        resp = app_with_qdrant_big.get(
            "/api/search?q=cat&surprise=true&limit=10&favorites=true&surprise_seed=1"
        )
        assert resp.status_code == 200
        body = resp.json()
        # Only the favourited id should appear in the results.
        result_ids = {r["id"] for r in body["results"]}
        assert result_ids == {fav_id}
        assert body["session_total"] == 1
    finally:
        db.unmark_favorite(fav_id)


def test_surprise_favorites_with_empty_set_falls_through_to_qdrant(app_with_qdrant_big):
    # Documented edge case: when favorites=true is set but the
    # user has no favourites, the favourites filter passes an
    # empty allowed_ids to Qdrant, which returns the full pool.
    # This matches the existing non-surprise favourites behaviour
    # — the empty-set short-circuit is a filename-only check at
    # line 270 of routers/search.py, not a favourites check.
    # The user-visible effect is "Surprise Me shows everything
    # when you ask for favourites but have none" — same as the
    # non-surprise favourites path.
    resp = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=10&favorites=true"
    )
    assert resp.status_code == 200
    body = resp.json()
    # The pool is unfiltered because the favourite set is empty;
    # the deck contains all 50 corpus points.
    assert body["session_total"] == 50


def test_surprise_offset_past_end_returns_empty(app_with_qdrant_big):
    first = app_with_qdrant_big.get(
        "/api/search?q=cat&surprise=true&limit=10"
    ).json()
    session_id = first["session_id"]
    # Offset 100 past a 50-card deck: 0 results, has_more=False.
    resp = app_with_qdrant_big.get(
        f"/api/search?q=cat&surprise=true&limit=10&offset=100&surprise_session={session_id}"
    ).json()
    assert resp["results"] == []
    assert resp["has_more"] is False
    # session_total still echoes the deck size so the client knows.
    assert resp["session_total"] == 50


# ---------------- the empty-prompt direction ----------------


def test_surprise_without_a_query_uses_a_directional_vector(
    app_with_qdrant_big, monkeypatch
):
    """Surprise Me with no query must not search with a zero vector.

    A zero vector has no direction, so Qdrant cannot use the HNSW
    graph and falls back to a near-exact scan of the whole
    collection — measured 8-12x slower (20,000 candidates: 60.0s
    with a zero vector, 4.8s with a random unit vector), which is
    what made the deep pools time out. It also gave every candidate
    the score 0.0, so the Gumbel draw had nothing to work with.

    A random unit vector is an unbiased direction, and in a
    1152-dimension space the corpus is spread out enough that its
    nearest neighbours are scattered rather than clustered: the same
    random sample, drawn from the index instead of from a full scan.
    """
    from search.qdrant_client import QdrantSearch

    seen: list[list[float]] = []
    original = QdrantSearch.search

    def recording_search(self, vec, *args, **kwargs):
        seen.append(list(vec))
        return original(self, vec, *args, **kwargs)

    monkeypatch.setattr(QdrantSearch, "search", recording_search)
    resp = app_with_qdrant_big.get(
        "/api/search?surprise=true&surprise_pool=200&limit=10"
    )

    assert resp.status_code == 200, resp.text
    assert seen, "the search never ran"
    vec = seen[0]
    assert len(vec) == VECTOR_DIM
    assert any(v != 0.0 for v in vec), (
        "the surprise search used a zero vector — Qdrant falls back "
        "to a full scan and the pool builds time out"
    )
    # A unit vector: same direction, no magnitude change.
    magnitude = sum(v * v for v in vec) ** 0.5
    assert abs(magnitude - 1.0) < 1e-3, f"the vector is not a unit vector: |v|={magnitude}"


def test_surprise_direction_is_seeded_and_reproducible(app_with_qdrant_big, monkeypatch):
    """The same seed samples the same direction, so a session is stable."""
    from search.qdrant_client import QdrantSearch

    seen: list[list[float]] = []
    original = QdrantSearch.search

    def recording_search(self, vec, *args, **kwargs):
        seen.append(list(vec))
        return original(self, vec, *args, **kwargs)

    monkeypatch.setattr(QdrantSearch, "search", recording_search)

    def direction(seed: str) -> list[float]:
        seen.clear()
        app_with_qdrant_big.get(
            f"/api/search?surprise=true&surprise_pool=200&limit=10&surprise_seed={seed}"
        )
        return seen[0]

    assert direction("42") == direction("42")
    assert direction("42") != direction("43")
