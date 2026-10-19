"""
tests/test_search_api.py

Layer 2 — search API tests using FastAPI TestClient + in-memory Qdrant
+ mock text encoder.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from indexer import upsert
from indexer.upsert import VECTOR_DIM
from search import app as app_mod
from search.config import Config

# Test UUIDs (deterministic, valid UUID format that Qdrant accepts).
CAT_ID = "11111111-1111-1111-1111-111111111111"
DOG_ID = "22222222-2222-2222-2222-222222222222"
CAR_ID = "33333333-3333-3333-3333-333333333333"


# ---------------- fixtures ----------------


@pytest.fixture
def app_with_qdrant(qdrant_in_memory, nas_base, monkeypatch):
    """
    A FastAPI app wired to:
      - in-memory Qdrant
      - mock text encoder (deterministic)
      - NAS_IMAGES_BASE=nas_base
      - collection pre-populated with a few test points
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

    # Pre-populate the collection with a few points whose vectors are
    # known and recognizable.
    client = qdrant_in_memory.client
    upsert.ensure_collection(client, qdrant_in_memory.collection, dim=VECTOR_DIM)

    # Use the deterministic mock embedder so we can pick what we put in.
    from lustre_kernel.registry import MockEmbedder; _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text

    # Three test "images": cat, dog, car.
    q_cat = _mock_embed("cat")
    q_dog = _mock_embed("dog")
    q_car = _mock_embed("car")
    items = [
        (
            CAT_ID,
            q_cat,
            {"id": CAT_ID, "path": str(nas_base / "cat.jpg"), "collection": "general", "indexed_at": "2026-01-01T00:00:00Z"},
        ),
        (
            DOG_ID,
            q_dog,
            {"id": DOG_ID, "path": str(nas_base / "dog.jpg"), "collection": "general", "indexed_at": "2026-01-01T00:00:00Z"},
        ),
        (
            CAR_ID,
            q_car,
            {"id": CAR_ID, "path": str(nas_base / "car.jpg"), "collection": "general", "indexed_at": "2026-01-01T00:00:00Z"},
        ),
    ]
    upsert.upsert_batch(client, qdrant_in_memory.collection, items, wait=True)

    # Save the cat and dog PNGs to disk so /photo/.../raw works.
    Image.new("RGB", (16, 16), (255, 0, 0)).save(nas_base / "cat.jpg")
    Image.new("RGB", (16, 16), (0, 255, 0)).save(nas_base / "dog.jpg")
    # (car.jpg is intentionally not on disk — tests the "file missing" path)

    # Build the app.
    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)

    with TestClient(app) as client:
        yield client

    app_mod.reset_for_tests()


# ---------------- Layer 2: / ----------------
















# ---------------- Layer 2: /api/search ----------------


def test_api_search_happy_path(app_with_qdrant):
    resp = app_with_qdrant.get("/api/search?q=cat")
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "cat"
    assert data["positives"] == ["cat"]
    assert data["negatives"] == []
    assert isinstance(data["results"], list)
    assert len(data["results"]) >= 1
    r = data["results"][0]
    assert "id" in r and "path" in r and "score" in r
    # Cosine similarity can drift slightly outside [-1, 1] due to fp imprecision
    # in the mock embedder's L2 normalization. Allow a small epsilon.
    assert -1.001 <= r["score"] <= 1.001
    assert "took_ms" in data


def test_api_search_default_limit_is_35(app_with_qdrant):
    response = app_with_qdrant.get("/api/search?q=cat")
    assert response.status_code == 200
    assert response.json()["limit"] == 35


def test_api_search_q_param_still_works(app_with_qdrant):
    resp = app_with_qdrant.get("/api/search?q=cat")
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "cat"
    assert data["positives"] == ["cat"]
    assert data["results"][0]["id"] == CAT_ID


def test_api_search_diversity_is_stable_across_pages(app_with_qdrant):
    first = app_with_qdrant.get(
        "/api/search",
        params=[("q", "cat"), ("diversity", "balanced"), ("limit", "2")],
    )
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["diverse"] is True
    assert first_data["diversity"]["requested"] is True
    assert first_data["diversity"]["applied"] is True
    assert first_data["diversity"]["mode"] == "balanced"
    assert first_data["diversity"]["candidate_count"] == 3
    assert first_data["has_more"] is True

    second = app_with_qdrant.get(
        "/api/search",
        params=[
            ("q", "cat"),
            ("diversity", "balanced"),
            ("limit", "2"),
            ("offset", "2"),
        ],
    )
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["diversity"]["mode"] == "balanced"
    assert {
        result["id"] for result in first_data["results"]
    }.isdisjoint({result["id"] for result in second_data["results"]})
    assert second_data["has_more"] is False


def test_api_search_diversity_depth_is_independent_and_reported(app_with_qdrant):
    response = app_with_qdrant.get(
        "/api/search",
        params=[
            ("q", "cat"),
            ("diversity", "high"),
            ("diversity_depth", "2000"),
        ],
    )
    assert response.status_code == 200
    metadata = response.json()["diversity"]
    assert metadata["mode"] == "high"
    assert metadata["strength"] == pytest.approx(0.88)
    assert metadata["depth"] == "2000"
    # The fixture has three matching points, so the actual pool is smaller
    # than the requested depth.
    assert metadata["pool_depth"] == 3


def test_api_search_diversity_depth_auto_uses_mode_default(app_with_qdrant):
    response = app_with_qdrant.get(
        "/api/search?q=cat&diversity=high"
    )
    assert response.status_code == 200
    metadata = response.json()["diversity"]
    assert metadata["depth"] == "auto"
    assert metadata["pool_depth"] == 3


def test_api_search_rejects_unknown_diversity_depth(app_with_qdrant):
    response = app_with_qdrant.get(
        "/api/search?q=cat&diversity=balanced&diversity_depth=7500"
    )
    assert response.status_code == 400
    assert response.json()["code"] == "bad_request"


def test_api_search_accepts_depth_10000(app_with_qdrant):
    response = app_with_qdrant.get(
        "/api/search?q=cat&diversity=balanced&diversity_depth=10000"
    )
    assert response.status_code == 200
    assert response.json()["diversity"]["depth"] == "10000"


def test_api_search_legacy_diverse_alias_maps_to_balanced(app_with_qdrant):
    response = app_with_qdrant.get("/api/search?q=cat&diverse=true")
    assert response.status_code == 200
    data = response.json()
    assert data["diverse"] is True
    assert data["diversity"]["mode"] == "balanced"


def test_api_search_rejects_unknown_diversity_mode(app_with_qdrant):
    response = app_with_qdrant.get("/api/search?q=cat&diversity=random")
    assert response.status_code == 400
    assert response.json()["code"] == "bad_request"


def test_api_search_rejects_surprise_and_diversity_together(app_with_qdrant):
    response = app_with_qdrant.get(
        "/api/search?q=cat&diversity=balanced&surprise=true"
    )
    assert response.status_code == 400
    assert response.json()["code"] == "bad_request"


def test_api_search_positives_multi_value(app_with_qdrant):
    resp = app_with_qdrant.get(
        "/api/search",
        params=[("positives", "cat"), ("positives", "kitten")],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == ""
    assert data["positives"] == ["cat", "kitten"]
    assert data["results"][0]["id"] == CAT_ID


def test_api_search_q_unioned_into_positives(app_with_qdrant):
    resp = app_with_qdrant.get(
        "/api/search",
        params=[("q", "cat"), ("positives", "kitten")],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "cat"
    assert data["positives"] == ["kitten", "cat"]
    assert data["results"][0]["id"] == CAT_ID


def test_api_search_negatives_subtract(app_with_qdrant):
    resp = app_with_qdrant.get(
        "/api/search",
        params=[("positives", "cat"), ("negatives", "dog")],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["positives"] == ["cat"]
    assert data["negatives"] == ["dog"]
    assert data["results"][0]["id"] == CAT_ID


def test_api_search_negatives_only_returns_400(app_with_qdrant):
    resp = app_with_qdrant.get("/api/search?negatives=blurry")
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "bad_request"
    assert body["detail"] == "at least one positive prompt is required"


def test_api_search_all_empty_returns_400(app_with_qdrant):
    resp = app_with_qdrant.get("/api/search?q=%20%20&positives=")
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "bad_request"
    assert body["detail"] == "at least one positive prompt is required"


def test_api_search_response_echoes_prompts(app_with_qdrant):
    resp = app_with_qdrant.get(
        "/api/search",
        params=[
            ("positives", "cat"),
            ("positives", "kitten"),
            ("negatives", "blurry"),
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["positives"] == ["cat", "kitten"]
    assert data["negatives"] == ["blurry"]


def test_api_search_long_prompt_dropped(app_with_qdrant):
    long_prompt = "x" * 600
    resp = app_with_qdrant.get(
        "/api/search",
        params=[("positives", long_prompt), ("positives", "cat")],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["positives"] == ["cat"]
    assert data["results"][0]["id"] == CAT_ID


def test_api_search_prompts_in_url_preserved_on_paginate(app_with_qdrant):
    params = [
        ("positives", "cat"),
        ("negatives", "dog"),
        ("limit", "2"),
    ]
    first = app_with_qdrant.get("/api/search", params=params)
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["positives"] == ["cat"]
    assert first_data["negatives"] == ["dog"]
    assert first_data["has_more"] is True

    second = app_with_qdrant.get(
        "/api/search",
        params=params + [("offset", str(first_data["offset"] + len(first_data["results"])))],
    )
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["positives"] == ["cat"]
    assert second_data["negatives"] == ["dog"]
    assert second_data["offset"] == 2


def test_api_search_dedup_case_insensitive(app_with_qdrant):
    resp = app_with_qdrant.get(
        "/api/search",
        params=[("positives", "Cat"), ("positives", "cat")],
    )
    assert resp.status_code == 200
    assert resp.json()["positives"] == ["Cat"]


def test_api_search_cat_ranks_cat_first(app_with_qdrant):
    resp = app_with_qdrant.get("/api/search?q=cat")
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results, "expected at least one result"
    # The mock embedder returns the same vector for the same query,
    # so the first result MUST be the cat (id 'a' * 32).
    assert results[0]["id"] == CAT_ID


# ---------------- Layer 2: ?view= param ----------------


def test_api_search_view_default_is_grid(app_with_qdrant):
    """No `?view=` → response echoes the default ('grid')."""
    resp = app_with_qdrant.get("/api/search?q=cat")
    assert resp.status_code == 200
    assert resp.json()["view"] == "grid"


def test_api_search_view_feed_echoed(app_with_qdrant):
    """`?view=feed` → response echoes 'feed' (no error, just echoed)."""
    resp = app_with_qdrant.get("/api/search?q=cat&view=feed")
    assert resp.status_code == 200
    data = resp.json()
    assert data["view"] == "feed"
    # The data is the same — only the rendering is different.
    assert data["results"], "feed view should still return results"


def test_api_search_view_invalid_falls_back_to_grid(app_with_qdrant):
    """Unknown view values silently fall back to 'grid' (no 400).

    Reason: a malformed/old URL shouldn't block the search; the
    frontend can recover and the user gets a working page.
    """
    resp = app_with_qdrant.get("/api/search?q=cat&view=blah")
    assert resp.status_code == 200
    assert resp.json()["view"] == "grid"


def test_api_search_view_empty_string_falls_back_to_grid(app_with_qdrant):
    resp = app_with_qdrant.get("/api/search?q=cat&view=")
    assert resp.status_code == 200
    assert resp.json()["view"] == "grid"












def test_api_search_empty_query_returns_400(app_with_qdrant):
    resp = app_with_qdrant.get("/api/search?q=")
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "bad_request"


def test_api_search_missing_query_returns_400(app_with_qdrant):
    # With manual validation, missing q is treated as empty → 400.
    resp = app_with_qdrant.get("/api/search")
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


def test_api_search_limit_out_of_range(app_with_qdrant):
    # We use manual validation that returns 400 (not FastAPI's 422),
    # so the contract is "400 bad_request" for these.
    resp = app_with_qdrant.get("/api/search?q=cat&limit=0")
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"
    resp = app_with_qdrant.get("/api/search?q=cat&limit=99999")
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


def test_api_search_long_query_returns_400(app_with_qdrant):
    long_q = "x" * 600
    resp = app_with_qdrant.get(f"/api/search?q={long_q}")
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "bad_request"


def test_api_search_qdrant_unreachable(qdrant_in_memory, nas_base, monkeypatch):
    """Mock the qdrant client to raise ConnectionError on search."""
    from unittest.mock import MagicMock

    from search import app as app_mod
    from search.config import Config

    broken = MagicMock()
    broken.search.side_effect = ConnectionError("simulated")

    # healthz() is called at lifespan — make it not blow up the test
    broken.healthz.return_value = False

    cfg = Config(
        qdrant_url="memory://",
        qdrant_collection="images_test",
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
    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=broken)
    with TestClient(app) as client:
        resp = client.get("/api/search?q=cat")
    app_mod.reset_for_tests()
    assert resp.status_code == 502
    assert resp.json()["code"] == "qdrant_unreachable"


# ---------------- Layer 2: /photo/{id} ----------------




def test_get_photo_page_unknown_id_returns_404(app_with_qdrant):
    resp = app_with_qdrant.get("/photo/does-not-exist")
    assert resp.status_code == 404


def test_get_photo_raw_known_id(app_with_qdrant):
    resp = app_with_qdrant.get(f"/photo/{CAT_ID}/raw")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
    assert len(resp.content) > 0  # actual image bytes


def test_get_photo_raw_unknown_id_returns_404(app_with_qdrant):
    resp = app_with_qdrant.get("/photo/does-not-exist/raw")
    assert resp.status_code == 404


def test_get_photo_page_file_missing(app_with_qdrant):
    # 'c' * 32 is the car.jpg point, but car.jpg was NOT created on disk.
    # After the lazy-validation change, the /photo page 404s immediately
    # when the on-disk file is missing (cleaner UX than rendering the
    # detail page with "File not found" text in the middle).
    resp = app_with_qdrant.get(f"/photo/{CAR_ID}")
    assert resp.status_code == 404


def test_get_photo_raw_file_missing_returns_404(app_with_qdrant):
    resp = app_with_qdrant.get(f"/photo/{CAR_ID}/raw")
    assert resp.status_code == 404


# ---------------- Layer 2: "Most similar" feature ----------------




def test_photo_page_omits_similar_button_when_file_missing(app_with_qdrant):
    """When the file is missing on disk, the /photo page 404s entirely
    (lazy-validation change) — there's no detail page to render, so
    the similar-button question is moot. The raw-image route still
    404s too (see test_get_photo_raw_file_missing_returns_404)."""
    resp = app_with_qdrant.get(f"/photo/{CAR_ID}")
    assert resp.status_code == 404




def test_get_photo_similar_unknown_id_returns_404(app_with_qdrant):
    resp = app_with_qdrant.get("/photo/does-not-exist/similar")
    assert resp.status_code == 404










# ---------------- Layer 2: result stability ----------------


def test_api_search_result_stability(app_with_qdrant):
    r1 = app_with_qdrant.get("/api/search?q=cat").json()
    r2 = app_with_qdrant.get("/api/search?q=cat").json()
    assert [x["id"] for x in r1["results"]] == [x["id"] for x in r2["results"]]
    # Scores must match to a tight tolerance (the mock embedder is
    # deterministic, so identical inputs should yield bit-identical
    # cosine sims in principle, but allow tiny float drift).
    s1 = [x["score"] for x in r1["results"]]
    s2 = [x["score"] for x in r2["results"]]
    assert len(s1) == len(s2)
    for a, b in zip(s1, s2, strict=False):
        # Tolerate float drift across CPU/BLAS implementations. The
        # mock embedder is deterministic in principle, but reduction
        # order varies between x86 + OpenBLAS (local), GitHub Actions
        # x86, and ARM, producing ~1e-9 to ~1e-7 of drift even with
        # identical inputs. 1e-6 is still tight enough to catch any
        # real regression (a meaningful scoring change would move by
        # orders of magnitude, not micro-units).
        assert abs(a - b) < 1e-6, f"score drift: {a} vs {b}"


# ---------------- Layer 2: LRU cache ----------------


def test_api_search_lru_cache(app_with_qdrant):
    # /api/search now uses the multi-prompt path even for legacy q=.
    from search import text_encoder
    text_encoder.clear_cache_multi()
    assert text_encoder._embed_query_multi_cached.cache_info().currsize == 0

    r1 = app_with_qdrant.get("/api/search?q=cat").json()
    assert text_encoder._embed_query_multi_cached.cache_info().hits == 0
    assert text_encoder._embed_query_multi_cached.cache_info().misses == 1

    r2 = app_with_qdrant.get("/api/search?q=cat").json()
    assert text_encoder._embed_query_multi_cached.cache_info().hits == 1

    # took_ms is wall-clock and may differ — ignore it for equality.
    r1.pop("took_ms", None)
    r2.pop("took_ms", None)
    # Same ids + paths come back, with float32-tolerant score matching.
    assert [h["id"] for h in r1["results"]] == [h["id"] for h in r2["results"]]
    assert [h["path"] for h in r1["results"]] == [h["path"] for h in r2["results"]]
    for a, b in zip(r1["results"], r2["results"], strict=False):
        assert a["score"] == pytest.approx(b["score"], abs=1e-5)


def test_api_search_lru_cache_multi(app_with_qdrant):
    from search import text_encoder

    text_encoder.clear_cache_multi()
    assert text_encoder._embed_query_multi_cached.cache_info().currsize == 0

    app_with_qdrant.get(
        "/api/search",
        params=[("positives", "cat"), ("positives", "kitten"), ("negatives", "dog")],
    )
    info = text_encoder._embed_query_multi_cached.cache_info()
    assert info.hits == 0
    assert info.misses == 1

    app_with_qdrant.get(
        "/api/search",
        params=[("positives", "Cat"), ("positives", "kitten"), ("negatives", "dog")],
    )
    info = text_encoder._embed_query_multi_cached.cache_info()
    assert info.hits == 1
    assert info.misses == 1

    app_with_qdrant.get(
        "/api/search",
        params=[("positives", "cat"), ("positives", "kitten"), ("negatives", "car")],
    )
    info = text_encoder._embed_query_multi_cached.cache_info()
    assert info.hits == 1
    assert info.misses == 2


# ---------------- Layer 4: pagination ----------------


def test_api_search_offset_pagination(app_with_qdrant):
    """Offset returns the next page; has_more flips False on last page."""
    r1 = app_with_qdrant.get("/api/search?q=cat&limit=2").json()
    assert r1["limit"] == 2
    assert r1["offset"] == 0
    assert len(r1["results"]) == 2
    assert r1["has_more"] is True  # we inserted 3 points, so limit=2 -> has_more

    r2 = app_with_qdrant.get("/api/search?q=cat&limit=2&offset=2").json()
    assert r2["offset"] == 2
    assert r2["limit"] == 2
    assert len(r2["results"]) == 1  # only 1 point left after offset 2
    assert r2["has_more"] is False
    # Different page → no overlap with r1.
    r1_ids = {h["id"] for h in r1["results"]}
    r2_ids = {h["id"] for h in r2["results"]}
    assert r1_ids.isdisjoint(r2_ids)


def test_api_search_offset_beyond_results(app_with_qdrant):
    """Offset past the end returns empty + has_more=False (no error)."""
    r = app_with_qdrant.get("/api/search?q=cat&offset=1000").json()
    assert r["results"] == []
    assert r["has_more"] is False
    assert r["offset"] == 1000


def test_api_search_offset_beyond_max_total(app_with_qdrant):
    """Offset way past MAX_RESULTS_TOTAL is clamped to 0 results, no error.

    MAX_RESULTS_TOTAL is the defensive pagination ceiling (1M by default
    — see search/config.py) that catches runaway clients ignoring
    has_more. Real clients stop when has_more flips False; this test
    just confirms the ceiling itself is a clean clamp.
    """
    r = app_with_qdrant.get("/api/search?q=cat&offset=2_000_000&limit=50").json()
    assert r["results"] == []
    assert r["has_more"] is False


def test_api_search_offset_max_total_cap(app_with_qdrant):
    """Offset >= MAX_RESULTS_TOTAL returns empty + has_more=False.

    Default MAX_RESULTS_TOTAL is 1M (search/config.py). The cap is
    intentionally generous — the client is expected to honour has_more
    and stop paginating long before reaching it.
    """
    r = app_with_qdrant.get("/api/search?q=cat&offset=1_000_000&limit=50").json()
    assert r["results"] == []
    assert r["has_more"] is False


def test_api_search_offset_negative_rejected(app_with_qdrant):
    r = app_with_qdrant.get("/api/search?q=cat&offset=-1")
    assert r.status_code == 400
    assert r.json()["code"] == "bad_request"


def test_api_search_offset_non_integer_rejected(app_with_qdrant):
    # FastAPI/Pydantic validates `offset: int` at the Query level
    # and returns 422 before our manual int() conversion runs.
    r = app_with_qdrant.get("/api/search?q=cat&offset=foo")
    assert r.status_code == 422





# ---------------- Layer 5: cross-machine path prefix ----------------


def test_resolve_local_rewrites_prefix(tmp_path):
    """resolve_local replaces prefix with base for cross-machine lookups."""
    from search.image_resolver import resolve_local

    # Create a file at tmp_path/subdir/img.jpg to simulate the Linux mount
    dest = tmp_path / "subdir" / "img.jpg"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"fake image")

    # Simulate: payload has Z:\nas\subdir\img.jpg, search runs on Linux
    # with base=/mnt/nas, prefix=Z:\nas
    result = resolve_local(
        r"Z:\nas\subdir\img.jpg",
        base=str(tmp_path),
        prefix=r"Z:\nas",
    )
    assert result is not None
    assert result == dest


def test_resolve_local_prefix_no_match(tmp_path):
    """When payload path doesn't start with prefix, it's tried as-is."""
    from search.image_resolver import resolve_local

    dest = tmp_path / "img.jpg"
    dest.write_bytes(b"x")

    # Payload path doesn't match prefix → prefix is ignored, path tried directly
    result = resolve_local(
        str(dest),
        base="/mnt/nas",
        prefix=r"Z:\nas",
    )
    assert result is not None
    assert result == dest


def test_resolve_local_absolute_path_without_prefix(tmp_path):
    """Absolute payload path not starting with prefix → tried directly (same-machine)."""
    from search.image_resolver import resolve_local

    dest = tmp_path / "img.jpg"
    dest.write_bytes(b"x")

    result = resolve_local(str(dest), base="/mnt/nas", prefix="")
    assert result is not None
    assert result == dest



# ---------------- Layer 6: library filter (collections) ----------------


def test_api_collections_endpoint(app_with_qdrant):
    """/api/collections returns distinct library names + counts."""
    r = app_with_qdrant.get("/api/collections").json()
    assert "collections" in r
    # Test fixture inserts 3 points all with collection="general".
    by_name = {c["name"]: c["count"] for c in r["collections"]}
    assert by_name.get("general") == 3
    # Sorted by name (only one entry, but verify the key shape).
    for c in r["collections"]:
        assert "name" in c and "count" in c


def test_api_search_no_collection_filter_returns_all(app_with_qdrant):
    """No ?collection= param returns all 3 points (no library filter)."""
    r = app_with_qdrant.get("/api/search?q=cat").json()
    assert len(r["results"]) == 3
    assert r["has_more"] is False


def test_api_search_single_collection_filter(app_with_qdrant):
    """?collection=general returns only points in that library."""
    r = app_with_qdrant.get("/api/search?q=cat&collection=general").json()
    assert len(r["results"]) == 3
    assert r["has_more"] is False


def test_api_search_collection_filter_excludes(app_with_qdrant):
    """?collection=kpop returns nothing when no points are in kpop."""
    r = app_with_qdrant.get("/api/search?q=cat&collection=kpop").json()
    assert r["results"] == []
    assert r["has_more"] is False


def test_api_search_multi_collection_filter(qdrant_in_memory, nas_base):
    """
    Multi-value ?collection=kpop&collection=general returns points
    in EITHER library (MatchAny semantics, not intersection).
    """
    from search import app as app_mod
    from search.config import Config
    from lustre_kernel.registry import MockEmbedder; _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text

    # Re-create the app with a payload that has points in two
    # different collections.
    q_cat = _mock_embed("cat")
    items = [
        ("a" * 32, q_cat, {"id": "a" * 32, "path": str(nas_base / "a.jpg"), "collection": "kpop"}),
        ("b" * 32, q_cat, {"id": "b" * 32, "path": str(nas_base / "b.jpg"), "collection": "general"}),
        ("c" * 32, q_cat, {"id": "c" * 32, "path": str(nas_base / "c.jpg"), "collection": "portrait"}),
    ]
    upsert.ensure_collection(qdrant_in_memory.client, qdrant_in_memory.collection, dim=VECTOR_DIM)
    upsert.upsert_batch(qdrant_in_memory.client, qdrant_in_memory.collection, items, wait=True)

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
    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)
    with TestClient(app) as client:
        # kpop + general = 2 of 3 points.
        r = client.get("/api/search?q=cat&collection=kpop&collection=general").json()
        ids = {h["id"] for h in r["results"]}
        assert ids == {"a" * 32, "b" * 32}
        # All three: no filter.
        r2 = client.get("/api/search?q=cat").json()
        assert len(r2["results"]) == 3
    app_mod.reset_for_tests()


def test_api_collections_endpoint_distinct_counts(
    qdrant_in_memory, nas_base, monkeypatch,
):
    """/api/collections returns per-library counts when more than one is populated."""
    from search import app as app_mod
    from search.config import Config
    from lustre_kernel.registry import MockEmbedder; _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text

    q_cat = _mock_embed("cat")
    items = [
        ("a" * 32, q_cat, {"id": "a" * 32, "path": str(nas_base / "a.jpg"), "collection": "kpop"}),
        ("b" * 32, q_cat, {"id": "b" * 32, "path": str(nas_base / "b.jpg"), "collection": "kpop"}),
        ("c" * 32, q_cat, {"id": "c" * 32, "path": str(nas_base / "c.jpg"), "collection": "general"}),
    ]
    upsert.ensure_collection(qdrant_in_memory.client, qdrant_in_memory.collection, dim=VECTOR_DIM)
    upsert.upsert_batch(qdrant_in_memory.client, qdrant_in_memory.collection, items, wait=True)

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
    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)
    with TestClient(app) as client:
        r = client.get("/api/collections").json()
    app_mod.reset_for_tests()
    by_name = {c["name"]: c["count"] for c in r["collections"]}
    assert by_name == {"kpop": 2, "general": 1}


def test_api_collections_endpoint_uses_facet_aggregation(
    qdrant_in_memory, nas_base, monkeypatch,
):
    """/api/collections calls `client.facet()` instead of scrolling.

    Locks in the optimization: if anyone reverts to the scroll loop,
    this test catches it. The library/code split is the same as the
    distinct-counts test, so the assertion is the same — the value
    is that we verified `facet()` was the actual call site.
    """
    from search import app as app_mod
    from search.config import Config
    from lustre_kernel.registry import MockEmbedder; _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text

    q_cat = _mock_embed("cat")
    items = [
        ("a" * 32, q_cat, {"id": "a" * 32, "path": str(nas_base / "a.jpg"), "collection": "kpop"}),
        ("b" * 32, q_cat, {"id": "b" * 32, "path": str(nas_base / "b.jpg"), "collection": "kpop"}),
        ("c" * 32, q_cat, {"id": "c" * 32, "path": str(nas_base / "c.jpg"), "collection": "general"}),
    ]
    upsert.ensure_collection(qdrant_in_memory.client, qdrant_in_memory.collection, dim=VECTOR_DIM)
    upsert.upsert_batch(qdrant_in_memory.client, qdrant_in_memory.collection, items, wait=True)

    # Wrap the facet method so we can assert it was called.
    real_facet = qdrant_in_memory.client.facet
    calls = {"n": 0, "kwargs": None}
    def spy_facet(*args, **kwargs):
        calls["n"] += 1
        calls["kwargs"] = kwargs
        return real_facet(*args, **kwargs)
    monkeypatch.setattr(qdrant_in_memory.client, "facet", spy_facet)

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
    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)
    with TestClient(app) as client:
        r = client.get("/api/collections").json()
    app_mod.reset_for_tests()

    # The optimization is in effect: facet() was the call site, not scroll().
    assert calls["n"] == 1, "expected exactly one facet() call per /api/collections request"
    assert calls["kwargs"].get("key") == "collection"
    assert calls["kwargs"].get("limit", 0) >= 10  # we ask for 100, accept >= 10

    by_name = {c["name"]: c["count"] for c in r["collections"]}
    assert by_name == {"kpop": 2, "general": 1}
    # And the list is sorted by name — the chip filter UI relies on
    # stable order between page loads.
    assert [c["name"] for c in r["collections"]] == ["general", "kpop"]


def test_api_collections_endpoint_skips_empty_collection_values(
    qdrant_in_memory, nas_base, monkeypatch,
):
    """Points with `collection=""` must not show up as a fake library.

    The old scroll-based code already filtered these. Lock it in.
    """
    from search import app as app_mod
    from search.config import Config
    from lustre_kernel.registry import MockEmbedder; _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text

    q_cat = _mock_embed("cat")
    items = [
        ("a" * 32, q_cat, {"id": "a" * 32, "path": str(nas_base / "a.jpg"), "collection": "kpop"}),
        ("b" * 32, q_cat, {"id": "b" * 32, "path": str(nas_base / "b.jpg"), "collection": ""}),
    ]
    upsert.ensure_collection(qdrant_in_memory.client, qdrant_in_memory.collection, dim=VECTOR_DIM)
    upsert.upsert_batch(qdrant_in_memory.client, qdrant_in_memory.collection, items, wait=True)

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
    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)
    with TestClient(app) as client:
        r = client.get("/api/collections").json()
    app_mod.reset_for_tests()

    names = [c["name"] for c in r["collections"]]
    assert "kpop" in names
    assert "" not in names
    # And the empty-collection point didn't sneak into the kpop count.
    assert next(c["count"] for c in r["collections"] if c["name"] == "kpop") == 1


# ---------------- Static asset caching ----------------




def test_non_static_responses_have_no_cache_header(app_with_qdrant):
    """The no-cache middleware should only touch /static/* paths.
    API and HTML responses should be unaffected.
    """
    resp = app_with_qdrant.get("/api/search?q=cat")
    # The middleware shouldn't add a Cache-Control on non-static paths.
    assert "no-cache" not in (resp.headers.get("cache-control") or "")


# ---------------- Text query normalization ----------------


def test_normalize_query_for_siglip2_lowercases():
    """
    Regression test for the case-sensitivity bug. SigLIP2's text
    encoder was trained on lowercased text; the open_clip tokenizer
    does NOT lowercase automatically. Without this, a query like
    "Cat" produces a degraded embedding whose cosine similarity to
    image vectors is ~0.1-0.15 vs. the scorer's expected ~0.4-0.6
    (isaac-image-scoring does the same lowercase at
    aesthetic_scorer.py:230).
    """
    from search.text_encoder import _normalize_query_for_siglip2

    # Mixed case: lowercased.
    assert _normalize_query_for_siglip2("HELLO World") == "hello world"
    assert _normalize_query_for_siglip2("Cat Photo") == "cat photo"
    # Already lowercase: no-op.
    assert _normalize_query_for_siglip2("hello world") == "hello world"
    # Punctuation preserved, just case-folded.
    assert _normalize_query_for_siglip2("Hello, World!") == "hello, world!"
    # Empty string: returns empty.
    assert _normalize_query_for_siglip2("") == ""
    # Single character: trivial.
    assert _normalize_query_for_siglip2("A") == "a"


def test_normalize_query_is_idempotent():
    """Lowercasing is idempotent (calling it twice = once)."""
    from search.text_encoder import _normalize_query_for_siglip2

    text = "Mixed CASE"
    once = _normalize_query_for_siglip2(text)
    twice = _normalize_query_for_siglip2(once)
    assert once == twice == "mixed case"



# ---------------- ETag distinctness ----------------


def test_api_search_etag_distinct_for_filename_filter(app_with_qdrant):
    """Round-1 fix: ETag must include `filename` so toggling the
    filename filter never returns a 304 with stale results from
    the unfiltered query."""
    client = app_with_qdrant
    no_filter = client.get("/api/search?q=cat&limit=4")
    # `cat` is a valid substring token in FTS5; lands the same
    # rows without 400-ing the request.
    with_filter = client.get("/api/search?q=cat&limit=4&filename=cat")
    assert no_filter.status_code == 200, no_filter.text
    assert with_filter.status_code == 200, with_filter.text
    no_filter_etag = no_filter.headers.get("ETag")
    with_filter_etag = with_filter.headers.get("ETag")
    assert no_filter_etag is not None
    assert with_filter_etag is not None
    assert no_filter_etag != with_filter_etag, (
        "Same ETag for different filename filters — stale 304 risk"
    )


def test_api_search_etag_distinct_for_diversity(app_with_qdrant):
    """Round-1 fix: ETag must include diversity mode/strength/depth."""
    client = app_with_qdrant
    off = client.get("/api/search?q=cat&limit=4")
    on = client.get("/api/search?q=cat&limit=4&diversity=balanced")
    assert off.status_code == 200, off.text
    assert on.status_code == 200, on.text
    assert off.headers["ETag"] != on.headers["ETag"], (
        "Same ETag for diversity off vs balanced — stale 304 risk"
    )


def test_api_search_etag_distinct_for_collections(app_with_qdrant):
    """Round-1 fix: ETag must include the collections filter."""
    client = app_with_qdrant
    no_coll = client.get("/api/search?q=cat&limit=4")
    with_coll = client.get(
        "/api/search?q=cat&limit=4&collection=general"
    )
    assert no_coll.status_code == 200, no_coll.text
    # The test fixture seeds items with collection="general", so
    # this filter returns the same rows but the ETag must differ.
    assert with_coll.status_code == 200, with_coll.text
    no_coll_etag = no_coll.headers.get("ETag")
    with_coll_etag = with_coll.headers.get("ETag")
    assert no_coll_etag is not None
    assert with_coll_etag is not None
    assert no_coll_etag != with_coll_etag


def test_api_search_etag_stable_for_identical_requests(app_with_qdrant):
    """Round-1 sanity: identical requests still hash identically
    so the 304 fast path keeps working."""
    client = app_with_qdrant
    a = client.get("/api/search?q=cat&limit=4")
    b = client.get("/api/search?q=cat&limit=4")
    assert a.status_code == 200
    assert b.status_code == 200
    assert a.headers["ETag"] == b.headers["ETag"]
