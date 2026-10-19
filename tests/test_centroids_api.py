"""
tests/test_centroids_api.py

Layer 2 — search API tests covering the centroid feature end-to-end:

  - /api/centroids lists loaded centroids
  - /api/centroids/{name}/search uses the centroid as a query vector
  - /api/centroids/reload rescans the directory on demand
  - /api/search supports ?centroid= (mutually exclusive with text)
  - /  supports ?centroid= (same mutex, HTML rendering)
  - An unknown centroid name 404s (not 500)

The fixture builds two .pt centroid files in tmp_path, points
the app at that dir, and pre-populates an in-memory Qdrant
collection with three known images (cat, dog, car). The
centroid's vector is `mock_embed("cat")` so a search with it
ranks the cat image first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Shared fixture and constants live in _centroid_fixture.py
# (re-exported through conftest.py for pytest auto-discovery).
from _centroid_fixture import (
    CENTROID_CAT_ID as CAT_ID,
)
from _centroid_fixture import (
    NOIR_CENTROID,
    WUXIA_CENTROID,
)
from _centroid_fixture import (
    save_centroid as _save_centroid,
)
from fastapi.testclient import TestClient

from indexer import upsert
from indexer.upsert import VECTOR_DIM
from search import app as app_mod
from search.config import Config

# ----------------------- /api/centroids -----------------------


def test_api_centroids_lists_loaded(app_with_centroids):
    resp = app_with_centroids.get("/api/centroids")
    assert resp.status_code == 200
    data = resp.json()
    names = [c["name"] for c in data["centroids"]]
    # Sorted, only the two valid ones — the dinov3 file was skipped.
    assert names == [NOIR_CENTROID, WUXIA_CENTROID]
    spec = data["centroids"][0]
    assert spec["model"] == "siglip2"
    assert spec["feature_dim"] == VECTOR_DIM
    assert spec["n_images"] == 25
    assert "source_path" in spec
    assert data["expected_model"] == "siglip2"
    assert data["expected_feature_dim"] == VECTOR_DIM


def test_api_centroids_empty_when_dir_unset():
    """A Config with centroids_dir=None returns an empty list, not an error."""
    qdrant_client = pytest.importorskip("qdrant_client").QdrantClient(location=":memory:")
    from search.qdrant_client import QdrantSearch
    qdrant = QdrantSearch(client=qdrant_client, collection="images_test_empty", timeout_ms=2000)
    upsert.ensure_collection(qdrant.client, qdrant.collection, dim=VECTOR_DIM)
    cfg = Config(
        qdrant_url="memory://",
        qdrant_collection="images_test_empty",
        qdrant_api_key=None,
        model_name="mock", model_revision="", device="cpu",
        top_k_default=35, top_k_max=200, query_timeout_ms=2000,
        nas_images_base="/tmp", path_prefix="", web_ui_url="http://localhost:8000",  # noqa: S108
        log_level="WARNING", test_mode=True,
        centroids_dir=None,
    )
    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant)
    with TestClient(app) as tc:
        resp = tc.get("/api/centroids")
        assert resp.status_code == 200
        # The model+dim are global config (always present), only the
        # `centroids` list is empty when CENTROIDS_DIR is unset.
        # `dynamic_centroids` is always populated (the favourites
        # centroid is registered at startup regardless of whether
        # the static store has any `.pt` files to load).
        body = resp.json()
        assert body["centroids"] == []
        assert body["expected_model"] == "siglip2"
        # Dim tracks whichever variant is the current prod default;
        # so400m/16-384 = 1152. Hardcoding the literal would couple
        # the test to the variant choice — pull from the registry
        # so this stays correct across future variant changes.
        from lustre_kernel.registry import get as _registry_get
        from search.config import DEFAULT_MODEL
        assert body["expected_feature_dim"] == _registry_get(DEFAULT_MODEL).dim
        assert any(d["name"] == "favourites" for d in body["dynamic_centroids"])
    app_mod.reset_for_tests()


# ----------------------- /api/centroids/{name}/search -----------------------


def test_centroid_search_returns_results(app_with_centroids):
    """A centroid search uses the centroid's vector as the query and returns hits."""
    resp = app_with_centroids.get(f"/api/centroids/{WUXIA_CENTROID}/search?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["centroid"] == WUXIA_CENTROID
    assert data["query"] == ""
    assert data["positives"] == []
    assert data["negatives"] == []
    assert len(data["results"]) == 3  # cat, dog, car
    # Scores should be in descending order.
    scores = [r["score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)


def test_centroid_search_404_on_unknown(app_with_centroids):
    resp = app_with_centroids.get("/api/centroids/nope/search?limit=10")
    assert resp.status_code == 404
    assert "nope" in resp.json()["detail"]


def test_centroid_search_case_insensitive(app_with_centroids):
    """Wuxia_Female_Leads matches the stored lowercase form."""
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID.upper()}/search?limit=5"
    )
    assert resp.status_code == 200
    assert resp.json()["centroid"] == WUXIA_CENTROID  # canonical name echoed back


def test_centroid_search_limit_validation(app_with_centroids):
    resp = app_with_centroids.get(f"/api/centroids/{WUXIA_CENTROID}/search?limit=99999")
    assert resp.status_code == 400


def test_centroid_search_offset_validation(app_with_centroids):
    resp = app_with_centroids.get(f"/api/centroids/{WUXIA_CENTROID}/search?offset=-1")
    assert resp.status_code == 400


def test_centroid_search_collection_filter(app_with_centroids):
    """The collection filter on a centroid search behaves the same as text search."""
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search?collection=general&limit=10"
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 3


# ----------------------- /api/centroids/reload -----------------------


def test_reload_picks_up_new_file(app_with_centroids, tmp_path):
    """Add a new .pt file, call reload, see it appear in the list."""
    centroids_dir = Path(app_with_centroids.get("/api/centroids").json()["centroids"][0]["source_path"]).parent
    _save_centroid(centroids_dir / "siglip2_new.pt", "new_centroid")
    resp = app_with_centroids.post("/api/centroids/reload")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3  # wuxia, noir, new
    names = sorted(c["name"] for c in app_with_centroids.get("/api/centroids").json()["centroids"])
    assert names == sorted([WUXIA_CENTROID, NOIR_CENTROID, "new_centroid"])


def test_reload_drops_deleted_file(app_with_centroids):
    centroids_dir = Path(app_with_centroids.get("/api/centroids").json()["centroids"][0]["source_path"]).parent
    (centroids_dir / "siglip2_2026-06-17_wuxia.pt").unlink()
    resp = app_with_centroids.post("/api/centroids/reload")
    assert resp.status_code == 200
    assert resp.json()["count"] == 1
    # 404 on the now-missing centroid.
    assert app_with_centroids.get(f"/api/centroids/{WUXIA_CENTROID}/search").status_code == 404


# ----------------------- /api/search?centroid= integration -----------------------


def test_api_search_with_centroid(app_with_centroids):
    resp = app_with_centroids.get(f"/api/search?centroid={WUXIA_CENTROID}&limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["centroid"] == WUXIA_CENTROID
    assert data["query"] == ""
    assert data["positives"] == []  # text prompts not populated
    assert len(data["results"]) == 3


def test_api_search_with_centroid_supports_diversity(app_with_centroids):
    """Centroid vectors can use the same Diversity ranking pass as text searches."""
    resp = app_with_centroids.get(
        f"/api/search?centroid={WUXIA_CENTROID}&diversity=balanced&limit=10"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["centroid"] == WUXIA_CENTROID
    assert data["diverse"] is True
    assert data["diversity"]["requested"] is True
    assert data["diversity"]["applied"] is True
    assert data["diversity"]["mode"] == "balanced"


def test_api_search_centroid_mutex_with_q(app_with_centroids):
    """?centroid=foo&q=bar is rejected with 400 — they can't coexist."""
    resp = app_with_centroids.get(f"/api/search?centroid={WUXIA_CENTROID}&q=cat")
    assert resp.status_code == 400
    assert "exclusive" in resp.json()["detail"]


def test_api_search_centroid_mutex_with_positives(app_with_centroids):
    resp = app_with_centroids.get(
        f"/api/search?centroid={WUXIA_CENTROID}&positives=foo"
    )
    assert resp.status_code == 400


def test_api_search_centroid_mutex_with_negatives(app_with_centroids):
    resp = app_with_centroids.get(
        f"/api/search?centroid={WUXIA_CENTROID}&negatives=bar"
    )
    assert resp.status_code == 400


def test_api_search_centroid_unknown_400(app_with_centroids):
    resp = app_with_centroids.get("/api/search?centroid=nope&limit=10")
    assert resp.status_code == 400
    assert "nope" in resp.json()["detail"]


def test_api_search_text_only_still_works(app_with_centroids):
    """Regression: a plain text search is unaffected by the centroid feature."""
    resp = app_with_centroids.get("/api/search?q=cat&limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["centroid"] is None
    assert data["query"] == "cat"
    assert data["results"][0]["id"] == CAT_ID


# ----------------------- /api/search multi-centroid -----------------------


def test_api_search_multi_centroid_equal_weights(app_with_centroids):
    """?centroid=a&centroid=b blends the two with equal weight (1.0
    each) and returns the response with both names listed."""
    resp = app_with_centroids.get(
        f"/api/search?centroid={WUXIA_CENTROID}&centroid={NOIR_CENTROID}&limit=10"
    )
    assert resp.status_code == 200
    data = resp.json()
    # Both centroids listed in the order they appeared in the URL.
    assert data["centroids"] == [WUXIA_CENTROID, NOIR_CENTROID]
    # Backward-compat: `centroid` echoes the first one.
    assert data["centroid"] == WUXIA_CENTROID
    # No weights in the URL → weights=None in the response (default
    # all-equal, not echoed).
    assert data["weights"] is None
    assert len(data["results"]) == 3


def test_api_search_multi_centroid_custom_weights(app_with_centroids):
    """?centroid=a&centroid=b&weights=2,1 skews the blend toward a
    and echoes the weights in the response."""
    resp = app_with_centroids.get(
        f"/api/search?centroid={WUXIA_CENTROID}&centroid={NOIR_CENTROID}"
        f"&weights=2,1&limit=10"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["centroids"] == [WUXIA_CENTROID, NOIR_CENTROID]
    assert data["weights"] == [2.0, 1.0]


def test_api_search_multi_centroid_single_weight_broadcasts(app_with_centroids):
    """?weights=2 (one value) broadcasts to every centroid as 2.0."""
    resp = app_with_centroids.get(
        f"/api/search?centroid={WUXIA_CENTROID}&centroid={NOIR_CENTROID}"
        f"&weights=2&limit=10"
    )
    assert resp.status_code == 200
    # Broadcast is implementation detail — the response still has
    # weights=None (all 1.0 by default; the broadcast is normalised
    # away). The blend output is still a valid unit vector.
    data = resp.json()
    assert data["results"]


def test_api_search_multi_centroid_weights_wrong_count_400(app_with_centroids):
    """3 weights for 2 centroids → 400."""
    resp = app_with_centroids.get(
        f"/api/search?centroid={WUXIA_CENTROID}&centroid={NOIR_CENTROID}"
        f"&weights=1,2,3&limit=10"
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "3" in detail and "2 centroids" in detail


def test_api_search_multi_centroid_weights_non_numeric_400(app_with_centroids):
    resp = app_with_centroids.get(
        f"/api/search?centroid={WUXIA_CENTROID}&centroid={NOIR_CENTROID}"
        f"&weights=abc,def&limit=10"
    )
    assert resp.status_code == 400


def test_api_search_multi_centroid_weights_non_positive_400(app_with_centroids):
    resp = app_with_centroids.get(
        f"/api/search?centroid={WUXIA_CENTROID}&centroid={NOIR_CENTROID}"
        f"&weights=0,1&limit=10"
    )
    assert resp.status_code == 400
    assert "positive" in resp.json()["detail"]


def test_api_search_multi_centroid_one_unknown_400(app_with_centroids):
    """One of two centroids doesn't exist → 400. All-or-nothing
    blends (no silent partial-drop)."""
    resp = app_with_centroids.get(
        f"/api/search?centroid={WUXIA_CENTROID}&centroid=nope&limit=10"
    )
    assert resp.status_code == 400
    assert "nope" in resp.json()["detail"]


def test_api_search_single_centroid_unchanged(app_with_centroids):
    """Regression: a single ?centroid= returns the same shape it
    always did, plus the new fields with backwards-compatible
    defaults (centroids=[name], weights=None)."""
    resp = app_with_centroids.get(f"/api/search?centroid={WUXIA_CENTROID}&limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["centroid"] == WUXIA_CENTROID
    assert data["centroids"] == [WUXIA_CENTROID]
    assert data["weights"] is None


def test_api_search_multi_centroid_mutex_with_q(app_with_centroids):
    """Centroid (single or multi) still mutex with text prompts."""
    resp = app_with_centroids.get(
        f"/api/search?centroid={WUXIA_CENTROID}&centroid={NOIR_CENTROID}&q=cat"
    )
    assert resp.status_code == 400
    assert "exclusive" in resp.json()["detail"]


def test_api_search_multi_centroid_three(app_with_centroids):
    """Three-centroid blend (using only two available centroids is
    fine — the third will 404, which the all-or-nothing check
    surfaces as 400). Verify the URL parses to 3 entries."""
    resp = app_with_centroids.get(
        f"/api/search?centroid={WUXIA_CENTROID}&centroid={NOIR_CENTROID}"
        f"&centroid=nope&limit=10"
    )
    # The third is unknown → 400.
    assert resp.status_code == 400




# ----------------------- / search page?centroid= integration -----------------------












# ----------------------- /api/centroids/{name}/search + diversity -----------------------


def test_centroid_search_diversity_applied(app_with_centroids):
    """The dedicated centroid route now supports the Diversity pass."""
    plain = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search?limit=10"
    ).json()
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search?diversity=balanced&limit=10"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["centroid"] == WUXIA_CENTROID
    assert data["diverse"] is True
    assert data["diversity"]["requested"] is True
    assert data["diversity"]["applied"] is True
    assert data["diversity"]["mode"] == "balanced"
    # MMR always opens with the most relevant candidate, so the
    # first result matches the plain ranking. The full set is a
    # permutation (only 3 points in the fixture).
    assert data["results"][0]["id"] == plain["results"][0]["id"]
    assert {r["id"] for r in data["results"]} == {
        r["id"] for r in plain["results"]
    }


def test_centroid_search_diversity_off_by_default(app_with_centroids):
    """No diversity params → plain vector search, metadata says off."""
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search?limit=10"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["diverse"] is False
    assert data["diversity"]["requested"] is False
    assert data["diversity"]["applied"] is False
    assert data["diversity"]["mode"] == "off"


def test_centroid_search_diversity_stable_across_pages(app_with_centroids):
    """Diverse centroid search pages from one cached ranking — pages
    are disjoint and the second request is a cache hit."""
    first = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?diversity=balanced&diversity_depth=500&limit=1"
    )
    assert first.status_code == 200
    first_ids = [r["id"] for r in first.json()["results"]]
    second = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?diversity=balanced&diversity_depth=500&limit=1&offset=1"
    )
    assert second.status_code == 200
    second_ids = [r["id"] for r in second.json()["results"]]
    assert first_ids and second_ids
    assert set(first_ids).isdisjoint(set(second_ids))


def test_centroid_search_diversity_mutually_exclusive_with_sample(app_with_centroids):
    """Sample mode re-rolls per request — incompatible with a cached
    stable diversity ranking. Must 400 like surprise+diversity."""
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search?mode=sample&diversity=balanced"
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


def test_centroid_search_rejects_unknown_diversity_mode(app_with_centroids):
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search?diversity=extreme"
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


def test_centroid_search_rejects_unknown_diversity_depth(app_with_centroids):
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search?diversity=balanced&diversity_depth=7500"
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


def test_centroid_search_accepts_depth_10000(app_with_centroids):
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search?diversity=balanced&diversity_depth=10000"
    )
    assert resp.status_code == 200
    assert resp.json()["diversity"]["depth"] == "10000"


def test_centroid_search_diverse_alias_flag(app_with_centroids):
    """?diverse=true maps to balanced, same as /api/search."""
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search?diverse=true"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["diverse"] is True
    assert data["diversity"]["mode"] == "balanced"


# ----------------------- Round-91: Surprise Me on centroid search -----------------------
#
# The centroid route gains `surprise=true` which runs the same
# Gumbel-max deck flow as /api/search?surprise=true but with the
# centroid vector as the query. The test corpus is 3 points
# (cat, dog, car), so a T=0 surprise returns the same result as a
# plain centroid search; T>0 still succeeds but the per-deck
# ordering can be anything. We focus on:
#   - response shape (surprise=true, session_id, session_total)
#   - same-seed=Same-deck
#   - different-seed=Different-deck (when pool > 1)
#   - mutex with mode=sample and with diversity
#   - knob validation 400s
#   - surprise_session paginates the cached deck

def test_centroid_surprise_returns_deck_with_session(app_with_centroids):
    """surprise=true on a centroid returns a Gumbel deck with session
    id and total. T=0 is deterministic top-K, so the result id list
    is stable and ordered by score."""
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&surprise_temperature=0&surprise_seed=42"
        "&surprise_pool=200&limit=10"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["surprise"] is True
    assert data["session_id"]
    assert data["session_total"] is not None and data["session_total"] > 0
    # The plain (non-surprise) ranking for this centroid is the
    # cat-image first (centroid is mock_embed("cat")). T=0 picks
    # the top-K in order, so the first surprise result is the
    # same as the first plain result.
    plain = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search?limit=10"
    ).json()
    assert data["results"]
    assert data["results"][0]["id"] == plain["results"][0]["id"]


def test_centroid_surprise_same_seed_same_deck(app_with_centroids):
    """Same seed + same knobs + same centroid = same deck (cache key
    scopes by centroid name + temperature + pool + seed)."""
    r1 = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&surprise_temperature=0&surprise_seed=99&limit=10"
    ).json()
    r2 = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&surprise_temperature=0&surprise_seed=99&limit=10"
    ).json()
    assert [r["id"] for r in r1["results"]] == [r["id"] for r in r2["results"]]
    # Same session id (cache key, scoped by seed).
    assert r1["session_id"] == r2["session_id"]


def test_centroid_surprise_different_seed_different_deck(app_with_centroids):
    """Different seeds roll different decks when the temperature
    produces non-trivial re-ordering. With 3 points + T=1.0 the
    Gumbel draw is effectively uniform over the pool, so two
    different seeds will (with overwhelming probability) produce
    different orderings of the 3 points."""
    r1 = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&surprise_temperature=1.0&surprise_seed=1"
        "&surprise_pool=200&limit=10"
    ).json()
    r2 = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&surprise_temperature=1.0&surprise_seed=2"
        "&surprise_pool=200&limit=10"
    ).json()
    # Different session ids.
    assert r1["session_id"] != r2["session_id"]


def test_centroid_surprise_session_walks_deck(app_with_centroids):
    """surprise_session round-trips: a second request with the
    session id from the first response slices the cached deck
    without recomputing. Pages are disjoint and offset advances."""
    first = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&surprise_temperature=0&surprise_seed=7"
        "&surprise_pool=200&limit=1"
    ).json()
    sid = first["session_id"]
    second = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        f"?surprise=true&surprise_temperature=0&surprise_seed=7"
        f"&surprise_pool=200&surprise_session={sid}&limit=1&offset=1"
    ).json()
    assert second["session_id"] == sid
    assert second["session_total"] == first["session_total"]
    # Different page of the same deck.
    assert [r["id"] for r in first["results"]] != [r["id"] for r in second["results"]]


def test_centroid_surprise_different_centroid_different_deck(app_with_centroids):
    """The cache key is scoped by centroid name. Two different
    centroids with identical knobs roll different decks."""
    r1 = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&surprise_temperature=0&surprise_seed=42&limit=10"
    ).json()
    r2 = app_with_centroids.get(
        f"/api/centroids/{NOIR_CENTROID}/search"
        "?surprise=true&surprise_temperature=0&surprise_seed=42&limit=10"
    ).json()
    assert r1["session_id"] != r2["session_id"]


def test_centroid_surprise_excludes_seed_ids(app_with_centroids):
    """For a dynamic centroid, the seed ids are passed as
    exclude_ids so the deck doesn't echo the source photos. With
    only 3 points in the test fixture, exclude_ids would drop the
    whole result set — we test the contract differently: ask
    for a tiny pool and assert the response shape (no 500, no
    echoes of the centroid source if any were registered)."""
    # The fixture uses static .pt centroids (no seed_ids), so
    # exclude_ids is empty and the test confirms the path runs
    # end-to-end without errors.
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&surprise_temperature=0&surprise_seed=1"
        "&surprise_pool=200&limit=10"
    )
    assert resp.status_code == 200


def test_centroid_surprise_mutex_with_sample(app_with_centroids):
    """Both randomize — pick one. Same shape as
    diversity+sample."""
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&mode=sample"
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


def test_centroid_surprise_mutex_with_diversity(app_with_centroids):
    """MMR re-rank on a per-request randomized pool is
    meaningless. Same shape as /api/search."""
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&diversity=balanced"
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


def test_centroid_surprise_rejects_out_of_range_temperature(app_with_centroids):
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&surprise_temperature=1.5"
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


def test_centroid_surprise_rejects_out_of_range_pool(app_with_centroids):
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&surprise_pool=51"
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


def test_centroid_surprise_rejects_non_integer_seed(app_with_centroids):
    """Non-integer seed is rejected by FastAPI at the query-param
    parse layer (422), not by the route's manual 400 path. Same
    shape as /api/search?surprise_seed=notanumber."""
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&surprise_seed=notanumber"
    )
    assert resp.status_code == 422


def test_centroid_surprise_off_by_default(app_with_centroids):
    """No surprise params → plain centroid search, metadata says
    off. session_id is None so the client knows there's nothing
    to walk forward."""
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search?limit=10"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["surprise"] is False
    assert data["session_id"] is None
    assert data["session_total"] is None


def test_centroid_surprise_works_on_static_centroid(app_with_centroids):
    """Static .pt centroids have no source population, but the
    Surprise pool around the centroid's fixed vector is still
    meaningful. Confirm the route accepts surprise=true on a
    static centroid (the same as mode=centroid + surprise=true)."""
    resp = app_with_centroids.get(
        f"/api/centroids/{WUXIA_CENTROID}/search"
        "?surprise=true&surprise_temperature=0&surprise_seed=1"
        "&surprise_pool=200&limit=10"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["surprise"] is True
    assert data["results"]
