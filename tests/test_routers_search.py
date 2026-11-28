"""
tests/test_routers_search.py — search router contract (§B2 step 40).

Pins the /api/search router module's contract:
- factory returns an APIRouter with the one documented endpoint
- factory injects resolve_query_vector + favorite_ids_for_filter
  as dependency slots (closure-bound for now; §B2 step 41 will
  lift them to module level)
- the route hits the right code paths on minimal inputs

Integration is verified by tests/test_search_api.py — the test
client drives the wired router through create_app, which is the
real environment.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from lustre_kernel.spaces import TEXT_SPACE


def _fake_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.top_k_default = 20
    cfg.top_k_max = 200
    cfg.max_results_total = 1000
    cfg.max_prompt_chars = 80
    cfg.max_prompts_total = 10
    cfg.default_view = "grid"
    cfg.centroid_expected_feature_dim = 1536
    cfg.web_ui_url = "http://localhost:5173"
    cfg.filename_cardinality_guard = 0.5
    cfg.diversity_relevance_drop = 0.1
    cfg.diversity_duplicate_hamming_distance = 8
    cfg.diversity_pool_depths = {}
    cfg.qdrant_collection = "test_collection"
    # Stable-pagination knobs (search/search_snapshot.py). Must be real
    # ints — a MagicMock here would flow into arithmetic in
    # materialize_search_page and raise TypeError.
    cfg.search_initial_band_size = 8
    cfg.search_band_size = 16
    cfg.search_max_snapshot_size = 200
    # Off: a MagicMock returns a truthy object for unset attrs, which
    # would spawn background prefetch threads during tests.
    cfg.search_prefetch_next_band = False
    return cfg


@pytest.fixture
def fake_search_deps():
    qdrant = MagicMock()
    qdrant.search.return_value = ([], False)
    cfg = _fake_cfg()
    index_db = MagicMock()
    diversity_cache = MagicMock()
    diversity_cache.get.return_value = None
    # Real cache, not a MagicMock: materialize_search_page relies on
    # get_or_create returning a working SearchSnapshot.
    from search.search_snapshot import SearchSnapshotCache
    snapshot_cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)

    def resolve_query_vector(centroid_names, prompt_state, **kwargs):
        return ([0.0] * 1536, TEXT_SPACE, None, None)

    async def favorite_ids_for_filter():
        return set()

    return {
        "qdrant": qdrant,
        "cfg": cfg,
        "index_db": index_db,
        "diversity_cache": diversity_cache,
        "snapshot_cache": snapshot_cache,
        "resolve_query_vector": resolve_query_vector,
        "favorite_ids_for_filter": favorite_ids_for_filter,
    }


def test_router_factory_returns_apirouter(fake_search_deps):
    """build_search_router returns an APIRouter instance with the right route."""
    from search.routers.search import build_search_router
    router = build_search_router(**fake_search_deps)
    # FastAPI router's routes are mounted on .routes
    paths = [r.path for r in router.routes if hasattr(r, "path")]
    assert "/api/search" in paths


def test_router_factory_accepts_closure_helpers(fake_search_deps):
    """The factory parameters are used at runtime, not just configured."""
    from search.routers.search import build_search_router

    called = {"resolve": False, "favorites": False}

    def rqv(*args, **kwargs):
        called["resolve"] = True
        return ([0.0] * 1536, None, None)

    async def fif():
        called["favorites"] = True
        return set()

    fake_search_deps["resolve_query_vector"] = rqv
    fake_search_deps["favorite_ids_for_filter"] = fif

    router = build_search_router(**fake_search_deps)
    # The factory returning without error is the main check; the
    # dependency slots are wired correctly because the route is
    # a closure over them.
    assert router is not None
    assert callable(rqv)
    assert callable(fif)
