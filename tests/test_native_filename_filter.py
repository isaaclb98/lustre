"""
tests/test_native_filename_filter.py

Tests for the native (Qdrant text-index) `?filename=` filter — the
replacement for the SQLite FTS5 id-list path.

  1. `parse_filename_pattern` — the shared pattern parser: exact vs
     prefix, the accepted character set, and the rejections that must
     surface as a 400.

  2. `build_path_filter` — the filter condition each pattern shape
     produces: a `MatchPhrase` on `path` for exact patterns (so a
     multi-token body matches only *adjacent* tokens, as FTS5 did),
     and a `MatchText` on `path_prefix` for prefix patterns (the
     `prefix` tokenizer is what makes `cha*` match `chaewon`).

  3. `resolve_filename_filter` — the guard: a pattern covering more
     than the configured fraction of the corpus drops the filter, a
     narrow one keeps it, an invalid one is a bad request, and an
     empty one is "no filter".

  4. End-to-end `/api/search?filename=` — the round trip from the
     URL param through the native filter to the response JSON,
     including the 400 and the empty-result cases.

  5. Cache-key isolation — two different filename patterns must not
     share a cached ranking.

The in-memory Qdrant used by the fixtures supports text indexes, so
these tests exercise the real native path rather than a stub.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from indexer import upsert
from indexer.upsert import VECTOR_DIM
from lustre_kernel.registry import MockEmbedder
from search import app as app_mod
from search.config import Config

_mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text

CHAEWON_ID = "11111111-1111-1111-1111-111111111111"
KAZUHA_ID = "22222222-2222-2222-2222-222222222222"
YUJIN_ID = "33333333-3333-3333-3333-333333333333"
ANOTHER_ID = "44444444-4444-4444-4444-444444444444"
GENERIC_ID = "55555555-5555-5555-5555-555555555555"


# ============================================================
#  1. parse_filename_pattern
# ============================================================


def test_exact_pattern_is_not_prefix():
    from search.filename_pattern import parse_filename_pattern

    pattern = parse_filename_pattern("chaewon")
    assert pattern is not None
    assert pattern.body == "chaewon"
    assert pattern.prefix is False


def test_trailing_star_is_prefix():
    from search.filename_pattern import parse_filename_pattern

    pattern = parse_filename_pattern("chaewon*")
    assert pattern is not None
    assert pattern.body == "chaewon"
    assert pattern.prefix is True


def test_empty_pattern_is_none():
    from search.filename_pattern import parse_filename_pattern

    assert parse_filename_pattern("") is None
    assert parse_filename_pattern("   ") is None


def test_leading_star_is_rejected():
    from search.filename_pattern import parse_filename_pattern

    with pytest.raises(ValueError):
        parse_filename_pattern("*chaewon")


def test_glob_in_the_middle_is_rejected():
    from search.filename_pattern import parse_filename_pattern

    with pytest.raises(ValueError):
        parse_filename_pattern("cha*won")


def test_fts5_operator_characters_are_rejected():
    from search.filename_pattern import parse_filename_pattern

    for pattern in ("cha+won", "cha^won", 'cha"won'):
        with pytest.raises(ValueError):
            parse_filename_pattern(pattern)


def test_underscore_is_a_token_separator():
    from search.filename_pattern import parse_filename_pattern

    pattern = parse_filename_pattern("img_2024")
    assert pattern is not None
    assert pattern.body == "img_2024"
    assert pattern.prefix is False


# ============================================================
#  2. build_path_filter
# ============================================================


def test_exact_pattern_builds_a_phrase_match_on_path():
    from search.filename_pattern import parse_filename_pattern
    from search.path_filter import PATH_FIELD, build_path_filter

    condition = build_path_filter(parse_filename_pattern("chaewon"))
    assert condition.key == PATH_FIELD
    # A phrase match, not a plain text match: the phrase form is what
    # preserves FTS5's adjacency semantics for multi-token bodies.
    assert condition.match.phrase == "chaewon"
    assert not hasattr(condition.match, "text")


def test_prefix_pattern_builds_a_text_match_on_path_prefix():
    from search.filename_pattern import parse_filename_pattern
    from search.path_filter import PATH_PREFIX_FIELD, build_path_filter

    condition = build_path_filter(parse_filename_pattern("chaewon*"))
    assert condition.key == PATH_PREFIX_FIELD
    assert condition.match.text == "chaewon"


def test_the_two_pattern_shapes_use_different_fields():
    from search.filename_pattern import parse_filename_pattern
    from search.path_filter import build_path_filter

    exact = build_path_filter(parse_filename_pattern("chaewon"))
    prefix = build_path_filter(parse_filename_pattern("chaewon*"))
    assert exact.key != prefix.key


# ============================================================
#  3. resolve_filename_filter — the cardinality guard
# ============================================================


class _GuardQdrant:
    """Counts how many points a filter matches, without a database."""

    def __init__(self, total: int, matches: int):
        self.total = total
        self.matches = matches
        self.counted: list[object] = []

    def count_points(self) -> int:
        return self.total

    def count_filtered(self, condition) -> int:
        self.counted.append(condition)
        return self.matches


class _Cfg:
    def __init__(self, guard: float):
        self.filename_cardinality_guard = guard


async def _resolve(qdrant, pattern, guard=0.5):
    from search._filename_filter import resolve_filename_filter

    return await resolve_filename_filter(qdrant, cfg=_Cfg(guard), pattern=pattern)


@pytest.mark.asyncio
async def test_narrow_pattern_keeps_the_filter():
    condition, err = await _resolve(_GuardQdrant(total=1000, matches=10), "chaewon")
    assert err is None
    assert condition is not None


@pytest.mark.asyncio
async def test_broad_pattern_drops_the_filter():
    qdrant = _GuardQdrant(total=1000, matches=900)
    condition, err = await _resolve(qdrant, "2024")
    assert err is None
    assert condition is None
    # The guard counted against Qdrant, so it knows what it skipped.
    assert len(qdrant.counted) == 1


@pytest.mark.asyncio
async def test_guard_threshold_is_inclusive_at_the_boundary():
    # Exactly at the threshold: 500/1000 = 0.5, and the guard fires
    # on `> guard`, so 0.5 does NOT drop the filter.
    qdrant = _GuardQdrant(total=1000, matches=500)
    condition, err = await _resolve(qdrant, "2024", guard=0.5)
    assert err is None
    assert condition is not None


@pytest.mark.asyncio
async def test_empty_pattern_is_no_filter():
    qdrant = _GuardQdrant(total=1000, matches=0)
    condition, err = await _resolve(qdrant, "")
    assert err is None
    assert condition is None
    # No count was needed — there was no pattern to guard.
    assert qdrant.counted == []


@pytest.mark.asyncio
async def test_invalid_pattern_is_a_bad_request():
    condition, err = await _resolve(_GuardQdrant(total=10, matches=1), "*chaewon")
    assert err == "bad_request"
    assert condition is None


@pytest.mark.asyncio
async def test_zero_match_pattern_still_returns_a_condition():
    # A pattern that matches nothing yields a condition that Qdrant
    # evaluates to an empty result set. The guard must not treat
    # "0 matches" as "skip the filter" — 0% coverage is narrow, so
    # the filter is exactly what should be applied.
    qdrant = _GuardQdrant(total=1000, matches=0)
    condition, err = await _resolve(qdrant, "nothingmatchesthis")
    assert err is None
    assert condition is not None


@pytest.mark.asyncio
async def test_zero_total_corpus_does_not_divide_by_zero():
    qdrant = _GuardQdrant(total=0, matches=0)
    condition, err = await _resolve(qdrant, "chaewon")
    assert err is None
    assert condition is not None


# ============================================================
#  4. End-to-end /api/search?filename=
# ============================================================


@pytest.fixture
def app_with_native_paths(qdrant_in_memory, nas_base, monkeypatch):
    """An app whose Qdrant collection carries the native text indexes."""
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

    # The payload must carry `path_prefix` — it is the field the
    # prefix-tokenized text index lives on, and the indexer writes it
    # for every point. A payload built by hand without it would make
    # the prefix filter match nothing, which is a fixture bug rather
    # than a behaviour under test.
    def _payload(name: str, collection: str) -> dict:
        path = str(nas_base / name)
        return {
            "id": "",
            "path": path,
            "path_prefix": path,
            "collection": collection,
            "indexed_at": "2026-01-01T00:00:00Z",
        }

    items = [
        (CHAEWON_ID, _mock_embed("chaewon"),
         _payload("chaewon_2024.jpg", "kpop")),
        (KAZUHA_ID, _mock_embed("kazuha"),
         _payload("kazuha_2024.jpg", "kpop")),
        (YUJIN_ID, _mock_embed("yujin"),
         _payload("yujin_2024.jpg", "kpop")),
        (ANOTHER_ID, _mock_embed("winter"),
         _payload("winter_portrait.jpg", "portrait")),
        (GENERIC_ID, _mock_embed("generic"),
         _payload("random.jpg", "general")),
    ]
    upsert.upsert_batch(client, qdrant_in_memory.collection, items, wait=True)

    for _pid, _vec, payload in items:
        Image.new("RGB", (16, 16), (0, 0, 0)).save(
            nas_base / payload["path"].rsplit("/", 1)[-1]
        )

    # The native text indexes the filename filter needs.
    qdrant_in_memory.ensure_path_text_index()

    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)
    with TestClient(app) as tc:
        yield tc
    app_mod.reset_for_tests()


def _ids(response) -> set[str]:
    return {r["id"] for r in response.json()["results"]}


def test_filename_exact_restricts_results(app_with_native_paths):
    resp = app_with_native_paths.get("/api/search?q=photo&filename=chaewon")
    assert resp.status_code == 200
    assert _ids(resp) == {CHAEWON_ID}


def test_filename_excludes_unmatched(app_with_native_paths):
    resp = app_with_native_paths.get("/api/search?q=photo&filename=chaewon")
    assert CHAEWON_ID in _ids(resp)
    assert KAZUHA_ID not in _ids(resp)
    assert GENERIC_ID not in _ids(resp)


def test_no_filename_returns_the_full_set(app_with_native_paths):
    resp = app_with_native_paths.get("/api/search?q=photo")
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 5


def test_empty_filename_is_treated_as_no_filter(app_with_native_paths):
    resp = app_with_native_paths.get("/api/search?q=photo&filename=")
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 5


def test_filename_prefix_match(app_with_native_paths):
    resp = app_with_native_paths.get("/api/search?q=photo&filename=chaewon*")
    assert resp.status_code == 200
    assert CHAEWON_ID in _ids(resp)
    assert GENERIC_ID not in _ids(resp)


def test_invalid_filename_pattern_returns_400(app_with_native_paths):
    resp = app_with_native_paths.get("/api/search?q=photo&filename=*chaewon")
    assert resp.status_code == 400


def test_no_match_returns_empty_results(app_with_native_paths):
    resp = app_with_native_paths.get("/api/search?q=photo&filename=nosuchthing")
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_filename_combined_with_a_collection_filter(app_with_native_paths):
    resp = app_with_native_paths.get(
        "/api/search?q=photo&filename=2024&collection=kpop"
    )
    assert resp.status_code == 200
    assert _ids(resp) <= {CHAEWON_ID, KAZUHA_ID, YUJIN_ID}


# ============================================================
#  5. Cache-key isolation
# ============================================================


def test_two_filename_patterns_get_different_snapshot_keys():
    from search.filename_pattern import parse_filename_pattern
    from search.path_filter import build_path_filter
    from search.search_snapshot import snapshot_key

    common = dict(
        collection="images",
        vector=[0.1, 0.2, 0.3],
        collections=["kpop"],
        allowed_ids=None,
        favorite_ids=None,
        band_size=100,
    )
    chaewon = snapshot_key(
        **common, path_filter=build_path_filter(parse_filename_pattern("chaewon"))
    )
    kazuha = snapshot_key(
        **common, path_filter=build_path_filter(parse_filename_pattern("kazuha"))
    )
    unfiltered = snapshot_key(**common, path_filter=None)
    assert chaewon != kazuha
    assert chaewon != unfiltered
    assert kazuha != unfiltered


def test_two_filename_patterns_get_different_diversity_cache_keys():
    from search.filename_pattern import parse_filename_pattern
    from search.path_filter import build_path_filter
    from search._indexed_helpers import diversity_cache_key

    cfg = type("C", (), {"qdrant_collection": "images"})()
    common = dict(
        cfg=cfg,
        vector=[0.1, 0.2, 0.3],
        mode="mmr",
        depth="standard",
        pool_depth=100,
        collections=["kpop"],
        allowed_ids=None,
        favorite_ids=None,
    )
    chaewon = diversity_cache_key(
        **common, path_filter=build_path_filter(parse_filename_pattern("chaewon"))
    )
    kazuha = diversity_cache_key(
        **common, path_filter=build_path_filter(parse_filename_pattern("kazuha"))
    )
    unfiltered = diversity_cache_key(**common, path_filter=None)
    assert chaewon != kazuha
    assert chaewon != unfiltered
    assert kazuha != unfiltered


def test_the_same_pattern_gets_the_same_cache_key():
    from search.filename_pattern import parse_filename_pattern
    from search.path_filter import build_path_filter
    from search.search_snapshot import snapshot_key

    common = dict(
        collection="images",
        vector=[0.1, 0.2, 0.3],
        collections=["kpop"],
        allowed_ids=None,
        favorite_ids=None,
        band_size=100,
    )
    a = snapshot_key(
        **common, path_filter=build_path_filter(parse_filename_pattern("chaewon"))
    )
    b = snapshot_key(
        **common, path_filter=build_path_filter(parse_filename_pattern("chaewon"))
    )
    assert a == b


# ============================================================
#  6. The index must be fully built before it is queried
# ============================================================


def test_ensure_text_index_waits_for_the_collection_to_settle():
    """A text index is only correct once every segment carries it.

    `create_payload_index` returns when the build is SCHEDULED, not
    when it is done. Until the collection goes `green`, the segments
    that are not indexed yet fall back to a looser in-memory match
    and a text filter returns the WRONG counts — measured on the
    2.2M-point corpus, `MatchPhrase('won')` reported 84,964 against
    FTS5's true 613 while the collection was still `yellow`.

    So the creation must wait for `green` before it reports success.
    """
    from unittest.mock import MagicMock

    from search.qdrant_client import QdrantSearch

    qdrant = QdrantSearch(client=MagicMock(), collection="images")
    qdrant.client.get_collection.side_effect = [
        MagicMock(status="yellow"),
        MagicMock(status="yellow"),
        MagicMock(status="green"),
    ]

    import search.qdrant_client as qc

    original_sleep = qc.time.sleep
    qc.time.sleep = lambda _s: None
    try:
        ready = qdrant._await_index_ready("path", timeout_s=60)
    finally:
        qc.time.sleep = original_sleep

    assert ready is True
    # It polled until green, and did not give up.
    assert qdrant.client.get_collection.call_count == 3


def test_ensure_text_index_gives_up_rather_than_hanging():
    """A collection stuck `yellow` must not block startup forever."""
    from unittest.mock import MagicMock

    from search.qdrant_client import QdrantSearch

    qdrant = QdrantSearch(client=MagicMock(), collection="images")
    qdrant.client.get_collection.return_value = MagicMock(status="yellow")

    import search.qdrant_client as qc

    original_sleep = qc.time.sleep
    qc.time.sleep = lambda _s: None
    try:
        ready = qdrant._await_index_ready("path", timeout_s=0.05)
    finally:
        qc.time.sleep = original_sleep

    assert ready is False


def test_ensure_text_index_warns_when_the_collection_does_not_settle(caplog):
    """A half-built index must be reported, not silently accepted."""
    import logging

    from unittest.mock import MagicMock

    from search.qdrant_client import QdrantSearch

    qdrant = QdrantSearch(client=MagicMock(), collection="images")
    qdrant.client.get_collection.return_value = MagicMock(status="yellow")

    import search.qdrant_client as qc

    original_sleep = qc.time.sleep
    qc.time.sleep = lambda _s: None
    try:
        with caplog.at_level(logging.WARNING):
            # A short timeout: the wait must give up rather than
            # block for the full production budget.
            qdrant._ensure_text_index("path", MagicMock(), timeout_s=0.05)
    finally:
        qc.time.sleep = original_sleep

    assert any("did not" in r.message for r in caplog.records)
