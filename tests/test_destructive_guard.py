"""
Tests for the destructive-filter guard.

Context: on 2026-10-28 a probe against the production Qdrant used
the filter `{"must": []}` to test whether a filter-delete endpoint
worked. In Qdrant an empty conjunction matches EVERY point, not none,
so the probe deleted a 2,280,814-point production collection in 0.6
seconds.

These tests pin the guard's behaviour on the shapes that caused it,
and exercise the real delete path against an IN-MEMORY Qdrant — never
a live service.

The expected match counts below were MEASURED against Qdrant 1.19,
not assumed. The asymmetry that matters: an empty `must` matches
everything, an empty `should` matches nothing.
"""
from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from lustre_kernel.destructive_guard import (
    DestructiveFilterError,
    assert_not_match_all,
    assert_scoped,
    build_filter,
    delete_source_points,
    is_match_all,
    source_delete_filter,
)


# ── The shapes that match everything ──────────────────────────────────────
# Each of these was either used in the 2026-10-28 incident or is the
# same class of mistake: a filter that looks restrictive but is not.
MATCH_ALL_FILTERS = [
    pytest.param(None, id="none"),
    pytest.param({}, id="empty-dict"),
    pytest.param({"must": []}, id="empty-must"),
    pytest.param({"must_not": []}, id="empty-must-not"),
    pytest.param({"must": [], "must_not": []}, id="empty-both"),
    pytest.param({"must": [{}]}, id="condition-without-predicate"),
]


@pytest.mark.parametrize("filter_", MATCH_ALL_FILTERS)
def test_match_all_shapes_are_detected(filter_) -> None:
    """Every match-all shape must be recognised as match-all."""
    assert is_match_all(filter_) is True


@pytest.mark.parametrize("filter_", MATCH_ALL_FILTERS)
def test_match_all_shapes_are_refused(filter_) -> None:
    """A match-all filter must never pass the guard."""
    with pytest.raises(DestructiveFilterError):
        assert_not_match_all(filter_, context="delete")


def test_empty_should_is_not_match_all() -> None:
    """An empty `should` matches NOTHING — the asymmetry with `must`.

    A disjunction over zero alternatives is false. This is the one
    empty-clause shape that is safe, and the guard must not treat it
    as match-all.
    """
    assert is_match_all({"should": []}) is False


# ── The shapes that are properly scoped ───────────────────────────────────
SCOPED_FILTERS = [
    pytest.param(
        {"must": [{"key": "collection", "match": {"value": "kpopping"}}]},
        id="single-source-match",
    ),
    pytest.param(
        {"must": [{"key": "collection", "match": {"any": ["a", "b"]}}]},
        id="match-any",
    ),
    pytest.param(
        {"must": [{"key": "size", "range": {"gte": 100}}]},
        id="range",
    ),
    pytest.param(
        {"must": [{"key": "has_id", "has_id": ["abc"]}]},
        id="has-id",
    ),
    pytest.param(
        {"should": [{"key": "collection", "match": {"value": "a"}}]},
        id="should-only",
    ),
]


@pytest.mark.parametrize("filter_", SCOPED_FILTERS)
def test_scoped_filters_are_allowed(filter_) -> None:
    """A filter that can actually exclude points must pass."""
    assert is_match_all(filter_) is False
    assert_not_match_all(filter_, context="delete")
    assert_scoped(filter_, context="delete")


def test_a_scoped_must_with_an_empty_must_not_is_still_scoped() -> None:
    """An empty `must_not` alongside a real `must` does not unscope it."""
    filter_ = {
        "must": [{"key": "collection", "match": {"value": "kpopping"}}],
        "must_not": [],
    }
    assert is_match_all(filter_) is False


def test_a_must_not_only_filter_is_refused_by_assert_scoped() -> None:
    """`must_not` excludes rather than selects — it is never enough.

    It is not match-all (it excludes the named points), but it still
    selects everything else, which is the wrong way round for a
    scoped delete.
    """
    filter_ = {"must_not": [{"key": "collection", "match": {"value": "a"}}]}
    assert is_match_all(filter_) is False
    with pytest.raises(DestructiveFilterError):
        assert_scoped(filter_, context="delete")


# ── The sanctioned filter builder ─────────────────────────────────────────
def test_source_delete_filter_targets_exactly_one_source() -> None:
    """The builder produces a single-source filter."""
    filter_ = source_delete_filter("kpopping")
    assert filter_ == {
        "must": [{"key": "collection", "match": {"value": "kpopping"}}]
    }
    assert is_match_all(filter_) is False


@pytest.mark.parametrize("bad", ["", "   ", None, 42])
def test_source_delete_filter_rejects_empty_names(bad) -> None:
    """An empty source name is refused — it would match points with no
    `collection` field, which is not a source."""
    with pytest.raises(DestructiveFilterError):
        source_delete_filter(bad)


def test_build_filter_produces_a_client_usable_model() -> None:
    """The Python client rejects plain dicts — the guard's dicts must be
    converted to qdrant-client models."""
    model = build_filter(source_delete_filter("kpopping"))
    assert isinstance(model, qmodels.Filter)
    assert len(model.must) == 1
    assert model.must[0].key == "collection"


# ── The real delete path, against an in-memory Qdrant ────────────────────
@pytest.fixture
def qdrant_with_sources():
    """An in-memory Qdrant holding two sources' points."""
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="images",
        vectors_config={"size": 4, "distance": "Cosine"},
    )

    points = [
        qmodels.PointStruct(
            id=uuid.UUID(int=i),
            vector=[1.0, 0.0, 0.0, 0.0] if i < 5 else [0.0, 1.0, 0.0, 0.0],
            payload={
                "collection": "kpopping" if i < 5 else "data",
                "path": f"/nas/{'kpopping' if i < 5 else 'data'}/{i}.jpg",
            },
        )
        for i in range(10)
    ]
    client.upsert(collection_name="images", points=points, wait=True)
    return client


def test_delete_source_points_removes_only_that_source(qdrant_with_sources) -> None:
    """Deleting one source leaves the other source's points intact."""
    client = qdrant_with_sources
    assert client.count(collection_name="images", exact=True).count == 10

    result = delete_source_points(client, "images", "kpopping")

    assert result["matched"] == 5
    assert result["deleted"] == 5
    assert client.count(collection_name="images", exact=True).count == 5

    survivors = client.scroll(
        collection_name="images",
        scroll_filter=build_filter(source_delete_filter("data")),
        limit=100,
        with_payload=["collection"],
    )[0]
    assert len(survivors) == 5
    assert {p.payload["collection"] for p in survivors} == {"data"}


def test_delete_source_points_dry_run_changes_nothing(qdrant_with_sources) -> None:
    """A dry run reports the count and deletes nothing."""
    client = qdrant_with_sources

    result = delete_source_points(client, "images", "kpopping", dry_run=True)

    assert result["matched"] == 5
    assert result["deleted"] == 0
    assert client.count(collection_name="images", exact=True).count == 10


def test_delete_source_points_on_unknown_source_is_a_no_op(qdrant_with_sources) -> None:
    """Removing a source that is not present deletes nothing."""
    client = qdrant_with_sources

    result = delete_source_points(client, "images", "does-not-exist")

    assert result["matched"] == 0
    assert result["deleted"] == 0
    assert client.count(collection_name="images", exact=True).count == 10


def test_an_empty_must_filter_would_match_everything(qdrant_with_sources) -> None:
    """The 2026-10-28 incident, reproduced in miniature.

    This is the shape that was sent to production. It is asserted
    here so the failure mode stays documented and the guard's value
    stays measurable — the in-memory collection here holds 10 points,
    not 2.28 million.
    """
    client = qdrant_with_sources

    matched = client.count(
        collection_name="images",
        count_filter=build_filter({"must": []}),
        exact=True,
    ).count

    assert matched == 10, (
        "an empty `must` conjunction matches EVERY point in Qdrant — "
        "this is the filter that deleted production on 2026-10-28"
    )
    # And the guard refuses it.
    with pytest.raises(DestructiveFilterError):
        assert_not_match_all({"must": []}, context="delete")
