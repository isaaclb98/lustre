"""
tests/test_qdrant_client_unit.py — Unit tests for search/qdrant_client.py.

The QdrantSearch wrapper hides client details and provides a typed
search API. Tests use the in-memory Qdrant (location=':memory:')
which works as a drop-in for local verification.
"""
from __future__ import annotations


import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from search.qdrant_client import QdrantSearch, SearchHit


# ----- Fixtures -----

@pytest.fixture
def in_memory_qdrant() -> QdrantClient:
    """In-memory Qdrant client — no disk, no network."""
    return QdrantClient(location=":memory:")


@pytest.fixture
def populated_collection(in_memory_qdrant):
    """Create a collection with 5 points, return (client, coll, point_ids)."""
    coll = "test_imgs"
    in_memory_qdrant.create_collection(
        collection_name=coll,
        vectors_config=qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE),
    )
    point_ids = [str(uuid.uuid4()) for _ in range(5)]
    points = [
        qmodels.PointStruct(
            id=pid,
            vector=[float(i), 0.0, 0.0, 0.0],
            payload={
                "path": f"/images/photo_{i}.jpg",
                "collection": "library-a",
                "model_name": "ViT-L-16-SigLIP2-256",
                "model_revision": "r0",
            },
        )
        for i, pid in enumerate(point_ids)
    ]
    in_memory_qdrant.upsert(collection_name=coll, points=points, wait=True)
    return in_memory_qdrant, coll, point_ids


@pytest.fixture
def multi_collection(in_memory_qdrant):
    """Two collections for testing collection filtering."""
    for coll in ["lib-a", "lib-b"]:
        in_memory_qdrant.create_collection(
            collection_name=coll,
            vectors_config=qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE),
        )
        points = [
            qmodels.PointStruct(
                id=f"{uuid.uuid4()}",
                vector=[float(i), 0.0, 0.0, 0.0],
                payload={
                    "path": f"/{coll}/photo_{i}.jpg",
                    "collection": coll,
                },
            )
            for i in range(3)
        ]
        in_memory_qdrant.upsert(collection_name=coll, points=points, wait=True)
    return in_memory_qdrant


# ----- SearchHit dataclass -----

class TestSearchHit:
    """SearchHit is the typed result returned by all queries."""

    def test_construction_minimal(self):
        hit = SearchHit(id="abc", path="/img.jpg", score=0.9)
        assert hit.id == "abc"
        assert hit.path == "/img.jpg"
        assert hit.score == 0.9
        assert hit.payload is None

    def test_construction_with_payload(self):
        hit = SearchHit(
            id="abc",
            path="/img.jpg",
            score=0.9,
            payload={"collection": "lib-a", "extra": "data"},
        )
        assert hit.payload["collection"] == "lib-a"

    def test_payload_default_is_none(self):
        """payload defaults to None — callers must handle missing payload."""
        hit = SearchHit(id="x", path="/p", score=0.5)
        assert hit.payload is None


# ----- QdrantSearch.search() -----

class TestSearch:
    """The main search() method — top-K with filters."""

    def test_basic_search(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        results, _ = search.search([1.0, 0.0, 0.0, 0.0], limit=3)
        assert len(results) == 3
        # All hits should have the right shape
        for hit in results:
            assert isinstance(hit, SearchHit)
            assert hit.id
            assert hit.path

    def test_search_returns_more_flag(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        # 5 points, limit=2 → more available
        _, has_more = search.search([1.0, 0.0, 0.0, 0.0], limit=2)
        assert has_more is True

    def test_search_exhausted(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        # 5 points, limit=10 → exhausted
        _, has_more = search.search([1.0, 0.0, 0.0, 0.0], limit=10)
        assert has_more is False

    def test_search_with_offset(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        first, _ = search.search([1.0, 0.0, 0.0, 0.0], limit=2, offset=0)
        second, _ = search.search([1.0, 0.0, 0.0, 0.0], limit=2, offset=2)
        # Offset should produce different results
        first_ids = {h.id for h in first}
        second_ids = {h.id for h in second}
        assert first_ids.isdisjoint(second_ids)

    def test_search_with_collections_filter(self, multi_collection):
        client = multi_collection
        # Search in lib-a only — QdrantSearch uses single collection,
        # so the collections filter is applied to the payload field
        search = QdrantSearch(client, "lib-a")
        results, _ = search.search(
            [1.0, 0.0, 0.0, 0.0],
            limit=10,
            collections=["lib-a"],
        )
        # All results should have collection="lib-a" in payload
        for hit in results:
            assert hit.payload.get("collection") == "lib-a"

    def test_search_with_exclude_ids(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        results, _ = search.search(
            [1.0, 0.0, 0.0, 0.0],
            limit=5,
            exclude_ids=[point_ids[0], point_ids[1]],
        )
        ids = {h.id for h in results}
        assert point_ids[0] not in ids
        assert point_ids[1] not in ids

    def test_search_with_allowed_ids(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        results, _ = search.search(
            [1.0, 0.0, 0.0, 0.0],
            limit=5,
            allowed_ids=[point_ids[0], point_ids[2]],
        )
        ids = {h.id for h in results}
        # Only allowed_ids should appear
        assert ids.issubset({point_ids[0], point_ids[2]})

    def test_search_zero_limit_raises(self, populated_collection):
        """Qdrant requires limit >= 1 — 0 raises ValueError."""
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        with pytest.raises(ValueError):
            search.search([1.0, 0.0, 0.0, 0.0], limit=0)


# ----- QdrantSearch.retrieve_batch -----

class TestRetrieveBatch:
    """Fetch specific points by id."""

    def test_basic_retrieve(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        results = search.retrieve_batch([point_ids[0], point_ids[2]])
        assert len(results) == 2
        ids = {h.id for h in results}
        assert ids == {point_ids[0], point_ids[2]}

    def test_retrieve_missing_ids_returns_empty(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        results = search.retrieve_batch(["nonexistent-1", "nonexistent-2"])
        assert results == []

    def test_retrieve_mix_existing_and_missing(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        results = search.retrieve_batch([point_ids[0], "nonexistent"])
        # Only existing points returned
        ids = {h.id for h in results}
        assert point_ids[0] in ids
        assert "nonexistent" not in ids

    def test_retrieve_empty_list(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        results = search.retrieve_batch([])
        assert results == []


# ----- QdrantSearch.retrieve -----

class TestRetrieve:
    """Fetch a single point by id."""

    def test_basic_retrieve(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        hit = search.retrieve(point_ids[0])
        assert hit is not None
        assert hit.id == point_ids[0]
        assert "/images/photo_0.jpg" in hit.path

    def test_retrieve_missing_returns_none(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        hit = search.retrieve("nonexistent")
        assert hit is None

    def test_retrieve_returns_payload(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        hit = search.retrieve(point_ids[0])
        assert hit is not None
        assert hit.payload is not None
        assert hit.payload.get("collection") == "library-a"


# ----- QdrantSearch.healthz -----

class TestHealthz:
    """Health check endpoint."""

    def test_healthz_returns_true_when_alive(self, in_memory_qdrant):
        search = QdrantSearch(in_memory_qdrant, "any")
        assert search.healthz() is True

    def test_healthz_handles_failure(self):
        """A client whose get_collections fails → returns False."""
        from unittest.mock import MagicMock
        bad_client = MagicMock()
        bad_client.get_collections.side_effect = Exception("connection refused")
        search = QdrantSearch(bad_client, "any")
        # The healthz wrapper should return False, not raise
        assert search.healthz() is False


# ----- QdrantSearch.list_collections_with_counts -----

class TestListCollections:
    """Enumerate collections with their point counts."""

    def test_list_collections_returns_list(self, multi_collection):
        client = multi_collection
        search = QdrantSearch(client, "lib-a")
        cols = search.list_collections_with_counts()
        assert isinstance(cols, list)

    def test_collection_counts_are_nonneg(self, multi_collection):
        client = multi_collection
        search = QdrantSearch(client, "lib-a")
        cols = search.list_collections_with_counts()
        for c in cols:
            # facet() returns dicts with value + count fields
            count = c.get("count") if isinstance(c, dict) else None
            if count is not None:
                assert count >= 0


# ----- QdrantSearch.scroll_all -----

class TestScrollAll:
    """Iterate over all points in a collection in batches."""

    def test_scroll_all_returns_all_points(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        all_points = []
        for batch in search.scroll_all(batch_size=2):
            all_points.extend(batch)
        assert len(all_points) == 5

    def test_scroll_all_batch_size(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        batches = list(search.scroll_all(batch_size=2))
        # With batch_size=2 and 5 points, we should get multiple batches
        assert len(batches) >= 2

    def test_scroll_all_empty_collection(self, in_memory_qdrant):
        in_memory_qdrant.create_collection(
            collection_name="empty",
            vectors_config=qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE),
        )
        search = QdrantSearch(in_memory_qdrant, "empty")
        all_points = []
        for batch in search.scroll_all(batch_size=10):
            all_points.extend(batch)
        assert all_points == []


# ----- QdrantSearch.random_window -----

class TestRandomWindow:
    """Random sampling from a collection."""

    def test_random_window_returns_limit_points(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        results = search.random_window(limit=3)
        assert len(results) == 3
        for hit in results:
            assert isinstance(hit, SearchHit)

    def test_random_window_default_limit(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        results = search.random_window()
        # Default limit should be a reasonable number (≤ total)
        assert len(results) <= 5

    def test_random_window_no_duplicates_within_batch(self, populated_collection):
        """The random sample should not contain duplicates."""
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        results = search.random_window(limit=5)
        ids = [h.id for h in results]
        assert len(ids) == len(set(ids))


# ----- QdrantSearch configuration -----

class TestQdrantSearchConfig:
    """Constructor and attribute configuration."""

    def test_default_timeout(self, in_memory_qdrant):
        search = QdrantSearch(in_memory_qdrant, "x")
        # Defaults bumped 2s -> 60s and 40s -> 60s to match
        # config.py (search/qdrant_client.py and config.py must
        # agree). The Surprise Me slider's max pool of 50k on
        # the 2M-point production corpus completes in 22-27s;
        # 30s was the cliff and produced intermittent 504s on
        # transient Qdrant load. 60s gives ~2x headroom without
        # binding gunicorn's 2-minute worker timeout.
        assert search.timeout_ms == 60000
        assert search.recommend_timeout_ms == 60000

    def test_custom_timeout(self, in_memory_qdrant):
        search = QdrantSearch(in_memory_qdrant, "x", timeout_ms=500)
        assert search.timeout_ms == 500

    def test_collection_attribute(self, in_memory_qdrant):
        search = QdrantSearch(in_memory_qdrant, "my-collection")
        assert search.collection == "my-collection"

    def test_client_attribute(self, in_memory_qdrant):
        search = QdrantSearch(in_memory_qdrant, "x")
        assert search.client is in_memory_qdrant


# ----- SearchHit payload access -----

class TestSearchHitPayloadAccess:
    """Common payload field access patterns."""

    def test_payload_get_with_default(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        results, _ = search.search([1.0, 0.0, 0.0, 0.0], limit=1)
        hit = results[0]
        # Standard payload fields
        assert hit.payload.get("path")
        assert hit.payload.get("collection") == "library-a"

    def test_payload_missing_field_returns_none(self, populated_collection):
        client, coll, point_ids = populated_collection
        search = QdrantSearch(client, coll)
        results, _ = search.search([1.0, 0.0, 0.0, 0.0], limit=1)
        hit = results[0]
        assert hit.payload.get("nonexistent_field") is None


# ----- lustre_kernel.qdrant_url.client_kwargs -----

class TestClientKwargsGrpc:
    """Pin the gRPC plumbing contract for the qdrant-client kwargs helper."""

    def test_default_is_rest(self):
        """No prefer_grpc flag means no gRPC kwargs."""
        from lustre_kernel.qdrant_url import client_kwargs

        kwargs = client_kwargs("http://localhost:6333")
        assert "prefer_grpc" not in kwargs
        assert "grpc_port" not in kwargs
        assert kwargs["port"] == 6333

    def test_prefer_grpc_adds_both_kwargs(self):
        """prefer_grpc=True emits both prefer_grpc and grpc_port."""
        from lustre_kernel.qdrant_url import client_kwargs

        kwargs = client_kwargs(
            "http://localhost:6333", prefer_grpc=True,
        )
        assert kwargs["prefer_grpc"] is True
        # Default grpc_port is REST port + 1 (6333 -> 6334).
        assert kwargs["grpc_port"] == 6334

    def test_prefer_grpc_with_explicit_port(self):
        """Explicit grpc_port wins over the +1 default."""
        from lustre_kernel.qdrant_url import client_kwargs

        kwargs = client_kwargs(
            "http://localhost:6333", prefer_grpc=True, grpc_port=17333,
        )
        assert kwargs["grpc_port"] == 17333

    def test_explicit_grpc_port_ignored_when_rest(self):
        """grpc_port without prefer_grpc does NOT leak into kwargs."""
        from lustre_kernel.qdrant_url import client_kwargs

        kwargs = client_kwargs(
            "http://localhost:6333", grpc_port=17333,
        )
        assert "grpc_port" not in kwargs
        assert "prefer_grpc" not in kwargs

    def test_https_default_grpc_port_is_443_plus_1(self):
        """HTTPS without explicit port still gets port + 1 as gRPC default."""
        from lustre_kernel.qdrant_url import client_kwargs

        kwargs = client_kwargs(
            "https://qdrant.example.com", prefer_grpc=True,
        )
        assert kwargs["port"] == 443
        assert kwargs["grpc_port"] == 444