"""
Tests for the admin orphan endpoints.

The purge is the second step of the two-step orphan lifecycle: the
indexer marks the unwanted points, and an operator reviews the count
and confirms the removal.

The purge removes orphans from Qdrant AND from the user-state
tables, in one action. The two halves are counted independently,
because they are different questions: a favourite whose photo was
deleted from disk is an orphan even when its Qdrant point was never
marked (the indexer may not have run since the deletion).
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from lustre_kernel.orphan_marking import mark_orphans
from search.index_db import IndexDB
from search.qdrant_client import QdrantSearch
from search.routers.admin_orphans import build_admin_orphans_router


@pytest.fixture
def photos_root(tmp_path) -> Path:
    root = tmp_path / "photos"
    root.mkdir()
    return root


@pytest.fixture
def qdrant_client() -> QdrantClient:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="images",
        vectors_config={"size": 2, "distance": "Cosine"},
    )
    points = []
    for src, count in (("telegram", 2), ("data", 1)):
        for i in range(count):
            points.append(qmodels.PointStruct(
                id=uuid.UUID(int=len(points)),
                vector=[1.0, 0.0],
                payload={
                    "collection": src,
                    "path": f"/nas/{src}/{i}.jpg",
                },
            ))
    client.upsert(collection_name="images", points=points, wait=True)
    return client


@pytest.fixture
def index_db(tmp_path, photos_root) -> IndexDB:
    """An IndexDB with a photos root the tests can write to."""
    return IndexDB(
        db_path=str(tmp_path / "index.db"),
        qdrant_client=_FakeQdrant(),
        photos_root=str(photos_root),
    )


class _FakeQdrant:
    """A Qdrant stand-in for the IndexDB's liveness checks."""

    def __init__(self) -> None:
        self.live_ids: set[str] = set()
        self.collection = "images"

    def count_points(self, exact: bool = True) -> int:
        return len(self.live_ids)

    def retrieve_batch(self, point_ids: list[str]):
        from search.qdrant_client import SearchHit

        return [
            SearchHit(id=pid, path="", score=0.0, payload={})
            for pid in point_ids
            if pid in self.live_ids
        ]


@pytest.fixture
def client(qdrant_client, index_db) -> TestClient:
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(build_admin_orphans_router(
        qdrant=QdrantSearch(client=qdrant_client, collection="images"),
        index_db=index_db,
    ))
    return TestClient(app)


def test_the_list_endpoint_reports_both_halves(
    client, qdrant_client, index_db, photos_root,
) -> None:
    """The review view shows the Qdrant count AND the user-state count."""
    ids = [str(p.id) for p in qdrant_client.scroll(
        collection_name="images", limit=100, with_payload=False,
    )[0]]
    mark_orphans(qdrant_client, "images", ids[:1])

    # A favourite whose file is gone from disk — its Qdrant point is
    # NOT marked, so this is the half the Qdrant count misses.
    path = photos_root / "gone.jpg"
    path.write_bytes(b"x")
    index_db.record_photo_path("id-gone", str(path))
    index_db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-gone", "2026-01-01T00:00:00Z"),
    )
    index_db._conn.commit()
    path.unlink()

    body = client.get("/api/admin/orphans").json()

    assert body["count"] == 1
    assert body["orphaned_user_data"]["favorites"] == 1


def test_the_purge_removes_both_halves(
    client, qdrant_client, index_db, photos_root,
) -> None:
    """One purge clears the embeddings AND the user-state rows."""
    ids = [str(p.id) for p in qdrant_client.scroll(
        collection_name="images", limit=100, with_payload=False,
    )[0]]
    mark_orphans(qdrant_client, "images", ids[:1])

    path = photos_root / "gone.jpg"
    path.write_bytes(b"x")
    index_db.record_photo_path("id-gone", str(path))
    index_db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-gone", "2026-01-01T00:00:00Z"),
    )
    index_db._conn.commit()
    path.unlink()

    resp = client.post("/api/admin/orphans/purge", json={"confirm": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == 1
    assert body["remaining"] == 0
    # The user-state half ran too.
    assert body["user_data_purged"]["favorites"] == 1
    assert index_db._conn.execute(
        "SELECT COUNT(*) AS c FROM favorites",
    ).fetchone()["c"] == 0


def test_the_purge_requires_confirmation(client) -> None:
    """There is no path to the delete without saying so explicitly."""
    resp = client.post("/api/admin/orphans/purge", json={})

    assert resp.status_code == 400
    assert "confirm" in resp.json()["detail"]


def test_the_purge_can_leave_the_user_data_alone(
    client, qdrant_client, index_db, photos_root,
) -> None:
    """`include_user_data: false` purges only the embeddings.

    The Qdrant half and the user-state half are separable, so an
    operator who wants to keep the favourites can.
    """
    ids = [str(p.id) for p in qdrant_client.scroll(
        collection_name="images", limit=100, with_payload=False,
    )[0]]
    mark_orphans(qdrant_client, "images", ids[:1])

    path = photos_root / "gone.jpg"
    path.write_bytes(b"x")
    index_db.record_photo_path("id-gone", str(path))
    index_db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-gone", "2026-01-01T00:00:00Z"),
    )
    index_db._conn.commit()
    path.unlink()

    resp = client.post("/api/admin/orphans/purge", json={
        "confirm": True,
        "include_user_data": False,
    })

    assert resp.status_code == 200
    assert resp.json()["deleted"] == 1
    assert resp.json()["user_data_purged"] == {}
    # The favourite survived.
    assert index_db._conn.execute(
        "SELECT COUNT(*) AS c FROM favorites",
    ).fetchone()["c"] == 1


def test_the_purge_leaves_live_photos_alone(
    client, qdrant_client, index_db, photos_root,
) -> None:
    """A photo that is alive keeps its favourite through a purge.

    The file-based rule: liveness is decided by the filesystem, so a
    photo that is on disk survives even when it is not currently in
    the collection.
    """
    path = photos_root / "alive.jpg"
    path.write_bytes(b"x")
    index_db.record_photo_path("id-alive", str(path))
    index_db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-alive", "2026-01-01T00:00:00Z"),
    )
    index_db._conn.commit()

    resp = client.post("/api/admin/orphans/purge", json={"confirm": True})

    assert resp.status_code == 200
    assert index_db._conn.execute(
        "SELECT COUNT(*) AS c FROM favorites",
    ).fetchone()["c"] == 1
