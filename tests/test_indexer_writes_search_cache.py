"""
test_indexer_writes_search_cache.py — the indexer used to mirror
Qdrant upserts into the search-side `images.db` so Likes /
Dislikes / Random saw fresh points during a long indexing run
without waiting for the periodic /api/cache/refresh.

It does not any more. The photo corpus lives in Qdrant and nowhere
else, so the indexer writes vectors and payloads there and stops.
These tests pin that down: the upsert path has no SQLite
dependency, and the side store keeps the WAL mode the concurrent
reader/writer setup needs.
"""

from __future__ import annotations

import sqlite3
import uuid


from indexer import upsert
from indexer.upsert import VECTOR_DIM


def _seed_payload(path: str = "/photos/a.jpg", collection: str = "kpop") -> dict:
    """Build a Qdrant payload shaped like build_payload()'s output."""
    return {
        "id": str(uuid.uuid4()),
        "path": path,
        "shard": "",
        "collection": collection,
        "blurhash": "L:abc:",
        "folder": "/photos",
        "width": 1024,
        "height": 768,
        "mtime": 1700000000,
        "size": 12345,
        "model_name": "ViT-B-16-SigLIP2-256",
        "model_revision": "",
        "model_dim": 768,
        "indexed_at": "2026-01-01T00:00:00Z",
    }


def test_upsert_batch_takes_no_sqlite_connection():
    """The upsert path has no SQLite parameter at all.

    The mirror is gone: there is no second copy of the corpus to
    keep in sync, and therefore no way for the two to disagree.
    """
    import inspect

    sig = inspect.signature(upsert.upsert_batch)
    assert "sqlite_conn" not in sig.parameters


def test_upsert_batch_writes_only_to_qdrant(tmp_path):
    """An upsert lands in Qdrant, and touches no SQLite file."""
    from qdrant_client import QdrantClient

    client = QdrantClient(location=":memory:")
    coll = "images_test_no_mirror"
    upsert.ensure_collection(client, coll, dim=VECTOR_DIM)

    payload = _seed_payload()
    items = [(payload["id"], [0.0] * VECTOR_DIM, payload)]
    upsert.upsert_batch(client, coll, items, wait=True)

    pts = client.retrieve(coll, ids=[payload["id"]])
    assert len(pts) == 1
    assert pts[0].payload["path"] == payload["path"]

    # No database file was created anywhere in tmp_path.
    assert list(tmp_path.glob("*.db")) == []


def test_upsert_batch_handles_missing_optional_payload_fields(tmp_path):
    """Older points (or non-standard callers) may not populate
    width/height. The upsert must still succeed.
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(location=":memory:")
    coll = "images_test_partial_payload"
    upsert.ensure_collection(client, coll, dim=VECTOR_DIM)

    payload = _seed_payload()
    payload.pop("width")
    payload.pop("height")
    items = [(payload["id"], [0.0] * VECTOR_DIM, payload)]
    upsert.upsert_batch(client, coll, items, wait=True)

    pts = client.retrieve(coll, ids=[payload["id"]])
    assert len(pts) == 1
    assert "width" not in pts[0].payload


def test_upsert_batch_is_idempotent(tmp_path):
    """Re-upserting the same id replaces the point."""
    from qdrant_client import QdrantClient

    client = QdrantClient(location=":memory:")
    coll = "images_test_idempotent"
    upsert.ensure_collection(client, coll, dim=VECTOR_DIM)

    payload = _seed_payload(path="/photos/v1.jpg")
    items = [(payload["id"], [0.0] * VECTOR_DIM, payload)]
    upsert.upsert_batch(client, coll, items, wait=True)

    # Re-upsert with the same id but a different path (e.g. user
    # moved the file).
    payload["path"] = "/photos/v2.jpg"
    upsert.upsert_batch(client, coll, items, wait=True)

    pts = client.retrieve(coll, ids=[payload["id"]])
    assert len(pts) == 1
    assert pts[0].payload["path"] == "/photos/v2.jpg"


def test_search_index_db_uses_wal_mode(tmp_path):
    """The search-side IndexDB connection enables WAL mode so the
    admin actions can write concurrently without blocking the
    search app's reads.
    """
    from unittest.mock import MagicMock

    from search.index_db import IndexDB

    db_path = tmp_path / "images.db"
    # IndexDB requires a QdrantSearch. The constructor doesn't
    # actually use it for the WAL test — we only care that the PRAGMA
    # is set on the freshly-opened connection.
    fake_qdrant = MagicMock()
    IndexDB(db_path=str(db_path), qdrant_client=fake_qdrant)

    # Read the journal_mode pragma directly from the file.
    conn = sqlite3.connect(str(db_path))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_search_index_db_has_no_images_table(tmp_path):
    """The side store holds only user state.

    The `images` table and its FTS5 structures are gone — the
    corpus lives in Qdrant. A database that still has them is a
    database that was opened by pre-migration code.
    """
    from unittest.mock import MagicMock

    from search.index_db import IndexDB

    db_path = tmp_path / "images.db"
    IndexDB(db_path=str(db_path), qdrant_client=MagicMock())

    conn = sqlite3.connect(str(db_path))
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "images" not in tables
    assert "images_fts" not in tables
    # The user-state tables are still there.
    for name in ("favorites", "dislikes", "albums", "album_memberships"):
        assert name in tables
