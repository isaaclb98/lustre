"""
Tests for the prune scope guard.

`prune_missing` deletes points whose files are missing from disk. That
inference is only sound when every managed source dir could actually
be walked. If a managed dir is missing — a dropped NFS mount, a typo,
a source that was just removed — its points are in the delete scope
(`source_names` covers them) but absent from the alive-set (the dir
could not be walked), so every one of them looks dead.

Before this guard, a missing dir produced a warning and the prune
carried on, deleting that source's entire point set. The guard aborts
the prune instead.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from indexer.upsert import PruneScopeError, prune_missing


@pytest.fixture
def source_dirs(tmp_path) -> dict[str, Path]:
    """Two source dirs on disk.

    `present` holds 2 of the 4 files its points claim exist.
    `vanished` does not exist at all — it is never created.
    """
    present = tmp_path / "present"
    present.mkdir()
    (present / "0.jpg").write_bytes(b"x")
    (present / "1.jpg").write_bytes(b"x")
    vanished = tmp_path / "vanished"
    return {"present": present, "vanished": vanished}


@pytest.fixture
def qdrant_with_points(source_dirs) -> QdrantClient:
    """A collection with 4 points per source, at real tmp_path paths."""
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="images",
        vectors_config={"size": 2, "distance": "Cosine"},
    )
    points = []
    for src, count in (("present", 4), ("vanished", 4)):
        for i in range(count):
            points.append(qmodels.PointStruct(
                id=uuid.UUID(int=len(points)),
                vector=[1.0, 0.0],
                payload={
                    "collection": src,
                    "path": str(source_dirs[src] / f"{i}.jpg"),
                },
            ))
    client.upsert(collection_name="images", points=points, wait=True)
    return client


def test_missing_source_dir_aborts_the_prune(qdrant_with_points, source_dirs) -> None:
    """A managed source dir that does not exist aborts the prune.

    Without the guard this deletes every point of the vanished
    source, because the dir could not be walked so none of its paths
    made it into the alive-set.
    """
    with pytest.raises(PruneScopeError) as excinfo:
        prune_missing(
            qdrant_with_points,
            "images",
            source_dirs=[str(source_dirs["present"]), str(source_dirs["vanished"])],
            source_names=["present", "vanished"],
        )

    assert "vanished" in str(excinfo.value)
    # Nothing was deleted — the abort happened before any delete.
    assert qdrant_with_points.count(collection_name="images", exact=True).count == 8


def test_all_source_dirs_present_still_prunes(qdrant_with_points, source_dirs) -> None:
    """The guard does not disable normal pruning.

    `present` has 4 points but only 2 files on disk, so 2 of its
    points are legitimately prunable. `vanished` is created empty —
    every one of its points is genuinely dead.
    """
    source_dirs["vanished"].mkdir()

    removed = prune_missing(
        qdrant_with_points,
        "images",
        source_dirs=[str(source_dirs["present"]), str(source_dirs["vanished"])],
        source_names=["present", "vanished"],
    )

    # 2 missing under `present` (files 2,3) + 4 under `vanished` (dir empty)
    assert removed == 6
    assert qdrant_with_points.count(collection_name="images", exact=True).count == 2


def test_prune_never_touches_unmanaged_sources(qdrant_with_points, source_dirs) -> None:
    """Points outside the managed source names survive the prune."""
    removed = prune_missing(
        qdrant_with_points,
        "images",
        source_dirs=[str(source_dirs["present"])],
        source_names=["present"],
    )

    # 2 of the 4 `present` points have no file -> pruned. The other 2
    # `present` points have files and survive, as do all 4 `vanished`
    # points — they are out of scope and must never be touched.
    assert removed == 2
    remaining = qdrant_with_points.scroll(
        collection_name="images", limit=100, with_payload=["collection"],
    )[0]
    assert len(remaining) == 6
    assert {p.payload["collection"] for p in remaining} == {"present", "vanished"}
    vanished_left = [p for p in remaining if p.payload["collection"] == "vanished"]
    assert len(vanished_left) == 4
