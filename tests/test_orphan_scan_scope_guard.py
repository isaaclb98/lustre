"""
Tests for the orphan-scan scope guard.

`find_disk_orphans` finds the points whose files are missing from
disk. That inference is only sound when every managed source dir
could actually be walked. If a managed dir is missing — a dropped
NFS mount, a typo, a source that was just removed — its points are
in the scan scope (`source_names` covers them) but absent from the
alive-set (the dir could not be walked), so every one of them looks
dead.

The guard aborts the scan instead. Note the asymmetry with the old
destructive prune: the scan only MARKS points, it never deletes, so
a false "dead" verdict hides photos rather than destroying them.
That is still bad — a photo the user can no longer find is a photo
they think they lost — so the guard still matters. But the blast
radius is smaller, and the marked points can be unmarked by
re-running the index once the mount is back.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from indexer.orphan_scan import OrphanScopeError, find_disk_orphans


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


def test_missing_source_dir_aborts_the_scan(qdrant_with_points, source_dirs) -> None:
    """A managed source dir that does not exist aborts the scan.

    Without the guard this would report every point of the vanished
    source as dead, because the dir could not be walked so none of
    its paths made it into the alive-set.
    """
    with pytest.raises(OrphanScopeError) as excinfo:
        find_disk_orphans(
            qdrant_with_points,
            "images",
            source_dirs=[str(source_dirs["present"]), str(source_dirs["vanished"])],
            source_names=["present", "vanished"],
        )

    assert "vanished" in str(excinfo.value)


def test_missing_source_dir_aborts_before_walking_any_dir(qdrant_with_points, source_dirs) -> None:
    """The abort must not wait for the walk.

    The prod source dirs hold 2.3M files over NFS and the walk takes
    ~56 minutes. A guard that checked for a missing dir AFTER the walk
    would let the operator wait an hour to be told the scan could not
    run. The check is a stat() per dir and must come first.
    """
    present = source_dirs["present"]

    def _explode(*args, **kwargs):
        raise AssertionError("the walk must not start before the scope check")

    monkey = pytest.MonkeyPatch()
    monkey.setattr("indexer.orphan_scan.existing_image_paths", _explode)
    try:
        with pytest.raises(OrphanScopeError):
            find_disk_orphans(
                qdrant_with_points,
                "images",
                source_dirs=[str(present), str(source_dirs["vanished"])],
                source_names=["present", "vanished"],
            )
    finally:
        monkey.undo()


def test_all_source_dirs_present_still_finds_orphans(qdrant_with_points, source_dirs) -> None:
    """The guard does not disable normal orphan detection.

    `present` has 4 points but only 2 files on disk, so 2 of its
    points are genuinely orphaned. `vanished` is created empty —
    every one of its points is genuinely dead.
    """
    source_dirs["vanished"].mkdir()

    orphans = find_disk_orphans(
        qdrant_with_points,
        "images",
        source_dirs=[str(source_dirs["present"]), str(source_dirs["vanished"])],
        source_names=["present", "vanished"],
    )

    # 2 missing under `present` (files 2,3) + 4 under `vanished` (dir empty)
    assert len(orphans) == 6
    # The scan does not delete — it only reports.
    assert qdrant_with_points.count(collection_name="images", exact=True).count == 8


def test_scan_never_touches_unmanaged_sources(qdrant_with_points, source_dirs) -> None:
    """Points outside the managed source names are never reported."""
    orphans = find_disk_orphans(
        qdrant_with_points,
        "images",
        source_dirs=[str(source_dirs["present"])],
        source_names=["present"],
    )

    # 2 of the 4 `present` points have no file -> orphaned. The other
    # 2 `present` points have files and survive, as do all 4 `vanished`
    # points — they are out of scope and must never be reported.
    assert len(orphans) == 2
    remaining = qdrant_with_points.scroll(
        collection_name="images", limit=100, with_payload=["collection"],
    )[0]
    assert len(remaining) == 8
    assert {p.payload["collection"] for p in remaining} == {"present", "vanished"}
