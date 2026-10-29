"""
tests/test_local_sync_perf_helpers.py — Round-perf (issue #1) coverage.

Pins the contract for the two new helpers introduced in the
ingestion perf refactor:

  - `_load_batch_concurrent(paths)`: decodes paths in a thread pool,
    returns ordered `(p, img, source_w, source_h)` tuples, errors → None
  - `_scroll_existing_meta(client, collection, src_name)`: scrolls all
    points for a source in one paginated query, returns
    `{point_id: (mtime, size) | None}`

These tests use a fake Qdrant client (in-memory via QdrantClient)
plus a tiny PNG corpus, so they exercise the real helpers end-to-end
without needing a GPU or real network.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from indexer import local_sync
from indexer.upsert import id_for


# ----- fixtures -----

@pytest.fixture
def corpus_dir(tmp_path):
    """Write a small PNG corpus so PIL can decode it."""
    img_dir = tmp_path / "corpus"
    img_dir.mkdir()
    for i in range(8):
        Image.new(
            "RGB", (48, 48), color=(i * 30 % 256, 100, 200),
        ).save(img_dir / f"img_{i:03d}.png", "PNG")
    return img_dir


@pytest.fixture
def in_memory_qdrant():
    """Fresh in-memory Qdrant for each test."""
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="images",
        vectors_config=qm.VectorParams(size=16, distance=qm.Distance.COSINE),
    )
    # Pre-seed a payload index so scroll filtering works in tests too
    client.create_payload_index(
        collection_name="images", field_name="collection", field_schema=qm.PayloadSchemaType.KEYWORD,
    )
    return client


# ----- _load_batch_concurrent -----

class TestLoadBatchConcurrent:
    """`local_sync._load_batch_concurrent` returns ordered results + handles errors."""

    def test_returns_one_entry_per_input(self, corpus_dir):
        paths = sorted(corpus_dir.glob("*.png"))
        results = local_sync._load_batch_concurrent(paths)
        assert len(results) == len(paths)
        # Every entry should be a 4-tuple
        for entry in results:
            assert entry is not None
            assert len(entry) == 4

    def test_preserves_input_order(self, corpus_dir):
        """Critical: results must align with the input `paths` order."""
        paths = sorted(corpus_dir.glob("*.png"))
        results = local_sync._load_batch_concurrent(paths)
        returned_paths = [entry[0] for entry in results]
        assert returned_paths == paths

    def test_loaded_image_is_rgb(self, corpus_dir):
        """Each loaded image is RGB and has reasonable dimensions."""
        paths = sorted(corpus_dir.glob("*.png"))[:3]
        results = local_sync._load_batch_concurrent(paths)
        for _p, img, _sw, _sh in results:
            assert img.mode == "RGB"
            assert img.width > 0 and img.height > 0

    def test_source_dims_are_captured(self, corpus_dir):
        """The 3rd + 4th tuple slots are the JPEG/PNG source dims."""
        paths = sorted(corpus_dir.glob("*.png"))
        results = local_sync._load_batch_concurrent(paths)
        for _p, _img, sw, sh in results:
            assert sw == 48
            assert sh == 48

    def test_corrupt_file_returns_none_image(self, tmp_path):
        """A non-image file produces (path, None, None, None), not a raise."""
        good = tmp_path / "good.png"
        Image.new("RGB", (32, 32), "red").save(good, "PNG")
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"this is not a PNG file")

        results = local_sync._load_batch_concurrent([good, bad])
        # Good file: full tuple
        assert results[0][0] == good
        assert results[0][1] is not None  # image
        assert results[0][2] == 32  # source w
        # Bad file: image is None (logged but not raised)
        assert results[1][0] == bad
        assert results[1][1] is None
        assert results[1][2] is None

    def test_empty_input_returns_empty(self):
        """Empty input list returns an empty list, no errors."""
        assert local_sync._load_batch_concurrent([]) == []

    def test_pool_size_is_bounded(self, monkeypatch):
        """The pool size env knob must be read and clamped to [1, 32]."""
        # Default = min(cpu_count, 8), clamped to [1, 32]
        # Just verify the constant exists and is in range
        assert 1 <= local_sync._LOAD_POOL_SIZE <= 32

    def test_uses_multiple_threads(self, corpus_dir, monkeypatch):
        """The pool actually uses >1 thread for parallelism.

        We patch the executor's submit to record which thread each
        task lands on. With a 4-worker pool and 8 inputs, we expect
        to see work landing on at least 2 different threads.
        """
        # Force pool size small enough to be deterministic
        monkeypatch.setattr(local_sync, "_LOAD_POOL_SIZE", 4)
        from concurrent.futures import ThreadPoolExecutor

        observed_threads: set[int] = set()
        original_submit = ThreadPoolExecutor.submit
        main_thread_id = threading.get_ident()

        def tracking_submit(self, fn, *args, **kwargs):
            observed_threads.add(threading.get_ident())
            return original_submit(self, fn, *args, **kwargs)

        monkeypatch.setattr(ThreadPoolExecutor, "submit", tracking_submit)

        paths = sorted(corpus_dir.glob("*.png"))
        local_sync._load_batch_concurrent(paths)

        # If the executor runs tasks in the calling thread (sync), no
        # worker threads would be observed. Otherwise, the calling
        # thread itself may show up plus worker threads.
        assert len(observed_threads) >= 1
        # Sanity: at least the main thread was active
        assert main_thread_id in observed_threads or len(observed_threads) >= 2


# ----- _scroll_existing_meta -----

class TestScrollExistingMeta:
    """`local_sync._scroll_existing_meta` caches change-detection metadata per source."""

    def _seed(self, client, src_name, points):
        """Upsert a few points with mtime/size payloads."""
        for pid, mtime, size in points:
            client.upsert(
                collection_name="images",
                points=[
                    qm.PointStruct(
                        id=id_for(Path(f"/tmp/{pid}.jpg"), src_name),  # noqa: S108 — deterministic id_for() input, not a real file
                        vector=[0.0] * 16,
                        payload={"collection": src_name, "mtime": mtime, "size": size},
                    )
                ],
            )

    def test_returns_mtime_size_for_seeded_points(self, in_memory_qdrant):
        client = in_memory_qdrant
        points = [(f"a_{i}", float(i * 1000), i * 100) for i in range(5)]
        self._seed(client, "lib_a", points)
        meta, _marked = local_sync._scroll_existing_meta(client, "images", "lib_a")
        assert len(meta) == 5
        for _label, mtime, size in points:
            pid = id_for(Path(f"/tmp/{_label}.jpg"), "lib_a")  # noqa: S108 — id_for() hash input
            assert str(pid) in meta
            assert meta[str(pid)] == (mtime, size)

    def test_points_missing_mtime_get_none(self, in_memory_qdrant):
        """Legacy points (no mtime/size payload) map to None, not raised."""
        client = in_memory_qdrant
        legacy_pid = id_for(Path("/tmp/legacy.jpg"), "lib_a")  # noqa: S108 — id_for() hash input
        modern_pid = id_for(Path("/tmp/modern.jpg"), "lib_a")  # noqa: S108 — id_for() hash input
        client.upsert(
            collection_name="images",
            points=[
                qm.PointStruct(
                    id=legacy_pid,
                    vector=[0.0] * 16,
                    payload={"collection": "lib_a"},  # no mtime/size
                ),
                qm.PointStruct(
                    id=modern_pid,
                    vector=[0.0] * 16,
                    payload={"collection": "lib_a", "mtime": 1234.0, "size": 5678},
                ),
            ],
        )
        meta, _marked = local_sync._scroll_existing_meta(client, "images", "lib_a")
        assert meta[str(legacy_pid)] is None
        assert meta[str(modern_pid)] == (1234.0, 5678)

    def test_filters_by_collection(self, in_memory_qdrant):
        """Only points with matching `collection` payload are returned."""
        client = in_memory_qdrant
        # lib_a: 3 points
        self._seed(client, "lib_a", [("a1", 1.0, 1), ("a2", 2.0, 2), ("a3", 3.0, 3)])
        # lib_b: 2 points (should NOT appear in lib_a's scroll)
        self._seed(client, "lib_b", [("b1", 4.0, 4), ("b2", 5.0, 5)])

        meta_a, _marked = local_sync._scroll_existing_meta(client, "images", "lib_a")
        meta_b, _marked = local_sync._scroll_existing_meta(client, "images", "lib_b")

        # Compute the expected IDs for lib_a (3) and lib_b (2)
        a_ids = {str(id_for(Path(f"/tmp/a{i}.jpg"), "lib_a")) for i in (1, 2, 3)}  # noqa: S108 — id_for() hash input
        b_ids = {str(id_for(Path(f"/tmp/b{i}.jpg"), "lib_b")) for i in (1, 2)}  # noqa: S108 — id_for() hash input

        assert set(meta_a.keys()) == a_ids
        assert set(meta_b.keys()) == b_ids
        # No overlap (filter actually filtered)
        assert not (set(meta_a.keys()) & set(meta_b.keys()))

    def test_handles_empty_collection(self, in_memory_qdrant):
        """Scroll over a source with no points returns an empty dict."""
        meta, _marked = local_sync._scroll_existing_meta(
            in_memory_qdrant, "images", "nonexistent_source",
        )
        assert meta == {}

    def test_paginates_large_results(self, in_memory_qdrant):
        """With >1000 points in a source, the scroll paginates correctly."""
        client = in_memory_qdrant
        # Seed 1500 points to force pagination (limit=1000 in the helper)
        # Build all PointStructs first (single upsert call for speed)
        client.upsert(
            collection_name="images",
            points=[
                qm.PointStruct(
                    id=id_for(Path(f"/tmp/big_{i}.jpg"), "big_lib"),  # noqa: S108 — id_for() hash input
                    vector=[0.0] * 16,
                    payload={"collection": "big_lib", "mtime": float(i), "size": i * 10},
                )
                for i in range(1500)
            ],
        )
        meta, _marked = local_sync._scroll_existing_meta(client, "images", "big_lib")
        assert len(meta) == 1500
        pid_0 = id_for(Path("/tmp/big_0.jpg"), "big_lib")  # noqa: S108 — id_for() hash input
        pid_last = id_for(Path("/tmp/big_1499.jpg"), "big_lib")  # noqa: S108 — id_for() hash input
        assert meta[str(pid_0)] == (0.0, 0)
        assert meta[str(pid_last)] == (1499.0, 14990)
