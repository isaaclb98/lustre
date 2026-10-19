"""
tests/test_run_pipeline.py — indexer/run_pipeline.py wrapper tests.

The wrapper at `indexer/run_pipeline.py` translates the simple
"index one source directory" use case into a `PipelineConfig` and
runs the new `IndexerPipeline`. These tests verify the wrapper
end-to-end against in-memory Qdrant.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PIL import Image
from qdrant_client import QdrantClient

from indexer.run_pipeline import (
    run_pipeline_source,
    set_active_model,
)


@pytest.fixture
def qdrant_in_memory():
    return QdrantClient(location=":memory:")


@pytest.fixture
def synth_corpus(tmp_path):
    """10 synthetic JPEGs in a flat directory."""
    n = 10
    images_dir = tmp_path / "corpus"
    images_dir.mkdir()
    for i in range(n):
        size = 32 + i * 8
        Image.new(
            "RGB", (size, size),
            color=((i * 25) % 256, (i * 50) % 256, (i * 75) % 256),
        ).save(images_dir / f"img_{i:03d}.jpg", "JPEG", quality=80)
    return images_dir


def test_run_pipeline_source_end_to_end(synth_corpus, qdrant_in_memory):
    """10 synthetic JPEGs index via the wrapper with v1 schema fields."""
    from lustre_kernel.payload_schema import (
        FIELD_FOLDER,
        FIELD_MODEL_DIM,
        FIELD_MODEL_NAME,
        FIELD_MODEL_REVISION,
        FIELD_PATH,
    )
    from indexer.upsert import VECTOR_DIM, ensure_collection

    set_active_model("mock-1536", "test-r0")
    ensure_collection(qdrant_in_memory, "images_run_pipe", dim=VECTOR_DIM)

    report = run_pipeline_source(
        source=synth_corpus,
        qdrant_client=qdrant_in_memory,
        collection="images_run_pipe",
        model_name="mock-1536",
        batch_size=4,
    )

    assert report.total_upserted == 10
    assert report.failures == []
    assert report.dry_run is False

    points, _ = qdrant_in_memory.scroll(
        collection_name="images_run_pipe",
        with_payload=True, with_vectors=False, limit=100,
    )
    assert len(points) == 10
    for p in points:
        pl = p.payload or {}
        assert pl.get(FIELD_MODEL_DIM) == VECTOR_DIM
        assert pl.get(FIELD_MODEL_NAME) == "mock-1536"
        assert pl.get(FIELD_MODEL_REVISION) == "test-r0"
        assert pl.get(FIELD_PATH)  # absolute path string
        assert pl.get(FIELD_FOLDER) == str(synth_corpus.resolve())


def test_run_pipeline_source_dry_run(synth_corpus, qdrant_in_memory):
    """dry_run=True returns the report without creating the collection."""
    set_active_model("mock-1536", "test-r0")
    report = run_pipeline_source(
        source=synth_corpus,
        qdrant_client=qdrant_in_memory,
        collection="images_dry_run",
        model_name="mock-1536",
        batch_size=4,
        dry_run=True,
    )
    assert report.total_upserted == 10
    assert report.dry_run is True
    cols = {c.name for c in qdrant_in_memory.get_collections().collections}
    assert "images_dry_run" not in cols


def test_run_pipeline_source_handles_corrupt_files(synth_corpus, qdrant_in_memory):
    """Corrupt files don't abort the pipeline; they're reported."""
    from indexer.upsert import VECTOR_DIM, ensure_collection

    (synth_corpus / "corrupt.jpg").write_bytes(b"\xff\xff not a real jpeg")

    set_active_model("mock-1536", "test-r0")
    ensure_collection(qdrant_in_memory, "images_corrupt", dim=VECTOR_DIM)

    report = run_pipeline_source(
        source=synth_corpus,
        qdrant_client=qdrant_in_memory,
        collection="images_corrupt",
        model_name="mock-1536",
        batch_size=4,
    )
    # 10 good files upserted; the corrupt file is reported as a load failure.
    assert report.total_upserted == 10
    failure_paths = {Path(f.path).name for f in report.failures}
    assert "corrupt.jpg" in failure_paths


def test_run_pipeline_source_emits_progress(synth_corpus, qdrant_in_memory):
    """`on_progress` is invoked when supplied."""
    from indexer.pipeline import ProgressEvent
    from indexer.upsert import VECTOR_DIM, ensure_collection

    set_active_model("mock-1536", "test-r0")
    ensure_collection(qdrant_in_memory, "images_progress", dim=VECTOR_DIM)

    events: list[ProgressEvent] = []
    run_pipeline_source(
        source=synth_corpus,
        qdrant_client=qdrant_in_memory,
        collection="images_progress",
        model_name="mock-1536",
        batch_size=2,
        on_progress=events.append,
    )
    # At least one progress event was emitted across the run.
    assert events


def test_run_pipeline_source_cancellation(synth_corpus, qdrant_in_memory):
    """Setting the cancel_event mid-run returns a partial report."""
    from indexer.upsert import VECTOR_DIM, ensure_collection

    set_active_model("mock-1536", "test-r0")
    ensure_collection(qdrant_in_memory, "images_cancel", dim=VECTOR_DIM)

    cancel = threading.Event()
    cancel.set()  # cancel before the run starts

    report = run_pipeline_source(
        source=synth_corpus,
        qdrant_client=qdrant_in_memory,
        collection="images_cancel",
        model_name="mock-1536",
        batch_size=4,
        cancel_event=cancel,
    )
    # Cancellation before any phase → zero upserted.
    assert report.total_upserted == 0
    assert report.dry_run is False


def test_run_pipeline_source_empty_source(qdrant_in_memory):
    """An empty source directory produces a zero-count report, not an error."""
    import tempfile

    from indexer.upsert import VECTOR_DIM, ensure_collection

    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp)
        set_active_model("mock-1536", "test-r0")
        ensure_collection(qdrant_in_memory, "images_empty", dim=VECTOR_DIM)

        report = run_pipeline_source(
            source=empty,
            qdrant_client=qdrant_in_memory,
            collection="images_empty",
            model_name="mock-1536",
        )
        assert report.total_upserted == 0
        assert report.failures == []


def test_set_active_model_persists_across_calls(synth_corpus, qdrant_in_memory):
    """`set_active_model` once is enough; subsequent runs use the pinned values."""
    from lustre_kernel.payload_schema import FIELD_MODEL_NAME, FIELD_MODEL_REVISION
    from indexer.upsert import VECTOR_DIM, ensure_collection

    ensure_collection(qdrant_in_memory, "images_pinned", dim=VECTOR_DIM)

    set_active_model("mock-1536", "pinned-rev-7")
    run_pipeline_source(
        source=synth_corpus,
        qdrant_client=qdrant_in_memory,
        collection="images_pinned",
        model_name="mock-1536",
        batch_size=4,
    )

    points, _ = qdrant_in_memory.scroll(
        collection_name="images_pinned",
        with_payload=True, with_vectors=False, limit=100,
    )
    for p in points:
        pl = p.payload or {}
        assert pl[FIELD_MODEL_NAME] == "mock-1536"
        assert pl[FIELD_MODEL_REVISION] == "pinned-rev-7"


# ---------------------------------------------------------------------------
# §C2 — concurrent PIL decode
# ---------------------------------------------------------------------------

def test_concurrent_load_completes_for_each_path(synth_corpus, qdrant_in_memory):
    """Concurrent decode processes every image exactly once."""
    from indexer.upsert import VECTOR_DIM, ensure_collection

    set_active_model("mock-1536", "test-r0")
    ensure_collection(qdrant_in_memory, "images_concurrent", dim=VECTOR_DIM)
    report = run_pipeline_source(
        source=synth_corpus,
        qdrant_client=qdrant_in_memory,
        collection="images_concurrent",
        model_name="mock-1536",
        batch_size=4,
    )
    n_images = len(list(synth_corpus.glob("*.jpg")))
    assert report.total_upserted == n_images
    assert report.failures == []


def test_concurrent_load_preserves_failure_aggregation(synth_corpus, qdrant_in_memory):
    """Concurrent decode: corrupt files reported via on_failure, not raised."""
    from indexer.upsert import VECTOR_DIM, ensure_collection

    (synth_corpus / "corrupt.jpg").write_bytes(b"\xff\xff not a real jpeg")
    set_active_model("mock-1536", "test-r0")
    ensure_collection(qdrant_in_memory, "images_concurrent_fail", dim=VECTOR_DIM)
    report = run_pipeline_source(
        source=synth_corpus,
        qdrant_client=qdrant_in_memory,
        collection="images_concurrent_fail",
        model_name="mock-1536",
        batch_size=4,
    )
    failure_paths = {Path(f.path).name for f in report.failures}
    assert "corrupt.jpg" in failure_paths
    # 10 good images still got indexed despite the corrupt one.
    assert report.total_upserted == 10


def test_concurrent_load_pool_size_configurable(monkeypatch):
    """`IMAGE_LOAD_POOL_SIZE` env var overrides the default pool size."""
    import importlib

    import indexer.run_pipeline as rp_module

    # Reload to pick up the env var at module import time.
    monkeypatch.setenv("IMAGE_LOAD_POOL_SIZE", "8")
    importlib.reload(rp_module)
    assert rp_module._LOAD_POOL_SIZE == 8
    # Reset for downstream tests.
    monkeypatch.undo()
    importlib.reload(rp_module)
