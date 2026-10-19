"""
tests/test_pipeline_integration.py — pipeline + real indexer modules.

Pins the §B1 contract by plugging the real `indexer/scan`,
`indexer/image_loader`, and `indexer/upsert` modules into the
`IndexerPipeline` Protocols defined in `indexer/pipeline.py`. A
test using fake phases proves the driver; this file proves the
contract holds when the production modules are the consumers.

If a future refactor changes any of these modules in a way that
breaks the pipeline contract, these tests fail before the change
ships.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image
from qdrant_client import QdrantClient

# ---------------------------------------------------------------------------
# Adapter phases: wrap real indexer modules to satisfy the Protocols.
# ---------------------------------------------------------------------------

def _scan_adapter(source: Path) -> Iterator[Path]:
    """`indexer.scan.snapshot` returns `list[Path]`; the Protocol
    expects an `Iterator[Path]`. Adapting here keeps the contract
    honest (the real consumer must yield, not return a list)."""
    from indexer.scan import snapshot

    return iter(snapshot(source))


def _load_adapter(
    paths: Iterator[Path], *, on_failure,
) -> Iterator[tuple[Path, object, int | None, int | None]]:
    """Wrap `indexer.image_loader.load` so per-file `LoaderError`s
    are reported via `on_failure` rather than aborting the pipeline.

    Round‑30: `load()` returns `(img, source_w, source_h)`; we
    propagate the source dims through to the upsert phase.
    """
    from indexer.image_loader import LoaderError, load

    for p in paths:
        try:
            img, source_w, source_h = load(p)
            yield (p, img, source_w, source_h)
        except LoaderError as e:
            on_failure(p, e)
        except Exception as e:  # defensive: PIL's zoo of exceptions
            on_failure(p, e)


def _embed_adapter(
    items: Iterator[tuple[Path, object]], *, embedder,
) -> Iterator[tuple[Path, object, list[float]]]:
    """Embed each loaded PIL.Image via the registered embedder.

    The Embedder Protocol's `embed_image(image)` returns a list[float];
    we surface the image along so downstream code can correlate.
    """
    for path, image in items:
        vec = embedder.embed_image(image)
        yield (path, image, vec)


def _upsert_adapter(items, *, client, collection, dry_run, batch_size, on_failure):
    """Wrap `indexer.upsert.upsert_batch` so the pipeline contract
    (per-item failure reporting) is honored at the qdrant boundary."""
    from qdrant_client.http import models as qmodels

    from lustre_kernel.payload_schema import (
        FIELD_PATH,
    )
    from lustre_kernel.registry import get as _registry_get
    from indexer.pipeline import WriteResult
    from indexer.upsert import (
        build_payload,
        id_for,
    )

    def _registry_model_dim_for(model_name: str) -> int:
        return _registry_get(model_name).dim

    batch: list = []
    for path, image, vec in items:  # noqa: B007
        try:
            model_name = "mock-1536"  # tests register this in conftest
            model_revision = "test-r0"
            dim = _registry_model_dim_for(model_name)
            payload = build_payload(
                path=path, shard="",
                model_name=model_name, model_revision=model_revision,
                collection=collection,
                model_dim=dim,
            )
            point_id = id_for(path, shard="")
            batch.append(qmodels.PointStruct(
                id=point_id, vector=vec, payload=payload,
            ))
            yield WriteResult(
                path=path, point_id=point_id, dry_run=dry_run,
                vector_dim=len(vec),
            )
        except Exception as e:
            on_failure(path, e)

    if batch and not dry_run:
        try:
            client.upsert(collection_name=collection, points=batch, wait=False)
        except Exception as e:
            for point in batch:
                on_failure(Path(point.payload.get(FIELD_PATH, "<unknown>")), e)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

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


def test_pipeline_runs_end_to_end_with_real_modules(synth_corpus, qdrant_in_memory):
    """Real indexer modules + mock embedder + in-memory Qdrant:
    10 synthetic images index end-to-end with v1 schema fields."""
    from indexer.pipeline import IndexerPipeline, PipelineConfig
    from indexer.upsert import ensure_collection

    pipeline = IndexerPipeline(
        scan=_scan_adapter,      # type: ignore[arg-type]
        load=_load_adapter,      # type: ignore[arg-type]
        embed=_embed_adapter,    # type: ignore[arg-type]
        upsert=_upsert_adapter,  # type: ignore[arg-type]
    )

    # Create the target collection at the mock embedder's dim. The
    # registry's mock-1536 spec is patched at registry-build time
    # to match the active prod variant (so400m/16-384 = 1152 today;
    # was 1536 pre-migration), so the right dim to use is whatever
    # `indexer.upsert.VECTOR_DIM` resolves to at this moment — not
    # a hardcoded literal.
    from indexer.upsert import VECTOR_DIM
    ensure_collection(
        qdrant_in_memory, "images_pipeline_test",
        dim=VECTOR_DIM,
    )

    config = PipelineConfig(
        source=synth_corpus,
        model_name="mock-1536",
        collection="images_pipeline_test",
        batch_size=4,
        dry_run=False,
        qdrant_client=qdrant_in_memory,
        progress_every=5,
    )
    # The pipeline records the active model via `_resolve_active_model_name`,
    # which falls back to the production default unless the test sets the
    # active-model global. Without this, payloads would carry
    # "ViT-so400m-patch16-384" instead of "mock-1536".
    from indexer.run_pipeline import set_active_model
    set_active_model("mock-1536", "test-r0")
    report = pipeline.run(config)

    # 10 synthetic images, no failures expected.
    assert report.total_upserted == 10, (
        f"expected 10 upserted, got {report.total_upserted}; "
        f"failures: {report.failures}"
    )
    assert report.failures == []
    assert report.dry_run is False

    # Verify every point landed in qdrant with the schema fields.
    from lustre_kernel.payload_schema import (
        FIELD_FOLDER,
        FIELD_MODEL_DIM,
        FIELD_MODEL_NAME,
        FIELD_MODEL_REVISION,
        FIELD_PATH,
    )

    points, _ = qdrant_in_memory.scroll(
        collection_name="images_pipeline_test",
        with_payload=True, with_vectors=False, limit=100,
    )
    assert len(points) == 10

    for p in points:
        pl = p.payload or {}
        # Schema fields are present. model_dim mirrors the embedder's
        # dim — mock-1536 is patched at registry build to match the
        # active prod variant (1152 today), so we assert against
        # VECTOR_DIM rather than a hardcoded literal.
        assert pl.get(FIELD_MODEL_DIM) == VECTOR_DIM
        assert pl.get(FIELD_MODEL_NAME) == "mock-1536"
        assert pl.get(FIELD_MODEL_REVISION) == "test-r0"
        assert pl.get(FIELD_PATH)  # absolute path string
        assert pl.get(FIELD_FOLDER) == str(synth_corpus.resolve())


def test_pipeline_handles_corrupt_files(synth_corpus, qdrant_in_memory):
    """Corrupt files are reported via `on_failure`, not raised."""
    from indexer.pipeline import IndexerPipeline, PipelineConfig
    from indexer.upsert import VECTOR_DIM, ensure_collection

    # Inject a corrupt file alongside the synth corpus.
    (synth_corpus / "corrupt.jpg").write_bytes(b"\xff\xff not a real jpeg")

    pipeline = IndexerPipeline(
        scan=_scan_adapter,      # type: ignore[arg-type]
        load=_load_adapter,      # type: ignore[arg-type]
        embed=_embed_adapter,    # type: ignore[arg-type]
        upsert=_upsert_adapter,  # type: ignore[arg-type]
    )

    ensure_collection(qdrant_in_memory, "images_pipeline_corrupt", dim=VECTOR_DIM)
    config = PipelineConfig(
        source=synth_corpus,
        model_name="mock-1536",
        collection="images_pipeline_corrupt",
        batch_size=4,
        dry_run=False,
        qdrant_client=qdrant_in_memory,
    )
    # Mirror the end-to-end test's active-model pin so the embedder
    # is mock-1536 (1152 today) and not the prod variant.
    from indexer.run_pipeline import set_active_model
    set_active_model("mock-1536", "test-r0")
    report = pipeline.run(config)

    # 10 good + 1 corrupt; 10 upserted, 1 failure reported.
    assert report.total_upserted == 10
    failures_by_path = {Path(f.path).name for f in report.failures}
    assert "corrupt.jpg" in failures_by_path


def test_pipeline_dry_run_does_not_upsert(synth_corpus, qdrant_in_memory):
    """`dry_run=True` returns the report's count without writing to Qdrant."""
    from indexer.pipeline import IndexerPipeline, PipelineConfig

    pipeline = IndexerPipeline(
        scan=_scan_adapter,
        load=_load_adapter,
        embed=_embed_adapter,
        upsert=_upsert_adapter,
    )
    config = PipelineConfig(
        source=synth_corpus,
        model_name="mock-1536",
        collection="images_pipeline_dryrun",
        batch_size=4,
        dry_run=True,
        qdrant_client=qdrant_in_memory,
    )
    report = pipeline.run(config)

    assert report.total_upserted == 10
    assert report.dry_run is True
    assert report.failures == []

    # Collection does not exist (dry-run didn't upsert, didn't create).
    cols = {c.name for c in qdrant_in_memory.get_collections().collections}
    assert "images_pipeline_dryrun" not in cols
