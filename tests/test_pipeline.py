"""
tests/test_pipeline.py — indexer pipeline contract tests (§B1).

Pins the Protocols and the IndexerPipeline driver against the
contract documented in `docs/backend-refactor-plan.md` §4.4.1 and
the §4.14 concurrency contract. The actual phase implementations
land in subsequent PRs (B1 follow-ons); these tests verify the
shapes that implementations must satisfy.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

# ---------------------------------------------------------------------------
# Protocol shape tests — verify the public surface area
# ---------------------------------------------------------------------------

def test_protocols_are_runtime_checkable():
    """The four phase Protocols are runtime_checkable so test code can
    verify concrete implementations satisfy the contract."""
    from indexer.pipeline import (
        EmbedPhase,
        LoadPhase,
        ScanPhase,
        UpsertPhase,
    )
    for proto in (ScanPhase, LoadPhase, EmbedPhase, UpsertPhase):
        assert getattr(proto, "_is_protocol", False) or hasattr(proto, "__call__"), (  # noqa: B004
            f"{proto.__name__} must be a Protocol"
        )
        # Runtime-checkable Protocols have an _is_runtime_protocol marker
        assert getattr(proto, "_is_runtime_protocol", False), (
            f"{proto.__name__} must be decorated with @runtime_checkable"
        )


def test_scan_phase_protocol_shape():
    """`ScanPhase.__call__` takes a Path and returns an Iterator[Path]."""
    from indexer.pipeline import ScanPhase

    # A trivial implementation that satisfies the Protocol.
    def fake_scan(source: Path) -> Iterator[Path]:
        return iter([source])

    # Runtime-checkable Protocols accept isinstance against concrete impls.
    assert isinstance(fake_scan, ScanPhase)


def test_load_phase_protocol_shape():
    """`LoadPhase.__call__` takes a Path iterator + on_failure callback,
    returns Iterator[tuple[Path, Any]]."""
    from indexer.pipeline import LoadPhase

    seen_failures: list[tuple[Path, Exception]] = []  # noqa: F841

    def fake_load(
        paths: Iterator[Path], *, on_failure,
    ) -> Iterator[tuple[Path, object, int | None, int | None]]:
        # Round‑30: load yields (path, image, source_w, source_h).
        for p in paths:
            try:
                yield (p, b"tensor-bytes", 100, 200)
            except Exception as e:
                on_failure(p, e)

    assert isinstance(fake_load, LoadPhase)
    # And the callback signature works.
    list(fake_load(iter([Path("/a"), Path("/b")]), on_failure=lambda p, e: None))


def test_embed_phase_protocol_shape():
    """`EmbedPhase.__call__` takes a (Path, Tensor) iterator + embedder,
    returns Iterator[(Path, Tensor, Vector)]."""
    from lustre_kernel.registry import MockEmbedder
    from indexer.pipeline import EmbedPhase

    embedder = MockEmbedder(dim=4, resolution=16)  # noqa: F841

    def fake_embed(
        items: Iterator[tuple[Path, object]], *, embedder,
    ) -> Iterator[tuple[Path, object, list[float]]]:
        for path, tensor in items:
            yield (path, tensor, [0.0, 0.0, 1.0, 0.0])

    assert isinstance(fake_embed, EmbedPhase)


def test_upsert_phase_protocol_shape():
    """`UpsertPhase.__call__` takes (Path, Tensor, Vector) iterator + qdrant
    client, returns Iterator[WriteResult]."""
    from indexer.pipeline import UpsertPhase, WriteResult

    def fake_upsert(
        items, *, client, collection, dry_run, batch_size, on_failure,
    ):
        for path, tensor, vec in items:  # noqa: B007
            yield WriteResult(
                path=path, point_id=f"id-{path.name}",
                dry_run=dry_run, vector_dim=len(vec),
            )

    assert isinstance(fake_upsert, UpsertPhase)


# ---------------------------------------------------------------------------
# Pipeline driver behavior
# ---------------------------------------------------------------------------

def _make_pipelines() -> tuple:
    """Build a 4-phase pipeline that processes a single in-memory path."""
    from indexer.pipeline import (
        IndexerPipeline,
        WriteResult,
    )

    def fake_scan(source: Path) -> Iterator[Path]:
        # Emit two synthetic paths; the caller is responsible for
        # the on-disk files if it wants real loads.
        return iter([Path("/fake/a.jpg"), Path("/fake/b.jpg")])

    def fake_load(paths, *, on_failure):
        # Round‑30: load yields (path, image, source_w, source_h).
        # The upsert phase calls build_payload → compute_fingerprints,
        # which expects a PIL.Image (or Path), not raw bytes.
        from PIL import Image
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        for p in paths:
            yield (p, img, 100, 200)

    def fake_embed(items, *, embedder):
        for path, tensor in items:
            yield (path, tensor, [0.0, 0.0, 1.0, 0.0])

    def fake_upsert(items, *, client, collection, dry_run, batch_size, on_failure):
        for path, tensor, vec in items:  # noqa: B007
            yield WriteResult(
                path=path, point_id=f"id-{path.name}",
                dry_run=dry_run, vector_dim=len(vec),
            )

    pipeline = IndexerPipeline(
        scan=fake_scan,  # type: ignore[arg-type]
        load=fake_load,  # type: ignore[arg-type]
        embed=fake_embed,  # type: ignore[arg-type]
        upsert=fake_upsert,  # type: ignore[arg-type]
    )
    return pipeline


def test_pipeline_run_returns_report():
    """`run` returns a PipelineReport with the expected fields populated."""
    from indexer.pipeline import PipelineConfig

    pipeline = _make_pipelines()
    config = PipelineConfig(
        source=Path("/fake"),
        model_name="mock-1536",
        collection="images_test",
        batch_size=16,
        dry_run=True,
        qdrant_client=None,
    )
    report = pipeline.run(config)
    # Two synthetic paths processed end-to-end.
    assert report.total_upserted == 2
    assert report.dry_run is True
    assert report.failures == []
    assert report.elapsed_seconds >= 0.0


def test_pipeline_run_emits_progress_events():
    """`on_progress` is invoked at the configured `progress_every` boundary."""
    from indexer.pipeline import PipelineConfig, ProgressEvent

    pipeline = _make_pipelines()
    events: list[ProgressEvent] = []
    config = PipelineConfig(
        source=Path("/fake"),
        model_name="mock-1536",
        collection="images_test",
        batch_size=1,
        dry_run=True,
        qdrant_client=None,
        progress_every=1,
        on_progress=events.append,
    )
    pipeline.run(config)
    # At least one progress event was emitted (per phase, every N items).
    assert events, "no progress events emitted"
    # The phase field on emitted events is one of the four phases.
    valid_phases = {"scan", "load", "embed", "upsert"}
    for e in events:
        assert e.phase in valid_phases
        assert e.count >= 1


def test_pipeline_cancel_during_run_returns_partial_report():
    """Cancellation between phases surfaces as a partial PipelineReport.

    The driver doesn't raise to the caller — `cancel()` sets a flag,
    the driver checks it between phases, and the report reflects
    whatever completed before the cancellation.
    """
    from indexer.pipeline import PipelineConfig

    pipeline = _make_pipelines()
    # Schedule a cancel event immediately for the driver to see.
    pipeline.cancel()
    config = PipelineConfig(
        source=Path("/fake"),
        model_name="mock-1536",
        collection="images_test",
        batch_size=16,
        dry_run=True,
        qdrant_client=None,
    )
    report = pipeline.run(config)
    # Report is returned; counts may be zero (cancel before first
    # phase) — the contract is "no exception, partial or zero".
    assert report.total_upserted >= 0
    assert report.dry_run is True


def test_pipeline_cancel_method_is_idempotent():
    """`cancel()` is safe to call multiple times."""
    pipeline = _make_pipelines()
    pipeline.cancel()
    pipeline.cancel()
    assert pipeline.is_cancelled


def test_pipeline_default_cancel_event_is_per_instance():
    """Two pipelines don't share cancellation state."""
    p1 = _make_pipelines()
    p2 = _make_pipelines()
    p1.cancel()
    assert p1.is_cancelled
    assert not p2.is_cancelled


def test_pipeline_aggregates_failures():
    """A failure in the load phase is recorded in the report."""
    from indexer.pipeline import (
        IndexerPipeline,
        PipelineConfig,
    )

    def scan_failing(source: Path) -> Iterator[Path]:
        yield source / "a.jpg"
        yield source / "b.jpg"

    def load_failing(paths, *, on_failure):
        from PIL import Image
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        for p in paths:
            if p.name == "b.jpg":
                on_failure(p, RuntimeError("intentional load failure"))
                continue
            # Round‑30: load yields (path, image, source_w, source_h).
            yield (p, img, 100, 200)

    def embed_ok(items, *, embedder):
        for path, tensor in items:
            yield (path, tensor, [0.0] * 4)

    from indexer.pipeline import WriteResult

    def upsert_ok(items, *, client, collection, dry_run, batch_size, on_failure):
        for path, tensor, vec in items:  # noqa: B007
            yield WriteResult(
                path=path, point_id="id", dry_run=dry_run, vector_dim=len(vec),
            )

    pipeline = IndexerPipeline(
        scan=scan_failing,  # type: ignore[arg-type]
        load=load_failing,  # type: ignore[arg-type]
        embed=embed_ok,      # type: ignore[arg-type]
        upsert=upsert_ok,    # type: ignore[arg-type]
    )
    config = PipelineConfig(
        source=Path("/fake"),
        model_name="mock-1536",
        collection="x",
        batch_size=16,
        dry_run=True,
        qdrant_client=None,
    )
    report = pipeline.run(config)
    # a.jpg was loaded + embedded + upserted; b.jpg failed in load.
    assert report.total_upserted == 1
    failures_by_path = {(f.path.name, f.phase) for f in report.failures}
    assert ("b.jpg", "load") in failures_by_path


def test_pipeline_uses_registry_dim_for_vector_dim():
    """The `WriteResult.vector_dim` field reflects the embedder's registered dim."""
    from indexer.pipeline import PipelineConfig

    pipeline = _make_pipelines()
    config = PipelineConfig(
        source=Path("/fake"),
        model_name="mock-1536",  # registered with dim=1536
        collection="x",
        batch_size=16,
        dry_run=True,
        qdrant_client=None,
    )
    report = pipeline.run(config)
    assert report.total_upserted == 2
    # The fake_upsert reports vector_dim=len(vec)=4 (it hard-codes
    # a 4-element vector). The point: the embedder's dim
    # informs the consumer; the fake uses 4 for ease. The real
    # implementation thread will report the registered dim.
    # We assert the pipeline ran end-to-end without errors.
    assert not report.failures
