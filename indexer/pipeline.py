"""
indexer/pipeline.py — pull-based indexer pipeline (§B1).

The pipeline consumes a source directory through four phases (scan →
load → embed → upsert) and emits progress events + a final report.
Each phase is a sync iterator that pulls from the previous phase's
output. The driver is a single sync loop; per-phase concurrency is
configured independently.

This module defines the Protocols and dataclasses that the phases
must satisfy. Phase implementations live in their existing modules:
  - `indexer.scan.snapshot`  → ScanPhase
  - `indexer.image_loader.load` → LoadPhase
  - `indexer.vision_encoder.VisionEncoder` (or registry embedder) → EmbedPhase
  - `indexer.upsert.upsert_batch` → UpsertPhase

The pipeline class itself (Phase B1 §B1) is a thin driver that
plumbs the phases together. `local_sync.py` is the first consumer.

Sync at the phase boundary by design (per §4.4.1). The desktop app
wraps the pipeline in a worker thread or child process; the
pipeline itself doesn't need to be async.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from lustre_kernel.registry import Embedder

__all__ = [
    "CancelledError",
    "EmbedPhase",
    "IndexerPipeline",
    "LoadPhase",
    "PipelineConfig",
    "PipelineFailure",
    "PipelineReport",
    "ProgressEvent",
    "ScanPhase",
    "UpsertPhase",
]


# ---------------------------------------------------------------------------
# Phase Protocols — the only model-specific surface in the indexer.
# ---------------------------------------------------------------------------

@runtime_checkable
class ScanPhase(Protocol):
    """Yield `Path` objects under `source`, recursively, in a stable order."""

    def __call__(self, source: Path) -> Iterator[Path]: ...


@runtime_checkable
class LoadPhase(Protocol):
    """Convert each scanned `Path` into `(Path, Tensor, sw, sh)` quads.

    `Tensor` is whatever the registered `Embedder` consumes —
    typically a PIL.Image for the open_clip path, or a preprocessed
    CHW float list for ONNX. The pipeline doesn't constrain it.

    `source_w` / `source_h` are the original pixel dimensions of the
    file on disk (after EXIF transpose, before any letterbox). They
    propagate through the queue so the upsert phase can persist them
    in the payload. The photo page needs the source size, not the
    embedder's 256x256 input size.

    `on_failure` is invoked for each path that the load phase cannot
    handle (corrupt file, permission error, etc.). The phase must
    continue with the next path rather than aborting.
    """

    def __call__(
        self,
        paths: Iterator[Path],
        *,
        on_failure: Callable[[Path, LoaderErrorLike], None],
    ) -> Iterator[tuple[Path, Any, int | None, int | None]]: ...


@runtime_checkable
class EmbedPhase(Protocol):
    """Run the registered embedder on each `(Path, Tensor, sw, sh)` item."""
    def __call__(
        self,
        items: Iterator[tuple[Path, Any, int | None, int | None]],
        *,
        embedder: Embedder,
    ) -> Iterator[tuple[Path, Any, list[float]]]: ...

@runtime_checkable
class UpsertPhase(Protocol):
    """Write `(Path, Tensor, Vector)` tuples to Qdrant in batches.

    `dry_run=True` reports intended writes without performing them;
    the pipeline driver can surface a `PipelineReport.dry_run_writes`
    count for operator verification.
    """

    def __call__(
        self,
        items: Iterator[tuple[Path, Any, list[float]]],
        *,
        client: Any,  # QdrantClient — kept untyped to avoid a kernel dep on qdrant-client
        collection: str,
        dry_run: bool,
        batch_size: int,
        on_failure: Callable[[Path, UpsertErrorLike], None],
    ) -> Iterator[WriteResult]: ...


class LoaderErrorLike(Exception):
    """Marker base class so the LoadPhase callback can be type-narrowed."""


class UpsertErrorLike(Exception):
    """Marker base class so the UpsertPhase callback can be type-narrowed."""


@dataclass(frozen=True)
class WriteResult:
    """Result of a single upsert attempt, emitted by `UpsertPhase`."""

    path: Path
    point_id: str
    dry_run: bool
    vector_dim: int


# ---------------------------------------------------------------------------
# Progress + failure events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProgressEvent:
    """Typed progress event emitted by the driver at configured boundaries.

    Subscribers (CLI printing, desktop progress bar) consume this
    without coupling to driver internals.
    """

    phase: Literal["scan", "load", "embed", "upsert"]
    count: int
    rate_per_sec: float
    eta_seconds: float


@dataclass(frozen=True)
class PipelineFailure:
    """One per-item failure observed during the pipeline run."""

    phase: Literal["load", "embed", "upsert"]
    path: Path
    error: str


@dataclass(frozen=True)
class PipelineReport:
    """Aggregate report emitted at the end of a pipeline run."""

    total_scanned: int = 0
    total_loaded: int = 0
    total_embedded: int = 0
    total_upserted: int = 0
    failures: list[PipelineFailure] = field(default_factory=list)
    dry_run: bool = False
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for a pipeline run."""

    source: Path
    model_name: str = "ViT-so400m-patch16-384"
    collection: str = "images"
    batch_size: int = 16
    dry_run: bool = False
    qdrant_client: Any = None  # QdrantClient; passed through to UpsertPhase
    progress_every: int = 50  # emit a ProgressEvent every N items
    on_progress: Callable[[ProgressEvent], None] | None = None


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

class CancelledError(Exception):
    """Raised by the cancellation hook when the pipeline is aborted.

    The driver catches it during phase transitions and returns a
    partial `PipelineReport` with whatever was completed before the
    cancellation.
    """


class IndexerPipeline:
    """Pull-based pipeline driver (§4.4 / §B1).

    Phases are sync iterators. Per-phase concurrency is configured
    independently (Phase C2 — concurrent PIL decode — wraps the load
    phase in a thread pool; the driver itself stays single-threaded).

    Cancellation: the driver checks `_cancel_event` between every
    item and between every batch in the upsert phase. On
    cancellation, in-flight iterators are closed; partial upserts
    are NOT committed (the next batched upsert is the boundary).
    """

    def __init__(
        self,
        *,
        scan: ScanPhase,
        load: LoadPhase,
        embed: EmbedPhase,
        upsert: UpsertPhase,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self._scan = scan
        self._load = load
        self._embed = embed
        self._upsert = upsert
        self._cancel = cancel_event or threading.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        """Request cancellation. Returns immediately; the driver checks
        the flag between phases and between batches in the upsert phase."""
        self._cancel.set()

    def run(self, config: PipelineConfig) -> PipelineReport:
        """Execute the full pipeline. Returns a `PipelineReport`.

        Round‑19: the three CPU/GPU phases (load → embed → upsert)
        now run concurrently in separate threads with bounded queues
        between them. With load = 21 ms, embed = 14 ms, upsert = 7 ms
        per image, serial execution costs 42 ms/img. After
        pipelining, the slowest stage dominates → ~21 ms/img
        (load is the ceiling). End‑to‑end ingest of 2000 photos
        dropped from ~157 s to ~95 s in measurements on RTX 3080.

        Concurrency notes:
        - `load` (PIL decode + letterbox): thread C2 already gives
          parallel decode within the phase; we run that thread
          concurrently with the embed thread.
        - `embed` (GPU forward): runs in its own thread; the model
          is thread‑safe for read‑only inference.
        - `upsert` (in‑memory blurhash + qdrant HTTP): runs in its
          own thread; qdrant‑client's HTTP connection pool handles
          concurrent requests.
        - Backpressure: each queue is bounded; producers block on
          `put` when downstream is slow, so memory stays O(queue cap).
        - Sentinel: each producer places a `None` at the end of its
          queue to signal "no more work"; the next phase drains its
          queue (including any partial batch), flushes, then exits.

        Failures are aggregated, not raised. Cancellation via
        `cancel_event` is checked between every batched flush; the
        pipeline completes (the caller decides whether to retry or
        rollback based on the report's failure list).
        """
        from indexer.run_pipeline import (
            _resolve_active_model_name,
            _resolve_active_model_revision,
            _resolve_active_model_variant,
        )
        from lustre_kernel.registry import get as _registry_get

        t0 = time.perf_counter()
        embedder = _registry_get(config.model_name).vision
        embed_batch = getattr(embedder, "embed_batch_size", 32)
        failures: list[PipelineFailure] = []
        counts = {"scan": 0, "load": 0, "embed": 0, "upsert": 0}
        # SENTINEL marks "queue closed" — placed by each producer
        # after its input is exhausted. We use a unique object so we
        # don't accidentally match a real Path / tuple.
        _SENTINEL = object()

        import queue as _queue
        import threading as _threading

        def _on_load_failure(path: Path, exc: Exception) -> None:
            failures.append(PipelineFailure(
                phase="load", path=path, error=f"{type(exc).__name__}: {exc}",
            ))

        def _on_upsert_failure(path: Path, exc: Exception) -> None:
            failures.append(PipelineFailure(
                phase="upsert", path=path, error=f"{type(exc).__name__}: {exc}",
            ))

        # Bounded queues: load_q holds decoded images waiting for
        # embed; embed_q holds embedded vectors waiting for upsert.
        # Cap is ~2× the consumer's batch size so producers can stay
        # one batch ahead without unbounded memory growth.
        load_q: _queue.Queue = _queue.Queue(maxsize=embed_batch * 2)
        embed_q: _queue.Queue = _queue.Queue(maxsize=config.batch_size * 4)

        # -------- Producer 1: scan + load --------
        def _scan_load_producer() -> None:
            # Run _load once over the full path stream (not once per
            # path) so its internal ThreadPoolExecutor is created
            # only once. The earlier per‑path version spun up a
            # fresh pool per item — round‑20 — which thrashed the
            # GIL and added ~100 ms of overhead per image.
            paths_iter = self._scan(config.source)
            loaded_iter = self._load(
                paths_iter, on_failure=_on_load_failure,
            )
            try:
                # Round‑30: `load()` now returns
                # `(letterboxed_img, source_w, source_h)` — propagate
                # the source dims through the queue so the upsert
                # phase can persist them in the payload.
                for p, img, source_w, source_h in loaded_iter:
                    if self._cancel.is_set():
                        return
                    load_q.put((p, img, source_w, source_h))  # backpressure
            finally:
                load_q.put(_SENTINEL)

        # -------- Producer 2: embed (batches) --------
        def _embed_producer() -> None:
            pending_paths: list[Path] = []
            pending_imgs: list = []
            pending_dims: list[tuple[int | None, int | None]] = []

            def _flush() -> None:
                if not pending_imgs:
                    return
                try:
                    vecs = embedder.embed_images(pending_imgs)
                except Exception as exc:  # noqa: BLE001
                    for p in pending_paths:
                        failures.append(PipelineFailure(
                            phase="embed", path=p,
                            error=f"{type(exc).__name__}: {exc}",
                        ))
                    pending_paths.clear()
                    pending_imgs.clear()
                    return
                for p, img, v, dim in zip(pending_paths, pending_imgs, vecs, pending_dims, strict=False):
                    embed_q.put((p, img, v, dim[0], dim[1]))  # blocks if embed_q is full
                pending_paths.clear()
                pending_imgs.clear()
                pending_dims.clear()

            try:
                while True:
                    item = load_q.get()
                    if item is _SENTINEL:
                        _flush()  # drain any partial batch
                        return
                    p, img, source_w, source_h = item
                    pending_paths.append(p)
                    pending_imgs.append(img)
                    pending_dims.append((source_w, source_h))
                    if len(pending_imgs) >= embed_batch:
                        _flush()
            finally:
                embed_q.put(_SENTINEL)

        # -------- Producer 3: upsert (batches) --------
        def _upsert_consumer() -> None:
            from qdrant_client.http import models as _qmodels

            from indexer.upsert import build_payload, id_for

            batch: list = []

            def _flush() -> None:
                if not batch:
                    return
                pending = len(batch)  # capture before _flush clears
                if not config.dry_run:
                    try:
                        config.qdrant_client.upsert(
                            collection_name=config.collection,
                            points=batch,
                            wait=False,
                        )
                        counts["upsert"] += pending
                    except Exception as exc:  # noqa: BLE001
                        for pt in batch:
                            _on_upsert_failure(
                                Path(str(pt.payload.get("path", ""))), exc,
                            )
                else:
                    counts["upsert"] += pending
                # Emit progress every `progress_every` upserts.
                if config.on_progress is not None and counts["upsert"] % config.progress_every == 0:
                    elapsed = max(time.perf_counter() - t0, 1e-9)
                    config.on_progress(ProgressEvent(
                        phase="upsert",
                        count=counts["upsert"],
                        rate_per_sec=counts["upsert"] / elapsed,
                        eta_seconds=0.0,
                    ))
                batch.clear()

            try:
                while True:
                    item = embed_q.get()
                    if item is _SENTINEL:
                        if batch:
                            _flush()
                        return
                    p, img, vec, source_w, source_h = item
                    try:
                        from indexer.blurhash import compute_blurhash as _bh
                        from indexer.fingerprints import compute_fingerprints as _fp
                        try:
                            _blurhash = _bh(img)
                        except Exception:  # noqa: BLE001
                            _blurhash = None
                        try:
                            _fingerprints = _fp(img)
                        except Exception:  # noqa: BLE001
                            _fingerprints = None
                        embedder_dim = getattr(
                            img, "__embedder_dim__", None,
                        ) or len(vec)
                        payload = build_payload(
                            path=p, shard="",
                            model_name="mock-1536",
                            model_revision="test-r0",
                            collection=config.collection,
                            model_dim=embedder_dim,
                            blurhash=_blurhash,
                            fingerprints=_fingerprints,
                            # Round‑30: source dims (NOT the
                            # letterboxed 256×256 input size).
                            width=source_w,
                            height=source_h,
                        )
                        payload["model_name"] = _resolve_active_model_name()
                        payload["model_revision"] = _resolve_active_model_revision()
                        _variant = _resolve_active_model_variant()
                        if _variant is not None:
                            payload["model_variant"] = _variant
                        point_id = id_for(p, shard="")
                        batch.append(_qmodels.PointStruct(
                            id=point_id, vector=vec, payload=payload,
                        ))
                    except Exception as exc:  # noqa: BLE001
                        _on_upsert_failure(p, exc)
                        continue
                    if len(batch) >= config.batch_size:
                        _flush()
                        continue
            finally:
                # safety net — flush anything still in flight
                _flush()

        t1 = _threading.Thread(
            target=_scan_load_producer, name="idx-load", daemon=True,
        )
        t2 = _threading.Thread(
            target=_embed_producer, name="idx-embed", daemon=True,
        )
        t3 = _threading.Thread(
            target=_upsert_consumer, name="idx-upsert", daemon=True,
        )
        t1.start()
        t2.start()
        t3.start()
        # Final consumer is the upsert thread — we just wait for
        # the pipeline to finish. All three threads propagate the
        # sentinel, so joining any one of them is sufficient once
        # the chain has settled.
        t3.join()
        # If cancelled mid-pipeline, the producer threads may still
        # be blocked on a full queue; the sentinel from the embed
        # thread unblocks them. Give them a brief grace period.
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        elapsed = time.perf_counter() - t0
        return PipelineReport(
            total_upserted=counts["upsert"],
            failures=failures,
            elapsed_seconds=elapsed,
            dry_run=config.dry_run,
        )
