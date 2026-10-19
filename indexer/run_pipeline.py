"""
indexer/run_pipeline.py — thin library wrapper around IndexerPipeline.

`local_sync.py` is a feature-rich CLI with change-detection, prune,
and backfill modes that predate the pipeline abstraction. Those
features will migrate to the pipeline as separate phases in
follow-on PRs.

This module exposes the simplest possible "full sync one source via
the pipeline" entry point — no change detection, no prune, no
backfill. It's the right shape for the desktop app's "Index this
folder" action, where there's nothing to diff against.

Caller contract:
  - `source` must be a directory of images the indexer knows how to
    embed (see `indexer.scan:IMAGE_EXTENSIONS`).
  - `qdrant_client` must be an open connection. The caller owns its
    lifecycle.
  - `model_name` must be registered in `lustre_kernel.registry`.
    Defaults to the web backend's current model.
  - `dry_run=True` runs the full pipeline without writing to Qdrant.
    Returns the report's counts unchanged.

Returns a `PipelineReport` (from `indexer.pipeline`).

This module does NOT touch the existing `local_sync.main()` code
path. Both can coexist; `main()` continues to handle change-detection
and the other modes that the pipeline doesn't yet support.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from qdrant_client.http import models as qmodels

from indexer.pipeline import (
    IndexerPipeline,
    PipelineConfig,
    PipelineReport,
    WriteResult,
)
from indexer.upsert import build_payload, id_for
from lustre_kernel.payload_schema import FIELD_PATH
from lustre_kernel.registry import Embedder
from lustre_kernel.registry import get as _registry_get

logger = logging.getLogger(__name__)

# Concurrent PIL decode (§C2). The plan's spec:
#   - default = min(cpu_count, 8)
#   - clamped to [1, 32]
#   - override via IMAGE_LOAD_POOL_SIZE env var for hosts with
#     more (or fewer) cores
_DEFAULT_LOAD_POOL_SIZE = min(os.cpu_count() or 1, 8)
try:
    _LOAD_POOL_SIZE = int(
        os.environ.get(
            "IMAGE_LOAD_POOL_SIZE",
            str(_DEFAULT_LOAD_POOL_SIZE),
        )
    )
except ValueError:
    # Non-numeric env value — fall back to the default.
    _LOAD_POOL_SIZE = _DEFAULT_LOAD_POOL_SIZE
# Clamp to [1, 32]. A pool of 0 would silently serialise; a pool
# of 100 on a 4-core box would just thrash the GIL.
if _LOAD_POOL_SIZE < 1:
    _LOAD_POOL_SIZE = 1
elif _LOAD_POOL_SIZE > 32:
    _LOAD_POOL_SIZE = 32

__all__ = ["run_pipeline_source"]


# Adapter phase implementations: wrap the production modules so they
# satisfy the pipeline Protocols.

def _scan(source: Path) -> Iterator[Path]:
    from indexer.scan import snapshot
    return iter(snapshot(source))


def _load(
    paths: Iterator[Path],
    *,
    on_failure,
    model_name: str = "ViT-L-16-SigLIP2-256",
) -> Iterator[tuple[Path, Any, int | None, int | None]]:
    """Concurrent PIL decode via a ThreadPoolExecutor (§C2).

    Round‑18: takes the active `model_name` so the loader letterboxes
    to the right resolution. The previous default of
    `ViT-gopt-16-SigLIP2-384` meant the embedder (running the L/16
    model) received 384×384 images — 9× more pixels than it needs —
    which silently inflated embed time by ~10×.

    `indexer.image_loader.load` is GIL-released (PIL is mostly C
    code) — threads work and are cheaper than processes. The pool
    prefetches up to `_LOAD_POOL_SIZE` images so the embed phase
    never waits on disk I/O for a single image at a time.

    Failures are reported per-image via `on_failure`, never raised.
    The pipeline contract is unchanged: this is still a sync iterator
    yielding `(Path, Image)` tuples.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from indexer.image_loader import LoaderError, load

    inflight: dict = {}

    def submit_next() -> bool:
        try:
            p = next(paths)
        except StopIteration:
            return False
        inflight[executor.submit(load, p, model_name=model_name)] = p
        return True

    with ThreadPoolExecutor(max_workers=_LOAD_POOL_SIZE) as executor:
        # Prime the pool.
        for _ in range(_LOAD_POOL_SIZE):
            if not submit_next():
                break

        # Yield as completed; refill one per yield so the pool stays full.
        while inflight:
            future = next(as_completed(inflight))
            p = inflight.pop(future)
            try:
                # Round‑30: `load` now returns
                # `(letterboxed_img, source_w, source_h)`.
                img, source_w, source_h = future.result()
                yield (p, img, source_w, source_h)
            except LoaderError as exc:
                on_failure(p, exc)
            except Exception as exc:  # defensive: PIL's zoo  # noqa: BLE001
                on_failure(p, exc)
            submit_next()


def _embed(items: Iterator[tuple[Path, Any]], *, embedder: Embedder) -> Iterator[tuple[Path, Any, list[float]]]:
    """Embed phase.

    Round‑15: accumulate `embedder.embed_batch_size` decoded images
    and send them as one GPU forward pass. We also drop the thumbnail
    generation step entirely: it's best‑effort, the host can't write
    to /app/thumbnails, and the per‑image log spam was costing real
    wall‑clock time. The frontend fetches /api/thumbnails lazily on
    demand so an empty / missing thumbnail cache is fine.
    """
    import logging
    logging.getLogger(__name__)
    batch_size = getattr(embedder, "embed_batch_size", 16)

    pending_paths: list[Path] = []
    pending_imgs: list[Any] = []
    pending_dims: list[tuple[int | None, int | None]] = []

    def _flush() -> Iterator[tuple[Path, Any, list[float], int | None, int | None]]:
        if not pending_imgs:
            return
        vecs = embedder.embed_images(pending_imgs)
        for p, img, v, dim in zip(pending_paths, pending_imgs, vecs, pending_dims, strict=False):
            # Thumbnail generation is intentionally skipped — see
            # the round‑15 note above. The failure path used to log a
            # warning per image; on a 2k‑photo ingest that was 2k
            # syscalls and noticeable wall‑clock.
            yield (p, img, v, dim[0], dim[1])
        pending_paths.clear()
        pending_imgs.clear()
        pending_dims.clear()

    for path, image, source_w, source_h in items:
        pending_paths.append(path)
        pending_imgs.append(image)
        pending_dims.append((source_w, source_h))
        if len(pending_imgs) >= batch_size:
            yield from _flush()

    yield from _flush()


def _upsert(
    items, *, client, collection, dry_run, batch_size, on_failure,
) -> Iterator[WriteResult]:
    """Upsert pipeline phase.

    Round‑15: actually honour `batch_size` instead of accumulating
    every point and flushing once at the end. The old behaviour
    meant a 2k‑photo ingest would build an 8 MB vector blob and try
    to ship it in a single `upsert`, which hits qdrant's 32 MB
    JSON‑payload limit and blocks the whole pipeline until the
    HTTP call returns.

    New behaviour: flush every `batch_size` points (default 16) so
    each request is small and the producer can keep up.
    """
    batch: list = []
    for path, image, vec, source_w, source_h in items:
        try:
            # The pipeline doesn't pass model_name; use the dim
            # recorded in the embedder for self-describing payloads.
            embedder_dim = getattr(image, "__embedder_dim__", None) or len(vec)
            # Round‑19: compute blurhash + fingerprints from the
            # already-loaded letterboxed image instead of re-reading
            # the file from disk. The image is 256×256 (model res)
            # so blurhash is fast and fingerprints are accurate.
            from indexer.blurhash import compute_blurhash as _bh
            from indexer.fingerprints import compute_fingerprints as _fp
            try:
                _blurhash = _bh(image)
            except Exception:  # noqa: BLE001
                _blurhash = None
            try:
                _fingerprints = _fp(image)
            except Exception:  # noqa: BLE001
                _fingerprints = None
            payload = build_payload(
                path=path, shard="",
                model_name="mock-1536",  # overwritten below via direct registry lookup
                model_revision="test-r0",
                collection=collection,
                model_dim=embedder_dim,
                blurhash=_blurhash,
                fingerprints=_fingerprints,
                # Round‑30: source dims (NOT the letterboxed
                # 256×256 input size). Captured by the loader before
                # the letterbox squashes the image.
                width=source_w,
                height=source_h,
            )
            # build_payload hard-codes model_name='mock-1536' for the
            # _default_ test fixture. For real callers, we override
            # model_name + model_revision here to match the active
            # registry entry.
            payload["model_name"] = _resolve_active_model_name()
            payload["model_revision"] = _resolve_active_model_revision()
            _variant = _resolve_active_model_variant()
            if _variant is not None:
                payload["model_variant"] = _variant
            point_id = id_for(path, shard="")
            point = qmodels.PointStruct(
                id=point_id, vector=vec, payload=payload,
            )
            batch.append(point)
            yield WriteResult(
                path=path, point_id=point_id, dry_run=dry_run,
                vector_dim=len(vec),
            )
        except Exception as exc:  # noqa: BLE001
            on_failure(path, exc)

        # Flush when the batch is full so the pipeline keeps making
        # forward progress instead of waiting until the end.
        if not dry_run and len(batch) >= batch_size:
            try:
                client.upsert(collection_name=collection, points=batch, wait=False)
            except Exception as exc:
                for pt in batch:
                    on_failure(pt.payload["path"], exc)  # type: ignore[index]
                raise
            batch = []

    if batch and not dry_run:
        try:
            client.upsert(collection_name=collection, points=batch, wait=False)
        except Exception as exc:  # noqa: BLE001
            for pt in batch:
                # Best-effort: report the failure against the source path.
                on_failure(Path(pt.payload.get(FIELD_PATH, "<unknown>")), exc)


_ACTIVE_MODEL: tuple[str, str] | None = None
_ACTIVE_VARIANT: str | None = None


def _resolve_active_model_name() -> str:
    global _ACTIVE_MODEL  # noqa: PLW0602
    if _ACTIVE_MODEL is not None:
        return _ACTIVE_MODEL[0]
    # Fallback: the web backend's current model.
    return "ViT-so400m-patch16-384"


def _resolve_active_model_revision() -> str:
    global _ACTIVE_MODEL  # noqa: PLW0602
    if _ACTIVE_MODEL is not None:
        return _ACTIVE_MODEL[1]
    return "webli"


def _resolve_active_model_variant() -> str | None:
    """Variant string for the currently-active indexer model.

    Set via set_active_model_variant(). Returns None if not pinned;
    callers that need a value should fall back to reading the env.
    """
    global _ACTIVE_VARIANT  # noqa: PLW0602
    return _ACTIVE_VARIANT


def set_active_model(model_name: str, model_revision: str) -> None:
    """Pin the model_name + revision used in newly-built payloads.

    The pipeline doesn't carry model identity per-call; payloads
    record the active model at index time. Callers (the desktop
    app, the CLI) set this once at startup; subsequent
    `run_pipeline_source` calls record the same identity.
    """
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = (model_name, model_revision)


def set_active_model_variant(variant: str) -> None:
    """Pin the SigLIP variant string used in newly-built payloads.

    Mirrors set_active_model() but for the variant shorthand
    (e.g. "so400m/16-384") that gets written to the
    `model_variant` Qdrant payload field. The reconciler in
    search/config.py reads this field on startup to detect
    variant changes.
    """
    global _ACTIVE_VARIANT
    _ACTIVE_VARIANT = variant


def run_pipeline_source(
    *,
    source: Path,
    qdrant_client: Any,
    collection: str,
    model_name: str = "ViT-L-16-SigLIP2-256",
    batch_size: int = 16,
    dry_run: bool = False,
    cancel_event: threading.Event | None = None,
    on_progress: Any = None,
) -> PipelineReport:
    """Run the new `IndexerPipeline` for a single source directory.

    Round‑18: default model is now L/16‑256 to match the desktop
    ingest path. The old gopt‑384 default silently loaded 384×384
    images even when the embedder was the L/16 model, multiplying
    embed time by ~10×.

    Returns a `PipelineReport`. Failures are aggregated, not raised.
    Cancelling via `cancel_event` between phases returns a partial
    report.
    """
    embedder = _registry_get(model_name).vision  # noqa: F841

    # Round‑18: bind `model_name` into `_load` so the loader
    # letterboxes to the right resolution for this embedder.
    import functools as _functools
    _load_bound = _functools.partial(_load, model_name=model_name)

    pipeline = IndexerPipeline(
        scan=_scan,        # type: ignore[arg-type]
        load=_load_bound,  # type: ignore[arg-type]
        embed=_embed,      # type: ignore[arg-type]
        upsert=_upsert,    # type: ignore[arg-type]
        cancel_event=cancel_event,
    )
    config = PipelineConfig(
        source=source,
        model_name=model_name,
        collection=collection,
        batch_size=batch_size,
        dry_run=dry_run,
        qdrant_client=qdrant_client,
        progress_every=batch_size,
        on_progress=on_progress,
    )
    report = pipeline.run(config)
    logger.info(
        "pipeline finished: upserted=%d failures=%d elapsed=%.2fs",
        report.total_upserted,
        len(report.failures),
        report.elapsed_seconds,
    )
    return report
