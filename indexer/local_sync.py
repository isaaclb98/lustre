#!/usr/bin/env python3
"""
local_sync.py — Windows-friendly single-command sync+embed.
Unified: walk source(s) -> diff vs Qdrant -> embed on local GPU -> upsert.
No _pending queue. Tailscale-native.

When invoked by the search backend (via the admin Index API), the
runner spawns this module as a child process and reads its stdout for
structured progress. Two flags wire that up:

  --json-progress   emit one JSON line per progress event to stdout
                    (alongside the human-readable logger output)
  --rebuild         wipe the Qdrant collection first, then embed every
                    source file (no change-detection skip)

Cancel is cooperative: SIGTERM sets an internal cancellation flag that
the main loop checks at every batch boundary. The process exits cleanly
with code 130 and emits a {"event": "cancelled"} progress line.
"""  # noqa: EXE001

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from indexer import scan as scan_mod
from indexer import upsert
from indexer.image_loader import letterbox_resize, load_image_pil, peek_source_dims
from indexer.retry import RetryExhausted, retry_with_backoff
from indexer.scan import ScanCancelled
from indexer.thumbnails import generate_thumbnail_for_path, thumbnail_path
from indexer.vision_encoder import VisionEncoder
from lustre_kernel.qdrant_url import client_kwargs as _qdrant_client_kwargs
from lustre_kernel.registry import get as _registry_get

load_dotenv()
logger = logging.getLogger(__name__)


# --- Cancellation support ------------------------------------------------
# SIGTERM sets `_cancel_event`; the main loop checks it at every batch
# boundary and exits cleanly. Initialised lazily inside `main` so module
# import has no side effects (tests import this file repeatedly).

_cancel_event: threading.Event | None = None


def _install_signal_handlers() -> None:
    """Wire SIGTERM (and SIGINT, for dev Ctrl-C) to the cancel event."""
    global _cancel_event
    _cancel_event = threading.Event()

    def _on_signal(signum, _frame):
        if _cancel_event is not None:
            _cancel_event.set()
            logger.info("received signal %d; cancelling at next batch boundary", signum)

    # Signal handlers can only be installed from the main thread.
    # Spawned subprocess workers (rare) skip this silently.
    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)


def _is_cancelled() -> bool:
    return _cancel_event is not None and _cancel_event.is_set()


# --- JSON progress output ------------------------------------------------
# When --json-progress is on, we emit ONE JSON object per line to stdout
# for the runner to parse. Logger output is untouched (goes to stderr
# in non-interactive runs) so the runner can split the streams cleanly.

_json_progress: bool = False


def _emit_progress(payload: dict) -> None:
    """Write one JSON progress line to stdout if --json-progress is on."""
    if not _json_progress:
        return
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="local_sync", description="Single-command sync+embed for Windows")
    p.add_argument("--source", type=Path, action="append", required=True, help="Folder to scan. Repeatable.")
    p.add_argument("--source-name", type=str, action="append", default=None, dest="source_names", help="Logical name. Repeatable; if one given for many sources, applied to all.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--prune", action="store_true")
    p.add_argument("--model", type=str, default="ViT-so400m-patch16-384")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--qdrant-url", type=str, default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    p.add_argument("--qdrant-api-key", type=str, default=os.environ.get("QDRANT_API_KEY") or None)
    p.add_argument("--qdrant-collection", type=str, default=os.environ.get("QDRANT_COLLECTION", "images"))
    p.add_argument("--qdrant-in-memory", action="store_true")
    p.add_argument(
        "--reblurhash",
        action="store_true",
        help="Walk the existing collection, recompute blurhash for each point from its source "
             "file, and rewrite only the 'blurhash' payload field. Does NOT re-embed. Idempotent. "
             "Mutually exclusive with --refingerprint.",
    )
    p.add_argument(
        "--refingerprint",
        action="store_true",
        help="Same as --reblurhash but for content_sha256 + dhash (diversity / near-duplicate "
             "detection in the search side). Does NOT re-embed. Idempotent. Mutually exclusive "
             "with --reblurhash.",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Full sweep: embed new/changed files, backfill missing "
        "blurhash+fingerprint on legacy points, prune dead points.",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the Qdrant collection before indexing, then embed every "
        "source file (no change-detection skip). Used by the in-app Index "
        "Rebuild-from-scratch mode. Mutually exclusive with --reblurhash "
        "and --refingerprint.",
    )
    p.add_argument(
        "--json-progress",
        action="store_true",
        help="Emit one JSON object per progress event to stdout. Used by the "
        "admin Index runner to track job state. Has no effect on the "
        "human-readable logger output.",
    )
    return p.parse_args(argv)


def make_client(args):
    if args.qdrant_in_memory:
        return QdrantClient(location=":memory:")
    return QdrantClient(**_qdrant_client_kwargs(url=args.qdrant_url, api_key=args.qdrant_api_key, timeout=30))


def _active_variant() -> str | None:
    """Active SigLIP variant for the indexer subprocess.

    Resolution order:
      1. SIGLIP_VARIANT env (set by compose for prod, by tests).
      2. None — caller skips writing the payload field. The startup
         reconciler treats missing `model_variant` as "unknown legacy,
         requires reindex" (defensive, not silent).

    Reads from env (not from search.config) because the subprocess
    runs as a separate interpreter; search.config is importable but
    importing it in local_sync drags the FastAPI app in as a side
    effect. Keeping this env-only is intentional.
    """
    return os.environ.get("SIGLIP_VARIANT")


def resolve_source_names(sources, names):
    if not names:
        return [src.name for src in sources]
    if len(names) == 1 and len(sources) > 1:
        return names * len(sources)
    if len(names) != len(sources):
        raise ValueError("source-name count mismatch")
    return names


def _await_points_visible(client, name, ids, timeout_s=30.0, poll_s=0.2):
    """Poll until every id in `ids` is retrievable from the collection.

    Batched upserts are sent with wait=False; Qdrant acknowledges them
    asynchronously. This bounds the race for callers that read right
    after sync. Returns True when visible, False on timeout.
    """
    deadline = time.time() + timeout_s
    want = set(ids)
    while True:
        try:
            found = upsert.existing_ids(client, name, list(want))
        except Exception:  # noqa: BLE001
            found = set()
        if want <= found:
            return True
        if time.time() >= deadline:
            logger.warning(
                "timed out waiting for %d point(s) to become visible",
                len(want - found),
            )
            return False
        time.sleep(poll_s)


# Round-perf (issue #1): concurrent PIL decode for the local_sync hot
# path. Mirrors `indexer.run_pipeline._load` but exposed as a top-level
# helper so the per-batch loop here can use it without re-implementing
# the executor plumbing.
_DEFAULT_LOAD_POOL_SIZE = min(os.cpu_count() or 1, 8)
try:
    _LOAD_POOL_SIZE = int(os.environ.get("IMAGE_LOAD_POOL_SIZE", str(_DEFAULT_LOAD_POOL_SIZE)))
except ValueError:
    _LOAD_POOL_SIZE = _DEFAULT_LOAD_POOL_SIZE
_LOAD_POOL_SIZE = max(1, min(_LOAD_POOL_SIZE, 32))


# Cap the per-run "id_mismatch" sample so a 1M-file run doesn't
# emit a multi-MB progress payload. 20 is enough to see the
# pattern (path → id, id_alt, recorded mtime/size, current mtime/size)
# without dominating the runner's stdout buffer.
_ID_MISMATCH_SAMPLE_MAX = 20


def _load_batch_concurrent(paths):
    """Decode `paths` into `(path, image, source_w, source_h)` tuples concurrently.

    PIL's decoder releases the GIL on the heavy C code (libjpeg /
    libpng / WebP / HEIF), so a ThreadPoolExecutor gives real
    parallelism without the cost of spawning processes. Each worker
    performs the same work as the previous serial loop:
      1. `peek_source_dims(p)` — JPEG header read
      2. `load_image_pil(p)` — full RGB decode
    Failures are logged + counted (not raised) to preserve the
    contract that the indexer continues with whatever loaded
    successfully.

    Returns a list of tuples, ordered to match `paths`. Per-path
    failures are recorded as `(p, None, None, None)` so the caller
    can skip + count them without an extra exception pass.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from indexer.image_loader import LoaderError

    results: list = [None] * len(paths)

    def _do(idx, p):
        try:
            sw, sh = peek_source_dims(p)
            img = load_image_pil(p)
            return idx, (p, img, sw, sh)
        except (LoaderError, Exception) as exc:  # noqa: BLE001
            logger.warning("load error %s: %s", p, exc)
            return idx, (p, None, None, None)

    with ThreadPoolExecutor(max_workers=_LOAD_POOL_SIZE) as executor:
        futures = [executor.submit(_do, i, p) for i, p in enumerate(paths)]
        for fut in as_completed(futures):
            _idx, payload = fut.result()
            results[_idx] = payload

    return results


def _scroll_existing_meta(client, collection, src_name):
    """Scroll all points in `collection` with `src_name` as their `collection`.

    Filters on the `collection` payload field — that is the name
    `upsert.build_payload` writes for the source name. (The earlier
    revision of this helper filtered on `source`, which no production
    code path actually populated; the mismatch made every legacy
    point invisible to change-detection and forced a full re-embed
    on every run. The defence-in-depth fallback at the call site
    still triggers per-batch retrieve for any legacy data that
    predates the `collection` field being written.)

    Returns a dict `{point_id_str: (mtime, size) | None}`. Points
    with mtime/size populated in the payload map to `(mtime, size)`;
    points lacking those fields map to `None` (treated as changed by
    the caller so legacy points heal on next run).

    Replaces the previous per-batch `client.retrieve(...)` which
    issued one round-trip per batch — on a 902k-photo corpus that's
    ~900 extra RTTs to Qdrant per source. One scroll handles them
    all in a single round-trip (well, a paginated scroll, but it's
    O(pages) not O(batches)).
    """
    from qdrant_client.http import models as _qm

    meta: dict = {}
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=_qm.Filter(
                must=[
                    _qm.FieldCondition(
                        key="collection",
                        match=_qm.MatchValue(value=src_name),
                    )
                ]
            ),
            with_payload=["mtime", "size"],
            limit=1000,
            offset=offset,
        )
        if not points:
            break
        for pt in points:
            pl = pt.payload or {}
            if pl.get("mtime") is not None and pl.get("size") is not None:
                meta[str(pt.id)] = (int(pl["mtime"]), int(pl["size"]))
            else:
                meta[str(pt.id)] = None
        if next_offset is None:
            break
        offset = next_offset
    return meta


def main(argv=None):
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    # Wire subprocess-level concerns (cancel signals, JSON progress) BEFORE
    # any work begins. Both are no-ops if their respective flags are off.
    global _json_progress
    _json_progress = bool(args.json_progress)
    _install_signal_handlers()

    # Mutual-exclusion checks (rebuild is the new entrant).
    if args.rebuild and (args.reblurhash or args.refingerprint):
        logger.error("--rebuild is mutually exclusive with --reblurhash/--refingerprint")
        return 2
    if args.rebuild and args.full:
        logger.error("--rebuild implies a fresh index; do not combine with --full")
        return 2

    try:
        source_names = resolve_source_names(args.source, args.source_names)
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    for src in args.source:
        if not src.exists():
            logger.error("source does not exist: %s", src)
            return 2
        if not src.is_dir():
            logger.error("source is not a directory: %s", src)
            return 2

    client = make_client(args)
    # Round-35: search-side images.db mirror. None when dry-run or
    # when INDEX_DB_PATH isn't set (tests, standalone indexer runs).
    if not args.dry_run:
        # Resolve the collection's vector dim from the chosen model so a
        # non-default --model (e.g. ViT-L-16-SigLIP2-256) gets a
        # 1024-dim collection instead of upsert.ensure_collection's
        # hardcoded 1536 (the gopt default).
        model_dim = _registry_get(args.model).dim

        if args.rebuild:
            # Wipe-and-rebuild: drop the collection, then recreate via
            # ensure_collection. Done BEFORE the progress "start" event so
            # the runner's points_count reflects the post-wipe state (zero).
            try:
                client.delete_collection(collection_name=args.qdrant_collection)
                logger.info("rebuild: deleted existing collection %r", args.qdrant_collection)
            except Exception as exc:  # noqa: BLE001 — first run, collection may not exist
                logger.info("rebuild: no existing collection to delete (%s)", exc)

        upsert.ensure_collection(client, args.qdrant_collection, dim=model_dim)
        upsert.ensure_payload_index(client, args.qdrant_collection, "collection", "keyword")

    if args.reblurhash and args.refingerprint:
        logger.error("--reblurhash and --refingerprint are mutually exclusive")
        return 2

    if args.full and (args.reblurhash or args.refingerprint):
        logger.error("--full includes backfill; drop --reblurhash/--refingerprint")
        return 2

    # Lazy: encoder only built when we actually embed. Backfill
    # (--reblurhash / --refingerprint)and --prune / --dry-run paths
    # never embed, so those runs must not pay the multi-second
    # VisionEncoder (timm ViT) init cost — it made prune tests hang
    # on CPU-only machines.
    encoder: VisionEncoder | None = None


    total_indexed = 0


    total_reembedded = 0


    last_written_ids: list = []
    total_skipped = 0
    total_errors = 0
    # Round-50: count of thumbnails regenerated for files whose
    # embed was skipped (mtime/size match → up-to-date) but whose
    # thumb was missing on disk. This is the "I deleted all the
    # thumbs, click Index, and the site comes back" self-heal
    # path. The count is surfaced in the progress events so the
    # UI can show it; it's also a way to detect disk-level
    # corruption (a spike means thumbs are being lost between
    # runs).
    total_thumb_regens = 0
    # Diagnostic for the "indexer re-embeds everything on no-op runs"
    # investigation. When the indexer treats a file as new (id not in
    # source_meta) but the same path resolves to an id that IS in the
    # collection, we record a sample entry. If this counter is large
    # on a no-change run, the root cause is path normalisation
    # (resolve() vs as_posix() vs trailing slash, etc.).
    total_id_mismatch = 0
    id_mismatch_sample: list = []
    t0 = time.time()

    # Emit the "start" event for the runner. Doing it here (after the
    # rebuild wipe + collection re-create) so the runner sees a fresh
    # points_count from the very first status poll.
    _emit_progress({
        "event": "start",
        "mode": "rebuild" if args.rebuild else "incremental",
        "model": args.model,
        "device": args.device,
        "sources": [str(s) for s in args.source],
        "batch_size": args.batch_size,
    })

    if args.reblurhash or args.refingerprint:
        return _backfill_payload_field(
            client=client,
            collection=args.qdrant_collection,
            field="blurhash" if args.reblurhash else "fingerprint",
            sources=args.source,
            source_names=source_names,
            batch_size=args.batch_size,
            limit=args.limit,
        )

    # Prune once per run, scoped to the full set of sources this run
    # manages. Doing it inside the per-source loop would let each
    # source's prune delete points owned by the other sources (the
    # alive-set is built from one dir but the scroll covers the whole
    # collection). Walking all source dirs + passing every source-name
    # means only genuinely-missing files under the managed sources are
    # removed — points from unmanaged sources are never touched.
    if args.prune and not args.dry_run:
        removed = upsert.prune_missing(
            client, args.qdrant_collection,
            source_dirs=args.source,
            source_names=source_names,
        )
        logger.info("prune removed %d point(s)", removed)

    for src_path, src_name in zip(args.source, source_names, strict=True):
        logger.info("=== %s -> %s ===", src_path, src_name)
        # Estimate the total for this source from the collection's
        # per-source point count (cheap: filtered count against the
        # `collection` payload index) so the walk progress can show a
        # time-to-go.
        expected_total = None
        if not args.dry_run:
            try:
                from qdrant_client.http import models as _qm
                cnt = client.count(
                    collection_name=args.qdrant_collection,
                    count_filter=_qm.Filter(
                        must=[
                            _qm.FieldCondition(
                                key="collection",
                                match=_qm.MatchValue(value=src_name),
                            )
                        ]
                    ),
                    exact_count=False,
                )
                expected_total = int(cnt.count)
            except Exception as exc:  # noqa: BLE001
                logger.debug("count for %s failed: %s", src_name, exc)

        # Round-perf (issue #1): cache the change-detection metadata
        # for this entire source in one scroll, then look up per
        # batch from the in-memory dict. The previous per-batch
        # `client.retrieve(...)` issued one round-trip per batch — on
        # a 902k-photo corpus that's ~900 RTTs to Qdrant per source.
        # The single scroll replaces them all (paginated, but
        # O(pages), not O(batches)).
        source_meta: dict = {}
        source_meta_failed = False
        if not args.rebuild and not args.dry_run:
            try:
                source_meta = _scroll_existing_meta(
                    client, args.qdrant_collection, src_name,
                )
                logger.debug(
                    "scrolled %d existing points for source %s",
                    len(source_meta), src_name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "source-scoped scroll failed for %s: %s; "
                    "falling back to per-batch retrieve", src_name, exc,
                )
                source_meta_failed = True
            else:
                # Successful scroll that returned no points is ambiguous:
                # (a) this source has no points in the collection yet
                #     (fresh source, nothing to skip), or
                # (b) the collection has legacy points but none of them
                #     have the `collection` payload field the scroll
                #     filters on — change-detection would silently
                #     re-embed them all, which is the bug we're
                #     guarding against.
                # Disambiguate by reading the collection's overall
                # point count: if it's non-zero, treat as case (b)
                # and fall back to per-batch retrieve (id-based, no
                # `collection` field needed).
                if not source_meta:
                    try:
                        if client.get_collection(
                            args.qdrant_collection
                        ).points_count > 0:
                            logger.warning(
                                "source-scoped scroll returned 0 points "
                                "for %s but the collection is non-empty; "
                                "legacy data likely lacks the "
                                "`collection` payload field — falling "
                                "back to per-batch retrieve",
                                src_name,
                            )
                            source_meta_failed = True
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "could not read collection point count "
                            "for %s: %s", src_name, exc,
                        )
        try:
            snap = scan_mod.snapshot(
                src_path,
                expected_total=expected_total,
                should_cancel=_is_cancelled,
            )
        except ScanCancelled:
            # Cancel during the filesystem walk itself. Without this
            # handler, a cancel on a 1M-file library would wait
            # minutes for the walk to complete before the next batch
            # check fires — the user would think cancel is broken.
            logger.info("cancel requested during scan of %s", src_path)
            _emit_progress({
                "event": "cancelled",
                "stage": "scanning",
                "indexed": total_indexed,
                "reembedded": total_reembedded,
                "skipped": total_skipped,
                "errors": total_errors,
                "thumb_regens": total_thumb_regens,
                "duration_s": time.time() - t0,
            })
            return 130
        if args.limit:
            snap = snap[:args.limit]
        logger.info("found %d files in %s", len(snap), src_path)
        if not snap:
            continue

        for i in range(0, len(snap), args.batch_size):
            batch = snap[i:i+args.batch_size]

            # Cancel check at batch boundary. SIGTERM (from the runner)
            # sets _cancel_event; we bail BEFORE loading or embedding
            # another batch and emit a final "cancelled" event.
            if _is_cancelled():
                logger.info("cancel requested; stopping after %d indexed", total_indexed)
                _emit_progress({
                    "event": "cancelled",
                    "indexed": total_indexed,
                    "reembedded": total_reembedded,
                    "skipped": total_skipped,
                    "errors": total_errors,
                    "thumb_regens": total_thumb_regens,
                    "duration_s": time.time() - t0,
                })
                return 130

            # Round-50: files that are up-to-date (mtime/size
            # match, no re-embed needed) but whose thumbnail is
            # missing on disk. We regen the thumb without
            # re-embedding. Built inside the per-file loop in the
            # non-rebuild branch; initialised here so the regen
            # block below is bound in rebuild mode too (where it's
            # always empty because every file is in to_embed).
            to_thumb: list = []

            # Skip the change-detection scroll entirely in --rebuild
            # mode. We just wiped the collection, so every existing id
            # lookup would fail anyway — and even if it didn't, we want
            # to embed every file from scratch.
            if args.rebuild:
                to_embed = [(p, "new") for p in batch]
                n_new = len(batch)
                n_changed = 0
                n_id_mismatch = 0
            else:
                ids = [upsert.id_for(p) for p in batch]
                # Change detection: pull mtime/size for points that already
                # exist. Point ids are deterministic (upsert.id_for), so a
                # changed file re-embeds INTO its existing point — no
                # duplicates, favourites/album membership preserved.
                # Points lacking stored mtime/size (pre-change-detection
                # index) are treated as changed so they heal on next run.
                #
                # Round-perf (issue #1): use the source-scoped
                # `source_meta` dict (scrolled once above the batch
                # loop) instead of a per-batch `client.retrieve(...)`.
                # On a 902k corpus that's ~900 fewer RTTs per source.
                # Falls back to the old per-batch retrieve if the
                # source scroll failed above.
                existing_meta: dict = {}
                if source_meta_failed:
                    try:
                        points = client.retrieve(
                            collection_name=args.qdrant_collection,
                            ids=ids,
                            with_payload=["mtime", "size"],
                        )
                        for pt in points:
                            pl = pt.payload or {}
                            if pl.get("mtime") is not None and pl.get("size") is not None:
                                existing_meta[str(pt.id)] = (int(pl["mtime"]), int(pl["size"]))
                            else:
                                existing_meta[str(pt.id)] = None
                    except Exception:
                        if not args.dry_run:
                            raise
                        logger.warning("dry-run: could not read existing points; assuming all new")
                        existing_meta = {}
                else:
                    # O(1) lookups against the pre-scrolled dict.
                    for pid in ids:
                        if pid in source_meta:
                            existing_meta[pid] = source_meta[pid]
                        # else: point not in source → 'new' below

                to_embed: list = []
                n_new = 0
                n_changed = 0
                n_id_mismatch = 0
                for path, pid in zip(batch, ids, strict=False):
                    if pid not in existing_meta:
                        # Diagnostic: when the indexer treats a file as
                        # new (id not in the batch-filtered dict), check
                        # the *source-scoped* scroll dict for the same
                        # path under a different id. If found, the file
                        # is already indexed but the indexer can't see
                        # it via its own id computation — the smoking
                        # gun for a path-normalisation drift. We
                        # surface a sample so a follow-up can see the
                        # raw strings that produced each id.
                        #
                        # We deliberately check `source_meta` (the
                        # pre-scrolled source-level dict), not
                        # `existing_meta` (the batch-filtered dict):
                        # the latter only contains ids that appear in
                        # the current batch, so it can never contain
                        # the alternate id we're looking for. The
                        # fallback path (per-batch retrieve) is
                        # excluded here because it doesn't build a
                        # source-level view.
                        if not source_meta_failed:
                            try:
                                pid_alt = upsert.id_for(path.resolve())
                            except OSError:
                                pid_alt = pid
                            # DEBUG
                            if pid_alt != pid and pid_alt in source_meta:
                                n_id_mismatch += 1
                                total_id_mismatch += 1
                                if len(id_mismatch_sample) < _ID_MISMATCH_SAMPLE_MAX:
                                    rec = source_meta[pid_alt]
                                    cur = path.stat()
                                    id_mismatch_sample.append({
                                        "path": str(path),
                                        "path_resolved": str(path.resolve()),
                                        "id": pid,
                                        "id_alt": pid_alt,
                                        "recorded_mtime": rec[0] if rec else None,
                                        "recorded_size": rec[1] if rec else None,
                                        "current_mtime": int(cur.st_mtime),
                                        "current_size": int(cur.st_size),
                                    })
                        to_embed.append((path, "new"))
                        n_new += 1
                        continue
                    recorded = existing_meta[pid]
                    if recorded is None:
                        to_embed.append((path, "changed"))
                        n_changed += 1
                        continue
                    try:
                        st = path.stat()
                    except OSError:
                        total_errors += 1
                        continue
                    if int(st.st_mtime) != recorded[0] or int(st.st_size) != recorded[1]:
                        to_embed.append((path, "changed"))
                        n_changed += 1
                    else:
                        total_skipped += 1
                        # Round-50: self-heal. The file's embedding
                        # is up-to-date but the on-disk thumb is
                        # gone (operator deleted the thumb dir, NFS
                        # hiccup, partial restore, etc.). Regen the
                        # thumb without re-embedding. One
                        # os.path.exists call per up-to-date file
                        # per batch — negligible cost.
                        #
                        # The check is on the canonical thumb path
                        # only: generate_thumbnail_for_path writes
                        # the canonical + all sized variants in one
                        # call, so a missing canonical is sufficient
                        # evidence the whole set is gone.
                        if not (thumbnail_path(upsert.id_for(path, ""))).exists():
                            to_thumb.append(path)

            logger.info(
                "batch %d: %d new, %d changed, %d up-to-date, %d id_mismatch",
                i // args.batch_size + 1, n_new, n_changed,
                len(batch) - n_new - n_changed,
                n_id_mismatch,
            )

            if not to_embed and not to_thumb:
                continue

            if args.dry_run:
                logger.info(
                    "dry-run: would embed %d file(s) (%d new, %d changed)",
                    len(to_embed), n_new, n_changed,
                )
                if to_thumb:
                    logger.info(
                        "dry-run: would regen %d missing thumb(s)",
                        len(to_thumb),
                    )
                total_indexed += n_new
                total_reembedded += n_changed
                continue

            # Round-perf (issue #1): concurrent PIL decode via a
            # thread pool (size = min(cpu_count, 8), env-tunable
            # via IMAGE_LOAD_POOL_SIZE). PIL releases the GIL on
            # libjpeg/libpng/WebP/HEIF, so this gives real
            # parallelism without the cost of spawning processes.
            # On an 8-core host this typically cuts the load phase
            # from batch_size × 25ms to ~batch_size × 4ms.
            #
            # Skipped when the batch is thumb-only (to_thumb
            # non-empty, to_embed empty) — those files are
            # up-to-date and don't need an embed, so we save
            # the encoder's time and the round trip.
            loaded: list = []
            if to_embed:
                to_embed_paths = [p for p, _reason in to_embed]
                raw_loaded = _load_batch_concurrent(to_embed_paths)
                for entry in raw_loaded:
                    if entry is None:
                        continue
                    _p, _img, _sw, _sh = entry
                    if _img is None:
                        # _load_batch_concurrent logged + counted this one
                        total_errors += 1
                        continue
                    loaded.append(entry)
                if not loaded and any(e is None or e[1] is None for e in raw_loaded):
                    continue
                if not loaded:
                    continue

            if loaded:
                try:
                    if encoder is None:
                        # First-batch model load. Downloading the SigLIP2
                        # weights from HF Hub takes 30-60s on a fresh
                        # install; without this event the UI sits at
                        # "Running · 0 indexed" the whole time and looks
                        # frozen. Emitting "warming_up" before the load
                        # starts gives the UI a clear state.
                        _emit_progress({"event": "warming_up", "model": args.model})
                        encoder = VisionEncoder(arch=args.model, device=args.device)

                    vecs = encoder.embed_batch([letterbox_resize(img) for (_, img, _sw, _sh) in loaded])
                except Exception:
                    logger.exception("embed failed")
                    total_errors += len(loaded)
                    continue

                items = []
                for (path, img, source_w, source_h), vec in zip(
                    loaded, vecs, strict=False,
                ):
                    # Generate thumbnail (best-effort, non-fatal)
                    try:
                        upsert.id_for(path, "")
                        thumb_path = generate_thumbnail_for_path(img, path, "")
                        if thumb_path:
                            logger.debug(f"Generated thumbnail: {thumb_path}")
                    except (OSError, ValueError, RuntimeError) as e:
                        # Thumbnail generation can fail on corrupt JPEGs,
                        # PIL decode errors, or filesystem issues. The
                        # caller wants the index to continue regardless;
                        # narrow to the realistic failure modes rather than
                        # a blind `Exception` catch.
                        logger.warning(f"Failed to generate thumbnail for {path}: {e}")

                    payload = upsert.build_payload(
                        path, "", args.model, "", src_name,
                        width=source_w, height=source_h,
                    )
                    payload["path"] = path
                    # Record the SigLIP variant shorthand (e.g.
                    # "so400m/16-384") on each point so the search backend's
                    # startup reconciler can detect variant mismatches against
                    # Qdrant instead of a redundant on-disk JSON file.
                    _variant = _active_variant()
                    if _variant is not None:
                        payload["model_variant"] = _variant
                    items.append((upsert.id_for(path, ""), vec, payload))

                try:
                    # Wrap the upsert in retry-with-backoff. A single transient
                    # Qdrant failure (5xx, network blip) must not abort the
                    # whole job — 3 attempts, 2s/4s/8s between, then fail.
                    retry_with_backoff(
                        lambda items=items: upsert.upsert_batch(
                            client,
                            args.qdrant_collection,
                            [(pid, v, pl) for pid, v, pl in items],
                            wait=False,
                        ),
                        max_attempts=3,
                        base_delay_s=2.0,
                    )
                    total_indexed += sum(1 for (_p, r) in to_embed if r == "new")
                    total_reembedded += sum(1 for (_p, r) in to_embed if r == "changed")
                    last_written_ids = [iid for (iid, _v, _pl) in items]
                    # Emit per-batch progress so the runner can show live counts.
                    _emit_progress({
                        "event": "batch",
                        "batch_index": i // args.batch_size + 1,
                        "indexed": total_indexed,
                        "reembedded": total_reembedded,
                        "skipped": total_skipped,
                        "errors": total_errors,
                        "thumb_regens": total_thumb_regens,
                    })
                except RetryExhausted as exc:
                    # All retries failed — surface to the runner via a final
                    # "failed" event and exit non-zero. The runner records
                    # `last_error` from the event payload.
                    logger.exception("upsert failed after retries")
                    total_errors += len(items)
                    _emit_progress({
                        "event": "failed",
                        "stage": "upsert",
                        "error": f"{type(exc.last_exc).__name__}: {exc.last_exc}",
                        "indexed": total_indexed,
                        "reembedded": total_reembedded,
                        "skipped": total_skipped,
                        "errors": total_errors,
                        "thumb_regens": total_thumb_regens,
                        "duration_s": time.time() - t0,
                    })
                    return 1
                except Exception:
                    logger.exception("upsert failed")
                    total_errors += len(items)

            # Round-50: regenerate thumbnails for files whose embed
            # was skipped but whose on-disk thumb is missing. Runs
            # after the embed block (which may have created thumbs
            # for the to_embed paths) so we don't double-work any
            # path. Uses the same _load_batch_concurrent helper as
            # the embed path so the threading is consistent; the
            # throughput on libvips is high enough that the cost
            # here is dominated by the JPEG decode, not the encode.
            if to_thumb:
                raw_thumb_loaded = _load_batch_concurrent(to_thumb)
                for entry, path in zip(
                    raw_thumb_loaded, to_thumb, strict=False
                ):
                    if entry is None:
                        total_errors += 1
                        continue
                    _p, _img, _sw, _sh = entry
                    if _img is None:
                        # _load_batch_concurrent logged + counted this one
                        total_errors += 1
                        continue
                    try:
                        out = generate_thumbnail_for_path(_img, path, "")
                        if out:
                            total_thumb_regens += 1
                    except (OSError, ValueError, RuntimeError) as e:
                        logger.warning(
                            "Failed to regen thumb for %s: %s", path, e
                        )
                        total_errors += 1
                if total_thumb_regens:
                    logger.info(
                        "thumb-regen: %d up-to-date files were missing "
                        "their on-disk thumb; regenerated",
                        total_thumb_regens,
                    )

    # Visibility guarantee: batched upserts go out wait=False for

    # speed; before declaring done, block until the last written

    # batch is actually retrievable so callers never see stale state.

    if args.full and not args.dry_run:
        logger.info("=== full sweep: backfill blurhash ===")
        rc_b1 = _backfill_payload_field(
            client=client,
            collection=args.qdrant_collection,
            field="blurhash",
            sources=args.source,
            source_names=source_names,
            batch_size=args.batch_size,
            limit=0,
        )
        logger.info("=== full sweep: backfill fingerprint ===")
        rc_b2 = _backfill_payload_field(
            client=client,
            collection=args.qdrant_collection,
            field="fingerprint",
            sources=args.source,
            source_names=source_names,
            batch_size=args.batch_size,
            limit=0,
        )
        if rc_b1 or rc_b2:
            logger.warning("full sweep: some backfill rows failed; see log")
        removed = upsert.prune_missing(
            client, args.qdrant_collection,
            source_dirs=args.source,
            source_names=source_names,
        )
        logger.info("full sweep: prune removed %d point(s)", removed)
    elif args.full and args.dry_run:
        logger.info("dry-run --full: backfill and prune skipped (no writes)")

    if last_written_ids and not args.dry_run:

        _await_points_visible(client, args.qdrant_collection, last_written_ids)


    dt = time.time() - t0
    print(
        f"Done indexed={total_indexed} re-embedded={total_reembedded} "
        f"skipped={total_skipped} thumb_regens={total_thumb_regens} "
        f"errors={total_errors} "
        f"id_mismatch={total_id_mismatch} ({dt:.1f}s)"
    )
    _emit_progress({
        "event": "done",
        "indexed": total_indexed,
        "reembedded": total_reembedded,
        "skipped": total_skipped,
        "thumb_regens": total_thumb_regens,
        "errors": total_errors,
        # Diagnostic for path-normalisation regressions: when the
        # indexer re-embeds a file that's already in the collection
        # under a different id, the count of files actually re-uploaded
        # (`reembedded`) climbs even though no content changed. The
        # sample shows the raw strings that produced each id so we can
        # see whether the divergence is from trailing slashes,
        # resolve() vs as_posix(), drive-letter case, etc.
        "id_mismatch": total_id_mismatch,
        "id_mismatch_sample": id_mismatch_sample,
        "duration_s": dt,
    })
    # Round-35: close the search-side images.db mirror.
    return 0


def _backfill_payload_field(
    client,
    collection: str,
    field: str,  # "blurhash" or "fingerprint"
    sources: list,
    source_names: list,
    batch_size: int,
    limit: int,
) -> int:
    """
    Walk the existing collection, recompute `field` for each point from
    its source file, and `set_payload` only that field in-place. No
    re-embedding; the 1536-dim vector stays untouched.

    Used as a backfill: existing points indexed before [field] support
    was added to the indexer have `field=None` in their payload. The
    client side reads `field` for instant placeholder rendering (LQIP)
    or near-duplicate detection; without it, those features degrade
    silently. This is the cheap way to fix that without re-embedding
    1M+ points.

    For `--reblurhash`: reads the JPEG/PNG/HEIC, downsamples to 32x32,
    encodes to a ~28-char blurhash string. ~30-60s per 1k images on
    a fast disk (most of that is file I/O + Pillow decode).

    For `--refingerprint`: reads the file, computes SHA-256 of the
    bytes and a 64-bit dHash of a tiny downsampled version. Used
    by the diversity ranker in the search side. Cheaper than blurhash
    because the SHA-256 is bytes-only (no decode).

    Source-scoped: only points whose payload.path falls under one of
    the `--source` roots are touched. This is intentional — the Windows
    runs only know about the kpop sources, so a backfill shouldn't
    rewrite points that were indexed by a different machine with
    different path conventions.

    Idempotent: points whose computed value already matches the
    existing payload value are skipped.
    """
    if not sources:
        logger.error("--%s requires at least one --source", field)
        return 2

    if field == "blurhash":
        from indexer.blurhash import compute_blurhash

        def compute(path):
            return compute_blurhash(path)
    elif field == "fingerprint":
        from indexer.fingerprints import compute_fingerprints

        def compute(path):
            fp = compute_fingerprints(path)
            # compute_fingerprints returns the full dict; set_payload
            # expects a single key. We only rewrite the bytes-hash + dhash
            # pair; the 'blurhash' is a separate field.
            return {"content_sha256": fp.get("content_sha256"), "dhash": fp.get("dhash")}
    else:
        logger.error("unknown backfill field %r", field)
        return 2

    total_updated = total_skipped = total_failed = 0
    t0 = time.time()
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=batch_size * 8,  # scroll a chunk; we filter by source
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        # Filter to points under one of the configured sources.
        # The payload.path is the file path the indexer stored (absolute
        # host path). Backfill just reads from that path directly — no
        # translation needed because the indexer runs on the same host
        # as the backfill.
        scoped = [p for p in points if (p.payload or {}).get("collection") in source_names]
        if not scoped:
            if next_offset is None:
                break
            offset = next_offset
            continue

        for p in scoped:
            payload = p.payload or {}
            path_str = str(payload.get("path") or "")
            if not path_str:
                total_failed += 1
                continue
            try:
                # payload.path is the absolute host path the indexer
                # stored; the backfill reads from it directly.
                computed = compute(Path(path_str))
                if field == "blurhash":
                    new_value = computed  # str | None
                else:
                    new_value = {k: v for k, v in computed.items() if v}
            except Exception as exc:  # noqa: BLE001
                logger.warning("backfill: failed to compute %s for %s: %s", field, path_str, exc)
                total_failed += 1
                continue

            existing = payload.get(field)
            if field == "blurhash":
                if existing == new_value:
                    total_skipped += 1
                    continue
            else:
                # fingerprint: dict of {content_sha256, dhash}; skip
                # when both already match.
                if existing and all(existing.get(k) == new_value.get(k) for k in new_value):
                    total_skipped += 1
                    continue

            try:
                client.set_payload(
                    collection_name=collection,
                    payload={field: new_value} if field == "blurhash" else new_value,
                    points=[p.id],
                )
                total_updated += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("backfill: set_payload failed for %s: %s", p.id, exc)
                total_failed += 1

            if limit and total_updated + total_failed >= limit:
                break

        if limit and total_updated + total_failed >= limit:
            break
        if next_offset is None:
            break
        offset = next_offset

    dt = time.time() - t0
    print(
        f"backfill[{field}] done updated={total_updated} skipped={total_skipped} "
        f"failed={total_failed} ({dt:.1f}s)"
    )
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
