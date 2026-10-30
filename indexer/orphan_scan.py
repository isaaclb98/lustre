"""
indexer/orphan_scan.py

Find the points that should be marked as orphans, and mark them.

Two kinds of orphan, from the indexer's point of view:

1. **The file is gone.** The photo was deleted from the NAS. The
   point's stored path no longer resolves to a file.

2. **The source was retired.** The source directory was removed
   from the configured source list (`INDEXER_SOURCES`), so it is no
   longer part of the library — even though its files may still be
   sitting on disk. Nothing in the filesystem reveals this; only the
   configured source list does.

Both cases end the same way: the point is MARKED with the
`orphan_since` payload field, which excludes it from every read
path. Neither case deletes anything — that is the second step, and
it requires an operator's confirmation.

The retired-source case is the reason this is not just "prune with
extra steps". A retired source's files are still on disk, so a
filesystem liveness check says they are alive, and the old prune
would have left them in the index forever: invisible in the source
list, still occupying collection quota, still returned by a search
that did not filter by source.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from qdrant_client import QdrantClient

from indexer.scan import existing_image_paths

logger = logging.getLogger(__name__)


class OrphanScopeError(RuntimeError):
    """Raised when an orphan scan cannot be scoped safely.

    The scan infers "this file is gone" from the absence of a path
    in the walked set. That inference is only sound when every
    managed source dir could actually be walked — otherwise the
    missing dir's points ALL look gone. Callers should abort the
    scan rather than mark on that assumption.

    The same reasoning applies to the retired-source detection: it
    needs the complete list of managed sources. A partial source
    list would make every unlisted source look retired, which would
    mark the entire library.
    """

def find_retired_sources(
    client: QdrantClient,
    collection: str,
    managed_source_names: list[str],
) -> dict[str, int]:
    """Which sources in the collection are no longer managed?

    A source is RETIRED when the collection holds points whose
    `collection` payload names it, but it is not in the managed
    source list. That is the exact signature of "the directory was
    removed from `INDEXER_SOURCES`": its points are still indexed,
    but nothing is managing them any more.

    The managed list must be COMPLETE. A partial list — one source
    of three, say — would report the other two as retired, and every
    point in them would be marked. That is why the caller passes the
    full configured set, and why `OrphanScopeError` exists on the
    scan path.

    Returns a mapping of retired source name → number of points.
    """
    managed = set(managed_source_names)
    counts: dict[str, int] = {}
    try:
        res = client.facet(
            collection_name=collection,
            key="collection",
            limit=1000,
        )
    except Exception as e:  # noqa: BLE001
        # The facet is a server-side feature the in-memory client does
        # not support. Without it we cannot see the source names, so
        # we report no retired sources rather than guessing.
        logger.info("retired-source facet unavailable: %s", e)
        return {}

    for hit in res.hits:
        value = hit.value
        if value is None or value == "":
            continue
        name = str(value)
        if name in managed:
            continue
        counts[name] = int(hit.count)

    if counts:
        logger.info(
            "retired sources (not in managed list %s): %s",
            sorted(managed),
            counts,
        )
    return counts


def find_disk_orphans(
    client: QdrantClient,
    collection: str,
    source_dirs: list[str],
    source_names: list[str],
    batch_size: int = 1000,
) -> list[str]:
    """The ids of points whose files are gone from disk.

    Scoped to `source_names`: points belonging to other sources are
    never considered, because the alive-set is built from these dirs
    only and every other source's points would look dead.

    Returns the list of point ids to mark. The caller marks them —
    this function does not write.
    """
    # Validate EVERY managed dir BEFORE walking any of them. The walk
    # over the prod sources takes ~56 minutes over NFS, so a check
    # placed after it would let a missing dir burn that hour before
    # finally aborting — and the abort would arrive after the
    # operator had already waited for nothing. Validate first; it is
    # a stat() per dir, milliseconds.
    missing_dirs = [
        src for src in source_dirs
        if not Path(src).exists() or not Path(src).is_dir()
    ]
    if missing_dirs:
        raise OrphanScopeError(
            "refusing to scan for orphans: managed source dir(s) missing "
            f"or not a directory: {', '.join(missing_dirs)}. Their points "
            "would all be treated as deleted files. Fix the mount or the "
            "path and re-run."
        )

    alive: set[str] = set()
    for src in source_dirs:
        src_path = Path(src)
        logger.info("orphan scan: walking %s ...", src_path)
        t0 = time.monotonic()
        for walked, image_path in enumerate(
            existing_image_paths(src_path), start=1,
        ):
            alive.add(image_path)
            if walked % 50_000 == 0:
                elapsed = time.monotonic() - t0
                rate = walked / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "orphan scan: walked %d image files so far (%.0f/s, "
                    "%dm %02ds elapsed) in %s",
                    walked, rate, int(elapsed) // 60, int(elapsed) % 60,
                    src_path,
                )
    logger.info(
        "orphan scan: pre-walked %d files under %d source dir(s)",
        len(alive), len(source_dirs),
    )

    managed = set(source_names)
    orphans: list[str] = []
    next_offset: int | None = None
    while True:
        points = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        batch, next_offset = points
        if not batch:
            break
        for p in batch:
            payload = p.payload or {}
            src = payload.get("collection", "")
            if managed and src not in managed:
                continue
            path_str = payload.get("path", "")
            if not path_str:
                orphans.append(str(p.id))
                continue
            if path_str not in alive:
                orphans.append(str(p.id))
        if next_offset is None:
            break

    return orphans


def find_source_orphans(
    client: QdrantClient,
    collection: str,
    source_name: str,
    batch_size: int = 1000,
) -> list[str]:
    """The ids of every point belonging to one retired source.

    Used when a source is retired: the whole source is marked, even
    though its files are still on disk. The points are excluded from
    every read path immediately, and their embeddings are removed
    only when the operator confirms the purge.
    """
    from qdrant_client.http import models as qmodels

    orphans: list[str] = []
    next_offset: int | None = None
    query_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="collection",
                match=qmodels.MatchValue(value=source_name),
            ),
        ],
    )
    while True:
        points = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=next_offset,
            scroll_filter=query_filter,
            with_payload=False,
            with_vectors=False,
        )
        batch, next_offset = points
        if not batch:
            break
        orphans.extend(str(p.id) for p in batch)
        if next_offset is None:
            break
    return orphans


def clear_orphan_marker(
    client: QdrantClient,
    collection: str,
    point_ids: list[str],
) -> int:
    """Clear the orphan marker from points that are alive again.

    A file can come back — restored from a backup, moved back into
    the source dir, or the source re-added to the source list. The
    point's id is a uuid5 of its path, so it comes back as the SAME
    id, and the marker has to be cleared or the photo stays hidden
    forever.

    This is called on the indexer's upsert path, where the point is
    known to be alive.
    """
    if not point_ids:
        return 0

    try:
        client.set_payload(
            collection_name=collection,
            payload={"orphan_since": None},
            points=point_ids,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "failed to clear orphan marker on %d point(s): %s",
            len(point_ids), e,
        )
        return 0
    logger.info("cleared orphan marker on %d point(s)", len(point_ids))
    return len(point_ids)


