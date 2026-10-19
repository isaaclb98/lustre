"""
indexer/sync_meta.py — shim for the legacy sync_meta module.

The pre-Option-B design kept three Qdrant collections in play:
`images` (the search-side corpus), `images_pending` (the indexer's
write-side staging area), and `_sync_meta` (the SyncManager's
drift-detection marker point). The `--qdrant-collection` flag was
ambiguous between them, which is how the change-detection bug
(see commit 75aa9bb) crept in: every walk classified every file as
new because the indexer was reading from the staging area, not
the canonical one.

Option B drops the staging area entirely. The indexer writes
directly to `images`; the SyncManager is gone; `_sync_meta` and
`_pending` are no longer created or read. This module is kept as
a thin shim so existing test fixtures (`ensure_sync_collections`)
still import — the body just delegates to `upsert.ensure_collection`,
which is the single-collection primitive.

Anything that used to read or write `META_COLLECTION`,
`META_POINT_ID`, or `PENDING_COLLECTION` should be deleted, not
ported. They were only meaningful inside the sync loop.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from indexer.upsert import ensure_collection as _ensure_collection


def ensure_sync_collections(
    client: QdrantClient,
    images_collection: str = "images",
    *,
    dim: int | None = None,
) -> None:
    """Create the canonical images collection if missing.

    Legacy API: the old implementation also created `_sync_meta`
    and `_pending` collections. After Option B those are gone, so
    this is now a one-line delegation to `upsert.ensure_collection`.
    Kept under the old name so test fixtures that pre-date the
    refactor don't need to change.
    """
    _ensure_collection(client, images_collection, dim=dim)


# Constants kept as `None` so any leftover import fails loudly
# instead of silently referring to a collection we no longer touch.
META_COLLECTION = None  # type: ignore[assignment]
META_POINT_ID = None  # type: ignore[assignment]
PENDING_COLLECTION = None  # type: ignore[assignment]


def _utc_now() -> str:
    """Unused after Option B; kept for import compatibility."""
    raise NotImplementedError(
        "sync_meta._utc_now is removed in Option B — call datetime.now() "
        "directly, or import from lustre_kernel.payload_schema."
    )


def pending_count(client: QdrantClient) -> int:
    """Removed in Option B — the `_pending` collection no longer exists."""
    raise NotImplementedError(
        "pending_count is removed in Option B. The indexer writes directly "
        "to the canonical collection; there is no staging queue to count."
    )


def write_meta(client: QdrantClient, **kwargs) -> None:
    """Removed in Option B — the SyncManager no longer publishes drift markers."""
    raise NotImplementedError(
        "write_meta is removed in Option B. The SyncManager that depended "
        "on it has been deleted; the indexer reads the same collection it "
        "writes to, so there is no drift to publish."
    )