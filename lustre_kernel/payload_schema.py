"""
lustre_kernel.payload_schema

Canonical Qdrant point payload schema. Single source of truth for
every field name and type stored on a Qdrant point.

Moved from `indexer/schema.py` so that both `search/` and `indexer/`
import the same constants. The `Payload` TypedDict lives here; the
writer (`indexer/upsert.py:build_payload`) and the readers (`search/*`)
both import from here, so a typo in one place can't silently break the
other side.

Adding a field:
  1. Add the constant below.
  2. Extend `Payload` TypedDict with the matching key/type.
  3. If readers need it, add the constant import to the consumer.

Renaming or removing a field is a breaking change to the on-disk
collection — drop the collection and reindex.
"""

from __future__ import annotations

from typing import TypedDict

__all__ = [  # noqa: RUF022 — grouped by category for readability; strict alphabetic sort would scatter related fields
    # Top-level payload fields
    "FIELD_ID",
    "FIELD_PATH",
    "FIELD_SHARD",
    "FIELD_COLLECTION",
    "FIELD_BLURHASH",
    "FIELD_FOLDER",
    "FIELD_MTIME",
    "FIELD_SIZE",
    "FIELD_MODEL_NAME",
    "FIELD_MODEL_REVISION",
    "FIELD_MODEL_DIM",
    "FIELD_INDEXED_AT",
    # Diversity fingerprints
    "FIELD_CONTENT_SHA256",
    "FIELD_DHASH",
    "FINGERPRINT_FIELDS",
    # TypedDict
    "Payload",
    "payload_field_names",
    "REQUIRED_FIELDS",
    "require_fields",
]


# --- Top-level payload fields --------------------------------------------

FIELD_ID = "id"
FIELD_PATH = "path"
FIELD_SHARD = "shard"
FIELD_COLLECTION = "collection"
FIELD_BLURHASH = "blurhash"
FIELD_FOLDER = "folder"
FIELD_MTIME = "mtime"
FIELD_SIZE = "size"
FIELD_MODEL_NAME = "model_name"
FIELD_MODEL_REVISION = "model_revision"
FIELD_MODEL_DIM = "model_dim"
FIELD_INDEXED_AT = "indexed_at"


# --- Diversity fingerprints (sub-keys; same flat payload) ---------------

FIELD_CONTENT_SHA256 = "content_sha256"
FIELD_DHASH = "dhash"

# All fingerprint keys in one place so callers can iterate without
# listing them by hand.
FINGERPRINT_FIELDS: tuple[str, ...] = (FIELD_CONTENT_SHA256, FIELD_DHASH)


class Payload(TypedDict, total=False):
    """Canonical Qdrant point payload (Qdrant stores payload as JSON).

    All fields are optional in the type system because Qdrant payload
    indexing is incremental — points written by older indexer versions
    may omit later-added fields. Readers tolerate that.
    """
    id: str
    path: str
    shard: str
    collection: str
    blurhash: str | None
    folder: str
    mtime: float
    size: int
    model_name: str
    model_revision: str
    model_dim: int
    indexed_at: str  # ISO-8601 UTC string
    # Diversity fingerprints
    content_sha256: str | None
    dhash: str | None


# Single required set. Adding a field that every point must have
# means bumping REQUIRED_FIELDS and reindexing — which is "wipe the
# collection and reindex", the same operation we do for any other
# breaking schema change.
REQUIRED_FIELDS: frozenset[str] = frozenset({
    FIELD_ID,
    FIELD_PATH,
    FIELD_FOLDER,
    FIELD_MODEL_NAME,
    FIELD_MODEL_REVISION,
    FIELD_MODEL_DIM,
})


def payload_field_names() -> list[str]:
    """Sorted list of every payload field name declared in this module.

    Used by `tests/test_schema.py` to cross-check that `SCHEMA.md`
    mentions every field.
    """
    return sorted({
        FIELD_ID,
        FIELD_PATH,
        FIELD_SHARD,
        FIELD_COLLECTION,
        FIELD_BLURHASH,
        FIELD_FOLDER,
        FIELD_MTIME,
        FIELD_SIZE,
        FIELD_MODEL_NAME,
        FIELD_MODEL_REVISION,
        FIELD_MODEL_DIM,
        FIELD_INDEXED_AT,
        FIELD_CONTENT_SHA256,
        FIELD_DHASH,
    })


def require_fields(payload: dict[str, object]) -> None:
    """Assert `payload` contains every REQUIRED_FIELDS entry.

    Raises:
        ValueError: with a structured message naming the missing fields.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValueError(
            f"payload missing required field(s) {missing!r}"
        )
