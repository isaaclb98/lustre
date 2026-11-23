#!/usr/bin/env python3
"""
Backfill: write model_variant to all points in a Qdrant collection.

Walks the collection in large scroll batches, writes via set_payload
in chunks. Idempotent on re-runs.

Usage:
    uv run python scripts/backfill_model_variant.py c-radio-v4/so400m
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

from qdrant_client import QdrantClient

SCROLL_BATCH = 10_000  # how many ids per scroll call
WRITE_BATCH = 1000     # how many ids per set_payload call


def _resolve_variant(variant: str) -> tuple[str, str, int]:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from search.config import (
        SIGLIP_VARIANTS,
        get_model_name_for_variant,
        get_vector_dim_for_variant,
    )
    if variant not in SIGLIP_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    return (
        get_model_name_for_variant(variant),
        variant,
        get_vector_dim_for_variant(variant),
    )


def _collection_dim(info: Any) -> Any:
    try:
        v = info.config.params.vectors
    except AttributeError:
        return None
    if isinstance(v, int):
        return v
    s = getattr(v, "size", None)
    if isinstance(s, int):
        return s
    if isinstance(v, dict):
        for vv in v.values():
            ss = getattr(vv, "size", None)
            if isinstance(ss, int):
                return ss
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_collection = os.environ.get("QDRANT_COLLECTION", "images")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None

    _model_name, variant, expected_dim = _resolve_variant(args.variant)
    print(f"variant={variant} dim={expected_dim}")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
    info = client.get_collection(collection_name=qdrant_collection)
    total = info.points_count
    stored_dim = _collection_dim(info)
    print(f"collection {qdrant_collection!r}: points={total} dim={stored_dim}")

    if stored_dim != expected_dim:
        raise SystemExit(
            f"REFUSING: collection dim {stored_dim} ≠ variant dim {expected_dim}"
        )
    if total == 0:
        print("empty; nothing to do")
        return 0

    t0 = time.perf_counter()
    offset: Any = None
    total_written = 0
    batch_count = 0
    while True:
        points, next_offset = client.scroll(
            collection_name=qdrant_collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            break
        ids = [p.id for p in points]
        for i in range(0, len(ids), WRITE_BATCH):
            chunk = ids[i:i + WRITE_BATCH]
            client.set_payload(
                collection_name=qdrant_collection,
                payload={"model_variant": variant},
                points=chunk,
            )
        total_written += len(ids)
        batch_count += 1
        elapsed = time.perf_counter() - t0
        rate = total_written / elapsed if elapsed > 0 else 0
        print(
            f"  batch {batch_count}: wrote {len(ids):,} ids, "
            f"running total {total_written:,} "
            f"({100 * total_written / total:.1f}%) "
            f"rate={rate:,.0f} pts/s",
            flush=True,
        )
        if next_offset is None:
            break
        offset = next_offset
        if args.dry_run:
            print("[dry-run] exiting after first scroll batch")
            return 0

    elapsed = time.perf_counter() - t0
    print(
        f"backfilled {total_written:,} points in {elapsed:.1f}s "
        f"({total_written / elapsed if elapsed > 0 else 0:,.0f} pts/s)"
    )

    sample, _ = client.scroll(
        collection_name=qdrant_collection, limit=1,
        with_payload=["model_variant"], with_vectors=False,
    )
    if sample and sample[0].payload.get("model_variant") == variant:
        print(f"verify OK: first point has model_variant={variant!r}")
        return 0
    print("verify FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())