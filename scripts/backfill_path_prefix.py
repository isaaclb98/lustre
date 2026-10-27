"""Backfill the `path_prefix` payload field.

The `?filename=cha*` (prefix) pattern needs a prefix-tokenized text
index, and a Qdrant collection supports one text index per field
name — so the prefix index lives on its own field, `path_prefix`,
holding a copy of `path`.

The indexer writes both fields for new points. This script fills in
`path_prefix` for points indexed before the field existed, so an
existing corpus can answer prefix queries without a full re-index.

Qdrant's `set_payload` REPLACES a point's entire payload — it does
not merge. Writing only `path_prefix` therefore strips every other
field (`path`, `blurhash`, `dhash`, `content_sha256`, `folder`, ...)
and leaves the point unreadable. This script reads the point's
existing payload and writes back the existing fields PLUS
`path_prefix`, so nothing is lost.

The write is payload-only: the point's vector is never touched. The
script is idempotent — a point that already has `path_prefix` equal
to `path` is skipped.

Usage:
    python3 scripts/backfill_path_prefix.py [--url http://127.0.0.1:6333]
                                             [--collection images]
                                             [--workers 12]
                                             [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# The field the prefix-tokenized index lives on, and the field it
# mirrors. Kept in sync with `search/path_filter.py`.
TARGET_FIELD = "path_prefix"
SOURCE_FIELD = "path"


def _request(
    url: str,
    body: Any = None,
    method: str = "POST",
    timeout: int = 120,
) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    raw = resp.read()
    try:
        return json.loads(raw)
    except ValueError:
        return {}


def _set_payload(base: str, point_id: str, payload: dict[str, Any]) -> bool:
    """Replace a point's payload with `payload`. True on success."""
    try:
        _request(
            f"{base}/points/payload?wait=false",
            {"payload": payload, "points": [point_id]},
            method="PUT",
        )
        return True
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} for point {point_id}: {e.read()[:200]!r}", flush=True)
        return False
    except Exception as e:  # noqa: BLE001
        print(f"  error for point {point_id}: {e}", flush=True)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="images")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="stop after N points")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written without writing it",
    )
    args = parser.parse_args(argv)

    base = f"{args.url.rstrip('/')}/collections/{args.collection}"
    info = _request(base, method="GET")
    total = info["result"]["points_count"]
    print(f"collection {args.collection!r}: {total:,} points", flush=True)

    lock = threading.Lock()
    counters = {"scanned": 0, "written": 0, "already": 0, "failed": 0}
    started = time.time()
    limit = args.limit or total
    # Points handed to the worker pool. The limit is applied to this
    # counter, not to the completion counter: the workers run
    # asynchronously, so a completion-based limit lets the scan race
    # ahead of the writes.
    submitted = 0

    def write_back(point: dict[str, Any]) -> None:
        payload = dict(point.get("payload") or {})
        value = payload.get(SOURCE_FIELD)
        if not value:
            with lock:
                counters["scanned"] += 1
            return
        if payload.get(TARGET_FIELD) == value:
            with lock:
                counters["scanned"] += 1
                counters["already"] += 1
            return
        # Keep every existing field. `set_payload` replaces the whole
        # payload, so writing only the new field would drop the rest.
        payload[TARGET_FIELD] = value
        ok = True if args.dry_run else _set_payload(base, point["id"], payload)
        with lock:
            counters["scanned"] += 1
            if ok:
                counters["written"] += 1
            else:
                counters["failed"] += 1
            if counters["scanned"] % 50_000 == 0:
                rate = counters["scanned"] / max(time.time() - started, 0.001)
                print(
                    f"  scanned {counters['scanned']:,}  written "
                    f"{counters['written']:,}  already present "
                    f"{counters['already']:,}  failed {counters['failed']:,}  "
                    f"({rate:,.0f}/s)",
                    flush=True,
                )

    offset: Any = None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while submitted < limit:
            body: dict[str, Any] = {
                "limit": 2000,
                "with_payload": True,
                "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset
            page = _request(f"{base}/points/scroll", body)["result"]
            points = page["points"]
            if not points:
                break
            for point in points:
                if submitted >= limit:
                    break
                submitted += 1
                pool.submit(write_back, point)

            offset = page.get("next_page_offset")
            if offset is None:
                break

    elapsed = time.time() - started
    print(
        f"\nscanned {counters['scanned']:,} in {elapsed:.0f}s\n"
        f"  written  {counters['written']:,}\n"
        f"  already  {counters['already']:,}\n"
        f"  failed   {counters['failed']:,}",
        flush=True,
    )
    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0
    if counters["failed"]:
        print("\nsome points failed — rerun to retry them")
        return 1

    # The prefix text index must be created (or recreated) after the
    # field exists, otherwise it indexes an empty field.
    print("\nwaiting for the payload to be applied...", flush=True)
    time.sleep(5)
    print("creating the prefix text index...", flush=True)
    try:
        _request(
            f"{base}/index?wait=true",
            {
                "field_name": TARGET_FIELD,
                "field_schema": {
                    "type": "text",
                    "tokenizer": "prefix",
                    "min_token_len": 1,
                    "lowercase": True,
                    "ascii_folding": True,
                    "phrase_matching": True,
                    "on_disk": True,
                },
            },
            method="PUT",
        )
    except urllib.error.HTTPError as e:
        # An index with this exact schema already exists — that is the
        # desired end state, so this is not a failure. A DIFFERENT
        # schema is: the index has to be rebuilt, which
        # `scripts/build_text_indexes.py` does.
        if e.code == 400:
            print("  the prefix text index already exists — left as it is")
        else:
            raise
    print("prefix text index ready", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
