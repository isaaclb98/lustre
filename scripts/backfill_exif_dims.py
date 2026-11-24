"""Backfill the `width` / `height` payload fields with the DISPLAY dims.

A camera held sideways writes the frame in the camera's own landscape
buffer and records the rotation in the EXIF orientation tag. The
picture is only portrait once that tag is applied.

The indexer applied the tag to the PIXELS but recorded the BUFFER's
dimensions, so a portrait phone photo was stored as `5472 x 3648`
instead of `3648 x 5472`. The grid lays each tile out from the
recorded ratio and then cover-crops the upright picture into it, so
every rotated photo showed a cropped horizontal band with its head
cut off. The thumbnail on disk was correct throughout — only the
numbers describing it were wrong.

`indexer/image_loader.py` now reports the display dims, so every
FUTURE index run writes them. This script fixes the points that are
already indexed, without touching a single vector: the embeddings
and the thumbnails were always computed from the upright picture.

The write is payload-only and idempotent — a point whose dims already
match the file is skipped, so the script can be rerun safely.

Only the two dimension fields are written. Qdrant's `set_payload`
replaces a point's payload rather than merging into it, so the write
names the keys it touches (`?key=width&key=height`); the rest of the
payload — `path`, `blurhash`, `dhash`, `orphan_since`, ... — is left
exactly as it is.

Usage:
    python3 scripts/backfill_exif_dims.py [--url http://127.0.0.1:6333]
                                          [--collection images]
                                          [--workers 16]
                                          [--dry-run]
                                          [--limit N]
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

from PIL import Image

# The EXIF orientation tag, and the values that rotate the frame by
# 90 degrees. 2 and 4 mirror without rotating, so they leave the
# axes alone.
EXIF_ORIENTATION = 274
ROTATED_ORIENTATIONS = frozenset({5, 6, 7, 8})


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


def _display_dims(path: str) -> tuple[int, int] | None:
    """The dims the photo at `path` DISPLAYS at, or None if unreadable.

    Reads the header and the EXIF block only — the pixels are never
    decoded. The file's own size is the source of truth, so the answer
    does not depend on what the payload currently claims.
    """
    try:
        with Image.open(path) as img:
            width, height = img.size
            orientation = img.getexif().get(EXIF_ORIENTATION)
    except Exception:  # noqa: BLE001 - a missing or broken file is not fatal
        return None
    if orientation in ROTATED_ORIENTATIONS:
        return height, width
    return width, height


def _set_dims(base: str, point_id: str, payload: dict[str, Any]) -> bool:
    """Write one point's payload back with the corrected dims.

    The WHOLE payload is written, not just the two fields. Qdrant's
    `set_payload` replaces a point's payload rather than merging into
    it, and its `?key=` query parameter is not honoured by the
    server — a keyed write still drops every other field, leaving the
    point without its `path`, `blurhash` or `dhash`. So the payload
    is read, the two dims are corrected, and the whole thing is put
    back.
    """
    try:
        _request(
            f"{base}/points/payload?wait=true",
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
    parser.add_argument("--workers", type=int, default=16)
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
    counters = {"scanned": 0, "written": 0, "already": 0, "missing": 0, "failed": 0}
    started = time.time()
    limit = args.limit or total
    # Points handed to the worker pool. The limit is applied to this
    # counter, not to the completion counter: the workers run
    # asynchronously, so a completion-based limit lets the scan race
    # ahead of the writes.
    submitted = 0

    def fix(point: dict[str, Any]) -> None:
        payload = point.get("payload") or {}
        path = payload.get("path")
        if not path:
            with lock:
                counters["scanned"] += 1
                counters["missing"] += 1
            return
        dims = _display_dims(path)
        if dims is None:
            with lock:
                counters["scanned"] += 1
                counters["missing"] += 1
            return
        width, height = dims
        if (payload.get("width"), payload.get("height")) == (width, height):
            with lock:
                counters["scanned"] += 1
                counters["already"] += 1
            return
        # Correct the two dims and put the whole payload back.
        corrected = dict(payload)
        corrected["width"] = width
        corrected["height"] = height
        ok = True if args.dry_run else _set_dims(base, point["id"], corrected)
        with lock:
            counters["scanned"] += 1
            if ok:
                counters["written"] += 1
            else:
                counters["failed"] += 1
            if counters["scanned"] % 25_000 == 0:
                rate = counters["scanned"] / max(time.time() - started, 0.001)
                print(
                    f"  scanned {counters['scanned']:,}  corrected "
                    f"{counters['written']:,}  already right "
                    f"{counters['already']:,}  missing {counters['missing']:,}  "
                    f"failed {counters['failed']:,}  ({rate:,.0f}/s)",
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
                pool.submit(fix, point)

            offset = page.get("next_page_offset")
            if offset is None:
                break

    elapsed = time.time() - started
    print(
        f"\nscanned {counters['scanned']:,} in {elapsed:.0f}s\n"
        f"  corrected  {counters['written']:,}\n"
        f"  already    {counters['already']:,}\n"
        f"  missing    {counters['missing']:,}\n"
        f"  failed     {counters['failed']:,}",
        flush=True,
    )
    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0
    if counters["failed"]:
        print("\nsome points failed — rerun to retry them")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
