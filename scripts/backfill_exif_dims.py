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

Only the two dimension fields change. Qdrant's `set_payload` replaces a
point's payload rather than merging into it, so the whole payload is
read, corrected and written back — the rest of it (`path`, `blurhash`,
`dhash`, `orphan_since`, ...) is left exactly as it is.

The orientation tag is read from the HEAD of each file rather than
through PIL: `getexif()` pulls the whole EXIF block, and a phone
photo's block runs to ~200 KB of maker notes, which made the backfill
read 209 KB per photo and crawl at ~40 photos/s on the NAS.

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

# The EXIF orientation tag, and the values that rotate the frame by
# 90 degrees. 2 and 4 mirror without rotating, so they leave the
# axes alone.
EXIF_ORIENTATION = 274
ROTATED_ORIENTATIONS = frozenset({5, 6, 7, 8})

# How much of the file's head is read to find the orientation tag.
#
# The tag lives in the FIRST IFD of the EXIF block, which starts two
# bytes into the file — so a few KB is plenty. It is not plenty
# enough for PIL: `getexif()` pulls the WHOLE EXIF block, and a
# phone photo's block runs to ~200 KB of maker notes and embedded
# thumbnails. Reading all of it for one tag made the backfill pull
# 209 KB per photo and crawl at ~40 photos/s on the NAS.
HEADER_BYTES = 8192


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


def _orientation_from_tiff(tiff: bytes) -> int | None:
    """The orientation tag out of a TIFF header's first IFD."""
    if len(tiff) < 8:
        return None
    endian = tiff[:2]
    if endian == b"II":
        order = "little"
    elif endian == b"MM":
        order = "big"
    else:
        return None
    ifd_offset = int.from_bytes(tiff[4:8], order)
    if ifd_offset + 2 > len(tiff):
        return None
    count = int.from_bytes(tiff[ifd_offset : ifd_offset + 2], order)
    for i in range(count):
        entry = ifd_offset + 2 + i * 12
        if entry + 12 > len(tiff):
            return None
        tag = int.from_bytes(tiff[entry : entry + 2], order)
        if tag == EXIF_ORIENTATION:
            value = int.from_bytes(tiff[entry + 8 : entry + 10], order)
            return value if 1 <= value <= 8 else None
    return None


# The JPEG frame headers — the marker that carries the picture's
# dimensions. 0xC4 (Huffman tables), 0xC8 (JPEG extension) and 0xCC
# (arithmetic coding) are not frame headers.
_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def _segments(head: bytes, start: int = 2):
    """Walk the JPEG's segment markers in a buffer.

    Yields `(marker, payload_offset, length)` for each segment whose
    marker and length are both inside the buffer.
    """
    pos = start
    end = len(head)
    while pos + 4 <= end:
        if head[pos] != 0xFF:
            pos += 1
            continue
        marker = head[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        if marker in (0xD9, 0xDA):
            return
        length = int.from_bytes(head[pos + 2 : pos + 4], "big")
        if length < 2:
            return
        yield marker, pos + 4, length
        pos += 2 + length


def _jpeg_facts(path: str) -> tuple[int, int, int | None] | None:
    """The photo's BUFFER dims and its EXIF orientation, or None.

    The buffer's dims are the camera's axes — the file's own frame
    header — and the orientation decides whether they are swapped for
    display. Both come from the FILE, so the answer does not depend on
    what the payload happens to say (a payload already corrected to
    the display dims must not be flipped back).

    PIL is not used for this: `Image.open()` on a JPEG reads the whole
    APP1 segment into memory, and a phone photo's EXIF block runs to
    ~200 KB of maker notes. The segment walk below reads the head of
    the file, SKIPS the EXIF payload with a seek, and reads the frame
    header — a few KB in all.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(HEADER_BYTES)
            if not head.startswith(b"\xff\xd8"):
                return None
            orientation: int | None = None
            for marker, offset, length in _segments(head):
                if marker == 0xE1:
                    payload = head[offset : offset + 6]
                    if payload.startswith(b"Exif\x00\x00"):
                        tiff = head[offset + 6 : offset + 6 + 4096]
                        orientation = _orientation_from_tiff(tiff)
                elif marker in _SOF_MARKERS:
                    data = head[offset : offset + 5]
                    if len(data) < 5:
                        return None
                    height = int.from_bytes(data[1:3], "big")
                    width = int.from_bytes(data[3:5], "big")
                    return width, height, orientation
            # The frame header is past the window — it is behind a
            # large EXIF block. Seek past the segment and read on.
            for marker, offset, length in _segments(head):
                if marker == 0xE1 and head[offset : offset + 6].startswith(b"Exif\x00\x00"):
                    fh.seek(offset + length - 2)
                    break
            else:
                return None
            tail = fh.read(HEADER_BYTES)
            if not tail:
                return None
            for marker, offset, length in _segments(tail, start=0):
                if marker in _SOF_MARKERS:
                    data = tail[offset : offset + 5]
                    if len(data) < 5:
                        return None
                    height = int.from_bytes(data[1:3], "big")
                    width = int.from_bytes(data[3:5], "big")
                    return width, height, orientation
            return None
    except OSError:
        return None


def _display_dims(path: str) -> tuple[int, int] | None:
    """The dims the photo at `path` DISPLAYS at, or None if unreadable.

    The file's own frame header gives the BUFFER dims; the EXIF
    orientation says whether they are swapped. The payload's dims are
    NOT trusted for this: a payload already corrected to the display
    dims would be flipped back by them.
    """
    facts = _jpeg_facts(path)
    if facts is None:
        return None
    file_w, file_h, orientation = facts
    if orientation is None:
        # No EXIF orientation: the buffer is what the photo shows.
        return file_w, file_h
    if orientation in ROTATED_ORIENTATIONS:
        return file_h, file_w
    return file_w, file_h


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
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="stop after N points")
    parser.add_argument(
        "--path-contains",
        default="",
        help="only points whose path contains this text (a Qdrant text match)",
    )
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
            if args.path_contains:
                body["filter"] = {
                    "must": [
                        {
                            "key": "path",
                            "match": {"text": args.path_contains},
                        }
                    ]
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
