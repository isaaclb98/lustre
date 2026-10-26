"""Restore Qdrant payloads that were stripped by a replace-semantics write.

What happened
-------------
Qdrant's `set_payload` REPLACES a point's whole payload — it does not
merge. A backfill that wrote only `path_prefix` therefore left those
points with a payload of exactly `{"path_prefix": ...}` and dropped
every other field (`path`, `blurhash`, `dhash`, `content_sha256`,
`folder`, `width`, `height`, ...).

Recovery
--------
The point's VECTOR was never touched (a payload update does not
re-encode), and the vector is what the collection is for. So the
repair is a payload-only write, per point, carrying the COMPLETE
payload.

Where the payload comes from
----------------------------
`search/index_db.py` mirrors the collection into SQLite, and that
mirror still holds every field for every point:

    id, path, shard, collection, mtime, size, indexed_at,
    width, height, blurhash

The remaining fields are recomputed or taken from constants:

  * `folder`          — parent directory of `path` (matches the
                         indexer, which stores `path.parent`).
  * `content_sha256`  — SHA-256 of the file bytes.
  * `dhash`           — 64-bit perceptual hash of the file.
  * `model_name`      — the configured embedding model.
  * `model_revision`  — the configured model revision.
  * `model_dim`       — the collection's vector dimension.
  * `model_variant`   — the SIGLIP variant for that dimension.
  * `path_prefix`     — copy of `path` (the prefix-index field).

`content_sha256` and `dhash` are the only fields that need the
source file, so they are computed with the indexer's own
`compute_fingerprints` — the same function the ingest path uses, so
the restored values are identical to what a fresh index would hold.
That means reading and decoding every affected image, which is slow
but is the only way to get the true values.

The script is safe to re-run: it only writes points whose payload is
missing a required field, and it is idempotent.

Usage:
    python3 scripts/repair_payload.py --collection images
    python3 scripts/repair_payload.py --collection images --limit 500
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Payload fields every point must carry. A point missing any of these
# is damaged and gets repaired.
REQUIRED_FIELDS = (
    "path",
    "blurhash",
    "dhash",
    "content_sha256",
    "folder",
    "width",
    "height",
    "size",
    "mtime",
    "shard",
    "collection",
    "indexed_at",
    "model_name",
    "model_revision",
    "model_dim",
    "model_variant",
)


def _request(
    url: str, body: Any = None, method: str = "POST", timeout: int = 120
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="images")
    parser.add_argument(
        "--db",
        default="/app/data/images.db",
        help="SQLite mirror holding id -> path and the other fields",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--model-name",
        default="",
        help="override the model name (use when every point is damaged)",
    )
    parser.add_argument(
        "--model-revision",
        default="",
        help="override the model revision (use with --model-name)",
    )
    parser.add_argument(
        "--model-variant",
        default="",
        help="override the model variant (use with --model-name)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report without writing"
    )
    args = parser.parse_args(argv)

    base = f"{args.url.rstrip('/')}/collections/{args.collection}"

    info = _request(base, method="GET")["result"]
    dim = info["config"]["params"]["vectors"]["size"]
    distance = info["config"]["params"]["vectors"]["distance"]
    print(
        f"collection {args.collection!r}: {info['points_count']:,} points, "
        f"dim={dim} distance={distance}",
        flush=True,
    )

    # The model identity comes from the collection's own payload, so
    # the repair matches whatever indexed the corpus. The first
    # point may itself be damaged, so scan for the first point that
    # actually carries the model fields.
    model_name = model_revision = model_variant = None
    if args.model_name:
        model_name = args.model_name
        model_revision = args.model_revision
        model_variant = args.model_variant
    else:
        offset: Any = None
        while True:
            body: dict[str, Any] = {
                "limit": 500,
                "with_payload": True,
                "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset
            page = _request(f"{base}/points/scroll", body)["result"]
            for p in page["points"]:
                payload = p.get("payload") or {}
                if payload.get("model_name"):
                    model_name = payload.get("model_name")
                    model_revision = payload.get("model_revision")
                    model_variant = payload.get("model_variant")
                    break
            if model_name or not page["points"]:
                break
            offset = page.get("next_page_offset")
            if offset is None:
                break
    if not model_name:
        print("could not determine the model from the collection", file=sys.stderr)
        return 1
    print(
        f"model: name={model_name} revision={model_revision} variant={model_variant}",
        flush=True,
    )

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    # Find the damaged points. `images` is the mirror of the
    # collection; join it against the live payloads to find the
    # points whose payload lost fields.
    print("scanning for damaged points...", flush=True)
    damaged: list[sqlite3.Row] = []
    offset: Any = None
    scanned = 0
    while True:
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
        ids = [p["id"] for p in points]
        placeholders = ",".join("?" * len(ids))
        rows = con.execute(
            f"SELECT * FROM images WHERE id IN ({placeholders})", ids
        ).fetchall()
        by_id = {r["id"]: r for r in rows}
        for p in points:
            scanned += 1
            payload = p.get("payload") or {}
            if all(f in payload for f in REQUIRED_FIELDS):
                continue
            row = by_id.get(p["id"])
            if row is None:
                continue
            damaged.append(row)
        offset = page.get("next_page_offset")
        if offset is None:
            break

    print(f"  scanned {scanned:,}, damaged {len(damaged):,}", flush=True)
    if not damaged:
        print("nothing to repair")
        return 0
    if args.limit:
        damaged = damaged[: args.limit]
        print(f"  limited to {len(damaged):,}", flush=True)
    if args.dry_run:
        print("(dry run — nothing written)")
        return 0

    # The fingerprints need the source file, and computing them means
    # importing the indexer's own implementation.
    from pathlib import Path

    from indexer.fingerprints import compute_fingerprints

    lock = __import__("threading").Lock()
    counters = {"done": 0, "failed": 0, "skipped": 0}
    started = time.time()

    def repair(row: sqlite3.Row) -> None:
        path = row["path"]
        try:
            fp = compute_fingerprints(Path(path))
        except Exception as e:  # noqa: BLE001
            with lock:
                counters["failed"] += 1
                counters["skipped"] += 1
            print(f"  fingerprint failed for {path}: {e}", flush=True)
            return

        payload = {
            "id": row["id"],
            "path": path,
            "path_prefix": path,
            "shard": row["shard"],
            "collection": row["collection"],
            "blurhash": row["blurhash"],
            "folder": str(Path(path).parent),
            "width": row["width"],
            "height": row["height"],
            "mtime": row["mtime"],
            "size": row["size"],
            "indexed_at": row["indexed_at"],
            "model_name": model_name,
            "model_revision": model_revision,
            "model_dim": dim,
            "model_variant": model_variant,
            "content_sha256": fp.get("content_sha256"),
            "dhash": fp.get("dhash"),
        }
        try:
            _request(
                f"{base}/points/payload?wait=false",
                {"payload": payload, "points": [row["id"]]},
                method="PUT",
            )
            with lock:
                counters["done"] += 1
                if counters["done"] % 2000 == 0:
                    rate = counters["done"] / max(time.time() - started, 0.001)
                    print(
                        f"  repaired {counters['done']:,}  failed "
                        f"{counters['failed']:,}  ({rate:,.0f}/s)",
                        flush=True,
                    )
        except urllib.error.HTTPError as e:
            with lock:
                counters["failed"] += 1
            print(f"  HTTP {e.code} for {row['id']}: {e.read()[:160]!r}", flush=True)
        except Exception as e:  # noqa: BLE001
            with lock:
                counters["failed"] += 1
            print(f"  error for {row['id']}: {e}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(repair, damaged))

    print(
        f"\nrepaired {counters['done']:,}  failed {counters['failed']:,}  "
        f"in {time.time() - started:.0f}s",
        flush=True,
    )
    return 0 if counters["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
