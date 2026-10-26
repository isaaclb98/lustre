"""Verify the native filename filter against the live production corpus.

Read-only. Compares, for a set of real patterns:

  * the native Qdrant text-index result count, and
  * the SQLite FTS5 result count (the system being replaced).

The counts must be identical for the filter to be a drop-in
replacement — the same patterns must match the same photos.

Usage:
    python3 scripts/verify_native_filter_live.py
    python3 scripts/verify_native_filter_live.py --terms chaewon sana 2024
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.request

# The Qdrant endpoint and the SQLite mirror. Both are environment-
# dependent: on the host the Qdrant port is published on localhost,
# inside the container the service is reached by its compose name.
import os

QDRANT = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
DB = os.environ.get("LUSTRE_DB_PATH", "/app/data/images.db")


def call(path, body=None, method="POST", timeout=300):
    req = urllib.request.Request(
        QDRANT + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    raw = resp.read()
    try:
        return json.loads(raw)
    except ValueError:
        return {}


def qdrant_count(field: str, match, exact: bool) -> int:
    result = call(
        "/collections/images/points/count",
        {"filter": {"must": [{"key": field, "match": match}]}, "exact": exact},
    )
    return int(result["result"]["count"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", nargs="*", default=None)
    args = parser.parse_args()

    terms = args.terms or [
        "chaewon", "sana", "2024", "kpop", "yujin", "winter",
        "img_2024", "twice_sana", "blackpink", "aespa",
    ]

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    print(f"{'pattern':<16} {'FTS5':>10} {'native':>10}  {'fts5 s':>8} {'native s':>9}  result")
    print("-" * 74)

    mismatches = 0
    for term in terms:
        is_prefix = term.endswith("*")
        body = term[:-1] if is_prefix else term

        # FTS5: the system being replaced.
        t0 = time.time()
        # The FTS5 index is `images_fts`; `images` is the content
        # table it shadows.
        sql = (
            "SELECT count(*) FROM images_fts WHERE images_fts MATCH ?"
        )
        fts_count = con.execute(
            sql, (body if not is_prefix else f"{body}*",)
        ).fetchone()[0]
        fts_s = time.time() - t0

        # Native: the replacement.
        if is_prefix:
            match = {"text": body}
            field = "path_prefix"
        else:
            match = {"phrase": body}
            field = "path"
        t0 = time.time()
        native_count = qdrant_count(field, match, exact=True)
        native_s = time.time() - t0

        ok = fts_count == native_count
        if not ok:
            mismatches += 1
        print(
            f"{term:<16} {fts_count:>10,} {native_count:>10,}  "
            f"{fts_s:>7.2f}s {native_s:>8.2f}s  {'OK' if ok else 'MISMATCH'}"
        )

    print()
    if mismatches:
        print(f"{mismatches} pattern(s) MISMATCHED")
        return 1
    print(f"all {len(terms)} patterns match FTS5 exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
