"""Build the two native text indexes the filename filter needs.

The `?filename=` filter is answered by two Qdrant text indexes:

  * `path` — `word` tokenizer with phrase matching. Serves exact
    patterns (`chaewon`). The settings reproduce FTS5's
    `unicode61 remove_diacritics 2` exactly: `word` splits on the
    same boundaries, `lowercase` folds case, `ascii_folding` strips
    diacritics, and phrase matching makes a multi-token body
    (`img_2024`) match only *adjacent* tokens, as FTS5 did.

  * `path_prefix` — `prefix` tokenizer. Serves prefix patterns
    (`chaewon*`). The `prefix` tokenizer indexes every prefix of
    every token, so a bare term matches any token that starts with
    it — which a `word`-tokenized index cannot express.

A collection supports one text index per field name, so the two
tokenizers need two fields. `path_prefix` holds a copy of `path`.

Both indexes are created with `on_disk=True`. Held in RAM, the
`prefix` tokenizer needs >16 GB on a 2.2M-point corpus: it OOM-kills
the container against a 4 GB limit, and a half-built index makes
every subsequent startup try to rebuild it, so Qdrant restart-loops
and never comes up. `on_disk=True` mmaps the index instead — the
same index builds in ~50s at 1.3 GB RSS and boots in ~10s at the
4 GB limit.

The `path_prefix` field must already be populated before its index
is created, or the index covers an empty field. Run
`scripts/backfill_path_prefix.py` first.

Usage:
    python3 scripts/build_text_indexes.py
    python3 scripts/build_text_indexes.py --skip-path   # prefix index only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

QDRANT = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "images")

WORD_PHRASE = {
    "type": "text",
    "tokenizer": "word",
    "min_token_len": 1,
    "lowercase": True,
    "ascii_folding": True,
    "phrase_matching": True,
    "on_disk": True,
}

PREFIX = {
    "type": "text",
    "tokenizer": "prefix",
    "min_token_len": 1,
    "lowercase": True,
    "ascii_folding": True,
    "phrase_matching": False,
    "on_disk": True,
}


def call(path, body=None, method="POST", timeout=1800):
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


def current_schema() -> dict:
    return call(f"/collections/{COLLECTION}", method="GET")["result"].get(
        "payload_schema", {}
    )


def build(field: str, schema: dict) -> None:
    """Create (or recreate) a text index on `field`."""
    existing = current_schema().get(field)
    params = (existing or {}).get("params", {})
    wanted = dict(schema)
    have = {k: v for k, v in params.items() if k in wanted}
    if have == wanted:
        print(f"  {field}: already correct ({params.get('tokenizer')})")
        return
    if existing:
        print(
            f"  {field}: replacing {params.get('tokenizer')} index "
            f"with {schema['tokenizer']}"
        )
        call(f"/collections/{COLLECTION}/index/{field}?wait=true", None, "DELETE")
    else:
        print(f"  {field}: creating {schema['tokenizer']} index")
    t = time.time()
    call(
        f"/collections/{COLLECTION}/index?wait=true",
        {"field_name": field, "field_schema": schema},
        "PUT",
    )
    print(f"    built in {time.time() - t:.0f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-path", action="store_true")
    parser.add_argument("--skip-prefix", action="store_true")
    args = parser.parse_args()

    print(f"collection {COLLECTION!r} at {QDRANT}")
    if not args.skip_path:
        build("path", WORD_PHRASE)
    if not args.skip_prefix:
        build("path_prefix", PREFIX)

    print("\nfinal schema:")
    for field, spec in current_schema().items():
        params = spec.get("params", spec)
        print(f"  {field}: {json.dumps(params)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
