"""Definitive FTS5-vs-Qdrant parity harness (sample-anchored).

Runs INSIDE the search container: reads the live images.db read-only
and talks to Qdrant at qdrant:6333.

Sample anchoring: both sides are restricted to the SAME random sample
of point ids, so corpus drift between the SQLite cache and the
Qdrant collection cannot produce a false mismatch. This matters —
the corpus grew 2,184,253 -> 2,200,027 points during this session.

Usage:
    python3 parity_harness.py exact     # word+phrase index
    python3 parity_harness.py prefix    # prefix tokenizer index
"""
import json
import random
import sqlite3
import sys
import time
import urllib.request

Q = "http://qdrant:6333"
DB = "/app/data/images.db"
COLL = "images"
SAMPLE = 40000

EXACT_TERMS = [
    "chaewon", "won", "2024", "kpop", "jpg", "png", "2023", "sakura",
    "img", "heejin", "yujin", "twice", "newjeans", "240331", "9f71e7",
    "d38e6e", "nayeon", "sana", "jeongyeon", "momo", "mina", "dahyun",
    "tzuyu", "jihyo", "chaeyeon", "wonder", "sakura",
]

PREFIX_TERMS = [
    "chaewon", "won", "2024", "kpop", "img", "sana", "twice", "2403",
    "9f7", "d38", "heejin", "sakur",
]


def call(path, body=None, method="POST", t=600):
    r = urllib.request.Request(
        Q + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    resp = urllib.request.urlopen(r, timeout=t)
    raw = resp.read()
    try:
        return json.loads(raw)
    except ValueError:
        return {"_raw": len(raw)}


def build_index(schema):
    call(f"/collections/{COLL}/index/path?wait=true", None, "DELETE")
    t = time.time()
    call(
        f"/collections/{COLL}/index?wait=true",
        {"field_name": "path", "field_schema": schema},
        "PUT",
    )
    return time.time() - t


WORD_PHRASE = {
    "type": "text", "tokenizer": "word", "min_token_len": 1,
    "lowercase": True, "ascii_folding": True, "phrase_matching": True,
}
PREFIX = {
    "type": "text", "tokenizer": "prefix", "min_token_len": 1,
    "lowercase": True, "ascii_folding": True, "phrase_matching": False,
}


def take_sample():
    rows = []
    offset = None
    while len(rows) < SAMPLE:
        body = {"limit": 4000, "with_payload": False, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        r = call(f"/collections/{COLL}/points/scroll", body)["result"]
        rows.extend(str(p["id"]) for p in r["points"])
        offset = r.get("next_page_offset")
        if offset is None:
            break
    random.seed(20261026)
    random.shuffle(rows)
    return set(rows[:SAMPLE])


def qdrant_matched(match, sample):
    out = set()
    offset = None
    while True:
        body = {
            "limit": 2000,
            "with_payload": False,
            "with_vector": False,
            "filter": {"must": [{"key": "path", "match": match}, {"has_id": list(sample)}]},
        }
        if offset is not None:
            body["offset"] = offset
        r = call(f"/collections/{COLL}/points/scroll", body)["result"]
        out.update(str(p["id"]) for p in r["points"])
        offset = r.get("next_page_offset")
        if offset is None:
            break
    return out


def fts_matched(query, sample):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        cur = con.execute(
            "SELECT images.id FROM images_fts "
            "INNER JOIN images ON images.rowid = images_fts.rowid "
            "WHERE images_fts MATCH ?",
            (query,),
        )
        return {str(r[0]) for r in cur.fetchall()} & sample
    finally:
        con.close()


def compare(terms, qdrant_match, fts_query):
    print(f"{'term':14} {'FTS5':>8} {'Qdrant':>8}  verdict", flush=True)
    bad = []
    for term in terms:
        f = fts_matched(fts_query(term), sample)
        q = qdrant_matched(qdrant_match(term), sample)
        same = f == q
        if not same:
            bad.append(term)
            print(
                f"{term:14} {len(f):>8,} {len(q):>8,}  DIFF "
                f"fts-only={len(f - q)} qdr-only={len(q - f)}",
                flush=True,
            )
    print(f"  -> {len(terms) - len(bad)}/{len(terms)} identical", flush=True)
    if bad:
        print(f"  -> mismatching: {bad}", flush=True)
    return bad


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "exact"
    print(f"=== building {'word+phrase' if mode == 'exact' else 'prefix'} index ===", flush=True)
    t = build_index(WORD_PHRASE if mode == "exact" else PREFIX)
    print(f"  built in {t:.0f}s", flush=True)

    print(f"taking a {SAMPLE:,}-point sample...", flush=True)
    sample = take_sample()
    print(f"  sample: {len(sample):,} ids\n", flush=True)

    if mode == "exact":
        # FTS5 bare term  vs  Qdrant MatchPhrase (word+phrase index)
        bad = compare(
            EXACT_TERMS,
            lambda t: {"phrase": t},
            lambda t: t,
        )
        print("\n=== ALSO: MatchText on the same index (expected looser) ===", flush=True)
        compare(
            ["chaewon", "2024", "kpop"],
            lambda t: {"text": t},
            lambda t: t,
        )
    else:
        # FTS5 term*  vs  Qdrant MatchText on a prefix-tokenized index
        bad = compare(
            PREFIX_TERMS,
            lambda t: {"text": t},
            lambda t: t + "*",
        )
        print("\n=== ALSO: MatchText on prefix index for EXACT terms (expected looser) ===", flush=True)
        compare(
            ["chaewon", "2024", "kpop"],
            lambda t: {"text": t},
            lambda t: t,
        )

    print("\n=== restoring word+phrase index ===", flush=True)
    t = build_index(WORD_PHRASE)
    print(f"  rebuilt in {t:.0f}s", flush=True)
    print("\nDONE", flush=True)
