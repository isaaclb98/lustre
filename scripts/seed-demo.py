#!/usr/bin/env python3
"""Populate the running demo server with realistic state.

Re-creates the seeded state used for visual QA against the lustre demo:
  - 5 favorites
  - 2 dislikes
  - 2 albums ("Studio portraits" with 2 members, "Sun + sky" with 1 member)
  - 2 saved searches ("Mountains at dusk", "Forest mood")

Usage:
  # against a demo server running on the default 127.0.0.1:8765
  ./bin/seed-demo.py

  # against a different host
  ./bin/seed-demo.py --base http://localhost:9000

Idempotent: re-running will fail loudly if albums with the same names already
exist (the API rejects duplicate album names). Delete via the API or restart
the demo server with --demo-data to reset.

This script is a development convenience. It is NOT a test, and is not invoked
by pytest.
"""
from __future__ import annotations

import argparse
import json
import sys
from urllib import error, request

SUBJECTS_USED_FOR_FAVORITES = 5  # demo_01..demo_05
SUBJECTS_USED_FOR_DISLIKES = 2   # demo_06..demo_07


def _post(base: str, path: str, body: dict | None = None) -> tuple[int, str]:
    data = json.dumps(body or {}).encode()
    req = request.Request(
        base + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req) as r:
            return r.status, r.read().decode()
    except error.HTTPError as e:
        return e.code, e.read().decode()


def _get(base: str, path: str) -> dict:
    with request.urlopen(base + path) as r:
        return json.loads(r.read())


def _seed_favorites_and_dislikes(base: str) -> tuple[list[str], list[str]]:
    """Like the first 5 demo photos, dislike the next 2."""
    random_payload = _get(base, "/api/random")
    fav_ids = [r["id"] for r in random_payload["results"][:SUBJECTS_USED_FOR_FAVORITES]]
    dis_ids = [r["id"] for r in random_payload["results"][
        SUBJECTS_USED_FOR_FAVORITES : SUBJECTS_USED_FOR_FAVORITES + SUBJECTS_USED_FOR_DISLIKES
    ]]
    for pid in fav_ids:
        code, _ = _post(base, f"/api/favorites/{pid}")
        if code not in (200, 204):
            print(f"  WARN: favorite {pid[:8]} returned {code}", file=sys.stderr)
    for pid in dis_ids:
        code, _ = _post(base, f"/api/dislikes/{pid}")
        if code not in (200, 204):
            print(f"  WARN: dislike {pid[:8]} returned {code}", file=sys.stderr)
    return fav_ids, dis_ids


def _seed_albums(base: str, favorite_ids: list[str]) -> list[dict]:
    albums_payload = [
        {"name": "Studio portraits", "description": "Clean, low-key, one face"},
        {"name": "Sun + sky", "description": "Anything where the sun is the subject"},
    ]
    out = []
    for payload in albums_payload:
        code, body = _post(base, "/api/albums", payload)
        if code != 200:
            print(f"  WARN: create album {payload['name']!r} returned {code}: {body}", file=sys.stderr)
            continue
        out.append(json.loads(body))
    # Add 2 favorites to album 1, 1 to album 2.
    if len(out) >= 2 and len(favorite_ids) >= 3:
        a1, a2 = out[0], out[1]
        for fid in favorite_ids[:2]:
            code, _ = _post(base, f"/api/albums/{a1['id']}/members/{fid}")
            if code != 200:
                print(f"  WARN: add fav to {a1['name']} returned {code}", file=sys.stderr)
        code, _ = _post(base, f"/api/albums/{a2['id']}/members/{favorite_ids[2]}")
        if code != 200:
            print(f"  WARN: add fav to {a2['name']} returned {code}", file=sys.stderr)
    return out


def _seed_saved_searches(base: str) -> None:
    """Note: API uses positives/negatives, not include/exclude."""
    for payload in [
        {"name": "Mountains at dusk", "positives": ["mountain", "dusk"], "negatives": []},
        {"name": "Forest mood", "positives": ["forest", "green", "trees"], "negatives": ["sunset"]},
    ]:
        code, body = _post(base, "/api/saved-searches", payload)
        if code != 201:
            print(f"  WARN: saved-search {payload['name']!r} returned {code}: {body}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8765", help="Demo server base URL")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    print(f"Seeding {base} ...")
    fav_ids, _ = _seed_favorites_and_dislikes(base)
    _seed_albums(base, fav_ids)
    _seed_saved_searches(base)

    final = {
        "favorites": _get(base, "/api/favorites").get("favorites", []),
        "albums": _get(base, "/api/albums").get("albums", []),
        "saved-searches": _get(base, "/api/saved-searches").get("saved_searches", []),
    }
    print(json.dumps({k: len(v) for k, v in final.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
