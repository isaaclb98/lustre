"""Fast local development server with optional model-free demo data."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run lustre locally for UI iteration."
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Use the deterministic mock text encoder; do not load/download SigLIP2.",
    )
    parser.add_argument(
        "--demo-data",
        action="store_true",
        help="Start with an in-memory Qdrant collection and N demo photos (--demo-count).",
    )
    parser.add_argument(
        "--demo-count",
        type=int,
        default=200,
        metavar="N",
        help="Number of generated demo photos (default: 200 — enough to exercise infinite scroll).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    return parser


def _make_demo_images(root: Path, count: int) -> list[Path]:
    """Generate synthetic demo photos using scripts/synth_photos.py.

    Earlier versions of this function wrote flat 2-tone PNGs with a
    single horizon line. The synth_photos generator produces
    realistic-ish JPEGs with sky gradients, mountain silhouettes,
    city skylines, etc. — much closer to what the UI is meant to
    render. The seeded RNG ensures the same demo set every run,
    so screenshots are stable.
    """
    import sys as _sys
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))
    from scripts.synth_photos import generate as synth_generate

    root.mkdir(parents=True, exist_ok=True)
    result = synth_generate(root, count=count, prefix="demo", seed=42)
    return [Path(item["path"]) for item in result.index]


def _make_demo_thumbnails(thumb_dir: Path, points: list) -> None:
    """Write the WebP thumbnails the thumbnail router serves.

    The router serves PRE-GENERATED files from THUMBNAIL_DIR and
    404s for anything missing — the frontend falls back to the
    blurhash. A demo seed that generates no thumbnails therefore
    renders every tile as a broken image, and the e2e suite ends up
    asserting against a page that never showed a photo.

    So the demo seed writes the same thumbnails the real indexer
    writes: one WebP per photo, at the same 384px long edge, in the
    same two-level prefix layout the router looks for.
    """
    from PIL import Image, ImageOps

    thumb_dir.mkdir(parents=True, exist_ok=True)
    for point in points:
        point_id = point.id
        source = Path(point.payload["path"])
        out_dir = thumb_dir / point_id[:2]
        out_path = out_dir / f"{point_id}.webp"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((384, 384), Image.Resampling.LANCZOS)
                im.save(out_path, "WEBP", quality=80, method=4)
        except Exception:
            # A missing thumbnail is the router's 404 path, which the
            # frontend already handles — it must not stop the seed.
            logger.warning("demo thumbnail failed for %s", point_id, exc_info=True)


def _build_demo_app(count: int) -> FastAPI:
    """Build the real app against an in-memory Qdrant with local demo photos."""
    # THUMBNAIL_DIR is read at import time by search.config, so it has
    # to be set BEFORE search.config is imported below. The demo
    # thumbnails live beside the demo photos, in their own directory.
    _thumb_dir = Path(tempfile.gettempdir()) / f"lustre-demo-thumbs-{os.getpid()}"
    os.environ["THUMBNAIL_DIR"] = str(_thumb_dir)

    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    from search import config
    from search.app import create_app
    from search.qdrant_client import QdrantSearch

    os.environ["SEARCH_NO_MODEL"] = "1"
    os.environ["SEARCH_TEST_MODE"] = "1"
    # Per-worker NAS_IMAGES_BASE so concurrent pytest-xdist workers
    # don't race on the same demo directory (one worker's
    # _make_demo_images clobbering another's mid-write produced
    # PIL.UnidentifiedImageError on /tmp/lustre-demo/*.jpg
    # at ~96% of pytest progress).
    os.environ["NAS_IMAGES_BASE"] = str(
        Path(tempfile.gettempdir()) / f"lustre-demo-{os.getpid()}-{id(object())}"
    )
    # Use a unique IndexDB path per invocation so re-running tests
    # (or _build_demo_app being called twice in one pytest session,
    # e.g. from test_v2_smoke and dev_server.py simultaneously) does
    # not collide on the albums/demodata UNIQUE constraints. The
    # demo DB is short-lived by design — it lives in tempfile and
    # is recreated every time someone runs `_build_demo_app`.
    _demo_db_path = Path(tempfile.gettempdir()) / f"lustre-demo-{os.getpid()}-{id(object())}.db"
    os.environ["INDEX_DB_PATH"] = str(_demo_db_path)
    os.environ["QDRANT_COLLECTION"] = "images_demo"
    # API result URLs must point at this local server, not the production
    # default (localhost:8000), otherwise Chromium shows empty cards.
    os.environ.setdefault("WEB_UI_URL", "http://127.0.0.1:8765")

    demo_root = Path(os.environ["NAS_IMAGES_BASE"])
    paths = _make_demo_images(demo_root, count)
    client = QdrantClient(location=":memory:")
    collection = "images_demo"
    from lustre_kernel.registry import get_default_registry
    _dim = get_default_registry().get(
        config.get_model_name_for_variant(config.get_siglip_variant())
    ).dim
    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(size=_dim, distance=qmodels.Distance.COSINE),
    )
    points = []
    for index, path in enumerate(paths):
        vector = [0.0] * _dim
        vector[index] = 1.0
        # Read image dimensions. Use a fixed valid blurhash per slot
        # so the frontend can still tint glass panels by photo without
        # paying the 6s-per-image encode cost on the demo seed path.
        # The blurhashes are real, valid, 4x3 component strings keyed
        # 7 distinct palettes so grid rows have clear color variety.
        from PIL import Image as _Img
        _img = _Img.open(path).convert("RGB")
        _sz = _img.size
        _BLURHASH_BY_TINT = [
            "LEHV6nWB2yk8pyo0adR*.7kCMdnj",  # warm orange
            "L6PZfSi_.AyE_3t7t7R**0o#DgR4",  # teal
            "LKO2:N%2Tw=w]~RBVZRi};RPxuwH",  # violet
            "LFE.@D9F01_2%L%MIVD*9GofRjWB",  # pink
            "L9AS}j_3?bD%fQM{ofof~qWBM_R*",  # blue
            "LjJ5LyWB?b~q?b%MWBoe.aaen$WB",  # lavender
            "L3JhgM?b00WB?b~q4n9F-;RjRjRj",  # olive
        ]
        bh = _BLURHASH_BY_TINT[index % len(_BLURHASH_BY_TINT)]
        points.append(
            qmodels.PointStruct(
                id=f"00000000-0000-4000-8000-{index + 1:012d}",
                vector=vector,
                payload={
                    "path": str(path),
                    "collection": "demo",
                    "shard": "demo",
                    "width": _sz[0],
                    "height": _sz[1],
                    **(({"blurhash": bh}) if bh else {}),
                },
            )
        )
    client.upsert(collection_name=collection, points=points)

    # The thumbnail router serves pre-generated files and 404s for
    # anything missing, so the seed has to write them — otherwise
    # every tile in the demo renders as a broken image.
    _make_demo_thumbnails(_thumb_dir, points)

    qdrant = QdrantSearch(client=client, collection=collection, timeout_ms=2000)
    app = create_app(cfg=config.load(), qdrant=qdrant)

    # ── Seed relational data (after app init creates tables) ────
    import sqlite3 as _sqlite3
    _db_path = os.environ["INDEX_DB_PATH"]
    with _sqlite3.connect(_db_path) as conn:
        fav_count = max(1, len(points) // 2)
        for i in range(fav_count):
            conn.execute(
                "INSERT OR IGNORE INTO favorites (id, favorited_at) VALUES (?, datetime('now'))",
                (points[i].id,),
            )
        for i in range(len(points) - 3, len(points)):
            conn.execute(
                "INSERT OR IGNORE INTO dislikes (id, disliked_at, source) VALUES (?, datetime('now'), 'manual')",
                (points[i].id,),
            )
        # Split the favourites between the two albums. An album needs
        # enough members to fill whole grid rows: a wall of three
        # photos cannot cover a five-column row, so the row would be
        # short and the background would show through at the edge.
        # The presentation wall is the album's reason to exist, so
        # the demo albums are sized to exercise it.
        per_album = max(1, min(24, fav_count // 2))
        cur = conn.execute(
            "INSERT INTO albums (name, description, created_at, updated_at) VALUES (?, ?, datetime('now'), datetime('now'))",
            ("Landscape picks", "Curated landscape photos from the library."),
        )
        a1 = cur.lastrowid
        for i in range(per_album):
            conn.execute(
                "INSERT OR IGNORE INTO album_memberships (album_id, favorite_id, added_at) VALUES (?, ?, datetime('now'))",
                (a1, points[i].id),
            )
        cur2 = conn.execute(
            "INSERT INTO albums (name, description, created_at, updated_at) VALUES (?, ?, datetime('now'), datetime('now'))",
            ("Night scenes", "Urban and night photography."),
        )
        a2 = cur2.lastrowid
        for i in range(per_album, min(2 * per_album, fav_count)):
            conn.execute(
                "INSERT OR IGNORE INTO album_memberships (album_id, favorite_id, added_at) VALUES (?, ?, datetime('now'))",
                (a2, points[i].id),
            )

    # Register a centroid for each seeded album. The app registers an
    # album's centroid when the album is created through the API, and
    # re-attaches the ones that already exist at STARTUP — but the
    # seed inserts its albums AFTER create_app() has run, so without
    # this the albums exist with no centroid and their Search button
    # leads to an empty result.
    from search.app import _register_album_centroid

    for album_id in (a1, a2):
        if album_id is None:
            raise RuntimeError("demo album insert did not return an id")
        _register_album_centroid(album_id)

    return app


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.no_model or args.demo_data:
        os.environ["SEARCH_NO_MODEL"] = "1"

    import uvicorn

    if args.demo_data:
        app = _build_demo_app(args.demo_count)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        uvicorn.run(
            "search.app:app",
            
            host=args.host,
            port=args.port,
            reload=args.reload,
        )


if __name__ == "__main__":
    main()
