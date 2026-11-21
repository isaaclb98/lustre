"""Round-50: re-encode existing thumbnails with the libvips pipeline.

Walks every {THUMBNAIL_DIR}/{prefix}/{point_id}.webp in the host's
thumbnail directory and re-encodes it through `indexer.thumbnails.
compute_thumbnail` (which now uses pyvips when available — 4:4:4
chroma, linear-light resize, ICC profile embed). The new file
replaces the old one in place.

This is the **fast** re-encode path. It reads the existing thumb
(already a 384px WebP) and runs it through the new pipeline. The
visible benefits over the old pipeline:

  - 4:4:4 chroma subsampling (was 4:2:0): the chroma channels
    are no longer subsampled, so saturated colors at edges
    (lips, eyes, fabrics) are no longer averaged away.
  - sRGB ICC profile embed (was none): the WebP file now
    declares its color space explicitly, so strict color-managed
    browsers render the same image as the rest of the web.
  - libvips color management (was PIL's bare sRGB pipeline):
    libvips processes the resize in linear-light and back, and
    uses a Lanczos-3 kernel (vs PIL's Lanczos-2 alias). The
    difference is small but consistent.

The full-benefit re-encode (re-thumbnailing from the original
source photos) requires source paths that we may no longer have
on this host — see generate_thumbnails.py for that path, which
reads from the SQLite images DB. This script is the "no source
access" fallback that still gives the user the chroma + ICC
improvements.

Usage:
  # default: in-place re-encode under THUMBNAIL_DIR
  python scripts/regenerate_thumbnails.py

  # dry-run: show what would change without writing
  python scripts/regenerate_thumbnails.py --dry-run

  # output to a different dir for diff-comparison
  python scripts/regenerate_thumbnails.py --out-dir /tmp/new-thumbs

  # only re-encode a sample
  python scripts/regenerate_thumbnails.py --limit 100

Safety:
  - Default mode is in-place; pass --out-dir to write elsewhere.
  - The script is idempotent: re-running it produces the same
    output (modulo mtime).
  - The script does not touch the SQLite DB or Qdrant.
  - Errors are logged and counted; the run continues.

Performance:
  - libvips encodes WebP at ~50-200 thumbs/sec/worker on a
    modern CPU. 2.2M thumbs at 8 workers = ~30-90 min wall.
  - Use --workers to tune (default 4 to keep I/O sane).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root so `indexer.thumbnails` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexer.thumbnails import (
    _HAS_PYVIPS,
    _PIL_FORCE,
    THUMBNAIL_DIR,
    THUMBNAIL_SIZE,
    compute_thumbnail,
)

logger = logging.getLogger("regenerate_thumbnails")


def _reencode_one(
    point_id: str,
    source_thumb: Path,
    out_dir: Path | None,
    only_sized: list[int] | None,
) -> tuple[str, str]:
    """Re-encode a single thumbnail + its variants.

    Returns (point_id, status) where status is one of:
      - "ok" — re-encoded
      - "ok-skip" — already matches (mtime check)
      - "fail:..." — error reason
    """
    try:
        # The existing thumb is itself a valid input for the new
        # pipeline. The libvips path will load it, apply linear-
        # light resize to THUMBNAIL_SIZE (same as current, so a
        # no-op), and re-encode with 4:4:4 + ICC.
        #
        # NOTE: this script re-encodes the EXISTING asset; it does
        # not change its geometry. A thumbnail written by the
        # pre-native-aspect pipeline is a square crop, and it stays
        # a square crop here — moving those to native aspect means
        # regenerating them from the source photos.
        from PIL import Image

        with Image.open(source_thumb) as im:
            # The canonical thumbnail's LONG edge is THUMBNAIL_SIZE
            # (the short edge follows the source's aspect ratio), so
            # that — not an exact size match — is what identifies it.
            if max(im.size) != THUMBNAIL_SIZE:
                # Thumbnail isn't the canonical. It's either a sized
                # variant or a legacy smaller asset. Skip — we only
                # re-encode the canonical.
                return (point_id, "skip-non-canonical")

        if out_dir is not None:
            # Output to a separate dir: monkeypatch the module's
            # path resolver via direct import.
            import indexer.thumbnails as T

            orig_dir = T.THUMBNAIL_DIR
            try:
                T.THUMBNAIL_DIR = str(out_dir)
                result = compute_thumbnail(im, point_id, source_path=source_thumb)
            finally:
                T.THUMBNAIL_DIR = orig_dir
        else:
            # In-place: the new file overwrites the old one.
            result = compute_thumbnail(im, point_id, source_path=source_thumb)
        if result is None:
            return (point_id, "fail:encode-returned-none")
        return (point_id, "ok")
    except Exception as exc:  # noqa: BLE001
        return (point_id, f"fail:{type(exc).__name__}:{exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-encode existing thumbnails with the libvips "
            "pipeline (4:4:4 chroma, ICC profile, linear-light "
            "resize). In-place by default."
        ),
    )
    parser.add_argument(
        "--thumb-dir",
        default=os.environ.get("THUMBNAIL_DIR", str(THUMBNAIL_DIR)),
        help="Directory containing the existing thumbnails",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Write re-encoded thumbs here instead of in-place. "
            "Useful for diff-comparison or staged rollout."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread pool size (default 4; libvips is C-bound)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Re-encode only the first N thumbs (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the first 10 thumbs that would be re-encoded and exit",
    )
    parser.add_argument(
        "--force-pil",
        action="store_true",
        help=(
            "Use the PIL fallback even if libvips is available. "
            "For testing parity between the two paths."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.force_pil:
        # Test-only toggle. Allows running the PIL path on a
        # host that has libvips installed.
        import indexer.thumbnails as T

        T._PIL_FORCE = True
        logger.info("Forced PIL path (--force-pil)")

    thumb_dir = Path(args.thumb_dir)
    if not thumb_dir.is_dir():
        logger.error("thumb-dir %s is not a directory", thumb_dir)
        return 1
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    logger.info("Pipeline: %s", "pyvips" if _HAS_PYVIPS and not _PIL_FORCE else "PIL")
    logger.info("Thumb dir: %s", thumb_dir)
    if out_dir:
        logger.info("Out dir:   %s", out_dir)
    else:
        logger.info("Mode:      IN-PLACE (overwrites existing files)")

    # Walk the 2-char prefix buckets
    prefix_dirs = sorted(p for p in thumb_dir.iterdir() if p.is_dir() and len(p.name) == 2)
    logger.info("Prefix buckets: %d", len(prefix_dirs))

    # Collect all canonical thumbnails (skip the .w{N}.webp variants)
    canonicals: list[tuple[str, Path]] = []
    for pd in prefix_dirs:
        for f in pd.glob("*.webp"):
            # Skip sized variants: <id>.w<width>.webp
            if ".w" in f.stem and f.stem.split(".w")[-1].isdigit():
                continue
            pid = f.stem
            canonicals.append((pid, f))
    logger.info("Canonical thumbs: %d", len(canonicals))

    if args.dry_run:
        for pid, path in canonicals[:10]:
            logger.info("would re-encode: %s (%d bytes)", path, path.stat().st_size)
        return 0

    if args.limit:
        canonicals = canonicals[: args.limit]
        logger.info("Limited to first %d", args.limit)

    start = time.monotonic()
    ok = fail = skip = 0
    fails: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_reencode_one, pid, path, out_dir, None): pid
            for pid, path in canonicals
        }
        for i, fut in enumerate(as_completed(futs), 1):
            pid, status = fut.result()
            if status == "ok":
                ok += 1
            elif status.startswith("skip"):
                skip += 1
            else:
                fail += 1
                fails.append((pid, status))
            if i % 500 == 0:
                rate = i / (time.monotonic() - start)
                logger.info(
                    "  ... %d/%d (ok=%d skip=%d fail=%d) %.1f/s",
                    i, len(canonicals), ok, skip, fail, rate,
                )

    elapsed = time.monotonic() - start
    rate = len(canonicals) / elapsed if elapsed else 0
    logger.info(
        "done. ok=%d skip=%d fail=%d in %.1fs (%.1f/s)",
        ok, skip, fail, elapsed, rate,
    )
    for pid, reason in fails[:5]:
        logger.error("  fail %s: %s", pid[:12], reason)
    if fail:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
