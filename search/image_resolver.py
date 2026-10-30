"""
search/image_resolver.py

Maps a Qdrant payload `path` to something the web UI can serve.

The NAS paths in Qdrant payloads are absolute source paths. The
container cannot serve arbitrary file paths. We expose images
through the `/photo/{id}/raw` route instead, so the resolver's
job is to map (id, payload_path) to that URL.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_url(point_id: str, web_ui_url: str) -> str:
    """
    Public URL for an image's raw bytes, given its Qdrant point id.

    This is what the grid thumbnail and detail page `<img src>` use.
    The id is opaque — storage layout stays an implementation detail.
    """
    base = web_ui_url.rstrip("/")
    return f"{base}/photo/{point_id}/raw"


def resolve_local(payload_path: str, base: str, prefix: str = "") -> Path | None:
    """
    Map a payload's stored path to a local file path on this machine.

    The indexer stores absolute Windows paths (e.g. Z:\\images\\photo.jpg)
    in Qdrant payloads. When the search app runs on a different machine
    (e.g. Linux with the NAS at /mnt/nas), those paths won't resolve.

    `prefix` handles this: if set and the payload path starts with it,
    the prefix is replaced with `base`. E.g.:
      prefix=Z:\\images, base=/mnt/nas
      payload Z:\\images\\kpop\\photo.jpg -> /mnt/nas/kpop/photo.jpg

    Returns None if the path doesn't exist on disk (file may have been
    deleted after indexing; the route renders a "file not found" notice
    but still returns 200 — the metadata is still useful).
    """
    if not payload_path:
        return None

    p = Path(payload_path)

    # Cross-machine prefix mapping: replace the indexer's mount path
    # with the local mount path. This handles both Windows-style absolute
    # paths (Z:\nas\...) and POSIX paths when running on different machines.
    # We match by prefix string rather than is_absolute() because on Linux
    # a Windows path like Z:\... is NOT absolute (no leading /), so the
    # is_absolute() check would fail.
    if prefix:
        payload_str = str(p)
        if payload_str.startswith(prefix):
            rel = payload_str[len(prefix):].lstrip("\\/").replace("\\", "/")
            p = Path(base) / rel
            logger.debug(
                "resolve_local: rewrote %s -> %s (prefix=%s, base=%s)",
                payload_str, p, prefix, base,
            )
            # Already resolved via prefix; skip the fallback join below.
            return _check_exists(p)

    # Try the path as-is (works when indexer and search share the same mount).
    if not p.is_absolute():
        p = Path(base) / p

    return _check_exists(p)


def _check_exists(p: Path) -> Path | None:
    try:
        if p.exists() and p.is_file():
            return p
    except OSError as e:
        logger.debug("resolve_local: check_exists(%s): %s", p, e)
        return None
    return None


def guess_content_type(local_path: Path) -> str:
    """
    Best-effort Content-Type for a file, falling back to application/octet-stream.
    """
    ctype, _ = mimetypes.guess_type(str(local_path))
    if ctype:
        return ctype
    # Sniff from extension when mimetypes doesn't know.
    ext = local_path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".jfif": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".heic": "image/heic",
        ".heif": "image/heif",
        ".avif": "image/avif",
    }.get(ext, "application/octet-stream")
