"""
search/theme.py — the deployment's UI theme, applied to the SPA shell.

The frontend is a single static SvelteKit build shared by every
deployment, so the theme cannot live in the bundle: switching themes
must not require a rebuild. The backend stamps the theme onto the
<html> element of the shell it serves

    <html lang="en" data-theme="dark">

and the SPA reads it back from there. The CSS selects the palette
off the same attribute, and the grid geometry (gutter, tile radius)
follows it in JS — so the two halves of the frontend cannot disagree
about which theme is running.

The attribute is the whole contract. Everything downstream — the
dark stylesheet, the grid config, the tile treatment — keys off it.
"""

from __future__ import annotations

import hashlib
import re
from email.utils import formatdate
from pathlib import Path

from fastapi.responses import Response

__all__ = ["THEMES", "THEME_ATTRIBUTE", "inject_theme", "shell_response"]

# The attribute the theme travels on. Must match the frontend's
# THEME_ATTRIBUTE in frontend/src/lib/theme.ts.
THEME_ATTRIBUTE = "data-theme"

# The themes the frontend ships. "light" is the default and is the
# current look; "dark" is the optional alternative.
THEMES = ("light", "dark")

# <html lang="en"> → <html lang="en" data-theme="dark">
_HTML_TAG = re.compile(rb"<html(\s[^>]*)?>", re.IGNORECASE)

# An existing data-theme attribute, so re-stamping replaces rather
# than duplicates it.
_DATA_THEME_ATTR = re.compile(rb"\s+data-theme=[\"'][^\"']*[\"']")


def inject_theme(html: bytes, theme: str) -> bytes:
    """Return `html` with the theme stamped onto the <html> element.

    Idempotent: an existing data-theme attribute is replaced rather
    than duplicated, so a shell that already carries one still ends
    up with exactly one.
    """
    if theme not in THEMES:
        raise ValueError(f"unknown theme {theme!r}")

    def replace(match: re.Match[bytes]) -> bytes:
        attrs = _DATA_THEME_ATTR.sub(b"", match.group(1) or b"")
        attr = f' {THEME_ATTRIBUTE}="{theme}"'.encode()
        return b"<html" + attrs + attr + b">"

    return _HTML_TAG.sub(replace, html, count=1)


# Stamped shells, keyed by (path, mtime, size, theme). The shell is
# static for the life of a deployment, so this is read and patched
# once, not once per request.
_SHELL_CACHE: dict[tuple[str, int, int, str], bytes] = {}


def _shell_bytes(path: Path, theme: str) -> bytes:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, theme)
    cached = _SHELL_CACHE.get(key)
    if cached is None:
        cached = inject_theme(path.read_bytes(), theme)
        # One deployment serves one shell; a rebuilt shell must not
        # leave the previous one behind.
        _SHELL_CACHE.clear()
        _SHELL_CACHE[key] = cached
    return cached


def shell_response(path: Path, theme: str) -> Response:
    """Serve the SPA shell with the deployment's theme applied.

    The response carries an ETag over the STAMPED bytes, so a client
    that already has the right shell revalidates to a 304, and
    `must-revalidate` so a theme change (a restart with a different
    LUSTRE_THEME) can never be masked by a stale cached shell.
    """
    stat = path.stat()
    body = _shell_bytes(path, theme)
    etag = hashlib.md5(body).hexdigest()  # noqa: S324
    return Response(
        content=body,
        media_type="text/html",
        headers={
            "ETag": etag,
            "Last-Modified": formatdate(stat.st_mtime, usegmt=True),
            "Cache-Control": "max-age=0, must-revalidate",
        },
    )
