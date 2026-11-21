"""
search/layout.py — the deployment's photo-grid layout mode.

The frontend is a single static SvelteKit build shared by every
deployment, so the grid's layout mode cannot live in the bundle:
switching a deployment between the uniform square grid and
Pinterest-style masonry must not require a rebuild.

The backend stamps the mode onto the <html> element of the shell it
serves

    <html lang="en" data-theme="dark" data-masonry="1">

and the SPA reads it back from there — the same contract the theme
uses. The user can override the deployment's default in the browser
(the Settings page); this attribute is the DEFAULT, not a lock.
"""

from __future__ import annotations

import re

__all__ = ["MASONRY_ATTRIBUTE", "inject_masonry"]

# The attribute the layout mode travels on. Must match the frontend's
# MASONRY_ATTRIBUTE in frontend/src/lib/layout.ts.
MASONRY_ATTRIBUTE = "data-masonry"

# <html lang="en"> → <html lang="en" data-masonry="1">
_HTML_TAG = re.compile(rb"<html(\s[^>]*)?>", re.IGNORECASE)

# An existing data-masonry attribute, so re-stamping replaces rather
# than duplicates it.
_DATA_MASONRY_ATTR = re.compile(rb"\s+data-masonry=[\"'][^\"']*[\"']")


def inject_masonry(html: bytes, enabled: bool) -> bytes:
    """Return `html` with the masonry marker stamped onto <html>.

    Idempotent: an existing data-masonry attribute is replaced rather
    than duplicated, so a shell that already carries one still ends
    up with exactly one.

    The attribute is only stamped when masonry is ON. A shell with no
    attribute means "the square grid" — the same default the frontend
    assumes when the marker is missing.
    """
    if not enabled:
        return html

    def replace(match: re.Match[bytes]) -> bytes:
        attrs = _DATA_MASONRY_ATTR.sub(b"", match.group(1) or b"")
        attr = f' {MASONRY_ATTRIBUTE}="1"'.encode()
        return b"<html" + attrs + attr + b">"

    return _HTML_TAG.sub(replace, html, count=1)
