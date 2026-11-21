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

__all__ = [
    "MASONRY_ATTRIBUTE",
    "THUMBNAIL_SIZE_ATTRIBUTE",
    "THUMBNAIL_VARIANTS_ATTRIBUTE",
    "inject_masonry",
    "inject_thumbnail_size",
    "inject_thumbnail_variants",
]

# The attribute the layout mode travels on. Must match the frontend's
# MASONRY_ATTRIBUTE in frontend/src/lib/layout.ts.
MASONRY_ATTRIBUTE = "data-masonry"

# <html lang="en"> → <html lang="en" data-masonry="1">
_HTML_TAG = re.compile(rb"<html(\s[^>]*)?>", re.IGNORECASE)

# An existing data-masonry attribute, so re-stamping replaces rather
# than duplicates it.
_DATA_MASONRY_ATTR = re.compile(rb"\s+data-masonry=[\"'][^\"']*[\"']")


# The attribute the thumbnail's long edge travels on.
#
# The frontend builds a `srcset` for the grid's thumbnails, and a
# `srcset` entry has to state the width the image actually IS. The
# thumbnail's long edge is a deployment-level number (THUMBNAIL_SIZE),
# and it is not the same on every deployment: the dev stack renders
# 512, production renders 384. A srcset that claims the wrong width
# makes the browser pick the wrong density — and a portrait photo's
# thumbnail is NARROWER than its long edge, so claiming the long edge
# is wrong for it too.
#
# So the deployment publishes the number and the frontend reads it
# back, the same way the theme and the layout mode travel.
THUMBNAIL_SIZE_ATTRIBUTE = "data-thumbnail-size"

_DATA_THUMBNAIL_SIZE_ATTR = re.compile(
    rb"\s+data-thumbnail-size=[\"'][^\"']*[\"']"
)

# The attribute the DISPLAY thumbnail's smaller variants travel on.
#
# The frontend builds a `srcset` so the browser fetches the smallest
# asset that covers the tile it is about to paint. That needs the
# ladder's sizes, and they are a deployment-level number (the
# indexer generates them, the router serves them), so they ride
# the shell like the theme and the layout mode do.
#
# The value is a comma-separated list of long edges, e.g.
# "192,384". The canonical (data-thumbnail-size) is NOT in the list:
# it is the fallback the router serves when a variant is missing.
THUMBNAIL_VARIANTS_ATTRIBUTE = "data-thumbnail-variants"

_DATA_THUMBNAIL_VARIANTS_ATTR = re.compile(
    rb"\s+data-thumbnail-variants=[\"'][^\"']*[\"']"
)


def inject_thumbnail_variants(html: bytes, sizes: tuple[int, ...]) -> bytes:
    """Return `html` with the thumbnail's variant ladder stamped onto <html>.

    Idempotent: an existing data-thumbnail-variants attribute is
    replaced rather than duplicated. An empty ladder stamps nothing
    — the frontend then builds a single-entry srcset, which is what
    a deployment without variants wants.
    """
    if not sizes:
        return html

    def replace(match: re.Match[bytes]) -> bytes:
        attrs = _DATA_THUMBNAIL_VARIANTS_ATTR.sub(b"", match.group(1) or b"")
        value = ",".join(str(int(s)) for s in sorted(sizes))
        attr = f' {THUMBNAIL_VARIANTS_ATTRIBUTE}="{value}"'.encode()
        return b"<html" + attrs + attr + b">"

    return _HTML_TAG.sub(replace, html, count=1)


def inject_thumbnail_size(html: bytes, size: int) -> bytes:
    """Return `html` with the thumbnail's long edge stamped onto <html>.

    Idempotent: an existing data-thumbnail-size attribute is replaced
    rather than duplicated.
    """
    if size <= 0:
        return html

    def replace(match: re.Match[bytes]) -> bytes:
        attrs = _DATA_THUMBNAIL_SIZE_ATTR.sub(b"", match.group(1) or b"")
        attr = f' {THUMBNAIL_SIZE_ATTRIBUTE}="{int(size)}"'.encode()
        return b"<html" + attrs + attr + b">"

    return _HTML_TAG.sub(replace, html, count=1)


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
