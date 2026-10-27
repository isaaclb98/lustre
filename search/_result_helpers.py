"""
search/_result_helpers.py — small pure helpers shared across routers.

These were previously closure-bound to `app.create_app`, which
made `/api/search` and `/api/centroids/{name}/search`
unextractable. They're pure functions of their inputs (no
hidden state, no closure deps) so they live here.

What stays closure-bound (for now):
- `_results_from_hits`: needs `index_db` for fav/dis lookups
  + `_cfg.web_ui_url` for URL hydration. Will move here once
  the remaining closure helpers (`_favorite_id_set`,
  `_resolve_filename_filter`) are lifted.
- `_resolve_filename_filter`: needs `index_db` for
  `list_filename_match_ids`. Same blocker.
- `_favorite_id_set`: needs `index_db`. Same blocker.

Everything else is here.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from search.diversity import DiversityStats
from search.models import DiversityMetadata, ErrorResponse


def parse_filename(request: Request) -> str:
    """
    Read the optional `?filename=` query param, stripped.

    Single-value: the existing UI is a single text input. If the
    user ever sends multiple `?filename=` values, the first
    non-empty one wins (consistent with how single-valued
    `?centroid=` works in `parse_centroid`). Empty /
    whitespace-only returns "" so callers can do the standard
    "if not raw: skip" check.

    The pattern itself is validated later — `resolve_filename_filter`
    in `_indexed_helpers.py` turns the raw string into a native
    Qdrant path filter and surfaces 400 on invalid syntax.
    """
    for raw in request.query_params.getlist("filename"):
        value = raw.strip()
        if value:
            return value
    return ""


def parse_collections(request: Request) -> list[str]:
    """
    Read all `?collection=` query params from the request, in
    stable order. The multi-value shape is what powers the
    chip-style filter UI on the frontend.

    The query param is the singular `collection` (NOT the plural
    `collections`) — matches the canonical implementation in
    search/app.py and the frontend's URL builder. An earlier
    version of this helper read the plural, which silently
    no-op'd the collection filter (queries with `?collection=kpop`
    would parse to an empty list, so every photo was returned).
    This is a real-world bug, not a test fixture mismatch.

    `getlist()` preserves order and skips duplicates the way
    the URL is written; we don't dedupe here because the user
    might paste a duplicate and the search behavior is the
    same.
    """
    return [c for c in request.query_params.getlist("collection") if c]


def coerce_view(raw: str | None) -> str:
    """Map the `view` query param to a known value.

    Defaults to "grid" when the param is missing or
    unrecognised; the SPA currently renders only the grid but
    this lets a future feed view hook in without a contract
    change.
    """
    if raw in ("grid", "feed"):
        return raw
    return "grid"


def diversity_metadata(stats: DiversityStats) -> DiversityMetadata:
    """Convert a compute-side DiversityStats into the wire DiversityMetadata."""
    return DiversityMetadata(
        requested=stats.requested,
        applied=stats.applied,
        mode=stats.mode,
        strength=stats.strength,
        candidate_count=stats.candidate_count,
        result_count=stats.result_count,
        duplicate_images_collapsed=stats.duplicate_images_collapsed,
        semantic_groups_covered=stats.semantic_groups_covered,
        depth=stats.depth,
        pool_depth=stats.pool_depth,
    )


def bad_request(detail: str) -> JSONResponse:
    """Build a 400 JSONResponse with the documented error envelope."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="bad_request", detail=detail, code="bad_request",
        ).model_dump(),
    )


def internal_error(detail: str) -> JSONResponse:
    """Build a 500 JSONResponse with the documented error envelope."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error", detail=detail, code="internal_error",
        ).model_dump(),
    )


def qdrant_unreachable(detail: str) -> JSONResponse:
    """Build a 502 JSONResponse for Qdrant connectivity failures."""
    return JSONResponse(
        status_code=502,
        content=ErrorResponse(
            error="qdrant_unreachable", detail=detail, code="qdrant_unreachable",
        ).model_dump(),
    )


def qdrant_timeout(detail: str) -> JSONResponse:
    """Build a 504 JSONResponse for Qdrant timeout failures."""
    return JSONResponse(
        status_code=504,
        content=ErrorResponse(
            error="qdrant_timeout", detail=detail, code="qdrant_timeout",
        ).model_dump(),
    )


def parse_centroids(request: Request) -> list[str]:
    """Read every `?centroid=...` value, preserving URL order.

    Empty / whitespace-only values are dropped. Repeated names
    are NOT deduped — the user might want a centroid to count
    twice via weights, and deduping would silently change the
    arithmetic. (E.g. `?centroid=a&centroid=a&weights=1,2`
    legitimately gives weight 3 to a — deduping would shrink
    it to 1.) `blend_centroids` sums weights so a repeated
    name just contributes more to the mean.

    Single `?centroid=a` returns `["a"]` — the call site
    doesn't need a special-case branch.
    """
    return [
        raw.strip()
        for raw in request.query_params.getlist("centroid")
        if raw.strip()
    ]


def parse_weights(request: Request, n: int) -> list[float] | None:
    """
    Parse `?weights=` from the request, returning a list of
    length `n` or None to mean "use defaults (all 1.0)".

    Accepted forms (all result in weights for n=2):
      ?weights=1,2          — comma-separated (preferred)
      ?weights=1&weights=2  — repeated params
      ?weights=1.0          — single value broadcast to all n
      (omitted)             — None → defaults

    Negative, zero, and non-numeric values are rejected with
    HTTPException(400) so the user gets immediate feedback
    instead of a silent zero-vector blend downstream.
    """
    from fastapi import HTTPException

    if n == 0:
        return []
    raw_values = request.query_params.getlist("weights")
    if not raw_values:
        return None
    # Flatten: each param may itself be comma-separated.
    flat: list[str] = []
    for raw in raw_values:
        flat.extend(p.strip() for p in raw.split(",") if p.strip())
    if not flat:
        return None
    if len(flat) == 1:
        try:
            w = float(flat[0])
        except ValueError as err:
            raise HTTPException(
                status_code=400,
                detail=f"weight {flat[0]!r} is not a number",
            ) from err
        if w <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"weights must be positive (got {w})",
            )
        return [w] * n
    if len(flat) != n:
        raise HTTPException(
            status_code=400,
            detail=(
                f"got {len(flat)} weights for {n} centroids; "
                f"counts must match (or pass a single weight "
                f"to broadcast)"
            ),
        )
    try:
        out = [float(x) for x in flat]
    except ValueError as err:
        raise HTTPException(
            status_code=400,
            detail=f"weights must be numbers (got {flat!r})",
        ) from err
    if any(w <= 0 for w in out):
        raise HTTPException(
            status_code=400,
            detail=f"weights must be positive (got {out})",
        )
    return out



# Public surface — explicit so `from search._result_helpers import *`
# has a deterministic set and the docstring is one stop for
# "what's in this module?".
__all__ = [
    "bad_request",
    "coerce_view",
    "diversity_metadata",
    "internal_error",
    "parse_centroids",
    "parse_collections",
    "parse_filename",
    "parse_weights",
    "qdrant_timeout",
    "qdrant_unreachable",
]
