"""search.routers — FastAPI APIRouter modules per resource group (§B2).

The 39 routes in `search/app.py` are split into one `APIRouter` per
resource group: search, favorites, albums, photos, saved-searches,
centroids, system. Each router is a factory
function `build_<group>_router(...)` returning an `APIRouter` ready
to `app.include_router(...)`.

Currently landed: `system` (`/healthz`, `/api/cache/status`).
Following groups land in follow-up PRs.

See `search/routers/system.py` for the pattern: a factory function
that takes live dependencies (qdrant, cfg, index_db, ...) and
returns an `APIRouter`. Callers `app.include_router(build_router(...))`.
"""