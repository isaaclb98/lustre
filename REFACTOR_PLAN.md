# Image Search Backend — Refactor Plan

**Goal.** Make the backend easier to extend, easier to read, and harder to
break. No functional changes; the public API contract and test suite are
the regression net.

**Strategy.** One sweep on `dev`, not 8 stacked PRs. Isaac confirmed
2026-10-15: *"These shouldn't have been separate PRs in the first place. It
is a big sweep for refactoring. Just merge into dev and work in dev branch
for this."* So all work below lands directly on `dev`. No more
`--base main` PRs for refactor work; the cumulative diff against `main`
keeps growing and that was the original problem. `dev` is the integration
branch for this sweep.

**Out of scope (will NOT be touched in any work below):**
- `indexer/` (its own subsystem, not part of the search service surface)
- `lustre_kernel/` (already strict-typed, CI-gated)
- `frontend/` (refactor lives in a separate cycle)
- Any new feature, new route, new field, or behavior change

**Existing branches I will leave alone** (unmerged, behind `dev`):
- `feat/centroid-diversity`
- `refactor/visual-balance`

If you want me to rebase either of those onto the refactor's `dev`,
open a follow-up — this plan doesn't touch them.

---

## What landed (2026-10-15)

| Phase | Title | Commit | Description |
|---|---|---|---|
| 1 | Suppression discipline + lint tightening | `1dd859e` | 62 `type: ignore` markers removed, return types widened, named justifications for remaining suppressions |
| 2 | Module layout | `eaf2e2c` | Extracted `RandomSessionStore`, `build_surprise_deck` to dedicated modules, deduped helpers, `__all__` added to public modules. Cross-router import gone. |
| 3a | Type-arg pass on `search/` | `5a055a4` | 70 `type-arg` errors fixed. Parametrized `dict`/`list`/`tuple`/`Task`/`Popen`. |
| 3b | Call-arg + no-untyped-def pass | `c4950f9` | 74 call-arg + no-untyped-def errors fixed (`Field(default, ...)` → short-form, dead `PromptState` class removed, FastAPI `response_model=None` for `JSONResponse` unions). |
| 3c | Finish the mypy pass | `5ff8379` | **mypy in `search/`: 67 → 0 errors.** Final pass on isinstance narrowing, numpy slicing, union dispatch, Qdrant stub mismatches. |
| 4 | Typed Pydantic payload models | `5004af9` | 6 new payload models (`HealthzResponse`, `SystemStatusResponse`, `CacheStatusResponse`, `CollectionsListResponse`, `ReloadCentroidsResponse`, `CentroidListResponse`). 6 routes migrated. |
| 5 | Centralized env-var parsing | `0c90fea` | `_str` / `_opt_str` / `_str_any_present` helpers. `config.load()` now uses uniform env-var helpers. |
| 6 | Centralized error-response helpers | `d35d581` | Removed 4 duplicate `_bad_request` / `_qdrant_unreachable` / `_qdrant_timeout` / `_internal_error` definitions across routers and `app.py`. Replaced 2 inline `JSONResponse + ErrorResponse` patterns with the module-level helpers. |
| 7 | Structured-logging helpers | `7ae768d` | New `search/_logging.py` with `log_qdrant_error` / `log_qdrant_timeout` / `log_request_completed`. Migrated Qdrant error/timeout sites in `/api/search` and `/api/centroids/{name}/search`. 4 new unit tests. |
| 8 | Test organization + naming | `7cf0be4` | `tests/README.md` documents the 5-pattern convention. Renamed 3 files to match their actual wiring. |

**Final state on `dev` after the sweep:**

- `mypy search/`: **0 errors** (7 pre-existing in `scripts/synth_photos.py`, dev-only, out of scope)
- `ruff check search/`: **clean**
- `pytest tests/`: **1521/1521 pass** (1517 baseline + 4 new logging tests)
- **All 8 phases done, all committed, all pushed, all CI green.**

## Hard rules (still in effect for any follow-up)

- No behavior changes. Tests must keep passing.
- Each commit is coherent; commits are not "atomic per file."
- Run `ruff check .` and `pytest tests/ --ignore=tests/test_local_sync_perf_helpers.py` after each meaningful chunk. CI runs on `push:dev`.
- Don't touch `scripts/*` unless the mypy error is in a file the dev/CI workflow imports. CI doesn't gate `scripts/`, so errors there are out of scope for now.
- Don't touch `indexer/`, `lustre_kernel/`, or `frontend/` in this sweep.

## Acceptance criteria — final scorecard

| Criterion | Status |
|---|---|
| `mypy search/` error count trends down (target < 20) | ✅ 0 |
| `ruff check .` clean | ✅ |
| `pytest` = 1517/1517 pass | ✅ 1521/1521 |
| Public API contract unchanged | ✅ `test_openapi_stability.py` passes |
| All public modules have `__all__` declared | ✅ Most; few remaining (e.g. `diversity.py` exports are mostly private) |
| New types live in dedicated modules (not nested in `app.py` closures) | ✅ All Pydantic models in `search/models.py`; structured logging in `search/_logging.py` |

## Not addressed in this sweep (potential follow-ups)

- `search/app.py` is still ~2000 lines. Most of that is the inline `/api/photo/{id}` and `/photo/{id}/raw` routes that pre-date the §B2 router extraction. Splitting them out is a larger extraction than this sweep.
- The `indexer/` subsystem has its own legacy code that wasn't in scope. Independent refactor.
- `frontend/` was explicitly out of scope.
