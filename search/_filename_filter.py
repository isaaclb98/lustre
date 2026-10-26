"""Resolve a `?filename=` pattern into a Qdrant filter condition.

The filename filter used to be resolved by asking the SQLite FTS5
index for the matching image ids and handing them to Qdrant as a
`HasId` filter. That meant every filtered search serialised the
whole matching id set (up to ~143k ids for a broad prefix pattern)
into the request, and Qdrant had to evaluate a huge id list per
query.

Qdrant's native text indexes let the matching happen server-side
instead, so the id set never leaves the database. The observable
behaviour is unchanged — the same patterns match the same photos —
but the per-request cost no longer scales with the number of
matches.

This module owns the decision of *which* native index answers a
given pattern, and the cardinality guard that decides whether the
filter is worth applying at all.
"""

from __future__ import annotations

import logging
from typing import Any

from search.filename_pattern import FilenamePattern, parse_filename_pattern
from search.path_filter import build_path_filter

logger = logging.getLogger(__name__)


def resolve_filename_pattern(pattern: str) -> tuple[FilenamePattern | None, str | None]:
    """Parse a raw `?filename=` pattern.

    Returns `(pattern, None)` for a usable pattern, `(None, None)`
    when the pattern carries no filter, and `(None, "bad_request")`
    when it is syntactically invalid.
    """
    try:
        parsed = parse_filename_pattern(pattern)
    except ValueError:
        return None, "bad_request"
    return parsed, None


async def resolve_filename_filter(
    qdrant: Any,
    *,
    cfg: Any,
    pattern: str,
) -> tuple[Any | None, str | None]:
    """Translate a raw `?filename=` pattern into a Qdrant filter.

    Returns one of three outcomes (always a 2-tuple):

      (None, None)            — no filter requested (the pattern was
                                empty), OR the cardinality guard
                                decided to skip the filter because
                                the pattern covers > 50% of the
                                corpus. Caller passes `None` to
                                qdrant.search meaning "no filter".
      (<condition>, None)     — a Qdrant filter condition to AND
                                into the query. The condition is
                                evaluated server-side against the
                                native text index.
      (None, "bad_request")   — the pattern was syntactically
                                invalid; the caller surfaces a 400.

    Zero-match patterns are NOT special-cased here. A pattern that
    matches nothing simply yields no hits from Qdrant, which is the
    same outcome the old id-list path produced (an empty id list
    AND-ed into the query). The caller no longer needs to
    distinguish "matched nothing" from "no filter" — both end in
    the same place, and the previous distinction existed only
    because an empty `HasIdCondition` is not the same as no filter
    at all.

    The cardinality guard is the load-bearing piece. When a pattern
    like `2024` covers 60% of the corpus, narrowing the search to it
    doesn't meaningfully improve relevance, and the unfiltered
    ranking is at least as good. The guard is symmetric: it applies
    to super-narrow patterns too, because filtering to a handful of
    points out of two million is a degenerate request either way.

    The count that drives the guard comes from Qdrant's own filtered
    count, so the guard and the filter always agree on what the
    corpus contains — no chance of the SQLite cache and the Qdrant
    collection disagreeing about coverage.
    """
    parsed, err = resolve_filename_pattern(pattern)
    if err == "bad_request":
        return None, "bad_request"
    if parsed is None:
        return None, None

    condition = build_path_filter(parsed)

    total = qdrant.count_points()
    if total > 0:
        matches = qdrant.count_filtered(condition)
        coverage = matches / total
        if coverage > cfg.filename_cardinality_guard:
            logger.info(
                "filename filter %r matched %d/%d (%.1f%%); "
                "skipping filter per cardinality guard",
                pattern, matches, total, coverage * 100,
            )
            return None, None

    return condition, None
