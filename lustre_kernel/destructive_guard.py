"""
lustre_kernel.destructive_guard

Guards for destructive Qdrant operations.

The rule this module exists to enforce: **a delete filter must never
be able to match every point in a collection by accident.**

Qdrant's filter semantics make that easy to get wrong. A filter of
`{"must": []}` — an empty conjunction — matches ALL points, not none.
It looks like a no-op, reads like a no-op, and is the most natural
way to write "delete everything in this collection". On 2026-10-28
exactly that mistake, sent as a probe against the production sidecar,
deleted a 2,280,814-point collection in 0.6 seconds.

The guard is structural rather than advisory:

  - `is_match_all` recognises a filter that cannot discriminate.
  - `assert_not_match_all` refuses it.
  - `source_delete_filter` is the ONLY sanctioned way to build a
    filter for a point delete, and it names exactly one source.
  - `delete_source_points` is the only sanctioned way to delete
    points, and it refuses to run against an unscoped filter.

Filter semantics, measured against Qdrant 1.19 (see
`tests/test_destructive_guard.py` for the harness):

    filter                      points matched (of 10)
    --------------------------  ------------------------
    None                        10   (all)
    {}                          10   (all)
    {"must": []}                10   (all)   <- the 2026-10-28 shape
    {"must_not": []}            10   (all)
    {"must": [], "must_not":[]} 10   (all)
    {"should": []}               0   (none)
    {"must": [cond]}             5   (scoped)
    {"should": [cond]}           5   (scoped)
    {"must_not": [cond]}         5   (scoped)
    {"must": [], "should":[c]}   5   (scoped)

So an empty `should` matches NOTHING, while an empty `must` matches
EVERYTHING. That asymmetry is the trap: the two clauses look
interchangeable and are not.

A filter is considered SCOPED when it carries at least one condition
that can actually exclude points — a `match`/`range`/`values`/
`geo`/`full_text`/`is_*` predicate under a `key`. A `has_id` with an
empty id list selects nothing and so does not scope anything.

Everything here is pure: it builds and inspects filters. It never
talks to Qdrant. `delete_source_points` takes an
already-constructed client, so it can be exercised against an
in-memory Qdrant in tests without a network.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DestructiveFilterError",
    "assert_not_match_all",
    "assert_scoped",
    "build_filter",
    "delete_source_points",
    "is_match_all",
    "source_delete_filter",
]

# Qdrant filter clauses that select points. A condition carrying one
# of these can exclude points; a condition without one cannot.
_PREDICATE_CLAUSES = frozenset({
    "match", "range", "values", "geo", "full_text",
    "is_empty", "is_null", "has_id",
})


class DestructiveFilterError(ValueError):
    """Raised when a destructive filter is not safely scoped."""


def _has_predicate(condition: Any) -> bool:
    """True when `condition` carries a predicate that can exclude points.

    A condition is a dict with a `key` and at least one predicate
    clause. `has_id` counts only when it names at least one id — an
    empty id list selects nothing, so it scopes nothing.
    """
    if not isinstance(condition, dict):
        return False
    if "key" not in condition:
        return False
    for clause, value in condition.items():
        if clause == "key":
            continue
        if clause == "has_id":
            ids = (value or {}).get("has_id") if isinstance(value, dict) else value
            return bool(ids)
        if clause in _PREDICATE_CLAUSES:
            return True
    return False


def _clause_predicates(filter_: dict[str, Any], clause: str) -> list[Any]:
    value = filter_.get(clause)
    if not isinstance(value, list):
        return []
    return [c for c in value if _has_predicate(c)]


def is_match_all(filter_: Any) -> bool:
    """True when `filter_` would match every point in the collection.

    Mirrors Qdrant's own semantics: a filter with no usable positive
    selection AND no usable exclusion is the identity — it excludes
    nothing.

    Note the asymmetry: an empty `should` matches NOTHING (a
    disjunction over zero alternatives is false), so a filter carrying
    only an empty `should` is safe-ish — it deletes nothing. An empty
    `must` matches EVERYTHING, so it is not.
    """
    if filter_ is None:
        return True
    if not isinstance(filter_, dict):
        return True
    if not filter_:
        return True

    positive = _clause_predicates(filter_, "must") + _clause_predicates(filter_, "should")
    excluding = _clause_predicates(filter_, "must_not")

    if positive:
        return False
    if excluding:
        return False

    # No positive selection and no exclusion. The only way this can
    # still be safe is an explicitly empty `should` — a disjunction
    # over zero alternatives matches nothing.
    should = filter_.get("should")
    return not (isinstance(should, list) and not should)


def assert_not_match_all(filter_: Any, *, context: str = "delete") -> None:
    """Raise `DestructiveFilterError` if `filter_` would match everything.

    Args:
        filter_: The filter about to be sent to Qdrant.
        context: Short description used in the error message.
    """
    if is_match_all(filter_):
        raise DestructiveFilterError(
            f"refusing to {context} with an unscoped filter: {filter_!r}. "
            "An empty or predicate-free filter matches EVERY point in the "
            "collection — it does not match none. Use "
            "source_delete_filter(source_name) to scope a delete to a "
            "single source."
        )


def assert_scoped(filter_: Any, *, context: str = "delete") -> None:
    """Raise unless the filter positively selects a subset of points.

    Stricter than `assert_not_match_all`: the filter must carry at
    least one condition that can exclude points. A `must_not`-only
    filter is not enough — it excludes the named points but still
    selects everything else, which is the wrong way round for a
    scoped delete.
    """
    assert_not_match_all(filter_, context=context)
    if not isinstance(filter_, dict):
        raise DestructiveFilterError(
            f"refusing to {context} with a non-dict filter: {filter_!r}"
        )
    positive = _clause_predicates(filter_, "must") + _clause_predicates(filter_, "should")
    if not positive:
        raise DestructiveFilterError(
            f"refusing to {context} with a filter that selects nothing "
            f"specifically: {filter_!r}. It needs at least one condition "
            "with a match/range/values predicate that excludes points."
        )


def source_delete_filter(source_name: str) -> dict[str, Any]:
    """Build the filter that selects exactly one source's points.

    Args:
        source_name: The `collection` payload value — the source's
            logical name, e.g. "kpopping".

    Returns:
        A Qdrant filter dict matching only points whose payload
        `collection` field equals `source_name`.

    Raises:
        DestructiveFilterError: If `source_name` is empty or
            whitespace — an empty value would match points with no
            `collection` field set, which is not a source.
    """
    if not isinstance(source_name, str) or not source_name.strip():
        raise DestructiveFilterError(
            "source name must be a non-empty string — an empty name "
            "would match points with no collection field, which is not "
            "a source"
        )
    return {
        "must": [
            {"key": "collection", "match": {"value": source_name.strip()}},
        ]
    }


def build_filter(filter_: dict[str, Any]) -> Any:
    """Convert a filter dict into a qdrant-client `Filter` model.

    The Python client rejects plain dicts for `count_filter`,
    `scroll_filter` and `points_selector` — it wants model objects.
    The REST API accepts dicts. This helper is the bridge, so the
    guard can reason about plain dicts while the client gets what it
    needs.

    Raises:
        DestructiveFilterError: If the filter is not a dict.
    """
    if not isinstance(filter_, dict):
        raise DestructiveFilterError(
            f"filter must be a dict, got {type(filter_).__name__}"
        )
    from qdrant_client.http import models as qmodels

    kwargs: dict[str, Any] = {}
    for clause in ("must", "should", "must_not"):
        conds = filter_.get(clause)
        if not isinstance(conds, list):
            continue
        built = []
        for c in conds:
            if isinstance(c, dict) and c:
                built.append(qmodels.FieldCondition(**c))
            else:
                # A condition with no key is invalid to Qdrant anyway;
                # build the degenerate form so the server — not the
                # client — is the one that rejects it.
                built.append(qmodels.FieldCondition(key="", match=qmodels.MatchValue(value="")))
        kwargs[clause] = built
    return qmodels.Filter(**kwargs)


def delete_source_points(
    client: Any,
    collection: str,
    source_name: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delete every point belonging to one source, and nothing else.

    This is the ONLY sanctioned way to delete points from a collection
    by source. It refuses to run against a filter that is not scoped
    to exactly one source, and it reports what it matched before it
    deletes anything.

    Args:
        client: A QdrantClient (or anything with the same
            `count`/`delete` surface).
        collection: Collection name.
        source_name: The source whose points should go.
        dry_run: When True, count and report but delete nothing.

    Returns:
        A dict with the source name, the number of points matched,
        and whether a delete was performed.
    """
    filter_ = source_delete_filter(source_name)
    # Defence in depth: the filter was just built by the sanctioned
    # builder, but assert it anyway — this is the last checkpoint
    # before a destructive call.
    assert_scoped(filter_, context=f"delete source {source_name!r}")

    model = build_filter(filter_)
    count = client.count(
        collection_name=collection, count_filter=model, exact=True,
    ).count
    if dry_run:
        return {
            "source": source_name,
            "collection": collection,
            "matched": count,
            "deleted": 0,
            "dry_run": True,
        }

    if count:
        client.delete(
            collection_name=collection,
            points_selector=model,
            wait=True,
        )
    return {
        "source": source_name,
        "collection": collection,
        "matched": count,
        "deleted": count,
        "dry_run": False,
    }
