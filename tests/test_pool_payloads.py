"""The pool builders must not fetch payloads.

A candidate POOL is a source of ids and scores for a Gumbel draw.
The page that is actually returned is hydrated separately. Fetching
the payload of every one of 20k-50k candidates is pure waste — it
costs time and memory on both sides of the wire for data nobody
reads, and on the production corpus it is the difference between a
pool build that completes and one that times out.

These tests pin the shape so a future change cannot quietly put the
payloads back.
"""
from __future__ import annotations

import inspect
from typing import Any

import pytest


class _FakeQdrant:
    """Records the calls the pool builders make."""

    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.recommend_calls: list[dict[str, Any]] = []

    def search(self, *args: Any, **kwargs: Any) -> tuple[list[Any], bool]:
        self.search_calls.append(dict(kwargs))
        return [], False

    def recommend(self, **kwargs: Any) -> list[Any]:
        self.recommend_calls.append(dict(kwargs))
        return []


def test_qdrant_search_accepts_with_payload_false() -> None:
    from search.qdrant_client import QdrantSearch

    sig = inspect.signature(QdrantSearch.search)
    assert "with_payload" in sig.parameters
    assert sig.parameters["with_payload"].default is True


def test_qdrant_recommend_accepts_with_payload_false() -> None:
    from search.qdrant_client import QdrantSearch

    sig = inspect.signature(QdrantSearch.recommend)
    assert "with_payload" in sig.parameters
    assert sig.parameters["with_payload"].default is True


def test_surprise_pool_search_does_not_fetch_payloads() -> None:
    """The surprise deck's pool search must ask for ids + scores only."""
    import asyncio

    from search.surprise_deck import SurpriseDeckCache, build_surprise_deck

    qdrant = _FakeQdrant()
    asyncio.run(
        build_surprise_deck(
            qdrant=qdrant,
            vec=[0.0] * 8,
            collections=[],
            surprise_allowed_ids=None,
            surprise_pool=100,
            surprise_temperature=0.5,
            surprise_seed=1,
            surprise_session=None,
            surprise_cache_key="k",
            surprise_deck_cache=SurpriseDeckCache(),
            effective_limit=10,
            offset=0,
        )
    )
    assert qdrant.search_calls, "the pool search never ran"
    assert qdrant.search_calls[0].get("with_payload") is False


def test_for_you_pool_recommend_does_not_fetch_payloads() -> None:
    """The for-you deck's pool recommend must ask for ids + scores only."""
    import asyncio

    from search.for_you_deck import build_for_you_deck

    qdrant = _FakeQdrant()
    # The fake returns an empty pool, so the builder raises.
    with pytest.raises(RuntimeError):
        asyncio.run(
            build_for_you_deck(
                fav_ids=["a"],
                dis_ids=[],
                qdrant=qdrant,
                pool_size=100,
                deck_size=10,
                temperature=0.5,
                seed=1,
            )
        )
    assert qdrant.recommend_calls, "the pool recommend never ran"
    assert qdrant.recommend_calls[0].get("with_payload") is False


def test_the_deck_needs_only_ids_and_scores() -> None:
    """The pool hits are read for their id and their score — nothing else.

    If this ever fails, a pool builder started using the payload and
    the pool cost went back up.
    """
    from search import for_you_deck, surprise_deck

    for module, fn_name in (
        (surprise_deck, "build_surprise_deck"),
        (for_you_deck, "build_for_you_deck"),
    ):
        src = inspect.getsource(getattr(module, fn_name))
        # The pool hits are consumed as .id and .score.
        assert ".id" in src
        assert ".score" in src
        # ...and the payload is never read off a pool hit.
        assert ".payload" not in src
