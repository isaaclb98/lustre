"""
Tests for the orphan lifecycle.

The lifecycle has three parts, and this file covers all three:

1. **Marking.** The indexer marks points whose file is gone from
   disk, and points belonging to a retired source. The mark is the
   `orphan_since` payload field.

2. **Exclusion.** A marked point is excluded from every read path —
   search, the random deck, the collection chips, the collection
   counts. It stops being served immediately, but it is NOT deleted.

3. **The confirmed purge.** An operator reviews the count, confirms,
   and only then are the embeddings deleted and the matching
   SQLite rows removed.

The tests use an in-memory Qdrant, so they are fast and touch no
production state.
"""
from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from lustre_kernel.orphan_marking import (
    FIELD_ORPHAN_SINCE,
    is_orphan_payload,
    mark_orphans,
    orphan_filter,
    orphan_since_filter,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qdrant_client() -> QdrantClient:
    """An in-memory collection with a few points across two sources."""
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="images",
        vectors_config={"size": 2, "distance": "Cosine"},
    )
    points = []
    for src, count in (("telegram", 3), ("data", 2)):
        for i in range(count):
            points.append(qmodels.PointStruct(
                id=uuid.UUID(int=len(points)),
                vector=[1.0, 0.0],
                payload={
                    "collection": src,
                    "path": f"/nas/{src}/{i}.jpg",
                    "mtime": 1000,
                    "size": 500,
                },
            ))
    client.upsert(collection_name="images", points=points, wait=True)
    return client


def _payloads(client: QdrantClient) -> list[dict]:
    return [
        p.payload or {}
        for p in client.scroll(
            collection_name="images", limit=100, with_payload=True,
        )[0]
    ]


# ---------------------------------------------------------------------------
# marking
# ---------------------------------------------------------------------------


def test_marking_sets_the_marker_and_keeps_the_payload(qdrant_client) -> None:
    """A marked point keeps its path and collection.

    `set_payload` merges rather than replaces, so the marker is
    ADDED to the payload. This matters: the review endpoint shows
    the operator which photos are about to go, and the purge needs
    the path to clean up the SQLite side.
    """
    ids = [str(p.id) for p in qdrant_client.scroll(
        collection_name="images", limit=100, with_payload=False,
    )[0]]

    marked = mark_orphans(qdrant_client, "images", ids[:2])

    assert marked == 2
    marked_payloads = [p for p in _payloads(qdrant_client)
                       if is_orphan_payload(p)]
    assert len(marked_payloads) == 2
    for p in marked_payloads:
        assert p["path"].startswith("/nas/")
        assert p["collection"] in ("telegram", "data")
        assert isinstance(p[FIELD_ORPHAN_SINCE], float)


def test_marking_is_idempotent(qdrant_client) -> None:
    """Marking an already-marked point keeps the ORIGINAL timestamp.

    "How long has this been an orphan?" has to stay meaningful
    across repeated index runs, so a second mark must not reset the
    clock.
    """
    ids = [str(p.id) for p in qdrant_client.scroll(
        collection_name="images", limit=100, with_payload=False,
    )[0]]

    assert mark_orphans(qdrant_client, "images", ids[:1], when=1000.0) == 1
    # A second mark is a no-op: the point is already marked.
    assert mark_orphans(qdrant_client, "images", ids[:1], when=2000.0) == 0

    stamps = [
        p[FIELD_ORPHAN_SINCE] for p in _payloads(qdrant_client)
        if is_orphan_payload(p)
    ]
    assert stamps == [1000.0]


def test_unmarked_points_have_no_marker(qdrant_client) -> None:
    """A point that was never marked does not carry the field."""
    assert not any(
        is_orphan_payload(p) for p in _payloads(qdrant_client)
    )


# ---------------------------------------------------------------------------
# exclusion from read paths
# ---------------------------------------------------------------------------


def test_the_orphan_filter_selects_only_marked_points(qdrant_client) -> None:
    """The positive filter matches exactly the marked points."""
    ids = [str(p.id) for p in qdrant_client.scroll(
        collection_name="images", limit=100, with_payload=False,
    )[0]]
    mark_orphans(qdrant_client, "images", ids[:2])

    matched = qdrant_client.scroll(
        collection_name="images",
        limit=100,
        scroll_filter=orphan_filter(),
        with_payload=True,
    )[0]
    assert len(matched) == 2


def test_the_exclusion_filter_hides_only_marked_points(qdrant_client) -> None:
    """The negative filter hides the marked points and keeps the rest."""
    ids = [str(p.id) for p in qdrant_client.scroll(
        collection_name="images", limit=100, with_payload=False,
    )[0]]
    mark_orphans(qdrant_client, "images", ids[:2])

    matched = qdrant_client.scroll(
        collection_name="images",
        limit=100,
        scroll_filter=orphan_since_filter(),
        with_payload=True,
    )[0]
    assert len(matched) == 3
    for p in matched:
        assert not is_orphan_payload(p.payload or {})


def test_a_cleared_marker_makes_the_point_visible_again(qdrant_client) -> None:
    """A point whose marker is cleared to null is served again.

    This is the "the file came back" path. The marker is set to
    null rather than removed, and null does not match the orphan
    range — so the point reappears without needing the payload key
    to be deleted.
    """
    ids = [str(p.id) for p in qdrant_client.scroll(
        collection_name="images", limit=100, with_payload=False,
    )[0]]
    mark_orphans(qdrant_client, "images", ids[:1])

    qdrant_client.set_payload(
        collection_name="images",
        payload={FIELD_ORPHAN_SINCE: None},
        points=ids[:1],
    )

    matched = qdrant_client.scroll(
        collection_name="images",
        limit=100,
        scroll_filter=orphan_since_filter(),
        with_payload=True,
    )[0]
    assert len(matched) == 5


# ---------------------------------------------------------------------------
# the confirmed purge
# ---------------------------------------------------------------------------


def test_the_confirmed_purge_deletes_only_marked_points(qdrant_client) -> None:
    """The purge removes the marked points and leaves the rest."""
    from search.qdrant_client import QdrantSearch

    ids = [str(p.id) for p in qdrant_client.scroll(
        collection_name="images", limit=100, with_payload=False,
    )[0]]
    mark_orphans(qdrant_client, "images", ids[:2])

    search = QdrantSearch(client=qdrant_client, collection="images")
    result = search.delete_orphans()

    assert result["deleted"] == 2
    assert len(result["ids"]) == 2
    remaining = qdrant_client.scroll(
        collection_name="images", limit=100, with_payload=True,
    )[0]
    assert len(remaining) == 3
    for p in remaining:
        assert not is_orphan_payload(p.payload or {})


def test_the_purge_can_be_scoped_to_one_source(qdrant_client) -> None:
    """Purging one source's orphans leaves the other source alone."""
    from search.qdrant_client import QdrantSearch

    ids = [str(p.id) for p in qdrant_client.scroll(
        collection_name="images", limit=100, with_payload=False,
    )[0]]
    mark_orphans(qdrant_client, "images", ids)  # mark everything

    search = QdrantSearch(client=qdrant_client, collection="images")
    result = search.delete_orphans(source_name="telegram")

    assert result["deleted"] == 3
    remaining = qdrant_client.scroll(
        collection_name="images", limit=100, with_payload=True,
    )[0]
    assert len(remaining) == 2
    assert {p.payload["collection"] for p in remaining} == {"data"}


def test_the_purge_respects_its_limit(qdrant_client) -> None:
    """A limited purge deletes only that many."""
    from search.qdrant_client import QdrantSearch

    ids = [str(p.id) for p in qdrant_client.scroll(
        collection_name="images", limit=100, with_payload=False,
    )[0]]
    mark_orphans(qdrant_client, "images", ids)

    search = QdrantSearch(client=qdrant_client, collection="images")
    result = search.delete_orphans(limit=2)

    assert result["deleted"] == 2
    assert qdrant_client.count(
        collection_name="images", exact=True,
    ).count == 3


def test_purging_nothing_is_a_no_op(qdrant_client) -> None:
    """With no orphans marked, the purge deletes nothing."""
    from search.qdrant_client import QdrantSearch

    search = QdrantSearch(client=qdrant_client, collection="images")
    result = search.delete_orphans()

    assert result["deleted"] == 0
    assert qdrant_client.count(
        collection_name="images", exact=True,
    ).count == 5


# ---------------------------------------------------------------------------
# the retired-source case
# ---------------------------------------------------------------------------


def test_a_retired_source_is_marked_even_though_its_files_exist(
    qdrant_client,
) -> None:
    """The retired-source case: files on disk, source not configured.

    Nothing in the filesystem distinguishes a retired source from a
    live one — the files are all still there. The configured source
    list is the only signal, so the detection compares the
    collection's `collection` facet against the managed names.
    """
    from indexer.orphan_scan import find_retired_sources

    # `telegram` is managed, `data` is not — it was removed from
    # the source list.
    retired = find_retired_sources(
        qdrant_client, "images", managed_source_names=["telegram"],
    )

    assert retired == {"data": 2}


def test_no_source_is_retired_when_all_are_managed(qdrant_client) -> None:
    """A complete managed list reports nothing retired."""
    from indexer.orphan_scan import find_retired_sources

    retired = find_retired_sources(
        qdrant_client, "images",
        managed_source_names=["telegram", "data"],
    )
    assert retired == {}


def test_retired_source_points_are_marked_and_hidden(qdrant_client) -> None:
    """A retired source's points are marked, then excluded from reads."""
    from indexer.orphan_scan import find_retired_sources, find_source_orphans
    from lustre_kernel.orphan_marking import mark_orphans

    retired = find_retired_sources(
        qdrant_client, "images", managed_source_names=["telegram"],
    )
    assert retired == {"data": 2}

    src_ids = find_source_orphans(qdrant_client, "images", "data")
    assert len(src_ids) == 2

    marked = mark_orphans(qdrant_client, "images", src_ids)
    assert marked == 2

    # The marked points are hidden; the managed source is not.
    visible = qdrant_client.scroll(
        collection_name="images",
        limit=100,
        scroll_filter=orphan_since_filter(),
        with_payload=True,
    )[0]
    assert len(visible) == 3
    assert {p.payload["collection"] for p in visible} == {"telegram"}


def test_a_retired_source_drops_off_the_collection_chips(
    qdrant_client,
) -> None:
    """A fully retired source disappears from the collection chips.

    The chip list comes from a facet on the `collection` field. The
    facet is filtered to exclude marked points, so a source whose
    every point is marked reports zero and drops off.
    """
    from indexer.orphan_scan import find_source_orphans
    from lustre_kernel.orphan_marking import mark_orphans

    src_ids = find_source_orphans(qdrant_client, "images", "data")
    mark_orphans(qdrant_client, "images", src_ids)

    from qdrant_client.http import models as qmodels

    facet = qdrant_client.facet(
        collection_name="images",
        key="collection",
        limit=100,
        facet_filter=qmodels.Filter(must_not=[qmodels.FieldCondition(
            key=FIELD_ORPHAN_SINCE, range=qmodels.Range(gte=0),
        )]),
    )
    assert {h.value for h in facet.hits} == {"telegram"}
