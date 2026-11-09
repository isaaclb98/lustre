"""
search/routers/explore.py — the Explore feedback loop.

GET /api/explore/{source}/round?session=&offset=&limit=:
    Serve a page of the current round. The round is materialised with
    one Qdrant `recommend()` call whose POSITIVES are the source's
    member photos and whose NEGATIVES are the photos the user
    down-voted so far. The client pages through the materialised
    round via `offset`; the server slices the cached id list rather
    than re-querying Qdrant, because offset-paging a recommend query
    returns duplicate rows at the page boundaries.

POST /api/explore/{source}/vote:
    Record a vote. `down` adds the photo to the session's negative
    set and invalidates the current round, so the next round is
    re-queried with the new negative. `up` adds the photo to the
    source itself, which makes it a positive for every later round.

`source` is a source key, not an album id: `album:{id}` for a user
album, `favourites` for the built-in one. The built-in album has no
row in the `albums` table — its members are the rows in the
`favorites` table — so it needs its own key and its own up-vote
behaviour. The vocabulary is the same one the centroid registry uses,
so `album:{id}` means the same thing in both places.

Up-votes mean different things per source. On a user album, an
up-vote adds the photo to the album's membership. On the built-in
Favourites album, an up-vote marks the photo as a favourite — the
built-in album's membership IS the favourites table, so "add it to
the album" and "favourite it" are the same act. Both then invalidate
the relevant centroid, because the source's own direction moved.

Why up-votes join the album: the album is what the user is curating,
so an up-vote is a curation act. It also compounds — each up-vote
moves the album's direction toward the user's taste, and the album
is the positive set for the next session's rounds. Measured on a
12-photo album, twelve up-votes from one artist moved the album
direction 13.2°; across five sessions of five up-votes each it
moved 4.95° → 10.79° cumulatively.

Why down-votes are temporary: they never reach the `dislikes` table.
That table feeds the permanent For You taste signal, and "not this
one right now" is not the same statement as "never show me this".
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from qdrant_client import models

from search._result_helpers import bad_request
from search.explore import (
    EXPLORE_ROUND_SIZE,
    FAVOURITES_SOURCE,
    ExploreSession,
    ExploreSessionStore,
)
from search.image_resolver import resolve_url
from search.models import SearchResponse, SearchResult

logger = logging.getLogger(__name__)


# The HNSW breadth pinned for the round query. The default breadth
# leaves a long cold tail at this limit — measured over distinct
# queries on a 2.8M-point collection, limit=240 costs p50 42ms /
# p90 63ms / max 83ms with 0/24 runs over 200ms, while limit=1000
# under the default breadth runs p50 292ms / p90 655ms / max 1117ms.
# 256 pulls the large limit back to p50 68ms for 11/1000 result-set
# differences. Pinned here so the round's latency is bounded
# regardless of which limit is configured.
HNSW_EF = 256


def _results_from_hits(
    hits: list[Any],
    *,
    web_ui_url: str,
    favorite_ids: set[str] | None = None,
    dislike_ids: set[str] | None = None,
) -> list[SearchResult]:
    """Build SearchResult objects from raw Qdrant hits.

    Mirrors the shape used by the other routers: the score is the
    cosine similarity to the effective recommend query, and the
    blurhash/dimensions come from the payload so the grid can render
    a placeholder before the thumbnail loads.
    """
    return [
        SearchResult(
            id=h.id,
            path=h.path,
            score=h.score,
            score_str=f"{h.score:.3f}",
            url=resolve_url(h.id, web_ui_url),
            is_favorite=h.id in (favorite_ids or set()),
            is_disliked=h.id in (dislike_ids or set()),
            blurhash=(h.payload or {}).get("blurhash") or None,
            width=(h.payload or {}).get("width"),
            height=(h.payload or {}).get("height"),
        )
        for h in hits
    ]


# The prefix on a user album's source key. `album:5` is album 5.
_ALBUM_SOURCE_PREFIX = "album:"


def parse_source(source: str) -> tuple[str, int | None]:
    """Split a source key into (kind, album_id).

    `kind` is "album" for a user album, "favourites" for the built-in
    one, or "unknown" for anything else. The album id is None for the
    built-in album, which has no row in the `albums` table.

    A malformed key is not an error here — it resolves to "unknown"
    and the route answers 404. That keeps the parsing total: every
    caller gets a value back and never has to handle an exception.
    """
    if source == FAVOURITES_SOURCE:
        return "favourites", None
    if source.startswith(_ALBUM_SOURCE_PREFIX):
        raw = source[len(_ALBUM_SOURCE_PREFIX):]
        if raw.isdigit():
            return "album", int(raw)
    return "unknown", None


def build_explore_router(
    *,
    index_db: Any,
    qdrant: Any,
    cfg: Any,
    invalidate_album_centroid: Any = None,
    invalidate_favourites_centroid: Any = None,
) -> APIRouter:
    """Build the Explore router with the live dependencies."""
    router = APIRouter()
    sessions = ExploreSessionStore()
    _web_ui_url = cfg.web_ui_url

    def _member_ids(kind: str, album_id: int | None) -> list[str]:
        """The photos that seed the round for this source."""
        if kind == "favourites":
            return list(index_db.list_favorite_ids())
        return list(index_db.list_album_member_ids(album_id))

    def _add_member(
        kind: str, album_id: int | None, point_id: str,
    ) -> bool:
        """Up-vote: make the photo a permanent member of the source.

        On a user album that is an album membership. On the built-in
        Favourites album it is a favourite — the same act, expressed
        against the table that backs that album.
        """
        if kind == "favourites":
            try:
                index_db.mark_favorite(point_id)
            except Exception:  # noqa: BLE001 — the photo left the cache
                return False
            if invalidate_favourites_centroid is not None:
                invalidate_favourites_centroid()
            return True
        return bool(index_db.add_album_member(album_id, point_id))

    def _liveness_filter(ids: list[str]) -> list[str]:
        """Drop ids Qdrant no longer knows about.

        `recommend()` is all-or-nothing on its seed ids: one missing
        id fails the whole call with a 404 rather than being skipped.
        Album membership outlives the photos (a membership row is
        kept so re-indexing re-attaches the photo), so this filter is
        what keeps one dead member from breaking the round.

        `retrieve_batch` skips unknown ids, and asking for them
        without their vectors keeps the cost at a few milliseconds
        even for a 3,000-member album.
        """
        if not ids:
            return []
        found = qdrant.retrieve_batch(ids)
        live = {h.id for h in found}
        return [i for i in ids if i in live]

    def _materialise_round_sync(sess: ExploreSession) -> list[Any]:
        """Query Qdrant for this session's next round. Blocking.

        Positives are the album's members; negatives are the session's
        down-votes. Both are liveness-filtered first — see
        `_liveness_filter`. The seen-set and the album's own members
        are excluded so a photo is never served twice and the album's
        contents do not fill the results.

        Returns the hits with payloads, in recommend order.

        The whole thing is synchronous and runs in one worker thread
        (see the `asyncio.to_thread` call in `explore_round`) — the
        SQLite read and the three Qdrant calls are all blocking, so
        one thread hop keeps them off the event loop without paying
        for a hop per call.
        """
        kind, album_id = parse_source(sess.source)
        member_ids = _member_ids(kind, album_id)
        if not member_ids:
            return []
        live_members = _liveness_filter(list(member_ids))
        if not live_members:
            return []
        live_negatives = _liveness_filter(list(sess.downvotes))
        exclude = sorted(sess.seen | set(live_members))
        return qdrant.recommend(
            positive=live_members,
            negative=live_negatives,
            limit=EXPLORE_ROUND_SIZE,
            exclude_ids=exclude,
            with_payload=True,
            search_params=_hnsw_params(),
        )

    def _hnsw_params() -> Any:
        """Build the pinned HNSW search params for the round query."""
        return models.SearchParams(hnsw_ef=HNSW_EF)

    def _hydrate_page(
        page_ids: list[str],
        scores: dict[str, float],
    ) -> tuple[list[Any], set[str], set[str]]:
        """Fetch one page's payloads and its like/dislike state.

        Blocking; runs in a worker thread. The hits come back in the
        round's order (the lookup is a dict rebuild), and the two
        SQLite lookups give the tile its persistent feedback badges.

        `retrieve_batch` carries no score — every hit it returns has
        a score of 0 — so the score the round earned is copied back
        onto the hit from `scores`. Without that, every tile in the
        Explore grid would display a similarity of 0.000.
        """
        hits = qdrant.retrieve_batch(page_ids)
        by_id = {h.id: h for h in hits}
        ordered = [by_id[i] for i in page_ids if i in by_id]
        for hit in ordered:
            hit.score = scores.get(hit.id, hit.score)
        fav_set = index_db.favorite_id_set(page_ids)
        dis_set = index_db.dislike_id_set(page_ids)
        return ordered, fav_set, dis_set

    @router.get("/api/explore/{source}/round", response_model=SearchResponse)
    async def explore_round(
        source: str,
        session: Annotated[
            str | None,
            Query(
                description=(
                    "Session id from a previous round response. Pass it "
                    "back to keep the same down-vote set. A new session "
                    "(or no session) starts a fresh round."
                ),
            ),
        ] = None,
        offset: Annotated[
            int,
            Query(ge=0, description="position within the current round"),
        ] = 0,
        limit: Annotated[
            int,
            Query(ge=1, le=EXPLORE_ROUND_SIZE, description="photos per page"),
        ] = 24,
    ) -> SearchResponse | JSONResponse:
        """Serve a page of the current Explore round for a source."""
        started = time.monotonic()

        kind, _album_id = parse_source(source)
        if kind == "unknown":
            return bad_request("unknown explore source")

        existing = sessions.get(session)
        if existing is None:
            session_id, sess = sessions.put_new(source)
        else:
            session_id, sess = existing
            if sess.source != source:
                return bad_request(
                    "session belongs to a different album; start a new "
                    "session to explore this one"
                )

        # Materialise the round if this session does not have one.
        # A round is invalidated by a down-vote (the negatives change),
        # so the vote handler clears `round_ids`.
        if not sess.round_ids:
            try:
                hits = await asyncio.to_thread(
                    _materialise_round_sync, sess
                )
            except Exception:
                logger.exception("explore: round materialisation failed")
                return bad_request("failed to build the explore round")
            if not hits:
                return bad_request(
                    "no photos to explore for this album — it may have "
                    "no members left in the index"
                )
            # The scores come from this recommend() call. They are the
            # only source of the similarity the tile displays — the
            # page's payload fetch carries none.
            sess.set_round(
                [h.id for h in hits],
                {h.id: float(h.score or 0.0) for h in hits},
            )

        # Slice the materialised round. The client pages through this
        # list; the server never re-queries Qdrant with an offset,
        # because offset-paging a recommend query returns duplicate
        # rows at the page boundaries.
        start = min(offset, len(sess.round_ids))
        end = min(start + limit, len(sess.round_ids))
        page_ids = sess.round_ids[start:end]
        # Only the ids actually SERVED are marked seen. Marking the
        # whole round would discard the unserved remainder every time
        # a vote invalidates the round, silently throwing away
        # candidates the user never actually looked at.
        sess.mark_seen(page_ids)

        results: list[SearchResult] = []
        if page_ids:
            # One thread hop for the page's payload fetch and the two
            # SQLite lookups — all three are blocking, and they are
            # cheap enough that one hop beats three.
            ordered, fav_set, dis_set = await asyncio.to_thread(
                _hydrate_page, page_ids, sess.round_scores
            )
            results = _results_from_hits(
                ordered,
                web_ui_url=_web_ui_url,
                favorite_ids=fav_set,
                dislike_ids=dis_set,
            )

        return SearchResponse(
            query="",
            positives=[],
            negatives=list(sess.downvotes),
            diverse=False,
            view="grid",
            centroid=None,
            centroids=[],
            weights=None,
            results=results,
            took_ms=int((time.monotonic() - started) * 1000),
            offset=start,
            limit=limit,
            has_more=end < len(sess.round_ids),
            session_id=session_id,
            session_total=len(sess.round_ids),
        )

    @router.post("/api/explore/{source}/vote", response_model=None)
    async def explore_vote(
        source: str,
        body: dict[str, Any],
        session: Annotated[
            str | None,
            Query(description="session id from a previous round response"),
        ] = None,
    ) -> dict[str, Any] | JSONResponse:
        """Record an up-vote or a down-vote for the current session.

        Body: `{"photo_id": "...", "vote": "up" | "down"}`.

        A down-vote is session-scoped: it joins the negative set and
        invalidates the round, so the next request re-queries Qdrant
        with the new negative. An up-vote is permanent: the photo
        joins the album, making it a positive for every later round
        and every later session.
        """
        kind, album_id = parse_source(source)
        if kind == "unknown":
            return bad_request("unknown explore source")

        existing = sessions.get(session)
        if existing is None:
            session_id, sess = sessions.put_new(source)
        else:
            session_id, sess = existing
            if sess.source != source:
                return bad_request(
                    "session belongs to a different album; start a new "
                    "session to explore this one"
                )

        photo_id = str(body.get("photo_id") or "").strip()
        vote = str(body.get("vote") or "").strip().lower()
        if not photo_id:
            return bad_request("photo_id is required")
        if vote not in ("up", "down"):
            return bad_request("vote must be 'up' or 'down'")

        if vote == "down":
            added = sess.add_downvote(photo_id)
            # The negatives changed, so the round is stale. Clearing
            # it makes the next request re-query Qdrant.
            sess.clear_round()
            return {
                "session_id": session_id,
                "photo_id": photo_id,
                "vote": vote,
                "added": added,
                "downvotes": len(sess.downvotes),
            }

        # Up-vote: join the source. The membership write is what
        # makes this photo a positive for later rounds.
        ok = await asyncio.to_thread(
            _add_member, kind, album_id, photo_id
        )
        if not ok:
            return bad_request("album not found")
        # The source's membership changed, so its cached centroid is
        # stale. The Explore round reads member ids directly and does
        # not need the centroid, but the source's own Search does.
        #
        # This is an in-memory registry drop, not a blocking call —
        # no thread hop needed. `_add_member` already invalidates the
        # favourites centroid on the built-in album's path, since a
        # favourite IS that album's membership.
        if kind == "album" and invalidate_album_centroid is not None:
            invalidate_album_centroid(album_id)
        # The album grew, so the current round's positives changed
        # too. Rebuild it on the next request.
        sess.clear_round()
        return {
            "session_id": session_id,
            "photo_id": photo_id,
            "vote": vote,
            "added": True,
            "downvotes": len(sess.downvotes),
        }

    # Expose the session store for tests via the router object.
    # `router._explore_sessions` is an implementation detail.
    # setattr (vs. router._x = store) because APIRouter has no
    # such attribute and we don't want to monkey-patch the
    # starlette class. B010 is "setattr with constant"; this
    # is the legitimate form for adding attrs at runtime.
    setattr(router, "_explore_sessions", sessions)  # noqa: B010

    return router
