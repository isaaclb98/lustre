"""
search/qdrant_client.py

Thin wrapper around the official qdrant-client. Hides search-knob
details and provides a typed interface.

This module never creates or modifies collections — that's the
indexer's job. The search side is read-only.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    id: str
    path: str
    score: float
    payload: dict[str, Any] | None = None


class QdrantSearch:
    """
    Read-only wrapper around QdrantClient.query_points.

    For tests, swap the client out via the `client` attribute or
    use a mock. The in-memory Qdrant (location=':memory:') works
    as a drop-in for local verification.
    """

    def __init__(
        self,
        client: Any,
        collection: str,
        timeout_ms: int = 60000,
        recommend_timeout_ms: int = 60000,
    ):
        self.client = client
        self.collection = collection
        self.timeout_ms = timeout_ms
        # Per-request timeout for the recommend() call. The default
        # 60s matches the query path — both run the same HNSW walk
        # but on different inputs (raw vector vs mean of
        # positive+negative ids). The 60s budget gives the larger
        # pool sizes (~50k in the Surprise Me slider) enough
        # headroom on the 2M-point production corpus.
        self.recommend_timeout_ms = recommend_timeout_ms

    def search(
        self, vector: list[float], limit: int, offset: int = 0,
        collections: list[str] | None = None,
        allowed_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        path_filter: Any | None = None,
    ) -> tuple[list[SearchHit], bool]:
        """
        Top-K search with optional offset and collection filter.

        `collections` is an optional whitelist of library names. When
        non-empty, only points whose payload `collection` field is in
        the list are returned. The filter is applied server-side via
        a `MatchAny` against the payload-indexed `collection` field
        (created by the indexer on first run), so it's O(log N) per
        query — no over-fetch, no Python post-filtering.

        `allowed_ids` is an optional whitelist of point ids, intended
        for the filename/path filter (see IndexDB.path_token_ids).
        When non-empty, only points whose id is in the list are
        returned. Like `collections`, this is a server-side
        `HasIdCondition` filter — O(log N) per query, no
        over-fetch. AND'd into the existing filter (a query that
        sets both `collections` and `allowed_ids` returns only
        points matching BOTH). Empty / None skips the filter.

        `exclude_ids` is an optional blacklist of point ids,
        applied via a `must_not` `HasIdCondition` so any hit whose
        id is in the list is dropped server-side. Used by the
        dynamic-centroid search route (Layer 1 of the
        near-duplicate exclusion) to remove exact-id matches to
        the seed set before the candidate set ever leaves Qdrant.
        Empty / None skips the filter. AND'd into the existing
        filter — `exclude_ids` and `allowed_ids` together
        naturally yield an empty result set when the seed ids
        are the only things that would match.

        `path_filter` is an optional pre-built Qdrant filter
        condition (typically a `FieldCondition` on the `path`
        payload field using the native text index). It is AND'd
        into the same `must` list as the other conditions, so a
        query that sets `path_filter` and `collections` returns
        only points matching BOTH. `None` skips it. This is the
        native replacement for materialising an id list from the
        SQLite FTS5 index and passing it as `allowed_ids` — the
        text index is evaluated server-side, so no id list is
        serialised per request.

        Pass None or an empty list to skip filtering (search all
        collections / all points, which is the default behavior).

        `has_more` is True when the result count equals the limit,
        which means there might be more results on a subsequent page.
        """
        from qdrant_client.http import models as qmodels

        must_conditions: list[Any] = []
        if collections:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="collection",
                    match=qmodels.MatchAny(any=collections),
                )
            )
        if allowed_ids:
            # HasIdCondition expects UUIDs or strings; we always
            # pass strings. A large `allowed_ids` list (50k+ points)
            # can hit a server-side limit on filter size — callers
            # should apply the cardinality guard before passing it
            # here (see app.py). Qdrant's default per-filter cap is
            # generous (4MB serialized) but a 1.5M-id list would
            # obviously overshoot it; we don't try to chunk.
            must_conditions.append(
                qmodels.HasIdCondition(has_id=cast("list[int|str|UUID]", allowed_ids))
            )
        if path_filter is not None:
            must_conditions.append(path_filter)
        must_not_conditions: list[Any] = []
        if exclude_ids:
            must_not_conditions.append(
                qmodels.HasIdCondition(has_id=cast("list[int|str|UUID]", exclude_ids))
            )
        query_filter = None
        if must_conditions or must_not_conditions:
            query_filter = qmodels.Filter(
                must=must_conditions or None,
                must_not=must_not_conditions or None,
            )

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            offset=offset,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
            timeout=self.timeout_ms // 1000,
        )

        hits: list[SearchHit] = [
            SearchHit(
                id=str(r.id),
                path=(r.payload or {}).get("path", ""),
                score=float(r.score) if r.score is not None else 0.0,
                payload=r.payload,
            )
            for r in response.points
        ]

        has_more = len(hits) >= limit
        return hits, has_more

    def recommend(
        self,
        positive: list[str],
        negative: list[str],
        limit: int = 20,
        collections: list[str] | None = None,
        allowed_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        with_vector: bool = False,
        score_threshold: float | None = None,
        offset: int = 0,
    ) -> list[SearchHit]:
        """
        Qdrant Recommend API: return the points nearest to
        `mean(positive_vecs) - mean(negative_vecs)`, restricted to
        the optional collection filter.

        `positive` and `negative` are lists of point IDs. Must be
        non-empty. Caller is responsible for sampling/deduping the
        results (e.g. /for-you draws 2 from the top-20 to
        keep the feed from converging to the same top-1 result).

        Score is the cosine similarity to the recomputed target
        vector — same scale and meaning as `search()`.

        Same collection-filter shape as `search()`: payload-keyword
        match against `collection`. Optional.

        `allowed_ids` mirrors the `search()` param: AND a `HasId`
        condition into the existing filter so the recommend path
        honours the filename filter too. Empty / None skips.

        Implementation note: qdrant-client 1.18 removed the
        direct `client.recommend(...)` method in favor of the
        universal `client.query_points(...)` endpoint, which
        accepts a `RecommendQuery` to drive the recommend logic.
        """
        from qdrant_client.http import models as qmodels

        must_conditions: list[Any] = []
        if collections:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="collection",
                    match=qmodels.MatchAny(any=collections),
                )
            )
        if allowed_ids:
            must_conditions.append(
                qmodels.HasIdCondition(has_id=cast("list[int|str|UUID]", allowed_ids))
            )
        must_not_conditions: list[Any] = []
        if exclude_ids:
            must_not_conditions.append(
                qmodels.HasIdCondition(has_id=cast("list[int|str|UUID]", exclude_ids))
            )
        if must_conditions or must_not_conditions:
            query_filter = qmodels.Filter(
                must=must_conditions or None,
                must_not=must_not_conditions or None,
            )
        else:
            query_filter = None

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=qmodels.RecommendQuery(
                    recommend=qmodels.RecommendInput(
                        positive=cast(Any, positive),
                        negative=cast(Any, negative),
                    ),
                ),
                limit=limit,
                offset=offset,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=with_vector,
                # Use the dedicated, more generous timeout — the
                # recommend path needs more headroom than the
                # default search path on the 2M-point corpus.
                timeout=self.recommend_timeout_ms // 1000,
            )
        except Exception as e:  # noqa: BLE001
            # Graceful fallback: if the recommend call times out
            # (or any other transient Qdrant error fires), return
            # an empty result set instead of propagating. /for-you's
            # fallback already handles `< 2 unseen candidates` by
            # using a random sample, so returning [] here means the
            # user gets a slightly-less-personalized next round. The
            # alternative — letting the exception bubble up to the
            # HTTP layer — turns a slow Qdrant into a 500, which
            # is strictly worse than a slightly-less-personalized
            # next round.
            logger.warning(
                "recommend failed (%d pos, %d neg, limit=%d): %s; "
                "returning empty result for graceful fallback",
                len(positive), len(negative), limit, e,
            )
            return []

        return [
            SearchHit(
                id=str(r.id),
                path=(r.payload or {}).get("path", ""),
                score=float(r.score) if r.score is not None else 0.0,
                payload=r.payload,
            )
            for r in response.points
        ]

    def retrieve_batch(self, point_ids: list[str]) -> list[SearchHit]:
        """
        Look up multiple points in a single call. Skips IDs that
        don't exist. Returns hits in the same order as `point_ids`
        (with missing IDs omitted, so call sites that need strict
        ordering should compare against the input).
        """
        if not point_ids:
            return []
        try:
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=point_ids,
                with_payload=True,
                with_vectors=False,
                timeout=self.timeout_ms // 1000,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("retrieve_batch(%d ids) failed: %s", len(point_ids), e)
            return []
        return [
            SearchHit(
                id=str(p.id),
                path=(p.payload or {}).get("path", ""),
                score=0.0,
                payload=p.payload,
            )
            for p in points
        ]

    def count_points(self) -> int:
        """
        Return the total number of points in the collection.

        Used by the Surprise Me route to cap the requested pool
        size at the actual corpus size — a slider set to "All"
        on a 2M-point corpus should sample from 2M, not error
        out because the requested pool exceeded the corpus.
        Returns 0 if the count call fails (the route then uses
        the requested pool as-is, which is the pre-fix
        behaviour).
        """
        try:
            result = self.client.count(
                collection_name=self.collection,
                exact=True,
                timeout=self.timeout_ms // 1000,
            )
            return int(result.count)
        except Exception as e:  # noqa: BLE001
            logger.debug("count_points failed: %s", e)
            return 0

    def count_filtered(self, path_filter: Any) -> int:
        """
        Count the points matching a filter condition.

        Used by the filename-filter cardinality guard, which needs
        to know what fraction of the corpus a pattern covers before
        deciding whether applying the filter is worth it. The count
        is exact — the approximate variant returns a wildly wrong
        number for a text filter (measured 390,953 vs the true
        83,098 for `chaewon`), which would make the guard fire on
        the wrong patterns.

        Returns 0 if the count call fails. The caller treats 0 as
        "no matches", which is the safe direction: the filter is
        skipped rather than applied to an unknown set.
        """
        from qdrant_client.http import models as qmodels

        try:
            result = self.client.count(
                collection_name=self.collection,
                count_filter=qmodels.Filter(must=[path_filter]),
                exact=True,
                timeout=self.timeout_ms // 1000,
            )
            return int(result.count)
        except Exception as e:  # noqa: BLE001
            logger.warning("count_filtered failed: %s", e)
            return 0

    def ensure_path_text_index(self) -> None:
        """
        Idempotently create the native text indexes the filename
        filter needs.

        Two indexes, because the two pattern shapes need different
        tokenizers:

        * `path` — `word` tokenizer with phrase matching. Serves
          exact patterns. Its settings reproduce FTS5's
          `unicode61 remove_diacritics 2` exactly: `word` splits
          on the same boundaries, `lowercase` folds case, and
          `ascii_folding` strips diacritics. Phrase matching is
          what makes a multi-token pattern (e.g. `img_2024`) match
          only adjacent tokens, as FTS5 does.

        * `path_prefix` — `prefix` tokenizer. Serves prefix
          patterns (`chaewon*`). The `prefix` tokenizer indexes
          every prefix of every token, so a bare term matches any
          token that starts with it — FTS5's `body*` behaviour,
          which a `word`-tokenized index cannot express.

        A collection supports one text index per field name, so the
        two need two fields. `path_prefix` holds a copy of `path`.

        Both indexes are created with `on_disk=True`. The prefix
        tokenizer indexes every prefix of every token, so holding it
        in RAM needs >16 GB on a 2.2M-point corpus — it OOM-kills
        the container against a 4 GB memory limit, and a half-built
        index makes every subsequent startup try to rebuild it, so
        Qdrant restart-loops and never comes up. `on_disk=True`
        mmaps the index instead: the same index builds in ~50s at
        1.3 GB RSS and boots in ~10s at the 4 GB limit.

        Both are created at startup so a fresh collection (or one
        restored from a snapshot taken before the indexes existed)
        can answer filtered searches. Safe to call repeatedly —
        Qdrant treats an existing index as a no-op.
        """
        from qdrant_client.http import models as qmodels

        self._ensure_text_index(
            "path",
            qmodels.TextIndexParams(
                type=qmodels.TextIndexType.TEXT,
                tokenizer=qmodels.TokenizerType.WORD,
                min_token_len=1,
                lowercase=True,
                ascii_folding=True,
                phrase_matching=True,
                on_disk=True,
            ),
        )
        self._ensure_text_index(
            "path_prefix",
            qmodels.TextIndexParams(
                type=qmodels.TextIndexType.TEXT,
                tokenizer=qmodels.TokenizerType.PREFIX,
                min_token_len=1,
                lowercase=True,
                ascii_folding=True,
                phrase_matching=False,
                on_disk=True,
            ),
        )

    def _ensure_text_index(self, field: str, params: Any) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=params,
            )
            logger.info("ensured text index on %s.%s", self.collection, field)
        except Exception as e:  # noqa: BLE001
            # Qdrant errors on a duplicate create; the index is
            # already there in that case, which is what we want.
            logger.debug(
                "text index ensure on %s.%s failed (likely already exists): %s",
                self.collection, field, e,
            )

    def search_with_vectors(
        self, vector: list[float], limit: int, offset: int = 0,
        collections: list[str] | None = None,
        allowed_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        path_filter: Any | None = None,
    ) -> tuple[list[tuple[SearchHit, list[float]]], bool]:
        """
        Like ``search()`` but returns ``(hit, vector)`` pairs so the
        caller can run MMR re-ranking without a second round-trip.

        The vector is the stored embedding (always a unit-norm
        ``list[float]``). ``has_more`` has the same semantics as
        ``search()`` — True when the result count equals the limit.

        ``exclude_ids`` mirrors ``search()``: server-side
        ``must_not`` `HasIdCondition` so any hit whose id is in the
        list is dropped before the result ever leaves Qdrant. Used
        by the dynamic-centroid search route's Layer 2 (over-fetch
        + numpy post-pass) so the over-fetch round-trip doesn't
        waste bandwidth shipping vectors we're going to drop on the
        python side anyway. Empty / None skips.

        ``path_filter`` mirrors ``search()``: an optional pre-built
        Qdrant filter condition (the native text-index condition
        for a ``?filename=`` pattern), AND'd into the same ``must``
        list. Empty / None skips.
        """
        from qdrant_client.http import models as qmodels

        must_conditions: list[Any] = []
        if collections:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="collection",
                    match=qmodels.MatchAny(any=collections),
                )
            )
        if allowed_ids:
            must_conditions.append(
                qmodels.HasIdCondition(has_id=cast("list[int|str|UUID]", allowed_ids))
            )
        if path_filter is not None:
            must_conditions.append(path_filter)
        must_not_conditions: list[Any] = []
        if exclude_ids:
            must_not_conditions.append(
                qmodels.HasIdCondition(has_id=cast("list[int|str|UUID]", exclude_ids))
            )
        query_filter = None
        if must_conditions or must_not_conditions:
            query_filter = qmodels.Filter(
                must=must_conditions or None,
                must_not=must_not_conditions or None,
            )

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            offset=offset,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,
            timeout=self.timeout_ms // 1000,
        )

        pairs: list[tuple[SearchHit, list[float]]] = []
        for r in response.points:
            vec = r.vector
            if vec is None:
                continue
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            pairs.append((
                SearchHit(
                    id=str(r.id),
                    path=(r.payload or {}).get("path", ""),
                    score=float(r.score) if r.score is not None else 0.0,
                    payload=r.payload,
                ),
                list(vec),
            ))

        has_more = len(pairs) >= limit
        return pairs, has_more

    def retrieve_batch_with_vectors(
        self, point_ids: list[str],
    ) -> list[tuple[str, list[float]]]:
        """
        Batch-retrieve `(id, vector)` pairs for the given ids.

        Used by the dynamic favourites-centroid compute: pull every
        favourited vector in one Qdrant round-trip, drop any ids the
        server doesn't know about (orphans — photo removed from
        Qdrant but favourite still in the cache). Returns the pairs
        in the order Qdrant returned them; callers that need
        ordering should compare against the input.
        """
        if not point_ids:
            return []
        try:
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=point_ids,
                with_payload=False,
                with_vectors=True,
                timeout=self.timeout_ms // 1000,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "retrieve_batch_with_vectors(%d ids) failed: %s",
                len(point_ids), e,
            )
            return []
        out: list[tuple[str, list[float]]] = []
        for p in points:
            vec = getattr(p, "vector", None)
            if vec is None:
                continue
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            out.append((str(p.id), list(vec)))
        return out

    def scroll_all(self, batch_size: int = 1000) -> Iterator[list[dict[str, Any]]]:
        """
        Paginate through every point in the collection.

        Yields batches of dictionaries with `id` and `payload`.
        Vectors are intentionally omitted; the search-side SQLite
        cache only needs rebuildable photo metadata.
        """
        scroll_offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=scroll_offset,
                with_payload=True,
                with_vectors=False,
                timeout=self.timeout_ms // 1000,
            )
            yield [
                {
                    "id": str(point.id),
                    "payload": point.payload or {},
                }
                for point in batch
            ]
            if next_offset is None:
                break
            scroll_offset = next_offset

    def scroll_ids_only(self, batch_size: int = 1000) -> Iterator[list[str]]:
        """
        Paginate through every point id in the collection.

        Yields batches of point id strings. Vectors and payloads
        are intentionally omitted — this is the cheapest way to
        enumerate ids for a diff against the SQLite cache (round-35).

        Note: there's no built-in Qdrant server-side filter that
        gives us 'only ids newer than X' without a payload index on
        a timestamp field, so we always scroll the full collection.
        The id-only scroll is ~10x cheaper than the payload scroll
        at our scale (2M points) because we skip payload deserialization.
        """
        scroll_offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=scroll_offset,
                with_payload=False,
                with_vectors=False,
                timeout=self.timeout_ms // 1000,
            )
            yield [str(point.id) for point in batch]
            if next_offset is None:
                break
            scroll_offset = next_offset

    def random_window(self, limit: int = 20) -> list[SearchHit]:
        """
        Return up to `limit` points sampled uniformly at random
        from the collection. Used as a cold-start sampler for
        the /random route and for /for-you when no positive/
        negative signal exists yet to drive a recommend query.

        Implementation:
          1. Paginate the collection to gather a list of all
             point ids (no payload/vector data — just ids).
          2. `random.sample()` `limit` ids from the list.
          3. Batch-retrieve the sampled ids with payload.

        The previous implementation passed a single random
        integer offset to scroll(). That had two problems:
          (a) in-memory mode ignored integer offsets, so the
              function always returned the same first-N points
              regardless of the random value;
          (b) in server mode, a single offset returned a
              contiguous window of `limit` points, which is a
              clump of consecutive points in insertion order —
              visibly not random when the collection is ordered
              by file path or upload time.

        The current implementation uniformly samples ids, then
        batch-retrieves them. Cost: one full id-pagination per
        call (O(N) network), but this is only invoked from
        cold-start paths, so the cost is amortized.
        """
        try:
            count_resp = self.client.get_collection(self.collection)
            count = count_resp.points_count
        except Exception as e:  # noqa: BLE001
            logger.warning("get_collection for random_window failed: %s", e)
            return []
        if not count:
            return []
        import random
        # Step 1: paginate to gather all point ids. No payload
        # or vector data — keeps the scroll response small.
        all_ids: list[str] = []
        scroll_offset = None
        while True:
            try:
                page, next_offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=1000,
                    offset=scroll_offset,
                    with_payload=False,
                    with_vectors=False,
                    timeout=self.timeout_ms // 1000,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("random_window id-scroll failed: %s", e)
                break
            all_ids.extend(str(p.id) for p in page)
            if next_offset is None or len(all_ids) >= count:
                break
            scroll_offset = next_offset
        if not all_ids:
            return []
        # Step 2: sample `limit` random ids uniformly.
        sampled_ids = random.sample(all_ids, min(limit, len(all_ids)))
        # Step 3: batch-retrieve the sampled ids (with payload).
        try:
            records = self.client.retrieve(
                collection_name=self.collection,
                ids=sampled_ids,
                with_payload=True,
                with_vectors=False,
                timeout=self.timeout_ms // 1000,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("random_window batch retrieve failed: %s", e)
            return []
        return [
            SearchHit(
                id=str(r.id),
                path=(r.payload or {}).get("path", ""),
                score=0.0,
                payload=r.payload,
            )
            for r in records
        ]

    def retrieve(self, point_id: str) -> SearchHit | None:
        """
        Look up a single point by id. Returns None if not found.
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
                timeout=self.timeout_ms // 1000,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("retrieve(%s) failed: %s", point_id, e)
            return None
        if not points:
            return None
        p = points[0]
        payload = p.payload or {}
        return SearchHit(
            id=str(p.id),
            path=payload.get("path", ""),
            score=0.0,
            payload=payload,
        )

    def retrieve_with_vector(
        self, point_id: str,
    ) -> tuple[list[float], SearchHit] | None:
        """
        Like `retrieve()` but also returns the stored embedding.

        Used by the photo "most similar" feature: fetch the source
        point's vector, then issue a second `query_points(query=<vec>)`
        against the same collection. Two round-trips total, no
        re-encoding on the search side.

        Returns (vector, hit) or None if the point is missing. Caller
        is responsible for surfacing a 404 when None comes back.
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=True,
                timeout=self.timeout_ms // 1000,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "retrieve_with_vector(%s) failed: %s", point_id, e
            )
            return None
        if not points:
            return None
        p = points[0]
        payload = p.payload or {}
        vec = p.vector
        # The qdrant-client returns a numpy array or a list depending
        # on version; normalize to a flat list[float].
        if vec is None:
            return None
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        return list(vec), SearchHit(
            id=str(p.id),
            path=payload.get("path", ""),
            score=0.0,
            payload=payload,
        )

    def list_collections_with_counts(self) -> list[dict[str, Any]]:
        """
        Return a list of distinct `collection` payload values in the
        index, with point counts.

        Implementation: a single `client.facet()` aggregation against
        the payload keyword index on `collection`. O(distinct values)
        on the server side, single round-trip, sub-50ms even at
        500k+ points. The chip-rendering JS calls this once per
        page load; not on the search hot path.

        Requires a payload keyword index on `collection` (the indexer
        creates one on every run via `upsert.ensure_payload_index`).
        Without the index Qdrant would have to scan everything anyway,
        so the speedup is real only when the index is present. Isaac
        confirmed via the qdrant dashboard that the index is in place
        on the live collection (269,887 points indexed as of
        2026-06-13).

        `exact=False` — counts are HNSW-approximated. Fine for a chip
        filter UI ("how many photos in this library?") and ~10x
        faster than exact. Test fixtures have ≤3 points so the
        approximation agrees with the truth there.

        Requires qdrant-client ≥ 1.10 (when `client.facet()` was
        added). Bumped in pyproject.toml.
        """
        response = self.client.facet(
            collection_name=self.collection,
            key="collection",
            limit=100,  # plenty for any sane library count
            timeout=self.timeout_ms // 1000,
        )
        items: list[dict[str, Any]] = []
        for h in response.hits:
            value = h.value
            # FacetValueHit.value is typed as union[str, int, bool]
            # but the indexer always writes a string. Be defensive
            # — coerce non-strings and skip empties.
            if value is None or value == "":
                continue
            items.append({"name": str(value), "count": h.count})
        # Stable sort by name so callers can rely on display order.
        items.sort(key=lambda x: x["name"])
        return items

    def healthz(self) -> bool:
        """Returns True if Qdrant is reachable. Logs on failure."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Qdrant healthz failed: %s", e)
            return False

    def ensure_payload_index(self, field: str, field_type: str = "keyword") -> None:
        """
        Idempotently create a payload index on `field` for the canonical
        collection if it doesn't already exist. Called at app startup
        so the collection always has the indexes the search API needs
        (e.g. `collection` keyword index for `list_collections_with_counts`
        via `client.facet()`).

        Option B (Sept 2026): there is no separate write collection
        anymore — the indexer writes here directly. The startup
        bootstrap is still necessary for fresh collections (no prior
        indexer run) so the search API doesn't 500 on the first
        `/api/collections` call.

        Safe to call repeatedly — Qdrant treats existing indexes as
        no-ops on duplicate create.
        """
        try:
            from qdrant_client.http import models as qmodels
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            logger.info("ensured payload index on %s.%s", self.collection, field)
        except Exception as e:  # noqa: BLE001
            # Qdrant returns an "already exists" error variant on
            # duplicate create; swallow ALL errors here so the app
            # still starts if Qdrant is being weird. The facet()
            # call below will surface a real index problem clearly.
            logger.debug("payload index ensure on %s.%s failed (likely already exists): %s",
                         self.collection, field, e)
