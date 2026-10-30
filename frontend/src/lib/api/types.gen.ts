export interface paths {
    "/albums/{album_id}/download.zip": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Album Download Zip
         * @description Stream an album's photos as a zip — same shape and rules
         *     as `/favorites/download.zip`. Membership comes from
         *     `index_db.list_album_members` (INNER JOIN against the cache,
         *     so orphan rows are hidden from the archive too). Files we
         *     can't resolve on disk are skipped and recorded in
         *     `_missing.txt`. Album id with no row → 404.
         */
        get: operations["album_download_zip_albums__album_id__download_zip_head_1"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        /**
         * Album Download Zip
         * @description Stream an album's photos as a zip — same shape and rules
         *     as `/favorites/download.zip`. Membership comes from
         *     `index_db.list_album_members` (INNER JOIN against the cache,
         *     so orphan rows are hidden from the archive too). Files we
         *     can't resolve on disk are skipped and recorded in
         *     `_missing.txt`. Album id with no row → 404.
         */
        head: operations["album_download_zip_albums__album_id__download_zip_head"];
        patch?: never;
        trace?: never;
    };
    "/api/admin/index": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Start Index
         * @description Start a new indexer job. Returns 202 with the initial status.
         *
         *     409 if a job is already running. The runner rejects
         *     concurrent starts — the UI must wait for IDLE before
         *     triggering another.
         */
        post: operations["start_index_api_admin_index_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/admin/index/cancel": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Cancel Index
         * @description Send SIGTERM to the running job.
         *
         *     200 with current status (now IDLE or FAILED). 400 if no
         *     job was running.
         */
        post: operations["cancel_index_api_admin_index_cancel_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/admin/index/log": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Log
         * @description Tail the indexer's stdout/stderr. `next_line` is the cursor
         *     for the next poll; pass it back as `since_line` to get
         *     only newer lines.
         */
        get: operations["get_log_api_admin_index_log_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/admin/index/status": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Status
         * @description Snapshot of the current job. The UI polls this every 1s
         *     while the job is RUNNING.
         */
        get: operations["get_status_api_admin_index_status_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/admin/orphans": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Orphans
         * @description How many orphans are marked, and which photos are they.
         *
         *     The review step of the two-step purge. The operator looks at
         *     the count and the sample here, and only then confirms.
         */
        get: operations["list_orphans_api_admin_orphans_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/admin/orphans/purge": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Purge Orphans
         * @description Delete the confirmed orphans from Qdrant and SQLite.
         *
         *     This is the irreversible half. It requires `confirm: true` in
         *     the request body — there is no way to reach the delete without
         *     saying so explicitly.
         *
         *     The Qdrant delete runs first, and it is scoped by the orphan
         *     marker: only points carrying `orphan_since` are in scope, so
         *     a purge can never reach a live photo.
         *
         *     The SQLite cleanup runs after, in two parts:
         *
         *     - the ids that were just deleted from Qdrant (their
         *       favourites, dislikes and album entries go with them);
         *     - the rows whose photo is gone from DISK, whether or not its
         *       Qdrant point was ever marked. The indexer may not have run
         *       since the file was deleted, so its point is still unmarked
         *       and the first part would miss it.
         *
         *     The second part is the file-based rule — liveness decided by
         *     the filesystem, never by "is it in Qdrant". A photo that is
         *     on disk but not currently indexed keeps its favourite, and
         *     re-attaches when the photo comes back.
         */
        post: operations["purge_orphans_api_admin_orphans_purge_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/albums": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Albums */
        get: operations["list_albums_api_albums_get"];
        put?: never;
        /** Create Album */
        post: operations["create_album_api_albums_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/albums/by-favorite/{favorite_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Albums For Favorite
         * @description Return every album that contains `favorite_id`.
         *
         *     Used by the per-photo UI to show which albums a photo is
         *     in. The summary shape omits member_count (always 1 for
         *     this view) so we re-use AlbumSummary with count=1.
         */
        get: operations["list_albums_for_favorite_api_albums_by_favorite__favorite_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/albums/{album_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Album */
        get: operations["get_album_api_albums__album_id__get"];
        put?: never;
        post?: never;
        /** Delete Album */
        delete: operations["delete_album_api_albums__album_id__delete"];
        options?: never;
        head?: never;
        /** Update Album */
        patch: operations["update_album_api_albums__album_id__patch"];
        trace?: never;
    };
    "/api/albums/{album_id}/members/{favorite_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Add Album Member */
        post: operations["add_album_member_api_albums__album_id__members__favorite_id__post"];
        /** Remove Album Member */
        delete: operations["remove_album_member_api_albums__album_id__members__favorite_id__delete"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/cache/refresh": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Api Cache Refresh
         * @description Purge user-state rows whose photo Qdrant no longer holds.
         *
         *     There is no derived cache to refresh any more. The photo
         *     corpus, its metadata, and the native text index all live in
         *     Qdrant; SQLite holds only the user's own state — favourites,
         *     dislikes, albums, memberships, saved searches, feedback.
         *
         *     What this endpoint still does is the one thing that needed a
         *     trigger: dropping user-state rows that point at photos the
         *     indexer has pruned. The purge checks liveness against
         *     Qdrant, so a favourite is only dropped when the COLLECTION
         *     says the photo is gone — never because a local mirror was
         *     stale.
         *
         *     The `full` parameter is accepted and ignored, so an old
         *     caller does not break.
         */
        get: operations["api_cache_refresh_api_cache_refresh_get"];
        put?: never;
        /**
         * Api Cache Refresh
         * @description Purge user-state rows whose photo Qdrant no longer holds.
         *
         *     There is no derived cache to refresh any more. The photo
         *     corpus, its metadata, and the native text index all live in
         *     Qdrant; SQLite holds only the user's own state — favourites,
         *     dislikes, albums, memberships, saved searches, feedback.
         *
         *     What this endpoint still does is the one thing that needed a
         *     trigger: dropping user-state rows that point at photos the
         *     indexer has pruned. The purge checks liveness against
         *     Qdrant, so a favourite is only dropped when the COLLECTION
         *     says the photo is gone — never because a local mirror was
         *     stale.
         *
         *     The `full` parameter is accepted and ignored, so an old
         *     caller does not break.
         */
        post: operations["api_cache_refresh_api_cache_refresh_get_1"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/cache/status": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Cache Status
         * @description Operator visibility into the corpus and the maintenance timer.
         *
         *     Returns the point count, the last maintenance timestamp, the
         *     configured maintenance interval, and the path-liveness cache
         *     size + cap.
         *
         *     `index_db_count` and `qdrant_count` are both read from
         *     Qdrant — there is no second copy of the corpus to drift
         *     from, so `drift` is 0 by construction. The field is kept so
         *     the response shape does not change for existing operators.
         *     Drift is "unknown" when Qdrant is unreachable
         *     (qdrant_count == -1) so operators don't see a misleading
         *     negative number.
         */
        get: operations["cache_status_api_cache_status_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/centroids": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Centroids
         * @description List static + dynamic centroids with metadata.
         */
        get: operations["list_centroids_api_centroids_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/centroids/reload": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Reload Centroids */
        post: operations["reload_centroids_api_centroids_reload_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/centroids/{name}/search": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Search By Centroid
         * @description Search using a loaded centroid as the query vector.
         */
        get: operations["search_by_centroid_api_centroids__name__search_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/collections": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Collections
         * @description List distinct library collections with point counts.
         */
        get: operations["list_collections_api_collections_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/dislikes": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Dislikes */
        get: operations["list_dislikes_api_dislikes_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/dislikes/{point_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Mark Dislike */
        post: operations["mark_dislike_api_dislikes__point_id__post"];
        /** Unmark Dislike */
        delete: operations["unmark_dislike_api_dislikes__point_id__delete"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/favorites": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Api Favorites */
        get: operations["api_favorites_api_favorites_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/favorites/{point_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Mark Favorite */
        post: operations["mark_favorite_api_favorites__point_id__post"];
        /** Unmark Favorite */
        delete: operations["unmark_favorite_api_favorites__point_id__delete"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/for-you/feed": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * For You Feed
         * @description Walk through a Gumbel-sampled deck of photos in the user's
         *     taste neighborhood.
         *
         *     The candidate pool is sourced from Qdrant
         *     `recommend(positive=likes, negative=dislikes)`. The
         *     recommend() results are ordered by score descending, and
         *     the Gumbel draw biases the deck toward high scores by
         *     the temperature. At T=0 the deck is the deterministic
         *     top-K; at T=1.0 the deck is near-uniform random over the
         *     pool.
         *
         *     Cache key: `compute_for_you_cache_key(fav_ids, dis_ids,
         *     temperature, pool_size, seed)`. A client that keeps
         *     these stable across multiple offsets walks through the
         *     same deck. A new seed (or a new fav_ids/dis_ids after a
         *     like/dislike) produces a fresh deck.
         *
         *     Cold start: 0 favourites returns 400. The /for-you page
         *     surfaces a "Like a few photos to seed" empty state
         *     before sending a request.
         */
        get: operations["for_you_feed_api_for_you_feed_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/photo/{point_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Photo Metadata
         * @description Fetch metadata for a single photo by ID.
         *
         *     Used by the frontend's dedicated photo page to render the
         *     large photo + metadata sidebar. Returns everything the
         *     sidebar can show: identity, file info, indexing info,
         *     favourite status. Does NOT return the vector — that's a
         *     different concern, and `payload` carries enough for the UI.
         */
        get: operations["photo_metadata_api_photo__point_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/random": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Api Random */
        get: operations["api_random_api_random_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/saved-searches": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Saved Searches */
        get: operations["list_saved_searches_api_saved_searches_get"];
        put?: never;
        /** Create Saved Search */
        post: operations["create_saved_search_api_saved_searches_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/saved-searches/{saved_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Saved Search */
        get: operations["get_saved_search_api_saved_searches__saved_id__get"];
        put?: never;
        post?: never;
        /** Delete Saved Search */
        delete: operations["delete_saved_search_api_saved_searches__saved_id__delete"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/search": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Api Search */
        get: operations["api_search_api_search_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/similar/{point_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Similar Photos
         * @description Most-similar photos: nearest neighbours of `point_id`.
         */
        get: operations["similar_photos_api_similar__point_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/system/status": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * System Status
         * @description System status with cache stats for the frontend dashboard.
         */
        get: operations["system_status_api_system_status_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/favorites/download.zip": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Favorites Download Zip */
        get: operations["favorites_download_zip_favorites_download_zip_head"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        /** Favorites Download Zip */
        head: operations["favorites_download_zip_favorites_download_zip_head_1"];
        patch?: never;
        trace?: never;
    };
    "/healthz": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Healthz */
        get: operations["healthz_healthz_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/photo/{point_id}/raw": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Photo Raw */
        get: operations["photo_raw_photo__point_id__raw_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/thumb/{point_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Thumbnail
         * @description Serve a pre-generated WebP thumbnail, optionally sized.
         *
         *     Args:
         *         point_id: Qdrant point ID (32-char hex)
         *         w: Optional target width in pixels. Looks for
         *             `{THUMBNAIL_DIR}/{prefix}/{point_id}.w{w}.webp`.
         *             Returns 404 if missing; the browser then loads the
         *             canonical 256px file.
         *
         *     Returns:
         *         WebP file with immutable cache headers
         *
         *     Raises:
         *         404 if thumbnail doesn't exist (frontend falls back to blurhash)
         */
        get: operations["get_thumbnail_thumb__point_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
}
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        /** AlbumCreateRequest */
        AlbumCreateRequest: {
            /**
             * Description
             * @default
             */
            description: string;
            /** Name */
            name: string;
        };
        /**
         * AlbumDetailResponse
         * @description Full album row + paginated members (UI rendering shape).
         *
         *     Members are photo metadata joined from the `images` cache, so
         *     orphan memberships (favourites whose photo is gone) are hidden.
         *     The album centroid compute still sees them via
         *     `list_album_member_ids`.
         */
        AlbumDetailResponse: {
            /** Cover Favorite Id */
            cover_favorite_id: string;
            /** Created At */
            created_at: string;
            /** Description */
            description: string;
            /** Id */
            id: number;
            /** Member Total */
            member_total: number;
            /** Members */
            members: components["schemas"]["AlbumMemberItem"][];
            /** Name */
            name: string;
            /** Updated At */
            updated_at: string;
        };
        /** AlbumMemberItem */
        AlbumMemberItem: {
            /** Added At */
            added_at: string;
            /** Id */
            id: string;
            /** Path */
            path: string;
        };
        /** AlbumMemberResponse */
        AlbumMemberResponse: {
            /** Added At */
            added_at: string;
            /** Album Id */
            album_id: number;
            /** Favorite Id */
            favorite_id: string;
        };
        /**
         * AlbumMembershipsResponse
         * @description List of albums containing a given favourite, used by the
         *     per-photo UI to show which albums a photo is in.
         */
        AlbumMembershipsResponse: {
            /** Albums */
            albums: components["schemas"]["AlbumSummary"][];
            /** Favorite Id */
            favorite_id: string;
        };
        /**
         * AlbumSummary
         * @description Lightweight album row for list views.
         *
         *     `member_count` is the count from `album_memberships`, which
         *     includes orphan memberships (favourites whose photo is no
         *     longer in the cache). For a UI count that hides orphans, use
         *     the detail endpoint.
         *
         *     `first_member_id` is the most RECENTLY added photo in the album
         *     (ORDER BY album_memberships.added_at DESC LIMIT 1). Drives the
         *     /albums index card thumbnail — prefer it over
         *     `cover_favorite_id` for display. Empty string when the album
         *     has no members yet.
         *
         *     The field name is historical — it was originally the
         *     chronologically-first member. The card cover now shows the
         *     newest addition.
         */
        AlbumSummary: {
            /** Cover Favorite Id */
            cover_favorite_id: string;
            /** Created At */
            created_at: string;
            /** Description */
            description: string;
            /**
             * First Member Id
             * @default
             */
            first_member_id: string;
            /** Id */
            id: number;
            /** Member Count */
            member_count: number;
            /** Name */
            name: string;
            /** Updated At */
            updated_at: string;
        };
        /** AlbumUpdateRequest */
        AlbumUpdateRequest: {
            /** Description */
            description?: string | null;
            /** Name */
            name?: string | null;
        };
        /** AlbumsListResponse */
        AlbumsListResponse: {
            /** Albums */
            albums: components["schemas"]["AlbumSummary"][];
        };
        /**
         * CacheStatusResponse
         * @description GET /api/cache/status — operator visibility into dual-store sync.
         *
         *     `drift` is the integer point-count gap between Qdrant and
         *     IndexDB, or the string "unknown" when Qdrant is unreachable
         *     (qdrant_count < 0). The string sentinel is intentional:
         *     returning -1 or 0 would be misleading — a negative number
         *     conflates "Qdrant down" with "0 drift", and 0 would imply
         *     the stores are in sync when we don't actually know.
         *
         *     `path_liveness_cache_size` / `_max` are the runtime liveness
         *     cache's current size and its eviction cap. These matter
         *     because the cache lives in memory and a leak would
         *     eventually OOM the container.
         */
        CacheStatusResponse: {
            /** Drift */
            drift: number | string;
            /** Index Db Count */
            index_db_count: number;
            /** Last Refresh */
            last_refresh: number | null;
            /** Path Liveness Cache Max */
            path_liveness_cache_max: number;
            /** Path Liveness Cache Size */
            path_liveness_cache_size: number;
            /** Path Liveness Ttl Seconds */
            path_liveness_ttl_seconds: number;
            /** Qdrant Count */
            qdrant_count: number;
            /** Refresh Interval Seconds */
            refresh_interval_seconds: number;
        };
        /**
         * CentroidListResponse
         * @description GET /api/centroids — static + dynamic centroid inventory.
         *
         *     `centroids` is the static (filesystem-backed) list.
         *     `dynamic_centroids` is the runtime registry (favourites,
         *     tag-based, custom). `expected_model` and
         *     `expected_feature_dim` are None when the static store
         *     isn't initialised; the frontend uses these to warn when
         *     the user has a config that doesn't match the current
         *     SigLIP2 model.
         */
        CentroidListResponse: {
            /** Centroids */
            centroids: {
                [key: string]: unknown;
            }[];
            /** Dynamic Centroids */
            dynamic_centroids: {
                [key: string]: unknown;
            }[];
            /** Expected Feature Dim */
            expected_feature_dim: number | null;
            /** Expected Model */
            expected_model: string | null;
        };
        /**
         * DiversityMetadata
         * @description What the search-side Diversity ranker actually did.
         */
        DiversityMetadata: {
            /**
             * Applied
             * @default false
             */
            applied: boolean;
            /**
             * Candidate Count
             * @default 0
             */
            candidate_count: number;
            /**
             * Depth
             * @default auto
             */
            depth: string;
            /**
             * Duplicate Images Collapsed
             * @default 0
             */
            duplicate_images_collapsed: number;
            /**
             * Mode
             * @default off
             */
            mode: string;
            /**
             * Pool Depth
             * @default 0
             */
            pool_depth: number;
            /**
             * Requested
             * @default false
             */
            requested: boolean;
            /**
             * Result Count
             * @default 0
             */
            result_count: number;
            /**
             * Semantic Groups Covered
             * @default 0
             */
            semantic_groups_covered: number;
            /**
             * Strength
             * @default 0
             */
            strength: number;
        };
        /** FavoriteToggleResponse */
        FavoriteToggleResponse: {
            /** Favorited At */
            favorited_at: string;
            /** Id */
            id: string;
        };
        /** HTTPValidationError */
        HTTPValidationError: {
            /** Detail */
            detail?: components["schemas"]["ValidationError"][];
        };
        /**
         * HealthzResponse
         * @description GET /healthz — operator + load-balancer probe.
         *
         *     Kept intentionally tiny: `qdrant` is the only required
         *     dependency the LB cares about (the rest of the system can be
         *     degraded without taking the container out of rotation). The
         *     `test_mode` flag is informational — it tells operators
         *     whether this is the prod-shaped config or a test fixture.
         */
        HealthzResponse: {
            /** Qdrant */
            qdrant: boolean;
            /** Test Mode */
            test_mode: boolean;
        };
        /**
         * IndexerLogResponse
         * @description Body for GET /api/admin/index/log.
         */
        IndexerLogResponse: {
            /** Lines */
            lines: string[];
            /** Next Line */
            next_line: number;
            /** Total */
            total: number;
        };
        /** IndexerProgressModel */
        IndexerProgressModel: {
            /** Errors */
            errors: number;
            /** Indexed */
            indexed: number;
            /** Reembedded */
            reembedded: number;
            /** Skipped */
            skipped: number;
            /**
             * Thumb Regens
             * @default 0
             */
            thumb_regens: number;
        };
        /**
         * IndexerRunRequest
         * @description Body for POST /api/admin/index.
         *
         *     `mode` is the only user-visible knob: `incremental` (default)
         *     embeds only new/changed files; `rebuild` wipes the Qdrant
         *     collection + SQLite side store first.
         */
        IndexerRunRequest: {
            /**
             * Mode
             * @default incremental
             * @enum {string}
             */
            mode: "incremental" | "rebuild";
        };
        /**
         * IndexerStatusResponse
         * @description Body for GET /api/admin/index/status.
         */
        IndexerStatusResponse: {
            /** Finished At */
            finished_at: string | null;
            /** Job Id */
            job_id: string | null;
            /** Last Error */
            last_error: string | null;
            /** Last Run At */
            last_run_at: string | null;
            /** Mode */
            mode: ("incremental" | "rebuild") | null;
            /** Phase */
            phase?: ("scanning" | "warming_up" | "embedding") | null;
            /** Pid */
            pid: number | null;
            /** Points Count */
            points_count: number | null;
            progress: components["schemas"]["IndexerProgressModel"];
            /** Started At */
            started_at: string | null;
            /**
             * State
             * @enum {string}
             */
            state: "idle" | "running" | "failed";
        };
        /**
         * OrphanListResponse
         * @description Body for GET /api/admin/orphans.
         */
        OrphanListResponse: {
            /** By Source */
            by_source?: {
                [key: string]: number;
            };
            /** Count */
            count: number;
            /** Orphaned User Data */
            orphaned_user_data?: {
                [key: string]: number;
            };
            /** Sample */
            sample?: components["schemas"]["OrphanSample"][];
        };
        /**
         * OrphanPurgeRequest
         * @description Body for POST /api/admin/orphans/purge.
         */
        OrphanPurgeRequest: {
            /**
             * Confirm
             * @default false
             */
            confirm: boolean;
            /**
             * Include User Data
             * @default true
             */
            include_user_data: boolean;
            /** Limit */
            limit?: number | null;
            /** Source */
            source?: string | null;
        };
        /**
         * OrphanPurgeResponse
         * @description Body for POST /api/admin/orphans/purge.
         */
        OrphanPurgeResponse: {
            /** Deleted */
            deleted: number;
            /** Remaining */
            remaining: number;
            /** Source */
            source?: string | null;
            /** User Data Purged */
            user_data_purged?: {
                [key: string]: number;
            };
            /** User State */
            user_state?: {
                [key: string]: number;
            };
        };
        /**
         * OrphanSample
         * @description One marked orphan, as shown to the operator.
         */
        OrphanSample: {
            /**
             * Collection
             * @default
             */
            collection: string;
            /** Id */
            id: string;
            /** Orphan Since */
            orphan_since?: number | null;
            /**
             * Path
             * @default
             */
            path: string;
        };
        /**
         * ReloadCentroidsResponse
         * @description POST /api/centroids/reload — operator-initiated centroid refresh.
         *
         *     `count` is the number of centroids the store successfully
         *     loaded. `centroids_dir` is the absolute filesystem path
         *     the store scanned (or None when the store has no
         *     configured directory — same shape as a config-disabled
         *     store).
         */
        ReloadCentroidsResponse: {
            /** Centroids Dir */
            centroids_dir: string | null;
            /** Count */
            count: number;
        };
        /**
         * SavedSearch
         * @description One saved-search row, as returned by every saved-search
         *     endpoint. `positives` / `negatives` are always Python lists of
         *     strings on the wire — the IndexDB serialises JSON on disk and
         *     deserialises on read so callers don't need to think about the
         *     on-disk shape.
         */
        SavedSearch: {
            /** Created At */
            created_at: string;
            /** Id */
            id: number;
            /** Name */
            name: string;
            /** Negatives */
            negatives: string[];
            /** Positives */
            positives: string[];
        };
        /**
         * SavedSearchCreateRequest
         * @description Body for POST /api/saved-searches.
         *
         *     `positives` / `negatives` are lists of free-text prompts. Both
         *     may be empty, but at least one prompt total must be present
         *     (the route enforces this with a 400 if neither list has a
         *     non-empty entry). Name is trimmed and length-checked in the
         *     route (1–80 chars after strip).
         */
        SavedSearchCreateRequest: {
            /** Name */
            name: string;
            /**
             * Negatives
             * @default []
             */
            negatives: string[];
            /**
             * Positives
             * @default []
             */
            positives: string[];
        };
        /**
         * SavedSearchListResponse
         * @description Paginated list response for GET /api/saved-searches.
         *
         *     Newest-first ordering matches the dropdown UX (most recently
         *     saved at the top of the list). `total` is the unpaginated row
         *     count so the UI can show a "showing N of M" hint if it ever
         *     wants to.
         */
        SavedSearchListResponse: {
            /** Limit */
            limit: number;
            /** Offset */
            offset: number;
            /** Saved Searches */
            saved_searches: components["schemas"]["SavedSearch"][];
            /** Total */
            total: number;
        };
        /** SearchResponse */
        SearchResponse: {
            /** Centroid */
            centroid?: string | null;
            /**
             * Centroids
             * @description Active centroid names in blend order. Empty when no centroid search is in play. One or more names blends via weighted mean.
             */
            centroids?: string[];
            /**
             * Diverse
             * @default false
             */
            diverse: boolean;
            /** @description Diagnostics for the search-only Diversity ranking pass. */
            diversity?: components["schemas"]["DiversityMetadata"];
            /**
             * Has More
             * @default false
             */
            has_more: boolean;
            /**
             * Limit
             * @description Max results requested for this page
             */
            limit: number;
            /** Negatives */
            negatives?: string[];
            /**
             * Offset
             * @default 0
             */
            offset: number;
            /** Positives */
            positives?: string[];
            /** Query */
            query: string;
            /** Results */
            results: components["schemas"]["SearchResult"][];
            /** Session Id */
            session_id?: string | null;
            /** Session Total */
            session_total?: number | null;
            /**
             * Surprise
             * @default false
             */
            surprise: boolean;
            /** Took Ms */
            took_ms: number;
            /**
             * View
             * @default grid
             */
            view: string;
            /** Weights */
            weights?: number[] | null;
        };
        /** SearchResult */
        SearchResult: {
            /** Blurhash */
            blurhash?: string | null;
            /** Height */
            height?: number | null;
            /**
             * Id
             * @description Qdrant point id (32-char hex prefix)
             */
            id: string;
            /**
             * Is Disliked
             * @default false
             */
            is_disliked: boolean;
            /**
             * Is Favorite
             * @default false
             */
            is_favorite: boolean;
            /**
             * Path
             * @description Absolute source path on the NAS
             */
            path: string;
            /**
             * Score
             * @description Cosine similarity in [-1, 1]
             */
            score: number;
            /**
             * Score Str
             * @description Score formatted to 3 decimals, e.g. '0.873'. Computed server-side so SSR + JS render identically.
             * @default
             */
            score_str: string;
            /**
             * Url
             * @description Public URL for the /photo/{id}/raw endpoint
             * @default
             */
            url: string;
            /** Width */
            width?: number | null;
        };
        /**
         * SystemStatusResponse
         * @description GET /api/system/status — frontend dashboard telemetry.
         *
         *     The frontend's settings panel reads this to surface model
         *     load state, point counts, and the last successful
         *     indexer→IndexDB refresh. Field types are chosen so the
         *     JS layer can `JSON.parse` and consume without coercion.
         */
        SystemStatusResponse: {
            /** Index Db Count */
            index_db_count: number;
            /** Last Refresh */
            last_refresh: number | null;
            /** Model Status */
            model_status: {
                [key: string]: unknown;
            };
            /** Qdrant Count */
            qdrant_count: number;
        };
        /** ValidationError */
        ValidationError: {
            /** Context */
            ctx?: Record<string, never>;
            /** Input */
            input?: unknown;
            /** Location */
            loc: (string | number)[];
            /** Message */
            msg: string;
            /** Error Type */
            type: string;
        };
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
    album_download_zip_albums__album_id__download_zip_head_1: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                album_id: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    album_download_zip_albums__album_id__download_zip_head: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                album_id: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    start_index_api_admin_index_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["IndexerRunRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            202: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["IndexerStatusResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    cancel_index_api_admin_index_cancel_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["IndexerStatusResponse"];
                };
            };
        };
    };
    get_log_api_admin_index_log_get: {
        parameters: {
            query?: {
                /** @description Return only lines after this 0-based line index. */
                since_line?: number;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["IndexerLogResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_status_api_admin_index_status_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["IndexerStatusResponse"];
                };
            };
        };
    };
    list_orphans_api_admin_orphans_get: {
        parameters: {
            query?: {
                /** @description How many orphans to include in the sample. */
                sample_size?: number;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["OrphanListResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    purge_orphans_api_admin_orphans_purge_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["OrphanPurgeRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["OrphanPurgeResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_albums_api_albums_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AlbumsListResponse"];
                };
            };
        };
    };
    create_album_api_albums_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["AlbumCreateRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AlbumSummary"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_albums_for_favorite_api_albums_by_favorite__favorite_id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                favorite_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AlbumMembershipsResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_album_api_albums__album_id__get: {
        parameters: {
            query?: {
                /** @description max members to return */
                limit?: number;
                /** @description offset into members */
                offset?: number;
            };
            header?: never;
            path: {
                album_id: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AlbumDetailResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_album_api_albums__album_id__delete: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                album_id: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_album_api_albums__album_id__patch: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                album_id: number;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["AlbumUpdateRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AlbumSummary"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    add_album_member_api_albums__album_id__members__favorite_id__post: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                album_id: number;
                favorite_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AlbumMemberResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    remove_album_member_api_albums__album_id__members__favorite_id__delete: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                album_id: number;
                favorite_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    api_cache_refresh_api_cache_refresh_get: {
        parameters: {
            query?: {
                /** @description Deprecated. There is no cache to refresh: the photo corpus lives in Qdrant and the SQLite side holds only authoritative user state. The endpoint now runs the orphan purge, which checks liveness against Qdrant. */
                full?: boolean;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    api_cache_refresh_api_cache_refresh_get_1: {
        parameters: {
            query?: {
                /** @description Deprecated. There is no cache to refresh: the photo corpus lives in Qdrant and the SQLite side holds only authoritative user state. The endpoint now runs the orphan purge, which checks liveness against Qdrant. */
                full?: boolean;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    cache_status_api_cache_status_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CacheStatusResponse"];
                };
            };
        };
    };
    list_centroids_api_centroids_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CentroidListResponse"];
                };
            };
        };
    };
    reload_centroids_api_centroids_reload_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ReloadCentroidsResponse"];
                };
            };
        };
    };
    search_by_centroid_api_centroids__name__search_get: {
        parameters: {
            query?: {
                /** @description max results */
                limit?: number;
                /** @description offset into the full result set */
                offset?: number;
                /** @description Retrieval mode. 'centroid' (default) uses the full mean of the seed set. 'sample' picks a random K-subset of the seeds and uses the mean of THAT subset. Each request re-rolls, so refreshing surfaces a different cluster. */
                mode?: string;
                /** @description Cluster count (k-means K) for sample mode. Defaults to 10. Only used when mode=sample. Round-75. */
                sample_k?: number;
                /** @description Number of cluster centroids to average per request. Defaults to 3. Must satisfy 1 <= n <= sample_k. Only used when mode=sample. Round-75. */
                sample_n?: number;
                /** @description apply MMR diversity re-ranking */
                diverse?: boolean;
                /** @description Diversity strength: off, low, balanced, or high */
                diversity?: string | null;
                /** @description Diversity candidate depth: auto, 500, 1000, 2000, 5000, or 10000 */
                diversity_depth?: string | null;
                /** @description Surprise Me — Gumbel-max over the top N candidates from this centroid. Mutually exclusive with mode=sample and with diversity. Same cache+session contract as /api/search?surprise=true. */
                surprise?: boolean;
                /** @description Gumbel noise scale for Surprise Me on centroid search. T=0 is deterministic top-K; higher = more variety. Ignored unless surprise=true. */
                surprise_temperature?: number;
                /** @description How many top candidates the Gumbel draw is taken from for centroid search. Larger = more variety, slower first page. Range 200..50000, default 5000. Ignored unless surprise=true. */
                surprise_pool?: number;
                /** @description Reseed knob for Surprise Me on centroid search. Pass a fresh random int for a new deck every request; pass the same value across requests for a shareable deterministic URL. The route generates a random int when this is absent so each new search gives a different deck by default. Ignored unless surprise=true. */
                surprise_seed?: number | null;
                /** @description Opaque session id from a prior Surprise Me response on this centroid. Pass back with offset to walk the cached deck without recomputing. Ignored unless surprise=true. */
                surprise_session?: string | null;
            };
            header?: never;
            path: {
                name: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SearchResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_collections_api_collections_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
        };
    };
    list_dislikes_api_dislikes_get: {
        parameters: {
            query?: {
                /** @description max dislikes */
                limit?: number;
                /** @description offset into dislikes */
                offset?: number;
                /** @description return SearchResponse-compatible shape */
                as_results?: boolean;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    mark_dislike_api_dislikes__point_id__post: {
        parameters: {
            query?: {
                source?: string;
            };
            header?: never;
            path: {
                point_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    unmark_dislike_api_dislikes__point_id__delete: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                point_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    api_favorites_api_favorites_get: {
        parameters: {
            query?: {
                /** @description max favourites */
                limit?: number;
                /** @description offset into favourites */
                offset?: number;
                /** @description return SearchResponse-compatible shape */
                as_results?: boolean;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    mark_favorite_api_favorites__point_id__post: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                point_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["FavoriteToggleResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    unmark_favorite_api_favorites__point_id__delete: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                point_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    for_you_feed_api_for_you_feed_get: {
        parameters: {
            query?: {
                /** @description Number of recommend() candidates the Gumbel draw is taken from. Default 2000. Range [200, 50000]. Higher = more variety, slower first page. */
                pool_size?: number;
                /** @description Photos per page. No upper bound. */
                limit?: number;
                /** @description Zero-based offset into the cached Gumbel deck. Pass the same `session` value to walk forward through the same deck without recomputing. */
                offset?: number;
                /** @description Gumbel noise scale. T=0 is deterministic top-K (curated). T=1.0 is near-uniform random over the pool. Default 0.05. */
                temperature?: number;
                /** @description Random seed for the Gumbel draw. Pass a fresh random int on every page reload so the user sees a different deck each time. Pass the same value for paginated scroll within a single page-mount to walk through the same deck. If omitted, the route generates a fresh random seed per request. */
                seed?: number | null;
                /** @description Opaque cache key from a prior /api/for-you/feed response. Pass back with offset=N to walk the same deck without recomputing. Server-cached for 30 minutes. */
                session?: string | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SearchResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    photo_metadata_api_photo__point_id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                point_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    api_random_api_random_get: {
        parameters: {
            query?: {
                /** @description max results */
                limit?: number;
                /** @description position in the shuffled deck */
                offset?: number;
                /** @description session id from a previous /api/random response */
                session?: string | null;
                /** @description restrict to one or more collections; empty = whole set */
                collections?: string[] | null;
                /** @description result view: 'grid' or 'feed' */
                view?: string;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SearchResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_saved_searches_api_saved_searches_get: {
        parameters: {
            query?: {
                /** @description max saved searches */
                limit?: number;
                /** @description offset into saved searches */
                offset?: number;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SavedSearchListResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_saved_search_api_saved_searches_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["SavedSearchCreateRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SavedSearch"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_saved_search_api_saved_searches__saved_id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                saved_id: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SavedSearch"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_saved_search_api_saved_searches__saved_id__delete: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                saved_id: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    api_search_api_search_get: {
        parameters: {
            query?: {
                /** @description text query */
                q?: string;
                /** @description max results */
                limit?: number;
                /** @description offset into the full result set */
                offset?: number;
                /** @description result view: 'grid' or 'feed' */
                view?: string;
                /** @description restrict results to favourites */
                favorites?: boolean;
                /** @description apply MMR diversity re-ranking */
                diverse?: boolean;
                /** @description Diversity strength: off, low, balanced, or high */
                diversity?: string | null;
                /** @description Diversity candidate depth: auto, 500, 1000, 2000, 5000, or 10000 */
                diversity_depth?: string | null;
                /** @description Surprise Me — random sample from deep pool */
                surprise?: boolean;
                /** @description Gumbel noise scale for Surprise Me. T=0 is deterministic; higher = more variety. */
                surprise_temperature?: number;
                /** @description How many top candidates the Gumbel draw is taken from. Larger = more variety, slower first page. */
                surprise_pool?: number;
                /** @description Reseed knob for Surprise Me. Omit (or pass a fresh random int) for a new deck every search; pass the same value across requests for a shareable deterministic URL. The route generates a random int when this is absent so each new search gives a different deck by default. */
                surprise_seed?: number | null;
                /** @description Opaque session id from a prior Surprise Me response. Pass back with offset to walk the cached deck without recomputing. */
                surprise_session?: string | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SearchResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    similar_photos_api_similar__point_id__get: {
        parameters: {
            query?: {
                /** @description max similar photos to return */
                limit?: number;
                /** @description offset into the result set */
                offset?: number;
            };
            header?: never;
            path: {
                point_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SearchResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    system_status_api_system_status_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SystemStatusResponse"];
                };
            };
        };
    };
    favorites_download_zip_favorites_download_zip_head: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
        };
    };
    favorites_download_zip_favorites_download_zip_head_1: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
        };
    };
    healthz_healthz_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HealthzResponse"];
                };
            };
        };
    };
    photo_raw_photo__point_id__raw_get: {
        parameters: {
            query?: {
                /** @description Optional target width in pixels. Server Lanczos-resizes if set. */
                w?: number | null;
            };
            header?: never;
            path: {
                point_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_thumbnail_thumb__point_id__get: {
        parameters: {
            query?: {
                w?: number | null;
            };
            header?: never;
            path: {
                point_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
}
