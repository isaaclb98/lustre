# lustre

Self-hosted semantic image search over a local photo library.

- **Embeddings:** SigLIP2 (open_clip `webli` pretrained). Four variants
  are registered — pick one via `SIGLIP_VARIANT` in `.env`:

  | Variant | Model arch | Embedding dim | Input | ~Weights | Trade-off |
  |---|---|---|---|---|---|
  | `B/16-256` (default) | `ViT-B-16-SigLIP2-256` | 768 | 256 | 370 MB | Smallest, fastest, runs on CPU. End-user default. |
  | `L/16-256` | `ViT-L-16-SigLIP2-256` | 1024 | 256 | 1.1 GB | Mid-tier. ~2× slower than B, better retrieval. |
  | `so400m/16-384` | `ViT-so400m-patch16-384` | 1152 | 384 | 1.8 GB | Highest quality per vector. Bigger model, higher input res. The default for this repo's prod. |
  | `gopt/16-384` | `ViT-gopt-16-SigLIP2-384` | 1536 | 384 | 3.5 GB | Largest. Highest-dim embeddings; diminishing returns over so400m in practice. |

  Switch by editing `.env` and `docker compose up -d` — the container
  detects the variant change, drops the now-incompatible Qdrant
  collection, and boots with the empty-index prompt. Click
  **Settings → Index** to re-embed with the new model. See
  "Switching the model variant" below for the full flow.
- **Vector store:** Qdrant (local container in dev, HTTPS reverse proxy in prod).
- **Backend:** FastAPI, single container, gunicorn + uvicorn workers.
- **Frontend:** SvelteKit 2 + Svelte 5 + TypeScript SPA. Speaks to the backend over an OpenAPI-typed client.
- **Auth:** None. Deploy behind a reverse proxy (caddy auth, oauth2-proxy, tailscale, etc.) if access control is needed.
- **Side store:** SQLite `index.db` for folder metadata, favorites, dislikes, saved searches, album membership. Background-refreshed from Qdrant.
- **Thumbnails:** Pre-baked WebP at 384×384, q80. Generated at index time, served from `/thumb/{id}?w=…`. See `indexer/thumbnails.py`.

## Set up

The image at `ghcr.io/isaaclb98/lustre:latest` is the single
container you need. It runs FastAPI on `:8000`, serves the bundled
SvelteKit SPA from the same port, and embeds photos by spawning the
indexer as an in-container subprocess (so the indexer has no host
dependency — see "Index your library" below).

### Production (pull from ghcr)

```bash
export NAS_IMAGES_PATH=/path/to/your/photos   # bind-mounted read-only at /nas
docker compose up -d                          # brings up Qdrant + search
```

`docker-compose.yml` (in this repo) pulls `lustre:latest` from
ghcr, runs a Qdrant sidecar, bind-mounts your photo library at
`/nas:ro`, and exposes `:8000` on the host.

Two named volumes hold state across image updates:

- **`qdrant_data`** — the vector index + payloads.
- **`lustre_search_data`** — the SQLite side store (favorites,
  dislikes, albums, saved searches), the SigLIP2 model cache (HF_HOME),
  and the runtime thumbnail cache. **Without this volume, every image
  update wipes your library state.** `docker compose down` keeps both;
  `docker compose down -v` wipes both.

First start downloads the SigLIP2 model (~30 s one-off, then cached in
`lustre_search_data`). Open <http://localhost:8000>.

### Local dev (faster iteration)

```bash
# Backend (loads SigLIP2 the first time, ~3 GB into HF cache)
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
docker run -p 6333:6333 qdrant/qdrant:v1.19.0 # vector DB
NAS_IMAGES_PATH=/path/to/your/photos \
QDRANT_URL=http://localhost:6333 \
    python -m search.dev_server

# Frontend (separate shell, HMR)
cd frontend && npm install && npm run dev
# http://localhost:5173
```

Two flags useful for UI work without a GPU or a real library:

```bash
python -m search.dev_server --no-model                     # mock encoder
python -m search.dev_server --no-model --demo-data --demo-count 500   # in-memory Qdrant + 500 synthetic photos
```

## Switching the model variant

The app reads `SIGLIP_VARIANT` from `.env` on every container start.
If you change the variant (or use the default for the first time),
the search backend's startup reconciler checks the live Qdrant
collection against the requested variant:

- **Dim matches and per-point payload matches** → boots normally.
- **No collection yet** → fresh install. Empty-index prompt.
- **Dim or payload mismatches** → drops the Qdrant collection
  (its vectors are in the wrong embedding space), logs a loud
  WARNING, and boots with the empty-index prompt.

Switching to a non-default variant:

```bash
# Edit .env (or docker-compose.yml's environment: block)
echo 'SIGLIP_VARIANT=so400m/16-384' >> .env

# Recreate the container with the new env
docker compose up -d

# Click Settings → Index to re-embed your library.
# The first time, this re-encodes everything (~minutes to hours
# depending on library size + model + GPU).
```

You can pin the matching model arch directly with `MODEL_NAME`,
or let the app derive it from the variant (the usual path). For
the four registered variants, the pair is:

| `SIGLIP_VARIANT` | matching `MODEL_NAME` |
|---|---|
| `B/16-256` | `ViT-B-16-SigLIP2-256` |
| `L/16-256` | `ViT-L-16-SigLIP2-256` |
| `so400m/16-384` | `ViT-so400m-patch16-384` |
| `gopt/16-384` | `ViT-gopt-16-SigLIP2-384` |

Setting just `SIGLIP_VARIANT` is enough — the app picks the right
`MODEL_NAME` automatically. Setting `MODEL_NAME` directly is for
fine-tune or non-registered variants; it overrides the derived
value.

## Use it

### Index your library

Open the SPA, go to **Settings → Index**, and click **Start**. The
search container spawns the indexer as an in-container subprocess —
no host Python needed. Settings polls status every second and shows
live progress + a cancel button.

For incremental runs (the default), re-running the indexer only
embeds photos that changed since the last run. Qdrant upserts are
idempotent (deterministic UUID5 per photo path), so this is safe
to run repeatedly. The first run on a fresh library takes a while;
subsequent runs are usually seconds.

### Index from the host CLI (advanced)

If you want to run the indexer directly (e.g. on a different host
or with GPU access the search container doesn't have), two shapes:

```bash
# Full sync + change-detection vs prior run. Idempotent — deterministic
# UUID5 per (shard, path). This is the right tool for a real library,
# run on the GPU host.
python -m indexer.local_sync --source /path/to/your/photos

# Thin wrapper around IndexerPipeline. No diff, no prune. Right shape
# for an "Index this folder" button in a desktop client.
python -m indexer.run_pipeline --source /path/to/folder
```

See `indexer/local_sync.py --help` for the full flag list (`--prune`, `--backfill`, `--dry-run`, ...).

### Query

Open the SPA and search by text (`"beach sunset"`) or jump from a result to its nearest neighbours. Other entry points:

- `/random` — walks the whole library in random order, no repeats per session.
- `/similar/{id}` — nearest neighbours of a given photo.
- `/albums` — favorites, dislikes, and saved searches as albums.
- `/api/photo/{id}` — JSON metadata for a point.
- `/photo/{id}/raw?w=1920` — original image, on-demand resize.

The frontend picks up new Qdrant points within `INDEX_DB_REFRESH_INTERVAL_SECONDS` (default 6h) via a background refresh. Force one now:

```bash
curl -X POST http://localhost:8000/api/cache/refresh
```

### Stop / restart

```bash
docker compose down      # stop, keep data
docker compose down -v   # stop, wipe data (forces a full reindex on next start)
```

## API

`http://localhost:8000/openapi.json` is the source of truth. CI enforces that the frontend's pinned snapshot is a subset of the live backend's spec — see `tests/test_openapi_stability.py` (6 contracts catching path/method drops and response-type narrowing).