## Scaling the AI recommender (training vs serving)

This project has two very different workloads:

- **Training / artifact building** (offline, heavy CPU/memory, infrequent)
- **Serving** (online, latency-sensitive, horizontally scalable)

Treat them as separate systems.

## Artifacts you must version

The serving API depends on three artifacts that must stay consistent with each other:

- `best_model.pkl` (Surprise SVD model)
- `song_vectors.index` (FAISS index over `model.qi`)
- `faiss_id_map.pkl` (mapping from FAISS row → canonical `song_id`)

If any of these are out of sync, recommendations will be wrong or crash.

## Recommended pipeline

1. **Train** (offline):
   - run `spotify-redux-be/main.py` to produce `best_model.pkl`
2. **Build vector DB** (offline):
   - run `spotify-redux-be/build_vector_db.py` to produce FAISS index + id map
3. **Validate**:
   - basic smoke test: pick a seed song and ensure top neighbors are returned and are not the seed itself
4. **Publish artifacts**:
   - store with a version tag, e.g. `artifacts/2026-01-12/…`
   - use object storage (S3-compatible) or a VM-mounted disk
5. **Deploy serving API**:
   - mount artifacts read-only in the backend container
   - set the backend to load from that mount path (or keep default paths if you copy into the image)

## Serving performance tuning

### Keep it stateless

- Store users/likes/seeds in a DB (Postgres recommended for production).
- Keep the API container stateless so you can run multiple replicas.

### Warm startup

- Load model + FAISS index at startup (or on first request, but that adds latency).
- Run multiple workers:
  - `gunicorn -k uvicorn.workers.UvicornWorker -w <cpu*2+1> api:app`

### Avoid slow metadata calls on the hot path

Album cover fetching depends on public APIs that can be slow or rate-limited.

Recommended options:
- **Cache** cover URLs:
  - in-memory cache works for a single instance
  - for multiple instances use Redis (shared cache)
- **Prefetch**:
  - build a small “cover URL table” offline for the entire catalog
- **Graceful degradation**:
  - return items without covers when upstream providers rate-limit

## Horizontal scaling model

Run multiple backend replicas behind a reverse proxy / load balancer:

- N replicas of FastAPI, each reading the same artifact set
- 1 Postgres instance (managed preferred)
- Optional Redis for shared caching

The recommender is read-heavy and parallelizes well.

## Memory sizing guidance (rule of thumb)

- Model + vectors:
  - `best_model.pkl` size + FAISS index size drive memory
- Add headroom:
  - 1–2 GB minimum for a small demo
  - more if you add precomputed metadata caches

## When you outgrow a single VM

Move to a managed setup:
- Managed Postgres
- Object storage for artifacts
- A container platform (Kubernetes or Cloud Run)

The key is still the same: serving stays stateless and artifacts remain versioned and consistent.

