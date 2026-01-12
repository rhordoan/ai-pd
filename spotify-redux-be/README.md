# spotify-redux-be (FastAPI backend)

FastAPI API that serves:
- Catalog-only search (`/catalog/search`)
- Cold-start onboarding recommendations (`/recommendations/cold-start`)
- “For you” shelves (`/recommendations/categories`)
- “Up next” queue (`/recommendations/similar`)
- Auth (`/auth/signup`, `/auth/login`) + likes (`/likes`, `/likes/toggle`)

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Health check: `GET http://localhost:8000/health`

## Model artifacts (AI “serving”)

The API loads these artifacts from the backend folder:

- `best_model.pkl` — trained Surprise SVD model
- `song_vectors.index` — FAISS index over SVD item vectors
- `faiss_id_map.pkl` — mapping from FAISS row → canonical `song_id`

If you retrain, rebuild the FAISS artifacts:

```bash
python main.py                 # trains + writes best_model.pkl
python build_vector_db.py      # writes song_vectors.index + faiss_id_map.pkl
```

## Storage (users / likes / seeds)

By default the backend uses SQLite:
- file: `spotify-redux-be/users.db`

You can switch to Postgres by setting `DATABASE_URL` (see `docs/deployment.md`).

## Configuration (env vars)

- **`API_SECRET_KEY`**: JWT secret key. Default: `dev-secret-change-me` (do not use in prod)
- **`TOKEN_EXPIRE_HOURS`**: JWT expiry. Default: `24`
- **`LIKED_WEIGHT`**: how strongly likes influence the profile vector. Default: `2.0`
- **`DATABASE_URL`**: SQLAlchemy URL. If unset, uses SQLite at `users.db`

Example:

```bash
API_SECRET_KEY=super-long-random
TOKEN_EXPIRE_HOURS=24
LIKED_WEIGHT=2.0
DATABASE_URL=postgresql+psycopg://spotify:spotify@db:5432/spotify
```

## Performance notes

- The model + FAISS index are loaded on demand. For production, run with multiple workers and preload artifacts:
  - `gunicorn -k uvicorn.workers.UvicornWorker -w <cpu*2+1> api:app`
- External metadata calls (covers) can be slow/rate-limited:
  - The backend caches cover URLs in-memory; for large scale consider a shared cache (Redis) and/or a metadata prefetch job.

## Scaling notes (high level)

- Keep the API stateless. Store:
  - user data in Postgres
  - artifacts in an object store or a mounted disk volume
- Scale horizontally by running multiple API replicas reading the same artifacts.

