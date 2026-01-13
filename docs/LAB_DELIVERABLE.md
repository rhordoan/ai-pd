# Spotify Redux — Lab Deliverable (Report + Documentation)

## 1) Project report

### 1.1 Executive summary
Spotify Redux is an end-to-end “Spotify-like” demo application that provides:

- **Music recommendations** from a trained **SVD latent-factor model** served via a **FastAPI** backend.
- **Search + playback simulation** (covers/metadata via public APIs) in an **Expo** (React Native) frontend.
- **JWT authentication**, **one-time cold-start onboarding per account**, **likes**, **For you shelves**, and an **Up next queue**.

The system is designed to be **stateless at the API layer** (scales horizontally) with **versioned AI artifacts** (model + FAISS index) and a DB for user data.

### 1.2 System design (architecture)

#### Components
- **Frontend** (`spotify-redux-fe/`): Expo Router app (mobile + web)
  - Login/signup modal (`/login`)
  - One-time onboarding (`/onboarding`) for cold start
  - Home (`/(tabs)`) with search, shelves, queue, player, likes
- **Backend** (`spotify-redux-be/`): FastAPI app (`api.py`)
  - Loads trained artifacts: `best_model.pkl`, `song_vectors.index`, `faiss_id_map.pkl`
  - Provides auth endpoints, catalog search, recommendations, likes
  - Persists users/likes/seeds to DB (SQLite by default, Postgres via `DATABASE_URL`)
- **AI artifacts**
  - Trained SVD model (Surprise) and FAISS index over item vectors (nearest neighbors)
  - Built offline and served read-only by the API

#### Request flow (high level)
1. User logs in → gets JWT token stored on device (SecureStore / web fallback).
2. If the account has no seeds (`/me/seeds` empty) → user is routed to `/onboarding`.
3. Onboarding posts seeds to `/recommendations/cold-start` which:
   - resolves seeds to catalog IDs
   - persists seeds for that user
   - returns immediate “You might like” list
4. Home fetches shelves from `/recommendations/categories` and queue from `/recommendations/similar`.

### 1.3 Development process (high level)
This project was built iteratively with emphasis on:
- **Functional product** (auth, onboarding, shelves, queue, likes, persistence)
- **Resilience** against metadata rate limits via fallbacks + caching
- **UX quality** (premium Spotify-like UI for login and onboarding)
- **Catalog consistency** (frontend displays only catalog songs; canonical ID resolution)

### 1.4 Model design (AI)

#### Data
Training input file: `spotify-redux-be/Last.fm_data.csv` (loaded as `Username`, `Artist`, `Track`).

#### Preprocessing
Implemented in `spotify-redux-be/main.py`:
- Create canonical song key: `song_id = Artist + " - " + Track`
- Aggregate play counts per user-song pair
- Transform play counts to an implicit “rating” via \(rating = \\log_{10}(play\\_count + 1)\)
- Optional random downsampling to `sample_size` for speed

#### Model
Algorithm: **Surprise SVD** (matrix factorization) trained on implicit ratings.

Hyperparameters are optimized using **Optuna** with a search space:
- `n_factors`: 20–100
- `n_epochs`: 10–30
- `lr_all`: 0.002–0.01
- `reg_all`: 0.02–0.1

#### Evaluation / model performance
Implemented in `spotify-redux-be/main.py` using Surprise `cross_validate`:
- **3-fold cross validation**
- Metric: **RMSE** over the implicit ratings
- Each Optuna trial logs its RMSE to **MLflow** (`mlflow.db`)

The “best” trial is selected by lowest mean RMSE, then retrained on the full dataset and saved as `best_model.pkl`.

> Note: RMSE is a proxy metric for implicit-feedback recommender quality. Online/user relevance can still differ (known limitation).

#### Serving-time recommendation algorithm
Core idea:
- Extract item vectors from `model.qi` (SVD item factors).
- Build a FAISS `IndexFlatL2` over those vectors.
- Cold-start: compute a pseudo-user vector as the (weighted) average of selected seed vectors (+ optional liked vectors) and retrieve nearest neighbors from FAISS.

Artifacts built in `spotify-redux-be/build_vector_db.py`:
- `song_vectors.index`
- `faiss_id_map.pkl`

### 1.5 Key engineering decisions

#### FastAPI + JWT
Reason: simple, widely supported client auth story; stateless API.

#### One-time onboarding per account
Seeds are persisted in the user row (`seed_selection`) so onboarding only occurs once per account.

#### Search constrained to catalog
Search uses `GET /catalog/search` which searches the model’s internal catalog only. This prevents UI from showing tracks not supported by the recommender model.

#### Metadata via public providers + caching
Cover art is fetched via public APIs (iTunes, Deezer, MusicBrainz) with caching to reduce repeated calls and mitigate rate-limits.

### 1.6 Known limitations
- **Recommendation relevance** depends heavily on the learned latent space; nearest neighbors may not always feel “sonically” similar.
- **Metadata providers** can rate-limit or return incomplete data; app degrades gracefully but covers may be missing.
- **Implicit ratings (log play counts)** are not a perfect proxy for preference.
- **FAISS IndexFlatL2** is exact but can be memory heavy for very large catalogs; a more scalable index may be needed for large-scale production.

## 2) User & technical documentation

### 2.1 System usage (user)
1. Open the app.
2. Create an account or log in.
3. If first time on the account, you will be taken to **Cold start onboarding**:
   - search songs in the catalog
   - select 3–5 songs
   - press **Continue** to get immediate “You might like” recommendations
4. Home:
   - Search at the top
   - “For you” shelves
   - Mini-player with queue and like controls

### 2.2 Running the system (developer)

#### Backend (local)
From repo root:
```bash
cd spotify-redux-be
pip install -r requirements.txt
uvicorn api:app --reload
```

#### Frontend (local)
From repo root:
```bash
cd spotify-redux-fe
npm install
setx EXPO_PUBLIC_API_BASE "http://localhost:8000"
npm start
```

> Important: if running `uvicorn` from the repo root, use `--app-dir spotify-redux-be`.

### 2.3 Docker / Compose (dev stack)
From repo root:
```bash
docker compose up --build
```

Services:
- backend: `http://localhost:8000`
- frontend (Expo): `http://localhost:19006`

### 2.4 Configuration
Backend env vars (see `spotify-redux-be/api.py`):
- `API_SECRET_KEY`
- `TOKEN_EXPIRE_HOURS`
- `LIKED_WEIGHT`
- `DATABASE_URL` (optional; if unset, uses SQLite)

Frontend env vars:
- `EXPO_PUBLIC_API_BASE`

### 2.5 Maintenance procedures
- **Retrain model**: run `spotify-redux-be/main.py` (writes `best_model.pkl`)
- **Rebuild index**: run `spotify-redux-be/build_vector_db.py` (writes FAISS index + map)
- **Deploy artifacts**: version and mount/copy them for the serving backend
- **DB maintenance**:
  - SQLite: ensure file persistence (volume)
  - Postgres: backups via `pg_dump` or managed DB snapshots

### 2.6 Optimal deployment & scaling (summary)
This section is fully inlined below to keep this document self-contained.

## 3) Deployment (single VM, Docker)

This targets a **single VM** deployment (e.g., DigitalOcean droplet, AWS EC2) using Docker + Compose.

### 3.1 Recommended architecture

- **Nginx**: TLS termination + reverse proxy to the API
- **Backend API**: FastAPI with multiple workers
- **Database**: Postgres (recommended for production)
- **Artifacts**: model + FAISS index stored on disk (or pulled from object storage on deploy)

### 3.2 Provision a VM

- Install Docker + Compose plugin
- Open ports: `80/443` (Nginx), optional `22` (SSH)

### 3.3 Configure environment

Create a `.env` next to your production `docker-compose.yml`:

```bash
API_SECRET_KEY=replace-with-a-strong-secret
TOKEN_EXPIRE_HOURS=24
LIKED_WEIGHT=2.0

# Postgres (recommended)
POSTGRES_USER=spotify
POSTGRES_PASSWORD=spotify
POSTGRES_DB=spotify
DATABASE_URL=postgresql+psycopg://spotify:spotify@db:5432/spotify

# Frontend clients should call this
EXPO_PUBLIC_API_BASE=https://your-domain.example
```

### 3.4 Compose for production

- Backend: run gunicorn instead of uvicorn `--reload`
- Frontend: typically you **do not** run the Expo dev server in prod

Recommended backend command:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 3 -b 0.0.0.0:8000 api:app
```

(Workers guideline: \(2 * CPU + 1\).)

### 3.5 Nginx reverse proxy (optional but recommended)

```nginx
server {
  listen 80;
  server_name your-domain.example;
  location / {
    proxy_pass http://backend:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }
}
```
Use Let’s Encrypt/certbot or managed TLS.

### 3.6 Database + backups
- Postgres volume for persistence.
- Backups: periodic `pg_dump` to object storage or infra snapshots.

### 3.7 Artifacts in production
- Required: `best_model.pkl`, `song_vectors.index`, `faiss_id_map.pkl`
- Recommended: build in CI → publish to object storage → pull/mount read-only in the backend container.

### 3.8 Observability
- Health: `GET /health`
- Logs: container stdout/stderr
- Consider: structured logs (JSON), request IDs, basic metrics (Prometheus) if you scale out.

## 4) Scaling & AI artifacts (training vs serving)

Two workloads:
- **Training / artifact building** (offline, heavy CPU/memory, infrequent)
- **Serving** (online, latency-sensitive, horizontally scalable)

Keep the API stateless; persist users/likes/seeds in DB; version artifacts.

### 4.1 Artifacts to version (must be consistent)
- `best_model.pkl`
- `song_vectors.index`
- `faiss_id_map.pkl`

If any are out of sync, recommendations may fail or degrade.

### 4.2 Recommended pipeline
1) Train: `spotify-redux-be/main.py` → `best_model.pkl`
2) Build FAISS: `spotify-redux-be/build_vector_db.py` → index + id map
3) Validate: simple neighbor sanity check (seed not returned; neighbors look plausible)
4) Publish: versioned location, e.g. `artifacts/2026-01-12/...` in object storage
5) Deploy: mount artifacts read-only; point backend to them (default paths or mounted paths)

### 4.3 Serving performance tuning
- **Stateless API**: DB for users/likes, artifacts read-only
- **Warm startup**: preload model + FAISS; multi-worker gunicorn
- **Metadata hot path**: cover fetching can be slow; use caching (Redis for multi-instance) or prefetch table; degrade gracefully if rate-limited.

### 4.4 Horizontal scaling model
- Multiple FastAPI replicas reading the same artifacts
- One Postgres instance (managed recommended)
- Optional Redis for shared caching

### 4.5 Memory sizing (rule of thumb)
- Driven by `best_model.pkl` size + FAISS index size
- Allow 1–2 GB for a small demo; more if caching metadata

### 4.6 When you outgrow a single VM
- Move to managed Postgres + object storage for artifacts
- Use a container platform (Kubernetes or Cloud Run)
- Keep serving stateless and artifacts versioned/consistent
