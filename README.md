# Spotify Redux (AI-powered music recommender)

An end-to-end “Spotify-like” demo app:

- **Backend**: FastAPI API serving recommendations from an SVD latent space + FAISS nearest-neighbor index, with JWT auth, cold-start onboarding, and likes.
- **Frontend**: Expo (React Native + Expo Router) with premium Spotify-like UI, search, mini-player, queue, and one-time cold-start onboarding per account.

## Repo layout

- `spotify-redux-be/` — FastAPI backend + model artifacts + training/index scripts
- `spotify-redux-fe/` — Expo app (mobile + web)
- `docs/` — deployment/scaling docs (added in this repo)

## Quickstart (Docker, full dev stack)

Prereqs:
- Docker Desktop (or Docker Engine) with Compose

From the repo root:

```bash
docker compose up --build
```

This starts:
- **Backend** on `http://localhost:8000`
- **Expo dev server** (frontend) on `http://localhost:19006` (web) and Metro on `:8081`

### Environment variables (recommended)

Create a `.env` file in the repo root (Compose will load it):

```bash
API_SECRET_KEY=change-me
TOKEN_EXPIRE_HOURS=24
LIKED_WEIGHT=2.0

# Optional: switch backend DB from SQLite to Postgres
# DATABASE_URL=postgresql+psycopg://spotify:spotify@db:5432/spotify

# Frontend: backend URL that the app should call
EXPO_PUBLIC_API_BASE=http://localhost:8000
```

## Local dev (without Docker)

### Backend (FastAPI)

```bash
cd spotify-redux-be
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn api:app --reload
```

Backend runs on `http://localhost:8000` and stores user data in `spotify-redux-be/users.db` by default.

### Frontend (Expo)

```bash
cd spotify-redux-fe
npm install
setx EXPO_PUBLIC_API_BASE "http://localhost:8000"
npm start
```

## Common troubleshooting

- **The app can’t reach the backend from a phone**:
  - `EXPO_PUBLIC_API_BASE` must be reachable from the device. Use your LAN IP, e.g. `http://192.168.1.50:8000`.
- **iTunes metadata failures (403/429)**:
  - The backend uses fallbacks (Deezer/MusicBrainz) and caching, but public APIs can still rate-limit.
- **Cold start doesn’t show**:
  - For logged-in users, cold start is a **one-time onboarding screen** and is skipped once seeds exist (`/me/seeds`).

## Docs

- Deployment (single VM + Docker): `docs/deployment.md`
- Scaling + AI artifacts (train vs serve): `docs/scaling-ai.md`

