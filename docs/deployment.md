## Deployment (single VM, Docker)

This guide targets a **single VM** deployment (e.g., DigitalOcean droplet, AWS EC2) using Docker + Compose.

### Recommended architecture

- **Nginx**: TLS termination + reverse proxy to the API
- **Backend API**: FastAPI with multiple workers
- **Database**: Postgres (recommended for production)
- **Artifacts**: model + FAISS index stored on disk (or pulled from object storage on deploy)

## 1) Provision a VM

- Install Docker + Compose plugin
- Open ports:
  - `80/443` (Nginx)
  - Optional: `22` (SSH)

## 2) Configure environment

Create a `.env` file next to your production `docker-compose.yml`:

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

## 3) Compose for production

You can start from the repo’s dev `docker-compose.yml` and adjust:

- **backend**: run gunicorn instead of uvicorn `--reload`
- **frontend**: in production you usually don’t run the Expo dev server

Recommended backend command:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 3 -b 0.0.0.0:8000 api:app
```

Rule of thumb for workers: \(2 * CPU + 1\).

## 4) Nginx reverse proxy (optional but recommended)

Example `nginx.conf` snippet:

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

For TLS, use Let’s Encrypt + certbot or your platform’s managed TLS.

## 5) Database + backups

- Use Postgres volume for persistence.
- Backups:
  - periodic `pg_dump` to object storage
  - or snapshots at the infrastructure level

## 6) Artifacts in production

Ensure the backend container can read:
- `best_model.pkl`
- `song_vectors.index`
- `faiss_id_map.pkl`

Recommended approach:
- Build artifacts in CI (or on a dedicated training machine)
- Publish to object storage (S3-compatible)
- Pull into the VM and mount read-only into the backend container

See `docs/scaling-ai.md` for the full artifact pipeline.

## Observability

- Health: `GET /health`
- Logs: container stdout/stderr
- Consider adding:
  - structured logs (JSON)
  - request IDs
  - basic metrics (Prometheus) if you scale out

