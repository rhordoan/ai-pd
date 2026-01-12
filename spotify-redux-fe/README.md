# spotify-redux-fe (Expo frontend)

Expo (React Native) client for the Spotify Redux recommender.

Features:
- Catalog-only search
- Elite login/signup modal
- One-time cold-start onboarding for logged-in users
- “For you” shelves + “Up next” queue
- Likes, persistent player state

## Run locally

```bash
npm install
```

Set the backend base URL for your environment:

### Web (local)

```bash
setx EXPO_PUBLIC_API_BASE "http://localhost:8000"
npm start
```

### Mobile device on LAN

Use your machine LAN IP so your phone can reach it:

```bash
setx EXPO_PUBLIC_API_BASE "http://192.168.1.50:8000"
npm start
```

## Routes

- `/login` — login/signup (modal)
- `/onboarding` — cold start onboarding (one-time per account)
- `/(tabs)` — main app tabs

## Auth storage

Auth token is persisted using `expo-secure-store` when available, with a safe fallback on web.

## Troubleshooting

- **App can’t reach backend**: `EXPO_PUBLIC_API_BASE` must be reachable from the device/browser.
- **Expo dev server in Docker**: use LAN host mode and ensure the ports are published (see root `docker-compose.yml`).

