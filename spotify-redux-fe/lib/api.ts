export type SearchResult = {
  song_id: string;
  title: string;
  artist: string;
  album?: string | null;
  cover_url?: string | null;
  preview_url?: string | null;
};

export type RecommendationItem = {
  song_id: string;
  title: string;
  artist: string;
  cover_url?: string | null;
  distance?: number | null;
};

export type RecommendationResponse = {
  results: RecommendationItem[];
  seed_count: number;
};

export type RecommendationBucket = {
  label: string;
  seed_used?: string | null;
  items: RecommendationItem[];
};

export type CategoriesResponse = {
  buckets: RecommendationBucket[];
};

export type SeedsResponse = {
  seeds: string[] | null;
};

export type LikesResponse = {
  likes: string[];
};

export type AuthResponse = {
  access_token: string;
  token_type: string;
};

const API_BASE = process.env.EXPO_PUBLIC_API_BASE ?? 'http://localhost:8000';

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export async function searchTracks(query: string, limit = 8): Promise<SearchResult[]> {
  if (!query.trim()) return [];
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  return apiFetch<SearchResult[]>(`/catalog/search?${params.toString()}`);
}

export async function getColdStartRecommendations(
  seeds: string[],
  limit = 10,
  token?: string
): Promise<RecommendationResponse> {
  return apiFetch<RecommendationResponse>(`/recommendations/cold-start`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ seeds, limit }),
  });
}

export async function getSimilarRecommendations(song: string, limit = 10, token?: string): Promise<RecommendationResponse> {
  return apiFetch<RecommendationResponse>(`/recommendations/similar`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ song, limit }),
  });
}

export async function getCategories(token?: string, seed?: string, limit = 8): Promise<CategoriesResponse> {
  const params = new URLSearchParams();
  if (seed) params.append('seed', seed);
  params.append('limit', String(limit));
  return apiFetch<CategoriesResponse>(`/recommendations/categories?${params.toString()}`, {
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });
}

export async function getSeeds(token?: string): Promise<SeedsResponse> {
  return apiFetch<SeedsResponse>(`/me/seeds`, {
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });
}

export async function getLikes(token: string): Promise<LikesResponse> {
  return apiFetch<LikesResponse>(`/likes`, {
    headers: { Authorization: `Bearer ${token}` },
  });
}

export async function toggleLike(token: string, song_id: string): Promise<{ liked: boolean }> {
  return apiFetch<{ liked: boolean }>(`/likes/toggle`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ song_id }),
  });
}

export async function login(username: string, password: string): Promise<AuthResponse> {
  const body = new URLSearchParams();
  body.append('username', username);
  body.append('password', password);
  return apiFetch<AuthResponse>(`/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  });
}

export async function signup(username: string, password: string): Promise<AuthResponse> {
  return apiFetch<AuthResponse>(`/auth/signup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
}
