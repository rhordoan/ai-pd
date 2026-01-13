import json
import os
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pickle
import requests
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField
from sqlmodel import Session, SQLModel, create_engine, select


# ==========================================
# Paths & constants
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
INDEX_PATH = BASE_DIR / "song_vectors.index"
MAP_PATH = BASE_DIR / "faiss_id_map.pkl"
SQLITE_PATH = os.getenv("SQLITE_PATH")
DB_PATH = Path(SQLITE_PATH) if SQLITE_PATH else (BASE_DIR / "data" / "users.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SECRET_KEY = os.getenv("API_SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("TOKEN_EXPIRE_HOURS", "24"))
LIKED_WEIGHT = float(os.getenv("LIKED_WEIGHT", "2.0"))

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
DEEZER_SEARCH_URL = "https://api.deezer.com/search"
MUSICBRAINZ_SEARCH_URL = "https://musicbrainz.org/ws/2/recording"
COVER_ART_URL = "https://coverartarchive.org/release"

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    # If using SQLite via env, keep check_same_thread disabled for FastAPI
    connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    db_engine = create_engine(DATABASE_URL, connect_args=connect_args)
else:
    db_engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
cover_cache: dict[str, Optional[str]] = {}
MAX_COVER_CACHE = 2000


# ==========================================
# Models
# ==========================================
class User(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    username: str = SQLField(index=True, sa_column_kwargs={"unique": True})
    hashed_password: str
    seed_selection: Optional[str] = SQLField(default=None, description="JSON array")


class Like(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(foreign_key="user.id", index=True)
    song_id: str = SQLField(index=True)
    created_at: datetime = SQLField(default_factory=datetime.utcnow, index=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=4, max_length=128)


class SearchResult(BaseModel):
    song_id: str
    title: str
    artist: str
    album: Optional[str] = None
    cover_url: Optional[str] = None
    preview_url: Optional[str] = None


class RecommendationItem(BaseModel):
    song_id: str
    title: str
    artist: str
    cover_url: Optional[str] = None
    distance: Optional[float] = None


class RecommendationBucket(BaseModel):
    label: str
    seed_used: Optional[str] = None
    items: List[RecommendationItem]


class ColdStartRequest(BaseModel):
    seeds: List[str]
    limit: int = Field(default=10, ge=1, le=50)


class SimilarRequest(BaseModel):
    song: str
    limit: int = Field(default=10, ge=1, le=50)
    seed: Optional[str] = None  # optional for deterministic flavor


def stable_indices_from_seed(total: int, k: int, seed: Optional[str]) -> List[int]:
    if total <= 0 or k <= 0:
        return []
    if seed:
        rng = np.random.default_rng(abs(hash(seed)) % (2**32))
    else:
        rng = np.random.default_rng()
    # sample without replacement but capped by total
    k = min(k, total)
    return rng.choice(total, size=k, replace=False).tolist()


class RecommendationResponse(BaseModel):
    results: List[RecommendationItem]
    seed_count: int


class CategoriesResponse(BaseModel):
    buckets: List[RecommendationBucket]


class LikesResponse(BaseModel):
    likes: List[str]


class ToggleLikeRequest(BaseModel):
    song_id: str


# ==========================================
# Helpers: auth & DB
# ==========================================
def get_session():
    with Session(db_engine) as session:
        yield session


def get_user_likes(session: Session, user: Optional[User]) -> List[str]:
    if not user:
        return []
    likes = session.exec(select(Like.song_id).where(Like.user_id == user.id)).all()
    return list(likes)


def verify_password(plain: str, hashed: str) -> bool:
    # bcrypt ignores everything after 72 bytes; reject overly long input explicitly
    if len(plain.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password too long for bcrypt (max 72 bytes)",
        )
    return pwd_context.verify(plain, hashed)


def hash_password(password: str) -> str:
    if len(password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password too long for bcrypt (max 72 bytes)",
        )
    return pwd_context.hash(password)


def create_access_token(subject: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode = {"sub": subject, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> Optional[User]:
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None

    user = session.exec(select(User).where(User.username == username)).first()
    return user


# ==========================================
# Recommender core
# ==========================================
class Recommender:
    def __init__(self):
        self.model = None
        self.index = None
        self.id_map: List[str] = []
        self._normalized_to_song_id: dict[str, str] = {}
        self._search_corpus: List[tuple[str, str]] = []

    def ensure_loaded(self):
        if self.model is not None and self.index is not None and self.id_map:
            return
        self.model = self._load_model()
        self.index, self.id_map = self._load_index_and_map(self.model)
        self._normalized_to_song_id = self._build_normalized_index(self.id_map)
        self._search_corpus = self._build_search_corpus(self.id_map)

    def _norm_text(self, s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = s.replace("???", "'")
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def _strip_featured(self, track: str) -> str:
        t = track
        # Remove (feat. ...), [feat. ...], (ft. ...), etc.
        t = re.sub(r"[\(\[\{]\s*(feat\.?|ft\.?|featuring)\b.*?[\)\]\}]", "", t, flags=re.IGNORECASE)
        # Remove trailing "feat. X" / "ft. X" / "featuring X"
        t = re.sub(r"\s*-\s*(feat\.?|ft\.?|featuring)\b.*$", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+(feat\.?|ft\.?|featuring)\b.*$", "", t, flags=re.IGNORECASE)
        # Remove remix/version tags in parentheses as a light normalization (keeps main title)
        t = re.sub(r"[\(\[\{]\s*(remix|mix|version|edit|live|remastered|mono|stereo)\b.*?[\)\]\}]", "", t, flags=re.IGNORECASE)
        return t

    def _normalize_song_key(self, artist: str, track: str) -> str:
        a = self._norm_text(artist)
        t = self._norm_text(self._strip_featured(track))
        # remove most punctuation
        a = re.sub(r"[^\w\s]", "", a)
        t = re.sub(r"[^\w\s]", "", t)
        a = re.sub(r"\s+", " ", a).strip()
        t = re.sub(r"\s+", " ", t).strip()
        return f"{a} - {t}"

    def _build_normalized_index(self, song_ids: List[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for sid in song_ids:
            artist, track = self._song_to_artist_track(sid)
            if not artist or not track:
                continue
            key = self._normalize_song_key(artist, track)
            # keep first encountered for determinism
            mapping.setdefault(key, sid)
        return mapping

    def _build_search_corpus(self, song_ids: List[str]) -> List[tuple[str, str]]:
        corpus: List[tuple[str, str]] = []
        for sid in song_ids:
            artist, track = self._song_to_artist_track(sid)
            if not artist or not track:
                continue
            key = self._normalize_song_key(artist, track)
            corpus.append((sid, key))
        return corpus

    def _load_model(self):
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. Run training to generate best_model.pkl."
            )
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    def _load_index_and_map(self, model):
        if INDEX_PATH.exists():
            idx = faiss.read_index(str(INDEX_PATH))
        else:
            item_vectors = model.qi.astype("float32")
            dimension = item_vectors.shape[1]
            idx = faiss.IndexFlatL2(dimension)
            idx.add(item_vectors)

        if MAP_PATH.exists():
            with open(MAP_PATH, "rb") as f:
                id_map = pickle.load(f)
        else:
            id_map = [model.trainset.to_raw_iid(i) for i in range(model.qi.shape[0])]

        if idx.ntotal != len(id_map):
            raise RuntimeError(
                f"FAISS index size ({idx.ntotal}) does not match id map ({len(id_map)}). Rebuild artifacts."
            )

        return idx, id_map

    def _song_to_artist_track(self, song_id: str):
        if " - " not in song_id:
            return None, None
        artist, track = song_id.split(" - ", 1)
        return artist.strip(), track.strip()

    def _vector_for_song(self, song_id: str):
        self.ensure_loaded()
        try:
            inner_id = self.model.trainset.to_inner_iid(song_id)
        except ValueError:
            return None
        return self.model.qi[inner_id]

    def has_song(self, song_id: str) -> bool:
        return self._vector_for_song(song_id) is not None

    def resolve_song_id(self, song_id: str) -> Optional[str]:
        """
        Map an external-ish song_id (possibly with feat/remix punctuation) to a canonical
        song_id in the model catalogue.
        """
        self.ensure_loaded()
        if self.has_song(song_id):
            return song_id
        artist, track = self._song_to_artist_track(song_id)
        if not artist or not track:
            return None
        key = self._normalize_song_key(artist, track)
        return self._normalized_to_song_id.get(key)

    def recommend_from_seeds(
        self,
        seeds: List[str],
        k: int = 10,
        liked_song_ids: Optional[List[str]] = None,
        liked_weight: float = LIKED_WEIGHT,
        seed_weights: Optional[dict[str, float]] = None,
    ) -> List[RecommendationItem]:
        self.ensure_loaded()
        valid_vectors = []
        weights = []
        seed_set = set(seeds)

        for s in seeds:
            vec = self._vector_for_song(s)
            if vec is not None:
                valid_vectors.append(vec)
                weights.append(float(seed_weights.get(s, 1.0)) if seed_weights else 1.0)

        if liked_song_ids:
            for s in liked_song_ids:
                vec = self._vector_for_song(s)
                if vec is not None:
                    valid_vectors.append(vec)
                    weights.append(liked_weight)

        if not valid_vectors:
            raise ValueError("No valid seeds found in model catalogue.")

        weights_arr = np.array(weights) if weights else None
        pseudo_user = (
            np.average(valid_vectors, axis=0, weights=weights_arr).astype("float32")
            if weights_arr is not None
            else np.mean(valid_vectors, axis=0).astype("float32")
        )
        query = np.expand_dims(pseudo_user, axis=0)
        distances, indices = self.index.search(query, k + len(seed_set))

        recs: List[RecommendationItem] = []
        for idx, dist in zip(indices[0], distances[0]):
            song_name = self.id_map[idx]
            if song_name in seed_set:
                continue
            artist, track = self._song_to_artist_track(song_name)
            recs.append(
                RecommendationItem(
                    song_id=song_name,
                    artist=artist or "",
                    title=track or song_name,
                    distance=float(dist),
                )
            )
            if len(recs) >= k:
                break
        return recs

    def build_profile_vector(
        self,
        seeds: List[str],
        *,
        liked_song_ids: Optional[List[str]] = None,
        liked_weight: float = LIKED_WEIGHT,
        seed_weights: Optional[dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Build a pseudo-user vector from seeds + likes (weighted).
        Returns a float32 vector suitable for FAISS queries.
        """
        self.ensure_loaded()
        valid_vectors: List[np.ndarray] = []
        weights: List[float] = []

        for s in seeds:
            vec = self._vector_for_song(s)
            if vec is not None:
                valid_vectors.append(vec)
                weights.append(float(seed_weights.get(s, 1.0)) if seed_weights else 1.0)

        if liked_song_ids:
            for s in liked_song_ids:
                vec = self._vector_for_song(s)
                if vec is not None:
                    valid_vectors.append(vec)
                    weights.append(float(liked_weight))

        if not valid_vectors:
            raise ValueError("No valid seeds found in model catalogue.")

        w = np.array(weights, dtype="float32")
        pseudo = np.average(np.stack(valid_vectors, axis=0), axis=0, weights=w).astype("float32")
        return pseudo

    def search_candidates_by_vector(
        self,
        vec: np.ndarray,
        *,
        k: int,
        exclude_song_ids: Optional[set[str]] = None,
    ) -> List[tuple[int, float]]:
        """
        Return raw (faiss_index, distance) candidates for a given vector.
        """
        self.ensure_loaded()
        exclude_song_ids = exclude_song_ids or set()
        query = np.expand_dims(vec.astype("float32"), axis=0)
        distances, indices = self.index.search(query, k + len(exclude_song_ids))
        out: List[tuple[int, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            sid = self.id_map[idx]
            if sid in exclude_song_ids:
                continue
            out.append((int(idx), float(dist)))
            if len(out) >= k:
                break
        return out

    def similar_to_song(
        self,
        song_id: str,
        k: int = 10,
        liked_song_ids: Optional[List[str]] = None,
        liked_weight: float = LIKED_WEIGHT,
    ) -> List[RecommendationItem]:
        self.ensure_loaded()
        base_vec = self._vector_for_song(song_id)
        if base_vec is None:
            raise ValueError(f"Song '{song_id}' not in model catalogue.")

        vectors = [base_vec]
        weights = [1.0]
        if liked_song_ids:
            for s in liked_song_ids:
                vec = self._vector_for_song(s)
                if vec is not None:
                    vectors.append(vec)
                    weights.append(liked_weight)

        query_vec = (
            np.average(vectors, axis=0, weights=np.array(weights)).astype("float32")
            if weights
            else base_vec.astype("float32")
        )
        query = np.expand_dims(query_vec, axis=0)
        distances, indices = self.index.search(query, k + 1)  # include self
        recs: List[RecommendationItem] = []
        for idx, dist in zip(indices[0], distances[0]):
            song_name = self.id_map[idx]
            if song_name == song_id:
                continue
            artist, track = self._song_to_artist_track(song_name)
            recs.append(
                RecommendationItem(
                    song_id=song_name,
                    artist=artist or "",
                    title=track or song_name,
                    distance=float(dist),
                )
            )
            if len(recs) >= k:
                break
        return recs


def diversify_recommendations(
    items: List[RecommendationItem],
    limit: int,
    *,
    min_title_len: int = 3,
    max_per_artist: int = 2,
    no_adjacent_same_artist: bool = True,
) -> List[RecommendationItem]:
    """
    Simple heuristics to improve perceived quality:
    - filter very short titles
    - cap repeats per artist
    - avoid consecutive same-artist recommendations
    """
    picked: List[RecommendationItem] = []
    artist_counts: dict[str, int] = {}
    last_artist: Optional[str] = None

    for r in items:
        title = (r.title or "").strip()
        if len(title) < min_title_len:
            continue
        artist = (r.artist or "").strip()
        if not artist:
            continue
        if no_adjacent_same_artist and last_artist and artist.lower() == last_artist.lower():
            continue
        cnt = artist_counts.get(artist, 0)
        if cnt >= max_per_artist:
            continue
        picked.append(r)
        artist_counts[artist] = cnt + 1
        last_artist = artist
        if len(picked) >= limit:
            break

    return picked


def blend_ranked_lists(
    primary: List[RecommendationItem],
    secondary: List[RecommendationItem],
    *,
    limit: int,
    primary_take: int = 2,
    secondary_take: int = 1,
) -> List[RecommendationItem]:
    """
    Interleave two ranked lists while deduping by song_id.
    Intended use: keep "up next" anchored to the current-song neighborhood,
    but still inject profile personalization to avoid a boring local loop.
    """
    out: List[RecommendationItem] = []
    seen: set[str] = set()
    i = 0
    j = 0
    while len(out) < limit and (i < len(primary) or j < len(secondary)):
        for _ in range(primary_take):
            if len(out) >= limit:
                break
            while i < len(primary) and primary[i].song_id in seen:
                i += 1
            if i < len(primary):
                out.append(primary[i])
                seen.add(primary[i].song_id)
                i += 1
        for _ in range(secondary_take):
            if len(out) >= limit:
                break
            while j < len(secondary) and secondary[j].song_id in seen:
                j += 1
            if j < len(secondary):
                out.append(secondary[j])
                seen.add(secondary[j].song_id)
                j += 1
    return out


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v
    return (v / n).astype("float32")


def build_seed_neighbor_candidate_indices(
    *,
    recommender: Recommender,
    seed_song_ids: List[str],
    per_seed_k: int,
    exclude_song_ids: Optional[set[str]] = None,
    min_seed_hits: int = 1,
) -> List[int]:
    """
    Build a candidate pool by taking top-K neighbors for EACH seed song.
    This tends to produce much more semantically coherent candidates than a single centroid search.
    """
    recommender.ensure_loaded()
    exclude_song_ids = exclude_song_ids or set()
    # idx -> (hit_count, best_distance)
    stats: dict[int, tuple[int, float]] = {}

    for sid in seed_song_ids:
        if not sid or sid in exclude_song_ids:
            continue
        vec = recommender._vector_for_song(sid)
        if vec is None:
            continue
        raw = recommender.search_candidates_by_vector(vec, k=per_seed_k, exclude_song_ids=exclude_song_ids | {sid})
        for idx, _dist in raw:
            hits, best = stats.get(idx, (0, float("inf")))
            hits += 1
            best = min(best, float(_dist))
            stats[idx] = (hits, best)

    # Prefer candidates that appear in many seed lists (overlap)
    eligible = [(idx, hits, best) for idx, (hits, best) in stats.items() if hits >= max(1, min_seed_hits)]
    eligible.sort(key=lambda t: (-t[1], t[2], t[0]))
    return [idx for idx, _hits, _best in eligible]


def build_allowed_artists_from_seed_neighbors(
    *,
    recommender: Recommender,
    seed_song_ids: List[str],
    per_seed_k: int,
    exclude_song_ids: Optional[set[str]] = None,
    min_seed_hits: int = 2,
    max_artists: int = 60,
) -> Optional[set[str]]:
    """
    Build an artist allowlist by looking at which artists appear in the neighbor lists
    of multiple seeds. This is a cheap proxy for "related artists" without external metadata.
    """
    recommender.ensure_loaded()
    exclude_song_ids = exclude_song_ids or set()
    # artist -> number of seed-lists it appears in
    artist_hits: dict[str, int] = {}

    for sid in seed_song_ids:
        if not sid or sid in exclude_song_ids:
            continue
        vec = recommender._vector_for_song(sid)
        if vec is None:
            continue
        raw = recommender.search_candidates_by_vector(vec, k=per_seed_k, exclude_song_ids=exclude_song_ids | {sid})
        artists_in_this_seed: set[str] = set()
        for idx, _dist in raw:
            cand_sid = recommender.id_map[idx]
            artist, _track = recommender._song_to_artist_track(cand_sid)
            if artist:
                artists_in_this_seed.add(artist.strip())
        for a in artists_in_this_seed:
            artist_hits[a] = artist_hits.get(a, 0) + 1

    eligible = [(a, h) for a, h in artist_hits.items() if h >= max(1, min_seed_hits)]
    if not eligible:
        return None
    eligible.sort(key=lambda t: (-t[1], t[0].lower()))
    return {a for a, _h in eligible[:max_artists]}


def mmr_rerank(
    *,
    recommender: Recommender,
    candidate_indices: List[int],
    target_vec: np.ndarray,
    limit: int,
    exclude_song_ids: Optional[set[str]] = None,
    allowed_artists: Optional[set[str]] = None,
    lambda_rel: float = 0.88,
    min_title_len: int = 3,
    max_per_artist: int = 2,
    no_adjacent_same_artist: bool = True,
) -> List[RecommendationItem]:
    """
    Maximal Marginal Relevance re-ranking over a candidate pool.

    score = lambda * relevance(target, cand) - (1-lambda) * max_sim(cand, picked)
    where similarities are cosine similarities.
    """
    recommender.ensure_loaded()
    exclude_song_ids = exclude_song_ids or set()
    allowed_artists_lc: Optional[set[str]] = None
    if allowed_artists:
        allowed_artists_lc = {a.strip().lower() for a in allowed_artists if a and a.strip()}

    # Precompute candidate metadata and normalized vectors
    cand: List[tuple[str, str, str, float, np.ndarray]] = []
    for idx in candidate_indices:
        if idx < 0 or idx >= len(recommender.id_map):
            continue
        sid = recommender.id_map[idx]
        if sid in exclude_song_ids:
            continue
        artist, track = recommender._song_to_artist_track(sid)
        artist = (artist or "").strip()
        title = (track or sid).strip()
        if allowed_artists_lc is not None and artist.lower() not in allowed_artists_lc:
            continue
        if len(title) < min_title_len or not artist:
            continue
        vec = recommender.model.qi[idx]
        cand.append((sid, artist, title, 0.0, _l2_normalize(vec)))

    if not cand:
        return []

    t = _l2_normalize(target_vec)

    picked: List[RecommendationItem] = []
    picked_vecs: List[np.ndarray] = []
    seen: set[str] = set()
    artist_counts: dict[str, int] = {}
    last_artist: Optional[str] = None

    # Greedy MMR selection
    for _ in range(limit):
        best_i = -1
        best_score = -1e9

        for i, (sid, artist, title, _dist, v) in enumerate(cand):
            if sid in seen:
                continue
            if no_adjacent_same_artist and last_artist and artist.lower() == last_artist.lower():
                continue
            if artist_counts.get(artist, 0) >= max_per_artist:
                continue

            rel = float(np.dot(v, t))  # cosine similarity
            if picked_vecs:
                div = max(float(np.dot(v, pv)) for pv in picked_vecs)
            else:
                div = 0.0
            score = lambda_rel * rel - (1.0 - lambda_rel) * div
            if score > best_score:
                best_score = score
                best_i = i

        if best_i < 0:
            break

        sid, artist, title, dist, v = cand[best_i]
        picked.append(
            RecommendationItem(
                song_id=sid,
                artist=artist,
                title=title,
                distance=dist,
            )
        )
        picked_vecs.append(v)
        seen.add(sid)
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
        last_artist = artist

    return picked


recommender = Recommender()
app = FastAPI(title="Spotify Redux Recommender", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# External search helpers
# ==========================================
def normalize_song_id(artist: str, track: str) -> str:
    return f"{artist.strip()} - {track.strip()}"


def fetch_itunes_search(query: str, limit: int = 10) -> List[SearchResult]:
    params = {
        "term": query,
        "media": "music",
        "entity": "song",
        "limit": limit,
    }
    try:
        res = requests.get(ITUNES_SEARCH_URL, params=params, timeout=10)
        res.raise_for_status()
    except requests.HTTPError as exc:
        # Gracefully degrade on 403/429/etc.
        if res.status_code in (403, 429, 500, 502, 503):
            return []
        raise exc

    payload = res.json()
    results = []
    for item in payload.get("results", []):
        artist = item.get("artistName") or ""
        track = item.get("trackName") or ""
        album = item.get("collectionName")
        cover = item.get("artworkUrl100") or item.get("artworkUrl60")
        preview = item.get("previewUrl")
        song_id = normalize_song_id(artist, track)
        results.append(
            SearchResult(
                song_id=song_id,
                title=track,
                artist=artist,
                album=album,
                cover_url=cover,
                preview_url=preview,
            )
        )
    return results


def fetch_deezer_search(query: str, limit: int = 10) -> List[SearchResult]:
    params = {"q": query, "limit": limit}
    res = requests.get(DEEZER_SEARCH_URL, params=params, timeout=10)
    res.raise_for_status()
    payload = res.json()
    results = []
    for item in payload.get("data", []):
        artist = (item.get("artist") or {}).get("name") or ""
        track = item.get("title") or ""
        album = (item.get("album") or {}).get("title")
        cover = (item.get("album") or {}).get("cover_medium") or (item.get("album") or {}).get("cover")
        preview = item.get("preview")
        song_id = normalize_song_id(artist, track)
        results.append(
            SearchResult(
                song_id=song_id,
                title=track,
                artist=artist,
                album=album,
                cover_url=cover,
                preview_url=preview,
            )
        )
    return results


def fetch_musicbrainz_search(query: str, limit: int = 10) -> List[SearchResult]:
    params = {
        "query": query,
        "fmt": "json",
        "limit": limit,
    }
    res = requests.get(MUSICBRAINZ_SEARCH_URL, params=params, timeout=10, headers={"User-Agent": "spotify-redux/1.0"})
    res.raise_for_status()
    payload = res.json()
    results = []
    for item in payload.get("recordings", []):
        title = item.get("title") or ""
        artist_credit = item.get("artist-credit") or item.get("artist_credit") or []
        artist = ""
        if artist_credit and isinstance(artist_credit, list):
            name = artist_credit[0].get("name")
            if name:
                artist = name
        # take first release for cover lookup
        release_list = item.get("releases") or []
        release_id = release_list[0].get("id") if release_list else None
        song_id = normalize_song_id(artist, title)
        results.append(
            SearchResult(
                song_id=song_id,
                title=title,
                artist=artist,
                album=None,
                cover_url=None,  # will enrich via cover fetch
                preview_url=None,
            )
        )
        # stash release_id on the object for potential cover fetch (not in schema, so skip storing)
    return results


def fetch_music_search(query: str, limit: int = 10) -> List[SearchResult]:
    primary = fetch_itunes_search(query, limit)
    if primary:
        return primary
    try:
        fallback = fetch_deezer_search(query, limit)
        if fallback:
            return fallback
    except Exception:
        pass
    try:
        return fetch_musicbrainz_search(query, limit)
    except Exception:
        return []


def catalog_search(query: str, limit: int = 10) -> List[SearchResult]:
    recommender.ensure_loaded()
    q = query.strip()
    if not q:
        return []
    # Normalize query similar to song keys: "artist - track"
    qn = recommender._norm_text(q)
    qn = re.sub(r"[^\w\s]", " ", qn)
    qn = re.sub(r"\s+", " ", qn).strip()
    terms = [t for t in qn.split(" ") if t]
    if not terms:
        return []

    matches: List[str] = []
    for sid, key in recommender._search_corpus:
        ok = True
        for t in terms:
            if t not in key:
                ok = False
                break
        if ok:
            matches.append(sid)
            if len(matches) >= limit:
                break

    results: List[SearchResult] = []
    for sid in matches:
        artist, track = recommender._song_to_artist_track(sid)
        results.append(
            SearchResult(
                song_id=sid,
                title=track or sid,
                artist=artist or "",
                album=None,
                cover_url=fetch_cover_for_song(sid),
                preview_url=None,
            )
        )
    return results


def fetch_cover_for_song(song_id: str) -> Optional[str]:
    # Simple in-memory cache to avoid repeated upstream calls
    if song_id in cover_cache:
        return cover_cache[song_id]

    artist, track = recommender._song_to_artist_track(song_id)
    if not artist or not track:
        cover_cache[song_id] = None
        return None
    try:
        results = fetch_itunes_search(f"{artist} {track}", limit=1)
        if not results:
            results = fetch_deezer_search(f"{artist} {track}", limit=1)
        if results:
            cover = results[0].cover_url
            # Maintain a simple size cap
            if len(cover_cache) >= MAX_COVER_CACHE:
                cover_cache.pop(next(iter(cover_cache)))
            cover_cache[song_id] = cover
            return cover
    except Exception:
        cover_cache[song_id] = None
        return None

    # Try MusicBrainz Cover Art Archive
    try:
        artist, track = recommender._song_to_artist_track(song_id)
        if artist and track:
            mb_results = fetch_musicbrainz_search(f"{artist} {track}", limit=1)
            if mb_results:
                # Attempt cover art via release id inside recording
                params = {
                    "query": f'artist:{artist} AND recording:{track}',
                    "fmt": "json",
                    "limit": 1,
                }
                res = requests.get(MUSICBRAINZ_SEARCH_URL, params=params, timeout=10, headers={"User-Agent": "spotify-redux/1.0"})
                res.raise_for_status()
                data = res.json()
                recordings = data.get("recordings", [])
                if recordings:
                    rels = recordings[0].get("releases") or []
                    if rels:
                        rel_id = rels[0].get("id")
                        if rel_id:
                            art_res = requests.get(f"{COVER_ART_URL}/{rel_id}", timeout=10)
                            if art_res.status_code == 200:
                                art_json = art_res.json()
                                images = art_json.get("images") or []
                                if images:
                                    image_url = images[0].get("image")
                                    if image_url:
                                        if len(cover_cache) >= MAX_COVER_CACHE:
                                            cover_cache.pop(next(iter(cover_cache)))
                                        cover_cache[song_id] = image_url
                                        return image_url
    except Exception:
        pass
    cover_cache[song_id] = None
    return None


def enrich_with_covers(items: List[RecommendationItem]) -> List[RecommendationItem]:
    enriched = []
    for r in items:
        cover = fetch_cover_for_song(r.song_id)
        enriched.append(
            RecommendationItem(
                song_id=r.song_id,
                title=r.title,
                artist=r.artist,
                cover_url=cover,
                distance=r.distance,
            )
        )
    return enriched


def dedupe_limit(items: List[RecommendationItem], seen: set[str], limit: int, offset: int = 0):
    unique: List[RecommendationItem] = []
    skipped = 0
    for r in items:
        if r.song_id in seen:
            continue
        if skipped < offset:
            skipped += 1
            continue
        unique.append(r)
        seen.add(r.song_id)
        if len(unique) >= limit:
            break
    return unique


# ==========================================
# Routes
# ==========================================
@app.on_event("startup")
def startup_event():
    SQLModel.metadata.create_all(db_engine)
    # Warm up model/index once to fail fast if missing
    recommender.ensure_loaded()


@app.get("/health")
def health():
    return {"status": "ok", "songs_indexed": len(recommender.id_map)}


@app.post("/auth/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, session: Session = Depends(get_session)):
    existing = session.exec(select(User).where(User.username == payload.username)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    user = User(username=payload.username, hashed_password=hash_password(payload.password))
    session.add(user)
    session.commit()
    session.refresh(user)
    token = create_access_token(user.username)
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)
):
    user = session.exec(select(User).where(User.username == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    token = create_access_token(user.username)
    return TokenResponse(access_token=token)


@app.get("/search", response_model=List[SearchResult])
def search(q: str = "", limit: int = 10):
    if not q:
        return []
    try:
        # External search filtered to model catalogue only
        raw = fetch_music_search(q, limit=min(limit, 25))
        dedup: dict[str, SearchResult] = {}
        for r in raw:
            resolved = recommender.resolve_song_id(r.song_id)
            if not resolved:
                continue
            artist, track = recommender._song_to_artist_track(resolved)
            dedup.setdefault(
                resolved,
                SearchResult(
                    song_id=resolved,
                    title=track or r.title,
                    artist=artist or r.artist,
                    album=r.album,
                    cover_url=r.cover_url or fetch_cover_for_song(resolved),
                    preview_url=r.preview_url,
                ),
            )
        return list(dedup.values())[: min(limit, 25)]
    except requests.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Search upstream error: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/catalog/search", response_model=List[SearchResult])
def catalog_search_route(q: str = "", limit: int = 10):
    return catalog_search(q, limit=min(limit, 25))


@app.get("/cover")
def cover(artist: str, track: str):
    try:
        results = fetch_itunes_search(f"{artist} {track}", limit=1)
        return results[0].cover_url if results else None
    except Exception:
        return None


@app.get("/me/seeds")
def get_seeds(user: Optional[User] = Depends(get_current_user)):
    if not user:
        return {"seeds": None}
    return {"seeds": json.loads(user.seed_selection) if user.seed_selection else None}


@app.get("/likes", response_model=LikesResponse)
def list_likes(
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_current_user),
):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    likes = get_user_likes(session, user)
    return LikesResponse(likes=likes)


@app.post("/likes/toggle")
def toggle_like(
    payload: ToggleLikeRequest,
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_current_user),
):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    raw_song_id = payload.song_id.strip()
    song_id = recommender.resolve_song_id(raw_song_id)
    if not song_id:
        raise HTTPException(status_code=400, detail="Song not in catalogue")

    existing = session.exec(
        select(Like).where(Like.user_id == user.id, Like.song_id == song_id)
    ).first()
    if existing:
        session.delete(existing)
        session.commit()
        return {"liked": False}
    new_like = Like(user_id=user.id, song_id=song_id)
    session.add(new_like)
    session.commit()
    session.refresh(new_like)
    return {"liked": True}


@app.post("/recommendations/cold-start", response_model=RecommendationResponse)
def cold_start(
    payload: ColdStartRequest,
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_current_user),
):
    if user and user.seed_selection:
        # Cold start already done for this account; reuse stored seeds
        seeds = json.loads(user.seed_selection)
    else:
        incoming = [s.strip() for s in payload.seeds if s.strip()]
        # normalize/resolve seeds against model catalogue
        seeds = []
        for s in incoming:
            resolved = recommender.resolve_song_id(s)
            if resolved:
                seeds.append(resolved)
        if not seeds:
            raise HTTPException(status_code=400, detail="At least one seed is required.")
        # Persist seeds for the authenticated user on first cold start
        if user:
            user.seed_selection = json.dumps(seeds)
            session.add(user)
            session.commit()

    liked_ids = get_user_likes(session, user)
    try:
        recs = recommender.recommend_from_seeds(
            seeds, k=payload.limit, liked_song_ids=liked_ids if liked_ids else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    recs = diversify_recommendations(recs, payload.limit)

    with_covers = enrich_with_covers(recs)
    return RecommendationResponse(results=with_covers, seed_count=len(seeds))


@app.post("/recommendations/similar", response_model=RecommendationResponse)
def similar(
    payload: SimilarRequest,
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_current_user),
):
    try:
        liked_ids = get_user_likes(session, user)
        resolved = recommender.resolve_song_id(payload.song.strip())
        if not resolved:
            raise ValueError(f"Song '{payload.song.strip()}' not in model catalogue.")
        # Build a blended target vector: mostly current-song, plus a bit of profile taste.
        current_vec = recommender._vector_for_song(resolved)
        if current_vec is None:
            raise ValueError(f"Song '{resolved}' not in model catalogue.")

        seeds_profile: List[str] = [resolved]
        if user and user.seed_selection:
            try:
                seeds_profile.extend(json.loads(user.seed_selection) or [])
            except Exception:
                pass

        profile_vec = recommender.build_profile_vector(
            seeds_profile,
            liked_song_ids=liked_ids if liked_ids else None,
            seed_weights={resolved: 3.0},
        )
        target_vec = _l2_normalize(_l2_normalize(current_vec) * 0.78 + _l2_normalize(profile_vec) * 0.22)

        exclude = {resolved}
        # Candidate pool: strongest neighbors of current song PLUS a little from each saved seed/like (profile),
        # then MMR rerank to avoid weird jumps.

        ctx_raw = recommender.search_candidates_by_vector(
            current_vec, k=max(payload.limit * 90, 540), exclude_song_ids=exclude
        )
        ctx_candidates: List[int] = []
        seen_idx: set[int] = set()
        for idx, _d in ctx_raw:
            if idx in seen_idx:
                continue
            seen_idx.add(idx)
            ctx_candidates.append(idx)

        # Per-seed neighbors (small) to keep "taste" influence without derailing local continuity
        taste_seeds: List[str] = []
        if user and user.seed_selection:
            try:
                taste_seeds.extend(json.loads(user.seed_selection) or [])
            except Exception:
                pass
        # also sample a few liked songs to represent taste
        if liked_ids:
            taste_seeds.extend(liked_ids[:3])
        taste_seeds = [s for s in taste_seeds if s and s != resolved]

        taste_candidates = build_seed_neighbor_candidate_indices(
            recommender=recommender,
            seed_song_ids=taste_seeds[:6],
            per_seed_k=40,
            exclude_song_ids=exclude,
        )

        # Small profile pool as glue
        prof_raw = recommender.search_candidates_by_vector(
            profile_vec, k=max(payload.limit * 25, 160), exclude_song_ids=exclude
        )
        prof_candidates: List[int] = []
        for idx, _d in prof_raw:
            if idx in seen_idx:
                continue
            seen_idx.add(idx)
            prof_candidates.append(idx)

        combined: List[int] = ctx_candidates + [i for i in taste_candidates if i not in seen_idx] + prof_candidates

        # Optional artist allowlist from current-song neighborhood (keeps "up next" coherent)
        allowed_artists: Optional[set[str]] = None
        try:
            top_ctx = ctx_candidates[:220]
            arts: set[str] = set()
            for idx in top_ctx:
                sid = recommender.id_map[idx]
                a, _t = recommender._song_to_artist_track(sid)
                if a:
                    arts.add(a.strip())
            allowed_artists = arts if arts else None
        except Exception:
            allowed_artists = None

        recs = mmr_rerank(
            recommender=recommender,
            candidate_indices=combined,
            target_vec=target_vec,
            limit=payload.limit,
            exclude_song_ids=exclude,
            allowed_artists=allowed_artists,
            lambda_rel=0.90,
            max_per_artist=1,
            no_adjacent_same_artist=True,
        )

        # If MMR gets too strict, fall back to the old blend+filter logic.
        if len(recs) < max(3, payload.limit // 2):
            context = recommender.similar_to_song(
                resolved,
                k=max(payload.limit * 10, 40),
                liked_song_ids=liked_ids if liked_ids else None,
            )
            profile = recommender.recommend_from_seeds(
                seeds_profile,
                k=max(payload.limit * 14, 60),
                liked_song_ids=liked_ids if liked_ids else None,
                seed_weights={resolved: 3.0},
            )
            context = [r for r in context if r.song_id != resolved]
            profile = [r for r in profile if r.song_id != resolved]
            recs = blend_ranked_lists(context, profile, limit=max(payload.limit * 6, 24))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Optional deterministic shuffle/sampling using seed
    if payload.seed:
        order = stable_indices_from_seed(len(recs), len(recs), payload.seed)
        recs = [recs[i] for i in order]

    # Diversify + trim down to requested size
    recs = diversify_recommendations(recs, payload.limit, max_per_artist=2, no_adjacent_same_artist=True)

    with_covers = enrich_with_covers(recs)
    return RecommendationResponse(results=with_covers, seed_count=1)


@app.get("/recommendations/categories", response_model=CategoriesResponse)
def categories(
    seed: Optional[str] = None,
    limit: int = 8,
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_current_user),
):
    """
    Returns multiple buckets of recommendations:
    - because_you_like: similar to the seed (user seed if present, else provided, else fallback)
    - more_from_seed: another similar batch with different offset
    - fresh_mix: random seeds sampled from catalogue to diversify
    """
    seeds_source: List[str] = []
    if user and user.seed_selection:
        try:
            seeds_source = json.loads(user.seed_selection)
        except Exception:
            seeds_source = []
    if not seeds_source and seed:
        resolved = recommender.resolve_song_id(seed)
        if resolved:
            seeds_source = [resolved]

    liked_ids = get_user_likes(session, user)
    # If no seed selection yet, use likes as personalization
    if not seeds_source and liked_ids:
        seeds_source = [liked_ids[0]]

    # Determine primary/secondary seeds
    primary_seed = seeds_source[0] if seeds_source else None
    secondary_seed: Optional[str] = None
    if len(seeds_source) > 1:
        secondary_seed = seeds_source[1]
    elif liked_ids and len(liked_ids) > 1:
        secondary_seed = liked_ids[1]
    buckets: List[RecommendationBucket] = []

    # Build one profile vector (seeds + likes centroid), then use MMR to create shelves
    profile_seeds: List[str] = []
    if primary_seed:
        profile_seeds.append(primary_seed)
    if secondary_seed and secondary_seed not in profile_seeds:
        profile_seeds.append(secondary_seed)
    for s in seeds_source:
        if s not in profile_seeds:
            profile_seeds.append(s)

    # If user is brand new (no seeds/likes), fall back to a catalogue seed but don't show "Based on ..."
    seed_used_label: Optional[str] = "your taste" if profile_seeds else None
    if not profile_seeds and recommender.id_map:
        profile_seeds = [recommender.id_map[np.random.randint(0, len(recommender.id_map))]]
        seed_used_label = None

    # Candidate pool: union of per-seed neighbors (strong local similarity) + profile neighbors (taste glue)
    base: List[RecommendationItem] = []
    try:
        profile_vec = recommender.build_profile_vector(
            profile_seeds,
            liked_song_ids=liked_ids if liked_ids else None,
        )
        exclude = set(profile_seeds)

        # 1) neighbors per seed (guarantees direct seed similarity)
        seed_candidates_overlap = build_seed_neighbor_candidate_indices(
            recommender=recommender,
            seed_song_ids=profile_seeds,
            per_seed_k=140,
            exclude_song_ids=exclude,
            # For the primary shelf, require overlap across multiple seeds when possible
            min_seed_hits=2 if len(profile_seeds) >= 3 else 1,
        )
        seed_candidates_all = build_seed_neighbor_candidate_indices(
            recommender=recommender,
            seed_song_ids=profile_seeds,
            per_seed_k=120,
            exclude_song_ids=exclude,
            min_seed_hits=1,
        )

        # 2) small profile pool (helps connect seeds/likes into one coherent taste)
        prof_raw = recommender.search_candidates_by_vector(
            profile_vec, k=max(limit * 40, 240), exclude_song_ids=exclude
        )
        prof_candidates: List[int] = []
        seen_idx2: set[int] = set(seed_candidates_all)
        for idx, _dist in prof_raw:
            if idx in seen_idx2:
                continue
            seen_idx2.add(idx)
            prof_candidates.append(idx)

        # Shelf 1 uses overlap-first candidates for tighter relevance
        candidate_indices_primary: List[int] = seed_candidates_overlap + seed_candidates_all + prof_candidates
        # Shelf 2/3 can be a bit broader
        candidate_indices_general: List[int] = seed_candidates_all + prof_candidates

        # Artist allowlist from seed-neighbor overlap (used for the first shelf only)
        allowed_artists = build_allowed_artists_from_seed_neighbors(
            recommender=recommender,
            seed_song_ids=profile_seeds,
            per_seed_k=220,
            exclude_song_ids=exclude,
            min_seed_hits=2 if len(profile_seeds) >= 3 else 1,
            max_artists=70,
        )

        seen_songs: set[str] = set()
        b1 = mmr_rerank(
            recommender=recommender,
            candidate_indices=candidate_indices_primary,
            target_vec=profile_vec,
            limit=limit,
            exclude_song_ids=exclude | seen_songs,
            allowed_artists=allowed_artists,
            lambda_rel=0.92,
            max_per_artist=1,
        )
        seen_songs.update([r.song_id for r in b1])
        if b1:
            buckets.append(
                RecommendationBucket(
                    label="Because you like",
                    seed_used=seed_used_label,
                    items=enrich_with_covers(b1),
                )
            )

        b2 = mmr_rerank(
            recommender=recommender,
            candidate_indices=candidate_indices_general,
            target_vec=profile_vec,
            limit=limit,
            exclude_song_ids=exclude | seen_songs,
            lambda_rel=0.84,
            max_per_artist=1,
        )
        seen_songs.update([r.song_id for r in b2])
        if b2:
            buckets.append(
                RecommendationBucket(
                    label="More from your vibe",
                    seed_used=seed_used_label,
                    items=enrich_with_covers(b2),
                )
            )

        b3 = mmr_rerank(
            recommender=recommender,
            candidate_indices=candidate_indices_general,
            target_vec=profile_vec,
            limit=limit,
            exclude_song_ids=exclude | seen_songs,
            lambda_rel=0.76,
            max_per_artist=1,
        )
        if b3:
            buckets.append(
                RecommendationBucket(
                    label="Fresh mix",
                    seed_used=None,
                    items=enrich_with_covers(b3),
                )
            )

        # Fallback if shelves are empty (MMR got too strict for sparse areas)
        if not buckets:
            base = recommender.recommend_from_seeds(
                profile_seeds,
                k=max(limit * 14, 60),
                liked_song_ids=liked_ids if liked_ids else None,
            )
            base = diversify_recommendations(base, limit=max(limit * 10, 40), max_per_artist=2)
            if base:
                buckets.append(
                    RecommendationBucket(
                        label="For you",
                        seed_used=seed_used_label,
                        items=enrich_with_covers(base[:limit]),
                    )
                )
    except Exception:
        # keep endpoint resilient; return empty buckets on unexpected issues
        buckets = []

    return CategoriesResponse(buckets=buckets)
