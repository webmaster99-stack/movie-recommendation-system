"""A six-movie bundle, an in-memory database and a fake TMDB: the API with nothing real
behind it."""

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jwt
import numpy as np
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from recsys_api.auth import TokenVerifier
from recsys_api.bundle import Bundle, load_bundle
from recsys_api.config import Settings
from recsys_api.db import make_engine
from recsys_api.main import create_app
from recsys_api.tmdb import MetadataCache

JWT_SECRET = "a-test-secret-that-is-long-enough-for-hs256"

# item_idx, movie_id, title as MovieLens writes it, year, genres, tmdb_id
MOVIES: list[tuple[int, int, str, int | None, list[str], int | None]] = [
    (0, 10, "Matrix, The", 1999, ["Action", "Sci-Fi"], 603),
    (1, 20, "Matrix Reloaded, The", 2003, ["Action", "Sci-Fi"], 604),
    (2, 30, "Amelie (Fabuleux destin d'Amélie Poulain, Le)", 2001, ["Comedy", "Romance"], 194),
    (3, 40, "Toy Story", 1995, ["Animation", "Children"], 862),
    (4, 50, "Heat", 1995, ["Action", "Crime"], 949),
    (5, 60, "Untitled", None, ["(no genres listed)"], None),
]
# EASE weights: row i says what liking movie i predicts. Liking The Matrix points strongly
# at its sequel, weakly at Heat and away from Amelie.
WEIGHTS = np.zeros((6, 6), dtype=np.float32)
WEIGHTS[0, [1, 2, 4]] = [0.9, -0.1, 0.2]
WEIGHTS[1, 0] = 0.8
WEIGHTS[2, 3] = 0.5
WEIGHTS[3, 2] = 0.4
WEIGHTS[4, 0] = 0.3
PRIOR = np.array([2.0, 0.5, 1.0, 1.5, 0.0, -1.0], dtype=np.float32)  # popularity order 0 3 2 1 4 5


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def write_bundle(path: Path) -> Path:
    """A bundle in the layout `recsys.registry.export` writes, small enough to check by hand."""
    records = [
        {
            "item_idx": idx,
            "movie_id": movie_id,
            "title": title,
            "year": year,
            "genres": genres,
            "tmdb_id": tmdb_id,
            "imdb_id": f"{movie_id:07d}",
        }
        for idx, movie_id, title, year, genres, tmdb_id in MOVIES
    ]
    write_json(path / "items.json", records)
    write_json(
        path / "metadata.json",
        {
            "model": "ease_recency",
            "preprocess": {"positive_threshold": 3.5},
            "trained_on": {"last_interaction": "2022-12-31"},
        },
    )
    model = path / "model"
    params = {"half_life_days": 11, "weight": 0.1}
    write_json(model / "meta.json", {"model": "ease_recency", "n_items": 6, "params": params})
    np.save(model / "prior.npy", PRIOR)
    base_params = {"l2": 1.0, "keep_per_item": 3}
    write_json(model / "base" / "meta.json", {"model": "ease", "n_items": 6, "params": base_params})
    sparse = sp.csr_matrix(WEIGHTS)
    np.save(model / "base" / "data.npy", sparse.data)
    np.save(model / "base" / "indices.npy", sparse.indices.astype(np.int32))
    np.save(model / "base" / "indptr.npy", sparse.indptr.astype(np.int64))
    return path


class FakeTmdb:
    """Stands in for `TmdbClient`: answers from memory and records what was asked."""

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.missing: set[int] = set()
        self.failing = False

    def movie(self, tmdb_id: int) -> dict[str, Any] | None:
        import httpx

        self.calls.append(tmdb_id)
        if self.failing:
            raise httpx.ConnectError("TMDB is down")
        if tmdb_id in self.missing:
            return None
        return {
            "title": f"TMDB title {tmdb_id}",
            "overview": f"Plot of {tmdb_id}.",
            "poster_path": f"/poster{tmdb_id}.jpg",
            "backdrop_path": f"/backdrop{tmdb_id}.jpg",
            "release_date": "1999-03-31",
            "runtime": 136,
            "vote_average": 8.2,
            "something_else": "ignored",
        }


@pytest.fixture(scope="session")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Bundle:
    return load_bundle(write_bundle(tmp_path_factory.mktemp("bundle")))


@pytest.fixture
def engine() -> Engine:
    return make_engine("sqlite://")


@pytest.fixture
def tmdb() -> FakeTmdb:
    return FakeTmdb()


@pytest.fixture
def client(bundle: Bundle, engine: Engine, tmdb: FakeTmdb) -> TestClient:
    settings = Settings(_env_file=None, model_version="7", cors_origins="https://app.example")
    app = create_app(
        settings,
        bundle=bundle,
        engine=engine,
        verifier=TokenVerifier(jwt_secret=JWT_SECRET),
        metadata=MetadataCache(tmdb),  # type: ignore[arg-type]
    )
    return TestClient(app)


def make_token(user_id: uuid.UUID, **overrides: Any) -> str:
    claims = {"sub": str(user_id), "aud": "authenticated", "exp": int(time.time()) + 3600}
    return jwt.encode(claims | overrides, JWT_SECRET, algorithm="HS256")


@pytest.fixture
def user_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-4000-8000-000000000001")


@pytest.fixture
def as_user() -> Callable[[uuid.UUID], dict[str, str]]:
    return lambda user: {"Authorization": f"Bearer {make_token(user)}"}


@pytest.fixture
def auth(user_id: uuid.UUID) -> dict[str, str]:
    return {"Authorization": f"Bearer {make_token(user_id)}"}
