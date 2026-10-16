"""Stage `monitoring_reference`: what the model saw and what it recommends, as the yardstick
the weekly drift report (`recsys.monitoring.drift`) holds production against.

Four small tables in `data/monitoring/reference/`, all from MovieLens and the selected model:

- `catalog.parquet`: each movie's year, genres and place in the popularity order, so that a
  production `movie_id` can be described in the same terms as a training one.
- `ratings.parquet`: a sample of the ratings, liked or not, that the model's users gave to
  catalog movies during its training period (training + validation).
- `users.parquet`: for each of those users, how many movies they rated and how many they
  liked: the history lengths the model was fitted to.
- `served.parquet`: the lists the model recommends to a sample of those users from their
  whole history, standing for "what this model serves when it sees its usual input".

The report needs nothing else from the pipeline, so the scheduled workflow pulls only this
directory. No production data is involved here: the stage is as deterministic as any other.
"""

from typing import Any

import numpy as np
import pandas as pd

from recsys.data.preprocess import filter_window
from recsys.evaluation.evaluate import load_split, recommend_all
from recsys.evaluation.protocol import _per_user
from recsys.models import load_model
from recsys.models.base import Recommender
from recsys.registry.export import read_json
from recsys.utils.config import load_params
from recsys.utils.io import write_json, write_parquet
from recsys.utils.paths import (
    INTERIM_DIR,
    METRICS_DIR,
    MONITORING_REFERENCE_DIR,
    PROCESSED_DIR,
    RETRAINED_DIR,
)
from recsys.utils.seed import make_deterministic


def catalog_table(items: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """One row per movie, in `item_idx` order.

    `popularity_percentile` is the movie's place when the catalog is ordered by likes in
    `interactions`, from 0 (the most liked) to 1 (the least); ties go to the lower index.
    """
    n_items = len(items)
    likes = np.bincount(interactions["item_idx"].to_numpy(), minlength=n_items)
    order = np.lexsort((np.arange(n_items), -likes))
    place = np.empty(n_items, dtype=np.int64)
    place[order] = np.arange(n_items)
    return pd.DataFrame(
        {
            "movie_id": items["movieId"].to_numpy(dtype=np.int32),
            "year": items["year"].astype("Int16").to_numpy(),
            "genres": items["genres"].map(list).to_numpy(),
            "popularity_percentile": place / max(n_items - 1, 1),
        }
    )


def training_ratings(
    ratings: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, start: str, end: str
) -> pd.DataFrame:
    """Every rating by one of the model's users on a catalog movie in [start, end)."""
    window = filter_window(ratings, start, end)
    known = window["userId"].isin(users["userId"]) & window["movieId"].isin(items["movieId"])
    return window[known]


def user_table(ratings: pd.DataFrame, positive_threshold: float) -> pd.DataFrame:
    """Per user, in ascending `userId` order: movies rated and movies liked. No ids."""
    liked = (ratings["rating"] >= positive_threshold).rename("liked")
    per_user = liked.groupby(ratings["userId"], sort=True).agg(["size", "sum"])
    return pd.DataFrame(
        {
            "n_ratings": per_user["size"].to_numpy(dtype=np.int32),
            "n_liked": per_user["sum"].to_numpy(dtype=np.int32),
        }
    )


def sample_rows(n_rows: int, n: int, rng: np.random.Generator) -> Any:
    """Positions of at most `n` of `n_rows` rows, in their original order."""
    return np.sort(rng.choice(n_rows, size=min(n, n_rows), replace=False))


def served_table(
    model: Recommender,
    interactions: pd.DataFrame,
    movie_ids: Any,
    n_users: int,
    k: int,
    batch_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """The top-`k` lists for a sample of users, one row per movie shown."""
    _, histories = _per_user(interactions)
    picked = [histories[i] for i in sample_rows(len(histories), n_users, rng)]
    recommended = recommend_all(model, picked, k, batch_size)
    return pd.DataFrame(
        {
            "movie_id": np.asarray(movie_ids, dtype=np.int32)[recommended.ravel()],
            "position": np.tile(np.arange(recommended.shape[1], dtype=np.int16), len(picked)),
        }
    )


def main() -> None:
    params = load_params()
    cfg = params["monitoring"]["reference"]
    rng = make_deterministic(params["seed"], params["runtime"]["num_threads"])
    threshold = params["preprocess"]["positive_threshold"]
    start, end = params["preprocess"]["start_date"], params["split"]["test_start"]

    items = pd.read_parquet(PROCESSED_DIR / "items.parquet")
    interactions, _ = load_split("test")  # everything before the test split
    catalog = catalog_table(items, interactions)

    ratings = training_ratings(
        pd.read_parquet(INTERIM_DIR / "ratings.parquet"),
        pd.read_parquet(PROCESSED_DIR / "user_map.parquet"),
        items,
        start,
        end,
    )
    users = user_table(ratings, threshold)
    sample = ratings.iloc[sample_rows(len(ratings), cfg["n_ratings"], rng)]
    sampled_ratings = pd.DataFrame(
        {
            "movie_id": sample["movieId"].to_numpy(dtype=np.int32),
            "rating": sample["rating"].to_numpy(dtype=np.float32),
        }
    )

    name: str = read_json(METRICS_DIR / "selection.json")["selected"]
    model = load_model(name, RETRAINED_DIR)  # the model the bundle serves
    served = served_table(
        model,
        interactions,
        catalog["movie_id"].to_numpy(),
        cfg["n_users"],
        cfg["k"],
        params["evaluation"]["batch_size"],
        rng,
    )

    write_parquet(catalog, MONITORING_REFERENCE_DIR / "catalog.parquet")
    write_parquet(sampled_ratings, MONITORING_REFERENCE_DIR / "ratings.parquet")
    write_parquet(users, MONITORING_REFERENCE_DIR / "users.parquet")
    write_parquet(served, MONITORING_REFERENCE_DIR / "served.parquet")
    meta = {
        "model": name,
        "period": {"start": start, "end": end},  # end is exclusive
        "positive_threshold": threshold,
        "k": cfg["k"],
        "n_movies": len(catalog),
        "n_ratings": len(ratings),
        "n_ratings_sampled": len(sampled_ratings),
        "n_users": len(users),
        "n_users_served": len(served) // cfg["k"],
    }
    write_json(meta, MONITORING_REFERENCE_DIR / "meta.json")
    print(meta)


if __name__ == "__main__":
    main()
