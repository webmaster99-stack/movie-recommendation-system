"""Stage `app_feedback`: the app's ratings as likes the models can be fitted on.

The snapshot (`data/app_feedback/ratings.parquet`, from `recsys.feedback.snapshot`) is
turned into rows shaped like the MovieLens splits (`user_idx`, `item_idx`, `timestamp`,
`rating`):

- Only likes are kept, with the same threshold as MovieLens (`preprocess.positive_threshold`).
- Movies outside the model's catalog are dropped; the catalog is not rebuilt for them.
- App users get `user_idx` values after the last MovieLens user, in order of their key.
- Of every app user with at least `min_likes` likes, the latest `holdout_fraction` (at least
  one) are held out: `recsys.registry.promote` asks whether a challenger predicts them better
  than the champion does. The rest, and all likes of users with fewer, are for training.

Outputs `data/processed/app_feedback/{train,holdout}.parquet` and the counts in
`metrics/app_feedback.json`. An empty snapshot gives empty tables, and the retrained model is
then the MovieLens one.
"""

from typing import Any

import numpy as np
import pandas as pd

from recsys.utils.config import load_params
from recsys.utils.io import write_json, write_parquet
from recsys.utils.paths import (
    APP_FEEDBACK_DIR,
    APP_FEEDBACK_PROCESSED_DIR,
    METRICS_DIR,
    PROCESSED_DIR,
)

_COLUMNS = ["user_idx", "item_idx", "timestamp", "rating"]
_ORDER = ["timestamp", "user_idx", "item_idx"]  # the order of the MovieLens files


def _date(timestamps: pd.Series) -> str | None:
    if timestamps.empty:
        return None
    return pd.Timestamp(int(timestamps.max()), unit="s", tz="UTC").date().isoformat()


def prepare(
    ratings: pd.DataFrame,
    item_map: pd.DataFrame,
    n_movielens_users: int,
    threshold: float,
    min_likes: int,
    holdout_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(train, holdout, counts) from a snapshot of app ratings."""
    likes = ratings[ratings["rating"] >= threshold]
    catalog = item_map.rename(columns={"movieId": "movie_id"})
    in_catalog = likes.merge(catalog, on="movie_id", how="inner")

    keys = np.sort(in_catalog["user_key"].unique().astype(str))
    users = pd.DataFrame(
        {
            "user_key": pd.Series(keys, dtype="string"),
            "user_idx": np.arange(n_movielens_users, n_movielens_users + len(keys), dtype=np.int32),
        }
    )
    df = in_catalog.merge(users, on="user_key")[_COLUMNS].astype(
        {"user_idx": np.int32, "item_idx": np.int32, "timestamp": np.int64, "rating": np.float32}
    )
    df = df.sort_values(["user_idx", "timestamp", "item_idx"], kind="mergesort")

    # Position of each like in its user's history, and how many of the user's likes are held out.
    position = df.groupby("user_idx").cumcount().to_numpy()
    n_likes = df.groupby("user_idx")["item_idx"].transform("size").to_numpy()
    n_held = np.where(
        n_likes >= min_likes,
        np.maximum(1, np.floor(n_likes * holdout_fraction + 1e-9)).astype(np.int64),
        0,
    )
    held = position >= n_likes - n_held
    train = df[~held].sort_values(_ORDER, kind="mergesort").reset_index(drop=True)
    holdout = df[held].sort_values(_ORDER, kind="mergesort").reset_index(drop=True)

    counts = {
        "n_ratings": len(ratings),
        "n_users": int(ratings["user_key"].nunique()),
        "n_likes": len(likes),
        "n_likes_outside_catalog": len(likes) - len(in_catalog),
        "n_train": len(train),
        "n_holdout": len(holdout),
        "n_holdout_users": int(holdout["user_idx"].nunique()),
        "last_rating": _date(ratings["timestamp"]),
    }
    return train, holdout, counts


def main() -> None:
    params = load_params()
    cfg = params["retraining"]["app_feedback"]
    if not 0 < cfg["holdout_fraction"] < 1:
        raise ValueError("retraining.app_feedback.holdout_fraction must be between 0 and 1")
    train, holdout, counts = prepare(
        pd.read_parquet(APP_FEEDBACK_DIR / "ratings.parquet"),
        pd.read_parquet(PROCESSED_DIR / "item_map.parquet"),
        len(pd.read_parquet(PROCESSED_DIR / "user_map.parquet")),
        params["preprocess"]["positive_threshold"],
        cfg["min_likes"],
        cfg["holdout_fraction"],
    )
    write_parquet(train, APP_FEEDBACK_PROCESSED_DIR / "train.parquet")
    write_parquet(holdout, APP_FEEDBACK_PROCESSED_DIR / "holdout.parquet")
    write_json(counts, METRICS_DIR / "app_feedback.json")
    print(counts)


if __name__ == "__main__":
    main()
