"""Stage `preprocess`: turn explicit ratings into the implicit-feedback dataset all models share.

Steps (all thresholds come from params.yaml):
1. Keep ratings inside the time window.
2. Keep "positive" ratings (rating >= threshold): we model *what users liked*, as an app
   recommending movies to watch would, rather than predicting star values.
3. Cap the catalog to the most-interacted movies (keeps EASE's item x item matrix in RAM).
4. Iterative k-core filter: drop users/items with too few interactions until stable.
5. Re-index users and items to contiguous 0..n-1 ids (rows/columns of sparse matrices).

Known trade-off: steps 3-4 look at the whole window, including the future test period, so
an item can survive the filter thanks to later interactions. This is standard practice and
only decides *which* items exist; no interaction ever crosses the temporal split.
"""

from typing import Any

import pandas as pd

from recsys.utils.config import load_params
from recsys.utils.io import write_json, write_parquet
from recsys.utils.paths import INTERIM_DIR, METRICS_DIR, PROCESSED_DIR


def to_unix(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp())


def filter_window(ratings: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    mask = ratings["timestamp"] >= to_unix(start)
    if end is not None:
        mask &= ratings["timestamp"] < to_unix(end)
    return ratings[mask]


def cap_items(df: pd.DataFrame, max_items: int) -> pd.DataFrame:
    """Keep the max_items most-interacted items; ties broken by movieId for determinism."""
    counts = df["movieId"].value_counts().rename("n").reset_index()
    counts = counts.sort_values(["n", "movieId"], ascending=[False, True], kind="mergesort")
    keep = counts["movieId"].head(max_items)
    return df[df["movieId"].isin(keep)]


def k_core(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Repeat until every user has >= min_user and every item >= min_item interactions.

    One pass isn't enough: dropping sparse users can push items below the threshold, and so on.
    """
    while True:
        item_counts = df["movieId"].map(df["movieId"].value_counts())
        user_counts = df["userId"].map(df["userId"].value_counts())
        mask = (item_counts >= min_item) & (user_counts >= min_user)
        if mask.all():
            return df
        df = df[mask]


def reindex(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map ids to contiguous indices, ordered by original id so the mapping is deterministic."""
    users = pd.DataFrame({"userId": sorted(df["userId"].unique())})
    users["user_idx"] = users.index.astype("int32")
    items = pd.DataFrame({"movieId": sorted(df["movieId"].unique())})
    items["item_idx"] = items.index.astype("int32")

    out = df.merge(users, on="userId").merge(items, on="movieId")
    out = out[["user_idx", "item_idx", "timestamp", "rating"]]
    # Stable, fully-specified order so the written file is byte-identical every run.
    out = out.sort_values(["timestamp", "user_idx", "item_idx"], kind="mergesort")
    return out.reset_index(drop=True), users, items


def preprocess(
    ratings: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = filter_window(ratings, cfg["start_date"], cfg.get("end_date"))
    df = df[df["rating"] >= cfg["positive_threshold"]]
    df = cap_items(df, cfg["max_items"])
    df = k_core(df, cfg["min_user_interactions"], cfg["min_item_interactions"])
    return reindex(df)


def summarize(interactions: pd.DataFrame, n_raw: int) -> dict[str, Any]:
    n_users = int(interactions["user_idx"].nunique())
    n_items = int(interactions["item_idx"].nunique())
    per_user = interactions.groupby("user_idx").size()
    return {
        "n_interactions": len(interactions),
        "n_users": n_users,
        "n_items": n_items,
        "density": round(len(interactions) / (n_users * n_items), 6),
        "kept_fraction_of_raw": round(len(interactions) / n_raw, 4),
        "median_interactions_per_user": float(per_user.median()),
    }


def main() -> None:
    cfg = load_params()["preprocess"]
    ratings = pd.read_parquet(INTERIM_DIR / "ratings.parquet")
    interactions, users, items = preprocess(ratings, cfg)

    write_parquet(interactions, PROCESSED_DIR / "interactions.parquet")
    write_parquet(users, PROCESSED_DIR / "user_map.parquet")
    write_parquet(items, PROCESSED_DIR / "item_map.parquet")
    stats = summarize(interactions, n_raw=len(ratings))
    write_json(stats, METRICS_DIR / "preprocess.json")
    print(stats)


if __name__ == "__main__":
    main()
