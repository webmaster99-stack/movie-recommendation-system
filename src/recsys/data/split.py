"""Stage `split`: global temporal split into train / validation / test.

Everything before `val_start` is train, [val_start, test_start) is validation, and
everything from `test_start` on is test. One cut-off date for *all* users mirrors
production: the model is trained on the past and judged on what happens next. A random or
per-user leave-last-out split lets a model see interactions that happened after the ones
it's asked to predict, which inflates offline metrics.

Turning the splits into (history, targets) pairs per user happens in evaluation (Phase 2):
validation users are scored from their train history, test users from train+validation.
"""

from typing import Any

import pandas as pd

from recsys.data.preprocess import to_unix
from recsys.utils.config import load_params
from recsys.utils.io import write_json, write_parquet
from recsys.utils.paths import METRICS_DIR, PROCESSED_DIR, SPLITS_DIR


def temporal_split(
    interactions: pd.DataFrame, val_start: str, test_start: str
) -> dict[str, pd.DataFrame]:
    t1, t2 = to_unix(val_start), to_unix(test_start)
    if t1 >= t2:
        raise ValueError("val_start must be before test_start")
    ts = interactions["timestamp"]
    return {
        "train": interactions[ts < t1],
        "val": interactions[(ts >= t1) & (ts < t2)],
        "test": interactions[ts >= t2],
    }


def target_stats(history: pd.DataFrame, targets: pd.DataFrame) -> dict[str, Any]:
    """How many target users/items can actually be evaluated given the history before them."""
    known_users = targets["user_idx"].isin(history["user_idx"])
    known_items = targets["item_idx"].isin(history["item_idx"])
    return {
        "n_interactions": len(targets),
        "n_users": int(targets["user_idx"].nunique()),
        "n_users_with_history": int(targets.loc[known_users, "user_idx"].nunique()),
        # Items never seen before the cut-off can't be recommended by collaborative models.
        "frac_targets_on_unseen_items": round(float((~known_items).mean()), 4),
    }


def summarize(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    train, val, test = splits["train"], splits["val"], splits["test"]
    return {
        "train": {
            "n_interactions": len(train),
            "n_users": int(train["user_idx"].nunique()),
            "n_items": int(train["item_idx"].nunique()),
        },
        "val": target_stats(history=train, targets=val),
        "test": target_stats(history=pd.concat([train, val]), targets=test),
    }


def main() -> None:
    cfg = load_params()["split"]
    interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
    splits = temporal_split(interactions, cfg["val_start"], cfg["test_start"])
    for name, df in splits.items():
        write_parquet(df.reset_index(drop=True), SPLITS_DIR / f"{name}.parquet")
    stats = summarize(splits)
    write_json(stats, METRICS_DIR / "split.json")
    print(stats)


if __name__ == "__main__":
    main()
