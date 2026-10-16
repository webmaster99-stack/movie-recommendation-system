"""`python -m recsys.feedback.snapshot`: copy the app's ratings into `data/app_feedback/`.

    uv run --env-file .env python -m recsys.feedback.snapshot
    uv run dvc add data/app_feedback

The ratings are read through the `monitoring.ratings` view as `monitoring_reader`
(`MONITORING_DATABASE_URL`), so the snapshot never holds an account id: a user is the MD5 of
theirs, enough to keep one person's ratings together and useless for finding the account.
Nothing else is copied (no titles, no TMDB data, no recommendation logs).

This is a command and not a pipeline stage, because its result depends on when it runs. The
pipeline reads the snapshot like any other input: `dvc add` records its hash, `dvc push`
uploads it, and anyone who checks out the same commit gets the same file and the same model.
"""

import argparse
import os
from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from recsys.utils.io import write_parquet
from recsys.utils.paths import APP_FEEDBACK_DIR

QUERY = "select user_key, movie_id, rating, updated_at from monitoring.ratings"
SNAPSHOT_FILE = "ratings.parquet"

Row = tuple[str, int, Decimal | float, datetime]


def to_frame(rows: Iterable[Row]) -> pd.DataFrame:
    """One row per rating, in a fixed order, with `timestamp` in Unix seconds like MovieLens.

    `updated_at` is used because it is when the rating took its current value, and it is the
    order the API reads a history in.
    """
    rows = list(rows)
    df = pd.DataFrame(
        {
            "user_key": pd.Series([r[0] for r in rows], dtype="string"),
            "movie_id": np.array([r[1] for r in rows], dtype=np.int32),
            "rating": np.array([float(r[2]) for r in rows], dtype=np.float32),
            "timestamp": np.array([int(r[3].timestamp()) for r in rows], dtype=np.int64),
        }
    )
    df = df.sort_values(["user_key", "timestamp", "movie_id"], kind="mergesort")
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=APP_FEEDBACK_DIR)
    args = parser.parse_args()

    url = os.environ.get("MONITORING_DATABASE_URL")
    if not url:
        raise SystemExit("MONITORING_DATABASE_URL is not set (see .env.example)")
    with psycopg.connect(url) as connection:
        rows = connection.execute(QUERY).fetchall()
    ratings = to_frame(rows)
    write_parquet(ratings, args.out / SNAPSHOT_FILE)
    print(
        f"{len(ratings)} ratings by {ratings['user_key'].nunique()} users -> "
        f"{args.out / SNAPSHOT_FILE}\nnext: uv run dvc add data/app_feedback"
    )


if __name__ == "__main__":
    main()
