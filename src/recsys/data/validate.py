"""Stage `validate`: read the raw CSVs with explicit dtypes, check them against schemas, and
write typed parquet files. Every later stage reads from here, so bad data fails loudly once
at the start instead of silently skewing metrics later.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa

from recsys.utils.io import write_json, write_parquet
from recsys.utils.paths import INTERIM_DIR, METRICS_DIR, RAW_DIR

# The README says ratings fall between January 09, 1995 and October 12, 2023.
MIN_TS = int(pd.Timestamp("1995-01-01", tz="UTC").timestamp())
MAX_TS = int(pd.Timestamp("2023-12-31", tz="UTC").timestamp())
VALID_RATINGS = [x / 2 for x in range(1, 11)]  # 0.5, 1.0, ..., 5.0

RATINGS_SCHEMA = pa.DataFrameSchema(
    {
        "userId": pa.Column("int32", pa.Check.gt(0)),
        "movieId": pa.Column("int32", pa.Check.gt(0)),
        "rating": pa.Column("float32", pa.Check.isin(VALID_RATINGS)),
        "timestamp": pa.Column("int64", pa.Check.in_range(MIN_TS, MAX_TS)),
    },
    unique=["userId", "movieId"],
    strict=True,
)

MOVIES_SCHEMA = pa.DataFrameSchema(
    {
        "movieId": pa.Column("int32", pa.Check.gt(0), unique=True),
        "title": pa.Column(str, pa.Check.str_length(min_value=1)),
        "genres": pa.Column(str),
    },
    strict=True,
)

LINKS_SCHEMA = pa.DataFrameSchema(
    {
        "movieId": pa.Column("int32", pa.Check.gt(0), unique=True),
        "imdbId": pa.Column(str, pa.Check.str_matches(r"^\d+$")),
        "tmdbId": pa.Column("Int64", pa.Check.gt(0), nullable=True),
    },
    strict=True,
)

TAGS_SCHEMA = pa.DataFrameSchema(
    {
        "userId": pa.Column("int32", pa.Check.gt(0)),
        "movieId": pa.Column("int32", pa.Check.gt(0)),
        "tag": pa.Column(str, nullable=True),
        "timestamp": pa.Column("int64", pa.Check.in_range(MIN_TS, MAX_TS)),
    },
    strict=True,
)


def read_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "ratings": pd.read_csv(
            raw_dir / "ratings.csv",
            dtype={
                "userId": "int32",
                "movieId": "int32",
                "rating": "float32",
                "timestamp": "int64",
            },
            engine="pyarrow",
        ),
        "movies": pd.read_csv(
            raw_dir / "movies.csv", dtype={"movieId": "int32", "title": "str", "genres": "str"}
        ),
        # imdbId stays a string to keep its leading zeros (e.g. "0114709").
        "links": pd.read_csv(
            raw_dir / "links.csv", dtype={"movieId": "int32", "imdbId": "str", "tmdbId": "Int64"}
        ),
        "tags": pd.read_csv(
            raw_dir / "tags.csv",
            dtype={"userId": "int32", "movieId": "int32", "tag": "str", "timestamp": "int64"},
        ),
    }


def check_referential_integrity(tables: dict[str, pd.DataFrame]) -> None:
    known = set(tables["movies"]["movieId"])
    for name in ("ratings", "links", "tags"):
        unknown = set(tables[name]["movieId"]) - known
        if unknown:
            raise ValueError(f"{name} references {len(unknown)} movieIds missing from movies")


def validate(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    schemas = {
        "ratings": RATINGS_SCHEMA,
        "movies": MOVIES_SCHEMA,
        "links": LINKS_SCHEMA,
        "tags": TAGS_SCHEMA,
    }
    # lazy=True collects every failing check before raising, not just the first one.
    validated = {name: schemas[name].validate(df, lazy=True) for name, df in tables.items()}
    check_referential_integrity(validated)
    return validated


def summarize(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    ratings = tables["ratings"]
    ts = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    return {
        "n_ratings": len(ratings),
        "n_users": int(ratings["userId"].nunique()),
        "n_movies_rated": int(ratings["movieId"].nunique()),
        "n_movies": len(tables["movies"]),
        "n_tags": len(tables["tags"]),
        "n_movies_without_tmdb_id": int(tables["links"]["tmdbId"].isna().sum()),
        "first_rating": ts.min().isoformat(),
        "last_rating": ts.max().isoformat(),
        "mean_rating": round(float(ratings["rating"].mean()), 4),
    }


def main() -> None:
    tables = validate(read_raw(RAW_DIR))
    for name, df in tables.items():
        write_parquet(df, INTERIM_DIR / f"{name}.parquet")
    stats = summarize(tables)
    write_json(stats, METRICS_DIR / "data_validation.json")
    print(stats)


if __name__ == "__main__":
    main()
