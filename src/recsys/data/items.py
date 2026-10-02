"""Stage `items`: the catalog table for the kept movies, built only from MovieLens.

The app uses `tmdbId` to fetch posters and plot summaries from TMDB at serving time
(TMDB data never enters this pipeline; see CLAUDE.md for why).
"""

import pandas as pd

from recsys.utils.io import write_parquet
from recsys.utils.paths import INTERIM_DIR, PROCESSED_DIR

# MovieLens titles end with the release year, e.g. "Toy Story (1995)". A few have none.
_YEAR_RE = r"^(?P<title>.*?)\s*\((?P<year>\d{4})(?:[–-]\d{0,4})?\)\s*$"


def split_title_year(titles: pd.Series) -> pd.DataFrame:
    parts = titles.str.extract(_YEAR_RE)
    return pd.DataFrame(
        {
            "title": parts["title"].fillna(titles).str.strip(),
            "year": pd.to_numeric(parts["year"]).astype("Int16"),
        }
    )


def build_items(item_map: pd.DataFrame, movies: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    df = item_map.merge(movies, on="movieId", how="left").merge(links, on="movieId", how="left")
    df = pd.concat([df.drop(columns="title"), split_title_year(df["title"])], axis=1)
    df["genres"] = df["genres"].map(lambda g: [] if g == "(no genres listed)" else g.split("|"))
    cols = ["item_idx", "movieId", "title", "year", "genres", "tmdbId", "imdbId"]
    return df[cols].sort_values("item_idx").reset_index(drop=True)


def main() -> None:
    items = build_items(
        pd.read_parquet(PROCESSED_DIR / "item_map.parquet"),
        pd.read_parquet(INTERIM_DIR / "movies.parquet"),
        pd.read_parquet(INTERIM_DIR / "links.parquet"),
    )
    write_parquet(items, PROCESSED_DIR / "items.parquet")
    print(f"{len(items)} items, {int(items['tmdbId'].isna().sum())} without a tmdbId")


if __name__ == "__main__":
    main()
