import zipfile
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from recsys.data.download import ChecksumError, extract, sha256_of, verify_checksum
from recsys.data.items import build_items, split_title_year
from recsys.data.preprocess import cap_items, k_core, preprocess, reindex, to_unix
from recsys.data.split import summarize, temporal_split
from recsys.data.validate import RATINGS_SCHEMA


def ratings_df(rows: list[tuple[int, int, float, int]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    return df.astype({"userId": "int32", "movieId": "int32", "rating": "float32"})


T0 = to_unix("2015-01-01")
DAY = 86_400


# --- download ----------------------------------------------------------------------------


def test_checksum_mismatch_and_missing_pin_fail(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    verify_checksum(f, sha256_of(f))  # correct pin passes
    with pytest.raises(ChecksumError, match="expected sha256"):
        verify_checksum(f, "0" * 64)
    with pytest.raises(ChecksumError, match="No checksum pinned"):
        verify_checksum(f, None)


def test_extract_flattens_and_requires_expected_files(tmp_path: Path) -> None:
    zip_path = tmp_path / "ml.zip"
    names = ["ratings.csv", "movies.csv", "tags.csv", "links.csv", "README.txt"]
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in names:
            zf.writestr(f"ml-32m/{name}", name)
    extract(zip_path, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(names)

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("ml-32m/ratings.csv", "")
    with pytest.raises(FileNotFoundError):
        extract(zip_path, tmp_path / "out2")


# --- validate ----------------------------------------------------------------------------


def test_schema_accepts_good_ratings() -> None:
    RATINGS_SCHEMA.validate(ratings_df([(1, 1, 4.0, T0), (1, 2, 0.5, T0)]))


@pytest.mark.parametrize(
    "bad",
    [
        [(1, 1, 4.2, T0)],  # rating not on the 0.5 grid
        [(1, 1, 4.0, 0)],  # timestamp outside the dataset's range
        [(1, 1, 4.0, T0), (1, 1, 3.0, T0 + 1)],  # duplicate (user, movie)
        [(0, 1, 4.0, T0)],  # invalid id
    ],
)
def test_schema_rejects_bad_ratings(bad: list[tuple[int, int, float, int]]) -> None:
    with pytest.raises(pandera.errors.SchemaErrors):
        RATINGS_SCHEMA.validate(ratings_df(bad), lazy=True)


# --- preprocess --------------------------------------------------------------------------


def test_cap_items_keeps_most_popular_with_deterministic_ties() -> None:
    df = ratings_df([(1, 10, 5, T0), (2, 10, 5, T0), (1, 20, 5, T0), (1, 30, 5, T0)])
    kept = cap_items(df, max_items=2)
    # 10 has 2 interactions; 20 and 30 tie at 1, so the smaller movieId wins.
    assert set(kept["movieId"]) == {10, 20}


def test_k_core_iterates_until_stable() -> None:
    # Item 3 only has user 3; dropping it leaves user 3 with 1 interaction, so user 3 goes too.
    rows = [(u, i, 5.0, T0) for u in (1, 2) for i in (1, 2)] + [(3, 1, 5.0, T0), (3, 3, 5.0, T0)]
    out = k_core(ratings_df(rows), min_user=2, min_item=2)
    assert set(out["userId"]) == {1, 2}
    assert set(out["movieId"]) == {1, 2}
    assert out.groupby("userId").size().min() >= 2
    assert out.groupby("movieId").size().min() >= 2


def test_reindex_is_contiguous_and_sorted() -> None:
    df = ratings_df([(50, 7, 5, T0 + 2), (9, 3, 5, T0 + 1), (50, 3, 5, T0)])
    out, users, items = reindex(df)
    assert users["userId"].tolist() == [9, 50] and users["user_idx"].tolist() == [0, 1]
    assert items["movieId"].tolist() == [3, 7] and items["item_idx"].tolist() == [0, 1]
    assert out["timestamp"].is_monotonic_increasing


def test_preprocess_applies_window_and_threshold() -> None:
    old = to_unix("2005-01-01")
    rows = [(u, i, 4.0, T0) for u in (1, 2) for i in (1, 2)]
    rows += [(1, 3, 2.0, T0), (2, 3, 2.0, T0)]  # negatives: dropped
    rows += [(3, 1, 5.0, old), (3, 2, 5.0, old)]  # before the window: dropped
    cfg = {
        "start_date": "2010-01-01",
        "positive_threshold": 3.5,
        "max_items": 100,
        "min_user_interactions": 2,
        "min_item_interactions": 2,
    }
    interactions, users, items = preprocess(ratings_df(rows), cfg)
    assert len(interactions) == 4
    assert users["userId"].tolist() == [1, 2]
    assert items["movieId"].tolist() == [1, 2]


# --- split -------------------------------------------------------------------------------


def test_temporal_split_has_no_leakage() -> None:
    ts = [to_unix(d) for d in ("2019-06-01", "2020-06-01", "2021-06-01")]
    df = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [0, 1, 2], "timestamp": ts})
    s = temporal_split(df, val_start="2020-01-01", test_start="2021-01-01")
    assert [len(s[k]) for k in ("train", "val", "test")] == [1, 1, 1]
    assert s["train"]["timestamp"].max() < s["val"]["timestamp"].min()
    assert s["val"]["timestamp"].max() < s["test"]["timestamp"].min()

    stats = summarize(s)
    assert stats["val"]["n_users_with_history"] == 1  # user 0 was seen in train
    assert stats["test"]["n_users_with_history"] == 0  # user 1 is new in test
    assert stats["test"]["frac_targets_on_unseen_items"] == 1.0

    with pytest.raises(ValueError):
        temporal_split(df, val_start="2021-01-01", test_start="2020-01-01")


# --- items -------------------------------------------------------------------------------


def test_split_title_year() -> None:
    out = split_title_year(pd.Series(["Toy Story (1995)", "Big Bang Theory (2007-)", "No Year"]))
    assert out["title"].tolist() == ["Toy Story", "Big Bang Theory", "No Year"]
    assert out["year"].tolist()[:2] == [1995, 2007]
    assert pd.isna(out["year"].iloc[2])


def test_build_items_joins_metadata() -> None:
    item_map = pd.DataFrame({"movieId": [1, 2], "item_idx": [0, 1]})
    movies = pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["A (2000)", "B (2001)"],
            "genres": ["X|Y", "(no genres listed)"],
        }
    )
    links = pd.DataFrame({"movieId": [1, 2], "imdbId": ["01", "02"], "tmdbId": [11, None]})
    items = build_items(item_map, movies, links)
    assert items["genres"].tolist() == [["X", "Y"], []]
    assert items["title"].tolist() == ["A", "B"]
