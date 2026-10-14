import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys.data.preprocess import to_unix
from recsys.models import build_model
from recsys.monitoring import drift, reference
from recsys.monitoring.drift import DRIFT, NOT_ENOUGH, OK, SECTIONS, Production, Reference

DAY = 86_400
T0 = to_unix("2020-01-01")

# --- reference ---------------------------------------------------------------------------


def test_catalog_table_orders_movies_by_likes_with_ties_to_the_lower_index() -> None:
    items = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "year": pd.array([1999, None, 2005, 2010], dtype="Int16"),
            "genres": [["Action"], [], ["Drama", "War"], ["Drama"]],
        }
    )
    interactions = pd.DataFrame({"item_idx": [2, 2, 2, 0, 3]})  # likes: 1, 0, 3, 1
    catalog = reference.catalog_table(items, interactions)
    assert catalog["movie_id"].tolist() == [10, 20, 30, 40]
    assert catalog["popularity_percentile"].tolist() == [1 / 3, 1.0, 0.0, 2 / 3]
    assert catalog["genres"].map(list).tolist() == [["Action"], [], ["Drama", "War"], ["Drama"]]
    assert pd.isna(catalog["year"][1]) and catalog["year"][0] == 1999


def test_training_ratings_keeps_the_models_users_movies_and_period() -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 3, 1],
            "movieId": [10, 99, 10, 10, 20],
            "rating": [4.0, 5.0, 1.0, 3.0, 2.0],
            "timestamp": [T0, T0, T0, T0, T0 + 400 * DAY],
        }
    )
    users = pd.DataFrame({"userId": [1, 2]})
    items = pd.DataFrame({"movieId": [10, 20]})
    kept = reference.training_ratings(ratings, users, items, "2020-01-01", "2021-01-01")
    # Dropped: a movie outside the catalog, a user the model never saw, a rating after the end.
    assert kept[["userId", "movieId"]].to_numpy().tolist() == [[1, 10], [2, 10]]


def test_user_table_counts_ratings_and_likes_per_user() -> None:
    ratings = pd.DataFrame({"userId": [7, 3, 7, 7], "rating": [3.5, 2.0, 5.0, 3.0]})
    users = reference.user_table(ratings, positive_threshold=3.5)
    assert users.to_dict("list") == {"n_ratings": [1, 3], "n_liked": [0, 2]}  # users 3, then 7


def test_sample_rows_is_seeded_sorted_and_never_more_than_there_is() -> None:
    first = reference.sample_rows(100, 10, np.random.default_rng(1))
    assert first.tolist() == reference.sample_rows(100, 10, np.random.default_rng(1)).tolist()
    assert first.tolist() == sorted(set(first.tolist())) and len(first) == 10
    assert reference.sample_rows(3, 10, np.random.default_rng(1)).tolist() == [0, 1, 2]


def test_served_table_lists_what_the_model_recommends_as_movie_ids() -> None:
    # Movie 2 is liked by all three users, movie 0 by two, movie 1 by one, movie 3 by none.
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 1, 2, 0, 1, 2],
            "item_idx": [2, 2, 2, 0, 0, 1],
            "timestamp": [T0 + i for i in range(6)],
        }
    )
    model = build_model("popularity", {"half_life_days": None}).fit(interactions, n_items=4)
    served = reference.served_table(
        model, interactions, [10, 20, 30, 40], 5, k=2, batch_size=2, rng=np.random.default_rng(0)
    )
    # Each user gets the two most liked movies they have not liked themselves.
    assert served.to_numpy().tolist() == [[20, 0], [40, 1], [20, 0], [40, 1], [10, 0], [40, 1]]


# --- drift -------------------------------------------------------------------------------

CATALOG = pd.DataFrame(
    {
        "movie_id": [1, 2, 3, 4],
        "year": pd.array([1990, 2000, 2010, None], dtype="Int16"),
        "genres": [["Action"], ["Action", "Drama"], ["Drama"], []],
        "popularity_percentile": [0.0, 1 / 3, 2 / 3, 1.0],
    }
)
CFG = {
    "window_days": 28,
    "min_rows": 30,
    "num_method": "wasserstein",
    "cat_method": "jensenshannon",
    "threshold": 0.1,
}
WHEN = datetime(2026, 10, 13, 6, 0, tzinfo=UTC)


def test_build_frames_describes_movies_from_the_catalog() -> None:
    ratings = pd.DataFrame({"movie_id": [2, 4, 99], "rating": [4.0, 1.5, 5.0]})
    users = pd.DataFrame({"n_ratings": [4, 10], "n_liked": [1, 10]})
    served = pd.DataFrame({"movie_id": [3, 2], "position": [0, 1]})
    frames = drift.build_frames(ratings, users, served, CATALOG)
    assert set(frames) == set(SECTIONS)
    for name, (_, _, numerical, categorical) in SECTIONS.items():
        assert frames[name].columns.tolist() == numerical + categorical

    # Movie 99 is not in the catalog; movie 4 has no year and no genres.
    assert frames["ratings"]["rating"].tolist() == [4.0, 1.5]
    assert frames["ratings"]["popularity_percentile"].tolist() == [1 / 3, 1.0]
    assert frames["ratings"]["movie_year"].fillna(-1).tolist() == [2000.0, -1.0]
    assert frames["rating_genres"]["genre"].tolist() == ["Action", "Drama"]
    assert frames["users"]["like_share"].tolist() == [0.25, 1.0]
    assert frames["served"]["movie_year"].tolist() == [2010.0, 2000.0]
    assert frames["served_genres"]["genre"].tolist() == ["Drama", "Action", "Drama"]


def test_user_table_groups_by_user_key() -> None:
    ratings = pd.DataFrame({"user_key": ["b", "a", "b"], "rating": [5.0, 3.0, 2.0]})
    assert drift.user_table(ratings, 3.5).to_dict("list") == {
        "n_ratings": [1, 2],
        "n_liked": [0, 1],
    }


def test_online_quality_is_computed_per_list() -> None:
    lists = pd.DataFrame(
        {
            "user_key": ["a", "a", "b", "c"],
            "strategy": ["personalised", "personalised", "popular", "personalised"],
            "n_items": [2, 2, 2, 2],
            "n_clicks": [2, 0, 0, 1],
            "best_rating_after": [4.5, None, 2.0, None],
            "popularity_percentile": [0.1, 0.3, 0.0, None],  # the last: from before monitoring
            "catalog_size": [4, 4, 4, None],
        }
    )
    served = pd.DataFrame({"movie_id": [1, 2, 1, 2, 1, 2, 1, 3]})
    assert drift.online_quality(lists, served, positive_threshold=3.5) == {
        "n_lists": 4,
        "n_users": 3,
        "personalised_share": 0.75,
        "click_through_rate": 0.5,
        "hit_rate": 0.5,
        "liked_hit_rate": 0.25,
        "coverage": 0.75,
        "popularity_percentile": pytest.approx(0.4 / 3),
    }
    assert drift.online_quality(lists.head(0), served.head(0), 3.5) == {"n_lists": 0}


def production(rng: np.random.Generator, n_users: int, weights: list[float]) -> Production:
    """What the views would return for users who rate movies 1 to 4 with the given preference
    and are recommended movies with the same one. Every rating falls inside the window."""
    n_rated = rng.integers(20, 200, n_users)
    n_rows = int(n_rated.sum())
    ratings = pd.DataFrame(
        {
            "user_key": np.repeat([f"user{i}" for i in range(n_users)], n_rated),
            "movie_id": rng.choice([1, 2, 3, 4], n_rows, p=weights),
            "rating": rng.choice([2.0, 3.0, 4.0, 5.0], n_rows),
            "in_window": True,
        }
    )
    served = pd.DataFrame({"movie_id": rng.choice([1, 2, 3, 4], 20 * n_users, p=weights)})
    lists = pd.DataFrame(
        {
            "user_key": ["user0"],
            "strategy": ["personalised"],
            "n_items": [20],
            "n_clicks": [1],
            "best_rating_after": [4.0],
            "popularity_percentile": [0.2],
            "catalog_size": [4],
        }
    )
    return Production(ratings, served.assign(position=0), lists)


@pytest.fixture(scope="module")
def training() -> Reference:
    drawn = production(np.random.default_rng(0), 3000, [0.4, 0.3, 0.2, 0.1])
    meta = {
        "model": "ease_recency",
        "period": {"start": "2015-01-01", "end": "2023-01-01"},
        "positive_threshold": 3.5,
    }
    return Reference(
        catalog=CATALOG,
        ratings=drawn.ratings[["movie_id", "rating"]],
        users=drift.user_table(drawn.ratings, 3.5),
        served=drawn.served,
        meta=meta,
    )


def test_production_like_the_training_data_is_not_drift(
    training: Reference, tmp_path: Path
) -> None:
    same = production(np.random.default_rng(1), 2000, [0.4, 0.3, 0.2, 0.1])
    summary = drift.build_report(training, same, CFG, tmp_path, WHEN)
    assert {name: s["status"] for name, s in summary["sections"].items()} == {
        "ratings": OK,
        "rating_genres": OK,
        "users": OK,
        "served": OK,
        "served_genres": OK,
    }
    assert summary["status"] == OK
    assert summary["sections"]["ratings"]["n_current"] == len(same.ratings)
    assert summary["sections"]["users"]["n_current"] == 2000
    assert set(summary["sections"]["ratings"]["columns"]) == set(SECTIONS["ratings"][2])
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary
    for name in SECTIONS:
        assert (tmp_path / f"{name}.html").stat().st_size > 10_000
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert '<a href="ratings.html">Ratings</a>' in page and "2026-10-13T06:00:00+00:00" in page


def test_a_shift_towards_obscure_movies_is_drift(training: Reference, tmp_path: Path) -> None:
    shifted = production(np.random.default_rng(1), 2000, [0.05, 0.05, 0.1, 0.8])
    summary = drift.build_report(training, shifted, CFG, tmp_path, WHEN)
    assert summary["status"] == DRIFT
    ratings = summary["sections"]["ratings"]["columns"]
    assert ratings["popularity_percentile"]["drifted"] and not ratings["rating"]["drifted"]
    assert summary["sections"]["served_genres"]["status"] == DRIFT
    assert summary["sections"]["users"]["status"] == OK  # the users themselves are as before

    metrics = drift.flat_metrics(summary)
    assert metrics["drift.ratings.popularity_percentile"] >= CFG["threshold"]
    assert metrics["drift.n_columns_drifted"] >= 2
    assert (metrics["online.n_lists"], metrics["online.click_through_rate"]) == (1.0, 1.0)
    row = f"| Ratings | {len(training.ratings):,} | {len(shifted.ratings):,} | drift |"
    assert row in drift.markdown_summary(summary)


def test_too_little_production_data_is_reported_as_such(
    training: Reference, tmp_path: Path
) -> None:
    everything = production(np.random.default_rng(1), 5, [0.0, 0.0, 0.0, 1.0])
    few = Production(
        everything.ratings.head(10), everything.served.head(10), everything.recommendations
    )
    summary = drift.build_report(training, few, CFG, tmp_path, WHEN)
    assert summary["status"] == NOT_ENOUGH
    assert all(section["columns"] == {} for section in summary["sections"].values())
    assert not (tmp_path / "ratings.html").exists()
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "not enough data" in page and "<a href" not in page
    assert drift.flat_metrics(summary)["drift.n_columns_drifted"] == 0.0


def test_load_reference_says_how_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dvc pull data/monitoring/reference"):
        drift.load_reference(tmp_path)


def test_supabase_urls_get_the_psycopg_driver() -> None:
    engine = drift.connect("postgresql://reader:secret@host:5432/postgres")
    assert (engine.url.drivername, engine.url.username) == ("postgresql+psycopg", "reader")
