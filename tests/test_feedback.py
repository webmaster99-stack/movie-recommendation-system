"""The retraining loop: snapshot, preparation of the app's likes, and the promotion gate."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import yaml

from recsys.feedback.prepare import prepare
from recsys.feedback.snapshot import to_frame
from recsys.models.base import Recommender
from recsys.registry.gate import Bundle, GateError, decide, paired_difference, run_gate
from recsys.utils.config import PROJECT_ROOT, load_params

# --- snapshot ----------------------------------------------------------------------------


def test_snapshot_rows_become_a_sorted_frame_in_unix_seconds() -> None:
    at = datetime(2026, 10, 13, 8, 40, 0, 500_000, tzinfo=UTC)
    rows = [
        ("b", 2, Decimal("4.5"), at),
        ("a", 5, Decimal("3.0"), at),
        ("a", 1, 5.0, at),
    ]
    df = to_frame(rows)
    assert df[["user_key", "movie_id"]].to_numpy().tolist() == [["a", 1], ["a", 5], ["b", 2]]
    assert df["rating"].tolist() == [5.0, 3.0, 4.5]
    assert df["timestamp"].tolist() == [int(at.timestamp())] * 3
    assert str(df["rating"].dtype) == "float32" and str(df["timestamp"].dtype) == "int64"


def test_an_empty_snapshot_has_the_columns() -> None:
    assert list(to_frame([]).columns) == ["user_key", "movie_id", "rating", "timestamp"]


# --- prepare -----------------------------------------------------------------------------

ITEM_MAP = pd.DataFrame(
    {"movieId": np.array([10, 20, 30, 40, 50, 60], dtype=np.int32), "item_idx": np.arange(6)}
).astype({"item_idx": np.int32})


def ratings(rows: list[tuple[str, int, float, int]]) -> pd.DataFrame:
    return to_frame(
        (key, movie, rating, datetime.fromtimestamp(ts, UTC)) for key, movie, rating, ts in rows
    )


def run_prepare(df: pd.DataFrame, fraction: float = 0.2) -> tuple[Any, Any, dict[str, Any]]:
    return prepare(df, ITEM_MAP, 100, threshold=3.5, min_likes=5, holdout_fraction=fraction)


def test_prepare_keeps_catalog_likes_and_holds_out_the_latest() -> None:
    rows = [("u2", movie, 4.0, ts) for ts, movie in enumerate([10, 20, 30, 40, 50, 60], 1)]
    rows += [
        ("u1", 10, 5.0, 5),
        ("u1", 20, 3.5, 3),
        ("u1", 30, 2.0, 4),  # not a like
        ("u1", 99, 5.0, 6),  # not in the catalog
        ("u3", 10, 1.0, 1),  # no likes at all
    ]
    train, holdout, counts = run_prepare(ratings(rows))

    # App users come after the 100 MovieLens users, in order of their key.
    assert holdout.to_numpy().tolist() == [[101, 5, 6, 4.0]]
    assert train[["user_idx", "item_idx", "timestamp"]].to_numpy().tolist() == [
        [101, 0, 1],
        [101, 1, 2],
        [100, 1, 3],  # u1 has too few likes to hold any out
        [101, 2, 3],
        [101, 3, 4],
        [100, 0, 5],
        [101, 4, 5],
    ]
    assert list(train.columns) == ["user_idx", "item_idx", "timestamp", "rating"]
    assert str(train["user_idx"].dtype) == "int32" and str(train["rating"].dtype) == "float32"
    assert counts == {
        "n_ratings": 11,
        "n_users": 3,
        "n_likes": 9,
        "n_likes_outside_catalog": 1,
        "n_train": 7,
        "n_holdout": 1,
        "n_holdout_users": 1,
        "last_rating": "1970-01-01",
    }


def test_the_holdout_is_the_given_share_of_each_users_likes() -> None:
    rows = [("u", 10 * (i % 6 + 1), 4.0, i) for i in range(6)]
    rows += [("v", 10 * (i % 6 + 1), 4.0, i) for i in range(5)]
    _, holdout, _ = run_prepare(ratings(rows), fraction=0.34)
    assert holdout.groupby("user_idx").size().tolist() == [2, 1]  # floor(6 * .34), at least 1


def test_an_empty_snapshot_gives_empty_tables() -> None:
    train, holdout, counts = run_prepare(ratings([]))
    assert train.empty and holdout.empty
    assert counts["n_ratings"] == 0 and counts["last_rating"] is None


# --- gate --------------------------------------------------------------------------------


def comparison(difference: float, ci_low: float, n_users: int = 1000) -> dict[str, Any]:
    return {"n_users": n_users, "difference": difference, "ci_low": ci_low, "ci_high": 0.01}


def test_paired_difference_is_challenger_minus_champion() -> None:
    rng = np.random.default_rng(0)
    champion = rng.random(500)
    out = paired_difference(champion + 0.1, champion, rng, 200, 0.95)
    assert out["difference"] == pytest.approx(0.1)
    assert out["ci_low"] == pytest.approx(0.1) and out["ci_high"] == pytest.approx(0.1)
    same = paired_difference(champion, champion, rng, 200, 0.95)
    assert same["difference"] == 0 and same["p_value"] == 1.0


def test_the_test_split_allows_a_loss_up_to_the_margin() -> None:
    assert decide(comparison(-0.001, -0.002), None, 0.002, 100)[0]
    passed, reasons = decide(comparison(-0.001, -0.0021), None, 0.002, 100)
    assert not passed and reasons[0].startswith("MovieLens test: FAILED")


def test_too_few_app_users_are_reported_but_not_applied() -> None:
    passed, reasons = decide(comparison(0.0, -0.001), comparison(-0.05, -0.1, 40), 0.002, 100)
    assert passed
    assert reasons[1].startswith("App holdout: not applied, 40 app users")


def test_enough_app_users_must_prefer_the_challenger() -> None:
    test = comparison(0.0, -0.001)
    assert decide(test, comparison(0.01, -0.001, 150), 0.002, 100)[0]
    assert not decide(test, comparison(0.0, 0.0, 150), 0.002, 100)[0]  # a tie is not better
    assert not decide(test, comparison(0.01, -0.01, 150), 0.002, 100)[0]


def test_bundles_with_different_catalogs_are_not_compared() -> None:
    model, rng = cast(Recommender, None), np.random.default_rng(0)
    with pytest.raises(GateError, match="different catalogs"):
        run_gate(Bundle(model, [1, 2], {}), Bundle(model, [2, 1], {}), load_params(), rng)


# --- pipeline ----------------------------------------------------------------------------


def test_retrain_depends_on_every_model_file() -> None:
    """The selected model can be any of them, and its code is listed file by file."""
    stages = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    model_files = {
        f"src/recsys/models/{f.name}" for f in (PROJECT_ROOT / "src/recsys/models").glob("*.py")
    }
    assert model_files <= set(stages["retrain"]["deps"])
    assert "models/retrained" in stages["export"]["deps"]
    assert "models/retrained" in stages["monitoring_reference"]["deps"]
