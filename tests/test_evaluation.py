import numpy as np
import pandas as pd
import pytest

from recsys.evaluation.evaluate import (
    evaluate,
    flatten,
    item_popularity,
    segment_names,
    summarize,
)
from recsys.evaluation.protocol import onboarding_eval_set, warm_eval_set
from recsys.models.popularity import Popularity
from recsys.utils.config import load_params


def interactions(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "timestamp"])


# History period: user 0 liked items 1 then 0 (note the timestamps); user 1 liked item 0.
HISTORY = interactions([(0, 0, 20), (0, 1, 10), (1, 0, 10)])
# Evaluation period: user 0 returns, user 1 doesn't, users 2 and 3 are new.
TARGETS = interactions(
    [(0, 2, 100), (0, 3, 101)]
    + [(2, 3, 103), (2, 0, 100), (2, 1, 101), (2, 2, 102)]
    + [(3, 0, 100), (3, 1, 101)]
)


def test_warm_eval_set_uses_prior_history_in_time_order() -> None:
    es = warm_eval_set(HISTORY, TARGETS)
    assert es.user_idx.tolist() == [0]  # user 1 has no targets; users 2 and 3 no history
    assert es.histories[0].tolist() == [1, 0]
    assert es.targets[0].tolist() == [2, 3]


def test_onboarding_eval_set_splits_new_users_first_likes() -> None:
    es = onboarding_eval_set(HISTORY, TARGETS, n_history=2)
    # User 3 has only 2 likes, so nothing would be left to predict: skipped.
    assert es.user_idx.tolist() == [2]
    assert es.histories[0].tolist() == [0, 1]
    assert es.targets[0].tolist() == [2, 3]


def test_item_popularity_is_share_of_users_with_a_floor() -> None:
    pop = item_popularity(HISTORY, n_items=3)
    # 2 users; item 0 liked by both, item 1 by one, item 2 by nobody (counted as one).
    assert pop.tolist() == [1.0, 0.5, 0.5]


def test_evaluate_end_to_end_on_hand_computed_case() -> None:
    # Popularity order is 0, 1, then 2 and 3 tied at zero (lower index first).
    model = Popularity().fit(HISTORY, n_items=4)
    es = warm_eval_set(HISTORY, TARGETS)
    per_user, catalog = evaluate(
        model, es, ks=[1, 2], popularity=item_popularity(HISTORY, 4), batch_size=1
    )
    # User 0 has seen items 0 and 1, so the list is [2, 3]: both targets, in order.
    row = per_user.iloc[0]
    assert (row["history_len"], row["n_targets"]) == (2, 2)
    assert row["recall_at_1"] == 0.5 and row["recall_at_2"] == 1.0
    assert row["ndcg_at_1"] == 1.0 and row["ndcg_at_2"] == pytest.approx(1.0)
    assert row["mrr_at_2"] == 1.0
    assert row["novelty_at_2"] == 1.0  # both items have popularity 0.5
    assert catalog == {"coverage_at_1": 0.25, "coverage_at_2": 0.5}


def test_batching_does_not_change_results() -> None:
    model = Popularity().fit(HISTORY, n_items=4)
    es = onboarding_eval_set(HISTORY, TARGETS, n_history=1)
    pop = item_popularity(HISTORY, 4)
    one, _ = evaluate(model, es, ks=[2], popularity=pop, batch_size=1)
    many, _ = evaluate(model, es, ks=[2], popularity=pop, batch_size=100)
    pd.testing.assert_frame_equal(one, many)


def test_segment_names() -> None:
    assert segment_names([20, 100]) == ["history_lt_20", "history_20_to_99", "history_ge_100"]


def test_summarize_means_cis_and_segments() -> None:
    per_user = pd.DataFrame(
        {
            "user_idx": [0, 1, 2, 3],
            "history_len": [5, 19, 20, 150],
            "n_targets": [1, 1, 1, 1],
            "recall_at_10": [0.0, 1.0, 0.5, 0.5],
        }
    )
    out = summarize(
        per_user,
        {"coverage_at_10": 0.1},
        np.random.default_rng(0),
        n_resamples=200,
        confidence=0.95,
        segment_edges=[20, 100],
    )
    assert out["n_users"] == 4 and out["coverage_at_10"] == 0.1
    recall = out["recall_at_10"]
    assert recall["mean"] == 0.5
    assert recall["ci_low"] <= 0.5 <= recall["ci_high"]
    assert out["segments"] == {
        "history_lt_20": {"n_users": 2, "recall_at_10": 0.5},
        "history_20_to_99": {"n_users": 1, "recall_at_10": 0.5},
        "history_ge_100": {"n_users": 1, "recall_at_10": 0.5},
    }


def test_flatten_keeps_only_numbers() -> None:
    nested = {"model": "x", "warm": {"n_users": 3, "ndcg_at_10": {"mean": 0.5}}}
    assert flatten(nested, "val") == {"val/warm/n_users": 3.0, "val/warm/ndcg_at_10/mean": 0.5}


def test_params_cover_every_model_and_the_evaluation() -> None:
    from recsys.models import MODELS, build_model

    params = load_params()
    for name in MODELS:
        build_model(name, params["models"][name])  # params match the constructor
    assert params["evaluation"]["ks"] == sorted(params["evaluation"]["ks"])
