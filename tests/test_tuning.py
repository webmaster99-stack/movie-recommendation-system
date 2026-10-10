import inspect
import json

import numpy as np
import optuna
import pandas as pd
import pytest

from recsys.evaluation.evaluate import item_popularity
from recsys.evaluation.protocol import onboarding_eval_set, warm_eval_set
from recsys.models import MODELS, build_model
from recsys.models.popularity import Popularity
from recsys.tuning.tune import make_sampler, suggest, validation_scores
from recsys.utils.config import load_params
from recsys.utils.paths import TUNING_DIR

SPACE = {
    "k": {"type": "int", "low": 1, "high": 100, "log": True},
    "similarity": {"type": "categorical", "choices": ["cosine", "bm25"]},
    "shrinkage": {"type": "float", "low": 0.0, "high": 200.0, "when": {"similarity": "cosine"}},
    "bm25_b": {"type": "float", "low": 0.0, "high": 1.0, "when": {"similarity": "bm25"}},
}


def test_suggest_skips_values_whose_condition_is_not_met() -> None:
    trial = optuna.trial.FixedTrial({"k": 7, "similarity": "cosine", "shrinkage": 12.0})
    assert suggest(trial, SPACE) == {"k": 7, "similarity": "cosine", "shrinkage": 12.0}
    trial = optuna.trial.FixedTrial({"k": 7, "similarity": "bm25", "bm25_b": 0.5})
    assert suggest(trial, SPACE) == {"k": 7, "similarity": "bm25", "bm25_b": 0.5}


def test_suggest_rounds_floats_to_four_significant_digits() -> None:
    trial = optuna.trial.FixedTrial({"k": 7, "similarity": "cosine", "shrinkage": 123.456789})
    assert suggest(trial, SPACE)["shrinkage"] == 123.5


def test_grid_sampler_tries_every_value_once() -> None:
    spec = {"grid": True, "space": {"x": {"type": "categorical", "choices": [None, 1, 2]}}}
    study = optuna.create_study(direction="maximize", sampler=make_sampler(spec, seed=0))
    seen = []
    study.optimize(lambda trial: float(seen.append(suggest(trial, spec["space"])["x"]) or 0))
    assert sorted(seen, key=str) == [1, 2, None]
    assert isinstance(make_sampler({"space": {}}, seed=0), optuna.samplers.TPESampler)


def test_suggest_rejects_unknown_types() -> None:
    with pytest.raises(ValueError, match="unknown type"):
        suggest(optuna.trial.FixedTrial({}), {"x": {"type": "bool"}})


def test_validation_scores_pool_all_users() -> None:
    def interactions(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["user_idx", "item_idx", "timestamp"])

    # Item popularity order is 0, 1, 2, 3. User 0 returns; users 2, 3 and 4 are new.
    history = interactions([(0, 0, 1), (1, 0, 1), (1, 1, 1), (5, 0, 1), (5, 1, 1), (5, 2, 1)])
    targets = interactions(
        [(0, 1, 10)]  # warm: top recommendation after excluding item 0 is item 1 -> hit
        + [(2, 0, 10), (2, 3, 11)]  # history [0], target 3; top is item 1 -> miss
        + [(3, 3, 10), (3, 1, 11)]  # history [3], target 1; top is item 0 -> miss
        + [(4, 1, 10), (4, 0, 11)]  # history [1], target 0; top is item 0 -> hit
    )
    model = Popularity().fit(history, n_items=4)
    eval_sets = {
        "warm": warm_eval_set(history, targets),
        "onboarding": onboarding_eval_set(history, targets, n_history=1),
    }
    scores = validation_scores(
        model, eval_sets, "ndcg_at_1", item_popularity(history, 4), batch_size=2
    )
    assert scores == {"warm": 1.0, "onboarding": round(1 / 3, 6), "all": 0.5}


def test_every_model_has_a_search_space_matching_its_constructor() -> None:
    params = load_params()
    assert set(params["tuning"]["models"]) == set(MODELS)
    for name, spec in params["tuning"]["models"].items():
        arguments = set(inspect.signature(MODELS[name].__init__).parameters)
        assert set(spec["space"]) <= arguments, name
        if spec.get("grid"):
            assert all(entry["type"] == "categorical" for entry in spec["space"].values()), name
        else:
            assert spec["n_trials"] >= 1
        fixed = {k: v for k, v in params["models"][name].items() if k not in spec["space"]}
        for start_point in spec.get("enqueue", []):
            assert set(start_point) <= set(spec["space"]), name
            build_model(name, {**fixed, **start_point})


@pytest.mark.parametrize("name", sorted(MODELS))
def test_params_use_the_tuned_values(name: str) -> None:
    """Once a model has been tuned, params.yaml must hold exactly the values that won."""
    path = TUNING_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{name} has not been tuned yet")
    result = json.loads(path.read_text(encoding="utf-8"))
    params = load_params()
    # Values outside the search space are fixed choices and may change after the search
    # (EASE's keep_per_item is set by the pruning study), so only searched values must match.
    searched = set(params["tuning"]["models"][name]["space"])
    tuned = {k: v for k, v in result["best_params"].items() if k in searched}
    assert {k: v for k, v in params["models"][name].items() if k in searched} == tuned
    assert result["best_scores"]["all"] == max(t["all"] for t in result["trials"])
    assert np.isfinite(result["best_scores"]["all"])
