"""Hyperparameter search for one model: `python -m recsys.tuning.tune <model>`.

Each trial trains the model on the training split with one set of hyperparameters and
scores it on the validation split, exactly as the `evaluate` stage does. (A recency variant
does not retrain anything: a trial recomputes its prior and reuses the base model saved in
`models/`.) The score is
`tuning.metric` averaged over all validation users, warm and onboarding together, so each
user counts once. The test split is never read.

The search is an Optuna TPE study with a fixed seed and one trial at a time, so running it
again proposes the same hyperparameters in the same order (`grid: true` tries every
combination of a small space instead). The search space, the number of
trials and the starting points are in `params.yaml` under `tuning.models.<model>`.

The result is written to `tuning/<model>.json` (committed to git). Tuning is not a DVC
stage: the best values are copied into `params.yaml` (`models.<model>`), which stays the one
place the pipeline reads hyperparameters from, and a test checks the two agree. Trials are
also logged to MLflow as nested runs, where the timings go.
"""

import argparse
import gc
import json
import logging
import time
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd

from recsys.evaluation.evaluate import evaluate, item_popularity, load_split
from recsys.evaluation.metrics import Floats
from recsys.evaluation.protocol import EvalSet, onboarding_eval_set, warm_eval_set
from recsys.models import MODELS, base_name, build_model, load_model
from recsys.models.base import Recommender
from recsys.models.recency import Recency
from recsys.utils.config import load_params
from recsys.utils.io import write_json
from recsys.utils.mlflow_utils import tracked_run
from recsys.utils.paths import MODELS_DIR, PROCESSED_DIR, TUNING_DIR
from recsys.utils.seed import make_deterministic

_DECIMALS = 6
_SIGNIFICANT_DIGITS = 4  # sampled floats are rounded so they can be copied into params.yaml


def suggest(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sample one value per entry of `space`.

    An entry with `when: {other: value}` is sampled only if `other` was given that value
    earlier in the same trial (e.g. `shrinkage` only when `similarity` is `cosine`);
    otherwise it is left out and the model uses its default.
    """
    values: dict[str, Any] = {}
    for name, spec in space.items():
        if any(values.get(key) != wanted for key, wanted in spec.get("when", {}).items()):
            continue
        if spec["type"] == "categorical":
            values[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec["type"] == "int":
            values[name] = trial.suggest_int(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif spec["type"] == "float":
            value = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            values[name] = float(f"{value:.{_SIGNIFICANT_DIGITS}g}")
        else:
            raise ValueError(f"unknown type {spec['type']!r} for {name}")
    return values


def make_sampler(spec: dict[str, Any], seed: int) -> optuna.samplers.BaseSampler:
    """TPE by default. `grid: true` tries every combination of an all-categorical space once."""
    if spec.get("grid"):
        grid = {name: entry["choices"] for name, entry in spec["space"].items()}
        return optuna.samplers.GridSampler(grid, seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def validation_scores(
    model: Recommender,
    eval_sets: dict[str, EvalSet],
    metric: str,
    popularity: Floats,
    batch_size: int,
) -> dict[str, float]:
    """Mean of `metric` (e.g. `ndcg_at_10`) per scenario and over all users (`all`)."""
    k = int(metric.rsplit("_", 1)[1])
    per_scenario = {
        scenario: evaluate(model, eval_set, [k], popularity, batch_size)[0][metric].to_numpy()
        for scenario, eval_set in eval_sets.items()
    }
    scores = {scenario: float(values.mean()) for scenario, values in per_scenario.items()}
    scores["all"] = float(np.concatenate(list(per_scenario.values())).mean())
    return {name: round(value, _DECIMALS) for name, value in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODELS))
    name = parser.parse_args().model

    logging.basicConfig(format="%(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    params = load_params()
    cfg = params["tuning"]
    spec = cfg["models"][name]
    seed = params["seed"]
    make_deterministic(seed, params["runtime"]["num_threads"])

    train, val = load_split("val")
    n_items = len(pd.read_parquet(PROCESSED_DIR / "item_map.parquet"))
    popularity = item_popularity(train, n_items)
    n_history = params["evaluation"]["onboarding"]["n_history"]
    eval_sets = {
        "warm": warm_eval_set(train, val),
        "onboarding": onboarding_eval_set(train, val, n_history),
    }
    # Hyperparameters that are not searched keep their value from `models.<model>`.
    fixed = {k: v for k, v in params["models"][name].items() if k not in spec["space"]}
    done: dict[str, dict[str, float]] = {}  # the sampler can propose the same values twice
    # A recency variant is tuned on top of its base model as trained by the pipeline.
    base = None
    if base_name(name) != name:
        base = load_model(base_name(name), MODELS_DIR / base_name(name))

    def objective(trial: optuna.Trial) -> float:
        model_params = {**fixed, **suggest(trial, spec["space"])}
        trial.set_user_attr("model_params", model_params)
        key = json.dumps(model_params, sort_keys=True)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(model_params)
            if key not in done:
                start = time.perf_counter()
                model = build_model(name, model_params, seed).fit(train, n_items)
                if isinstance(model, Recency) and base is not None:
                    model.wrap(base)
                mlflow.log_metric("train_seconds", time.perf_counter() - start)
                done[key] = validation_scores(
                    model, eval_sets, cfg["metric"], popularity, params["evaluation"]["batch_size"]
                )
                del model
                gc.collect()
            scores = done[key]
            mlflow.log_metrics(
                {f"val/{scenario}/{cfg['metric']}": v for scenario, v in scores.items()}
            )
        trial.set_user_attr("scores", scores)
        print(f"trial {trial.number}: {scores} {model_params}", flush=True)
        return scores["all"]

    study = optuna.create_study(direction="maximize", sampler=make_sampler(spec, seed))
    for start_point in spec.get("enqueue", []):
        study.enqueue_trial(start_point)
    with tracked_run(f"tune-{name}", tags={"model": name, "stage": "tune"}):
        study.optimize(objective, n_trials=spec.get("n_trials"))  # a grid stops by itself
        best = study.best_trial
        result = {
            "model": name,
            "metric": cfg["metric"],
            "seed": seed,
            "best_params": best.user_attrs["model_params"],
            "best_scores": best.user_attrs["scores"],
            "best_trial": best.number,
            "trials": [
                {
                    "number": t.number,
                    "params": t.user_attrs["model_params"],
                    **t.user_attrs["scores"],
                }
                for t in study.trials
            ],
        }
        write_json(result, TUNING_DIR / f"{name}.json")
        mlflow.log_params(result["best_params"])
        mlflow.log_metrics(
            {f"val/{scenario}/{cfg['metric']}": v for scenario, v in result["best_scores"].items()}
        )
        mlflow.log_artifact(str(TUNING_DIR / f"{name}.json"))
    print(f"{name}: best {cfg['metric']} {best.value:.4f} with {result['best_params']}")


if __name__ == "__main__":
    main()
