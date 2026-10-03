"""Stage `evaluate@<model>`: score a trained model on the validation split.

Every model is ranked against the whole catalog (no sampled negatives, which distort model
comparisons; Krichene & Rendle 2020) with the movies in the user's history excluded.

Outputs:
- `metrics/val_<model>.json`: means with bootstrap confidence intervals, per scenario.
- `data/evaluation/val/<model>.parquet`: the per-user values behind those means, kept so
  that models can be compared with paired significance tests in Phase 4.

Latency goes to MLflow only: it varies run to run and would break the reproducibility check
if it were written to a DVC-tracked file.

The test split is never touched here. It is evaluated once, in Phase 4, after tuning.
"""

import argparse
import time
from collections.abc import Sequence
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from recsys.evaluation import metrics
from recsys.evaluation.protocol import EvalSet, onboarding_eval_set, warm_eval_set
from recsys.models import MODELS, load_model
from recsys.models.base import History, ItemIds, Recommender
from recsys.models.train import dir_size_bytes
from recsys.utils.config import load_params
from recsys.utils.io import write_json, write_parquet
from recsys.utils.mlflow_utils import tracked_run
from recsys.utils.paths import EVAL_DIR, METRICS_DIR, MODELS_DIR, SPLITS_DIR
from recsys.utils.seed import make_deterministic

_SPLIT_ORDER = ("train", "val", "test")
_ID_COLUMNS = ("user_idx", "history_len", "n_targets")
_DECIMALS = 6


def load_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(history, targets): the target split and everything that came before it."""
    earlier = _SPLIT_ORDER[: _SPLIT_ORDER.index(split)]
    history = pd.concat(
        [pd.read_parquet(SPLITS_DIR / f"{s}.parquet") for s in earlier], ignore_index=True
    )
    return history, pd.read_parquet(SPLITS_DIR / f"{split}.parquet")


def item_popularity(history: pd.DataFrame, n_items: int) -> metrics.Floats:
    """Share of users who liked each movie; never-liked movies count as liked once."""
    counts = np.bincount(history["item_idx"].to_numpy(), minlength=n_items)
    return np.maximum(counts, 1) / history["user_idx"].nunique()


def recommend_all(
    model: Recommender, histories: Sequence[History], k: int, batch_size: int
) -> ItemIds:
    batches = [
        model.recommend_batch(histories[i : i + batch_size], k)
        for i in range(0, len(histories), batch_size)
    ]
    return np.concatenate(batches)


def evaluate(
    model: Recommender,
    eval_set: EvalSet,
    ks: Sequence[int],
    popularity: metrics.Floats,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-user metrics (one row per user) and catalog-level metrics, at every k in `ks`."""
    recs = recommend_all(model, eval_set.histories, max(ks), batch_size)
    hits = metrics.hit_matrix(recs, eval_set.targets)
    n_targets = np.array([len(t) for t in eval_set.targets], dtype=np.int64)

    columns: dict[str, Any] = {
        "user_idx": eval_set.user_idx,
        "history_len": np.array([len(h) for h in eval_set.histories], dtype=np.int32),
        "n_targets": n_targets.astype(np.int32),
    }
    catalog: dict[str, float] = {}
    for k in ks:
        columns[f"recall_at_{k}"] = metrics.recall(hits[:, :k], n_targets)
        columns[f"ndcg_at_{k}"] = metrics.ndcg(hits[:, :k], n_targets)
        columns[f"mrr_at_{k}"] = metrics.mrr(hits[:, :k])
        columns[f"novelty_at_{k}"] = metrics.novelty(recs[:, :k], popularity)
        columns[f"popularity_at_{k}"] = metrics.mean_popularity(recs[:, :k], popularity)
        catalog[f"coverage_at_{k}"] = round(metrics.coverage(recs[:, :k], model.n_items), _DECIMALS)
    return pd.DataFrame(columns), catalog


def segment_names(edges: Sequence[int]) -> list[str]:
    """[20, 100] -> history_lt_20, history_20_to_99, history_ge_100."""
    middle = [f"history_{a}_to_{b - 1}" for a, b in zip(edges, edges[1:], strict=False)]
    return [f"history_lt_{edges[0]}", *middle, f"history_ge_{edges[-1]}"]


def summarize(
    per_user: pd.DataFrame,
    catalog: dict[str, float],
    rng: np.random.Generator,
    n_resamples: int,
    confidence: float,
    segment_edges: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Mean and confidence interval per metric, plus means per history-length segment."""
    names = [c for c in per_user.columns if c not in _ID_COLUMNS]
    values = per_user[names].to_numpy(dtype=np.float64)
    low, high = metrics.bootstrap_ci(values, rng, n_resamples, confidence)

    summary: dict[str, Any] = {"n_users": len(per_user), **catalog}
    for j, name in enumerate(names):
        summary[name] = {
            "mean": round(float(values[:, j].mean()), _DECIMALS),
            "ci_low": round(float(low[j]), _DECIMALS),
            "ci_high": round(float(high[j]), _DECIMALS),
        }
    if segment_edges:
        segment = np.digitize(per_user["history_len"].to_numpy(), segment_edges)
        summary["segments"] = {}
        for i, segment_name in enumerate(segment_names(segment_edges)):
            rows = values[segment == i]
            means = {
                n: round(float(m), _DECIMALS) for n, m in zip(names, rows.mean(axis=0), strict=True)
            }
            summary["segments"][segment_name] = {"n_users": len(rows), **means}
    return summary


def measure_latency(model: Recommender, histories: Sequence[History], k: int) -> dict[str, float]:
    """p50/p95 milliseconds of single-user `recommend` calls, as the API will make them."""
    timings = []
    for history in histories:
        start = time.perf_counter()
        model.recommend(history, k)
        timings.append((time.perf_counter() - start) * 1000)
    p50, p95 = np.percentile(timings, [50, 95])
    return {"latency_p50_ms": float(p50), "latency_p95_ms": float(p95)}


def flatten(nested: dict[str, Any], prefix: str) -> dict[str, float]:
    """Nested results -> flat `a/b/c` metric names for MLflow; non-numeric values are dropped."""
    flat: dict[str, float] = {}
    for key, value in nested.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}/{key}"))
        elif isinstance(value, int | float):
            flat[f"{prefix}/{key}"] = float(value)
    return flat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODELS))
    name = parser.parse_args().model
    split = "val"

    params = load_params()
    cfg = params["evaluation"]
    rng = make_deterministic(params["seed"], params["runtime"]["num_threads"])
    history, targets = load_split(split)
    model = load_model(name, MODELS_DIR / name)
    popularity = item_popularity(history, model.n_items)
    bootstrap = cfg["bootstrap"]

    scenarios = {
        "warm": (warm_eval_set(history, targets), cfg["history_segments"]),
        "onboarding": (
            onboarding_eval_set(history, targets, cfg["onboarding"]["n_history"]),
            None,  # every onboarding history has the same length
        ),
    }
    results: dict[str, Any] = {
        "model": name,
        "split": split,
        "model_size_bytes": dir_size_bytes(MODELS_DIR / name),
    }
    frames = []
    for scenario, (eval_set, segment_edges) in scenarios.items():
        per_user, catalog = evaluate(model, eval_set, cfg["ks"], popularity, cfg["batch_size"])
        results[scenario] = summarize(
            per_user, catalog, rng, bootstrap["n_resamples"], bootstrap["confidence"], segment_edges
        )
        per_user.insert(0, "scenario", scenario)
        frames.append(per_user)

    write_parquet(pd.concat(frames, ignore_index=True), EVAL_DIR / split / f"{name}.parquet")
    write_json(results, METRICS_DIR / f"{split}_{name}.json")

    warm_histories = scenarios["warm"][0].histories[: cfg["latency_users"]]
    with tracked_run(f"evaluate-{name}-{split}", tags={"model": name, "stage": "evaluate"}):
        mlflow.log_params(model.params)
        mlflow.log_metrics(flatten(results, split))
        mlflow.log_metrics(measure_latency(model, warm_histories, max(cfg["ks"])))
        mlflow.log_artifact(str(METRICS_DIR / f"{split}_{name}.json"))

    for scenario in scenarios:
        ndcg = results[scenario]["ndcg_at_10"]
        print(
            f"{name} {split} {scenario}: NDCG@10 {ndcg['mean']:.4f} "
            f"[{ndcg['ci_low']:.4f}, {ndcg['ci_high']:.4f}], n={results[scenario]['n_users']}"
        )


if __name__ == "__main__":
    main()
