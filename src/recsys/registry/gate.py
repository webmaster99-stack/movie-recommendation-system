"""The promotion gate: may a challenger replace the champion?

`recsys.registry.promote` runs it before moving the `champion` alias. Both bundles are
downloaded from the registry and score the same users, so the comparison is paired, on
`tuning.metric`:

1. **MovieLens test split, no worse.** Every test user, returning and new pooled, as in
   `evaluate --split test`. The challenger passes when the lower end of the 95% confidence
   interval of (challenger - champion) is at least `-retraining.gate.margin`. This catches a
   retrain that broke something; it can't show that app feedback helped, since MovieLens
   users never saw the app.
2. **The app's held-out likes, better.** Each app user's latest likes, which neither model
   was fitted on (stage `app_feedback`), predicted from their earlier ones. The challenger
   must be ahead on average and not credibly behind (the same margin). With fewer than
   `retraining.gate.min_app_users` such users the difference is noise, so this check is
   reported but not applied.

Both bundles must use the same catalog: the users' movies are given to both by index. The
app feedback in the workspace must be the snapshot the challenger was fitted on, or its
held-out likes would not be held out; both are checked.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.data.download import sha256_of
from recsys.evaluation import metrics
from recsys.evaluation.compare import paired_p_value
from recsys.evaluation.evaluate import evaluate, item_popularity, load_split
from recsys.evaluation.protocol import EvalSet, onboarding_eval_set, warm_eval_set
from recsys.models import load_model
from recsys.models.base import Recommender
from recsys.registry.export import read_json
from recsys.utils.paths import APP_FEEDBACK_PROCESSED_DIR

_DECIMALS = 6


class GateError(RuntimeError):
    """The two bundles can't be compared."""


@dataclass(frozen=True)
class Bundle:
    model: Recommender
    movie_ids: list[int]
    metadata: dict[str, Any]


def load_bundle(path: Path) -> Bundle:
    metadata = read_json(path / "metadata.json")
    items = json.loads((path / "items.json").read_text(encoding="utf-8"))
    model = load_model(metadata["model"], path / "model")
    return Bundle(model, [int(item["movie_id"]) for item in items], metadata)


def paired_difference(
    challenger: metrics.Floats,
    champion: metrics.Floats,
    rng: np.random.Generator,
    n_resamples: int,
    confidence: float,
) -> dict[str, Any]:
    """Mean per-user difference (challenger - champion) with a bootstrap confidence interval."""
    difference = challenger - champion
    n = len(difference)
    resampled = rng.multinomial(n, np.full(n, 1.0 / n), size=n_resamples) @ difference / n
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(resampled, [tail, 1.0 - tail])
    return {
        "n_users": n,
        "challenger": round(float(challenger.mean()), _DECIMALS),
        "champion": round(float(champion.mean()), _DECIMALS),
        "difference": round(float(difference.mean()), _DECIMALS),
        "ci_low": round(float(low), _DECIMALS),
        "ci_high": round(float(high), _DECIMALS),
        "p_value": float(f"{paired_p_value(difference):.4g}"),
    }


def decide(
    test: dict[str, Any], app: dict[str, Any] | None, margin: float, min_app_users: int
) -> tuple[bool, list[str]]:
    """Whether the challenger passes, and one line per check saying why."""
    reasons = []
    test_ok = test["ci_low"] >= -margin
    reasons.append(
        f"MovieLens test: {'passed' if test_ok else 'FAILED'}, difference "
        f"{test['difference']:+.4f} [{test['ci_low']:+.4f}, {test['ci_high']:+.4f}] "
        f"over {test['n_users']} users (may lose at most {margin})"
    )
    n_app = 0 if app is None else app["n_users"]
    if app is None or n_app < min_app_users:
        shown = (
            ""
            if app is None
            else f", difference {app['difference']:+.4f} [{app['ci_low']:+.4f}, "
            f"{app['ci_high']:+.4f}]"
        )
        reasons.append(
            f"App holdout: not applied, {n_app} app users with held-out likes "
            f"(needs {min_app_users}){shown}"
        )
        return test_ok, reasons
    app_ok = app["difference"] > 0 and app["ci_low"] >= -margin
    reasons.append(
        f"App holdout: {'passed' if app_ok else 'FAILED'}, difference {app['difference']:+.4f} "
        f"[{app['ci_low']:+.4f}, {app['ci_high']:+.4f}] over {n_app} users (must be ahead)"
    )
    return test_ok and app_ok, reasons


def per_user(
    model: Recommender,
    eval_set: EvalSet,
    popularity: metrics.Floats,
    metric: str,
    params: dict[str, Any],
) -> metrics.Floats:
    cfg = params["evaluation"]
    values, _ = evaluate(model, eval_set, cfg["ks"], popularity, cfg["batch_size"])
    return values[metric].to_numpy(dtype=np.float64)


def run_gate(
    challenger: Bundle, champion: Bundle, params: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    if challenger.movie_ids != champion.movie_ids:
        raise GateError(
            "the two bundles have different catalogs, so the same history can't be given to "
            "both; promote with --skip-gate after comparing them another way"
        )
    app_file = APP_FEEDBACK_PROCESSED_DIR / "train.parquet"
    fitted_on = challenger.metadata.get("app_feedback", {}).get("sha256", {}).get("train.parquet")
    if fitted_on != sha256_of(app_file):
        raise GateError(
            f"{app_file} is not the app feedback the challenger was fitted on; check out the "
            "commit it was built from and run `dvc pull`"
        )

    metric = params["tuning"]["metric"]
    bootstrap = params["evaluation"]["bootstrap"]
    history, targets = load_split("test")
    popularity = item_popularity(history, challenger.model.n_items)
    n_history = params["evaluation"]["onboarding"]["n_history"]
    test_sets = [
        warm_eval_set(history, targets),
        onboarding_eval_set(history, targets, n_history),
    ]
    app_set = warm_eval_set(
        pd.read_parquet(app_file), pd.read_parquet(APP_FEEDBACK_PROCESSED_DIR / "holdout.parquet")
    )

    def compare(eval_sets: list[EvalSet]) -> dict[str, Any]:
        a, b = (
            np.concatenate([per_user(m, s, popularity, metric, params) for s in eval_sets])
            for m in (challenger.model, champion.model)
        )
        return paired_difference(a, b, rng, bootstrap["n_resamples"], bootstrap["confidence"])

    test = compare(test_sets)
    app = compare([app_set]) if len(app_set.user_idx) else None
    gate = params["retraining"]["gate"]
    passed, reasons = decide(test, app, gate["margin"], gate["min_app_users"])
    return {"metric": metric, "passed": passed, "reasons": reasons, "test": test, "app": app}
