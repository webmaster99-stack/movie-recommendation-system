"""Stage `compare@<split>`: are the differences between the models real?

Every model is scored on the same users, so two models can be compared user by user. For
each pair of models this stage reports, on `tuning.metric`:

- the mean difference, with a bootstrap confidence interval (users resampled, the same
  resamples for every model);
- a paired t-test p-value on the per-user differences (with thousands of users the mean
  difference is close enough to normal for the test to hold, even though the metric is not);
- the p-value after Holm's correction, which accounts for testing many pairs at once.

Results are given for returning users (`warm`), new users (`onboarding`) and both together
(`all`), and written to `metrics/comparison_<split>.json`.
"""

import argparse
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from recsys.evaluation.metrics import Floats
from recsys.models import MODELS
from recsys.utils.config import load_params
from recsys.utils.io import write_json
from recsys.utils.paths import EVAL_DIR, METRICS_DIR
from recsys.utils.seed import make_deterministic

_DECIMALS = 6
_P_DIGITS = 4  # significant digits kept of a p-value


def holm(p_values: Floats) -> Floats:
    """Holm's step-down correction: adjusted p-values, in the order given."""
    order = np.argsort(p_values, kind="stable")
    factors = len(p_values) - np.arange(len(p_values))
    adjusted = np.minimum(np.maximum.accumulate(p_values[order] * factors), 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def paired_p_value(difference: Floats) -> float:
    """Two-sided paired t-test; models that agree on every user give 1."""
    std = difference.std(ddof=1)
    if std == 0:
        return 1.0
    t = difference.mean() / (std / np.sqrt(len(difference)))
    return float(2.0 * stats.t.sf(abs(t), df=len(difference) - 1))


def compare(
    values: pd.DataFrame,
    rng: np.random.Generator,
    n_resamples: int,
    confidence: float,
    alpha: float,
) -> dict[str, Any]:
    """`values` has one row per user and one column per model."""
    n = len(values)
    names = list(values.columns)
    data = values.to_numpy(dtype=np.float64)
    means = data.mean(axis=0)
    # Best first; equal means keep the column order.
    ranking = [int(i) for i in np.argsort(-means, kind="stable")]
    resampled = rng.multinomial(n, np.full(n, 1.0 / n), size=n_resamples) @ data / n
    tail = (1.0 - confidence) / 2.0

    pairs: list[dict[str, Any]] = []
    for a, b in combinations(ranking, 2):
        low, high = np.quantile(resampled[:, a] - resampled[:, b], [tail, 1.0 - tail])
        pairs.append(
            {
                "better": names[a],
                "worse": names[b],
                "difference": round(float(means[a] - means[b]), _DECIMALS),
                "ci_low": round(float(low), _DECIMALS),
                "ci_high": round(float(high), _DECIMALS),
                "p_value": paired_p_value(data[:, a] - data[:, b]),
            }
        )
    adjusted = holm(np.array([pair["p_value"] for pair in pairs]))
    for pair, p_adjusted in zip(pairs, adjusted, strict=True):
        pair["p_value"] = float(f"{pair['p_value']:.{_P_DIGITS}g}")
        pair["p_adjusted"] = float(f"{p_adjusted:.{_P_DIGITS}g}")
        pair["significant"] = bool(p_adjusted < alpha)
    return {
        "n_users": n,
        "ranking": [names[i] for i in ranking],
        "means": {names[i]: round(float(means[i]), _DECIMALS) for i in ranking},
        "pairs": pairs,
    }


def load_metric(split: str, metric: str) -> pd.DataFrame:
    """One row per (scenario, user), one column per model."""
    columns = {}
    for name in MODELS:
        per_user = pd.read_parquet(EVAL_DIR / split / f"{name}.parquet")
        columns[name] = per_user.set_index(["scenario", "user_idx"])[metric]
    values = pd.concat(columns, axis=1)
    if values.isna().to_numpy().any():
        raise ValueError("the models were not all evaluated on the same users")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["val", "test"])
    split: str = parser.parse_args().split

    params = load_params()
    alpha = params["comparison"]["alpha"]
    bootstrap = params["evaluation"]["bootstrap"]
    metric = params["tuning"]["metric"]
    rng = make_deterministic(params["seed"], params["runtime"]["num_threads"])
    values = load_metric(split, metric)

    scenario = values.index.get_level_values("scenario")
    scopes = {name: values[scenario == name] for name in scenario.unique()}
    scopes["all"] = values
    result = {
        "split": split,
        "metric": metric,
        "alpha": alpha,
        "scopes": {
            scope: compare(rows, rng, bootstrap["n_resamples"], bootstrap["confidence"], alpha)
            for scope, rows in scopes.items()
        },
    }
    write_json(result, METRICS_DIR / f"comparison_{split}.json")
    for scope, summary in result["scopes"].items():
        best, runner_up = summary["ranking"][:2]
        pair = summary["pairs"][0]
        print(
            f"{split} {scope}: {best} beats {runner_up} by {pair['difference']:.4f} "
            f"[{pair['ci_low']:.4f}, {pair['ci_high']:.4f}], adjusted p = {pair['p_adjusted']}"
        )


if __name__ == "__main__":
    main()
