"""Stage `select`: pick the model to serve.

The winner is the model with the best validation `tuning.metric` over all users (returning
and new pooled) among those small enough to serve: the API runs on a 512 MB host, so a
model larger than `selection.max_model_size_mb` is not a candidate however well it scores.

Only validation results are read. The test split plays no part in the choice, which is what
keeps the winner's test score an honest estimate.

`metrics/selection.json` records the ranking, the winner, and whether its lead over the
runner-up is statistically significant (from the `compare@val` stage).
"""

import json
from typing import Any

from recsys.models import MODELS, base_name
from recsys.utils.config import load_params
from recsys.utils.io import write_json
from recsys.utils.paths import METRICS_DIR

_SCOPES = ("all", "warm", "onboarding")


def candidate(metrics: dict[str, Any], metric: str, max_size_mb: float) -> dict[str, Any]:
    size_mb = round(metrics["model_size_bytes"] / 1e6, 1)
    return {
        "model": metrics["model"],
        **{scope: metrics[scope][metric]["mean"] for scope in _SCOPES},
        "model_size_mb": size_mb,
        "within_budget": size_mb <= max_size_mb,
    }


def select(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Candidates best first: within budget before over budget, then by score, then by name."""
    return sorted(candidates, key=lambda c: (not c["within_budget"], -c["all"], c["model"]))


def lead(comparison: dict[str, Any], winner: str, runner_up: str) -> dict[str, Any]:
    """The winner-vs-runner-up entry of `compare`'s results over all users."""
    for pair in comparison["scopes"]["all"]["pairs"]:
        if {pair["better"], pair["worse"]} == {winner, runner_up}:
            keys = ("difference", "ci_low", "ci_high", "p_adjusted", "significant")
            return {key: pair[key] for key in keys}
    raise KeyError(f"no comparison of {winner} and {runner_up}")


def main() -> None:
    params = load_params()
    metric = params["tuning"]["metric"]
    max_size_mb = params["selection"]["max_model_size_mb"]

    candidates = [
        candidate(
            json.loads((METRICS_DIR / f"val_{name}.json").read_text(encoding="utf-8")),
            metric,
            max_size_mb,
        )
        for name in MODELS
    ]
    ranked = select(candidates)
    if not ranked[0]["within_budget"]:
        raise RuntimeError(f"no model fits in {max_size_mb} MB")
    winner, runner_up = ranked[0]["model"], ranked[1]["model"]
    comparison = json.loads((METRICS_DIR / "comparison_val.json").read_text(encoding="utf-8"))
    result = {
        "split": "val",
        "metric": metric,
        "max_model_size_mb": max_size_mb,
        "selected": winner,
        "base_model": base_name(winner),
        "runner_up": runner_up,
        "lead_over_runner_up": lead(comparison, winner, runner_up),
        "candidates": ranked,
    }
    write_json(result, METRICS_DIR / "selection.json")
    print(f"selected {winner} ({metric} {ranked[0]['all']:.4f}); runner-up {runner_up}")


if __name__ == "__main__":
    main()
