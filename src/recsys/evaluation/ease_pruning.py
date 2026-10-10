"""Stage `ease_pruning`: how many weights per movie does EASE need?

The full EASE matrix is 1.5 GB, far beyond what the API's 512 MB host can load, so the model
is stored with only its `keep_per_item` strongest weights per movie. This stage measures
what that costs: it fits the full matrix once, in memory (it is never written to disk), and
scores the validation split with the full matrix and with each pruned version of it.

`metrics/ease_pruning.json` lists every version and names the smallest one whose
`tuning.metric` stays within `ease_pruning.max_relative_drop` of the full matrix. That value
is copied by hand into `models.ease.keep_per_item`, and a test checks the two agree.
"""

from typing import Any

import numpy as np
import pandas as pd

from recsys.evaluation.evaluate import item_popularity, load_split
from recsys.evaluation.protocol import onboarding_eval_set, warm_eval_set
from recsys.models.ease import EASE, prune
from recsys.tuning.tune import validation_scores
from recsys.utils.config import load_params
from recsys.utils.io import write_json
from recsys.utils.paths import METRICS_DIR, PROCESSED_DIR
from recsys.utils.seed import make_deterministic

_DECIMALS = 6


def weights_size_mb(n_items: int, keep: int | None) -> float:
    """Size of the saved float32 weights: dense, or sparse with a column index per weight."""
    n_bytes = 4 * n_items * n_items if keep is None else 8 * n_items * min(keep, n_items - 1)
    return round(n_bytes / 1e6, 1)


def recommend_keep(variants: list[dict[str, Any]], max_relative_drop: float) -> int | None:
    """The smallest `keep_per_item` within the tolerated drop; None means keep everything."""
    allowed = [v for v in variants if v["relative_drop"] <= max_relative_drop]
    keep: int | None = min(allowed, key=lambda v: v["size_mb"])["keep_per_item"]
    return keep


def main() -> None:
    params = load_params()
    cfg = params["ease_pruning"]
    metric = params["tuning"]["metric"]
    l2 = params["models"]["ease"]["l2"]
    n_history = params["evaluation"]["onboarding"]["n_history"]
    make_deterministic(params["seed"], params["runtime"]["num_threads"])

    train, val = load_split("val")
    n_items = len(pd.read_parquet(PROCESSED_DIR / "item_map.parquet"))
    popularity = item_popularity(train, n_items)
    eval_sets = {
        "warm": warm_eval_set(train, val),
        "onboarding": onboarding_eval_set(train, val, n_history),
    }
    full = EASE(l2=l2).fit(train, n_items).weights
    assert isinstance(full, np.ndarray)

    variants: list[dict[str, Any]] = []
    for keep in [None, *sorted((k for k in cfg["keep_per_item"] if k is not None), reverse=True)]:
        model = EASE(l2=l2, keep_per_item=keep)
        model.n_items = n_items
        model.weights = full if keep is None else prune(full, keep)
        scores = validation_scores(
            model, eval_sets, metric, popularity, params["evaluation"]["batch_size"]
        )
        variants.append(
            {"keep_per_item": keep, "size_mb": weights_size_mb(n_items, keep), **scores}
        )
        print(variants[-1], flush=True)
    for variant in variants:
        variant["relative_drop"] = round(1.0 - variant["all"] / variants[0]["all"], _DECIMALS)

    result = {
        "metric": metric,
        "l2": l2,
        "max_relative_drop": cfg["max_relative_drop"],
        "recommended_keep_per_item": recommend_keep(variants, cfg["max_relative_drop"]),
        "variants": variants,
    }
    write_json(result, METRICS_DIR / "ease_pruning.json")
    print(f"recommended keep_per_item: {result['recommended_keep_per_item']}")


if __name__ == "__main__":
    main()
