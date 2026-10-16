"""Stage `retrain`: the selected model fitted on MovieLens plus the app's likes.

The model to serve keeps the hyperparameters the pipeline chose; what changes is the data.
It is fitted on training + validation, like `train_final`, plus the training part of the app
feedback (`data/processed/app_feedback/train.parquet`), and written to `models/retrained/`,
which is what `export` bundles.

For a recency variant only the base model sees the app's likes. The prior is still computed
from MovieLens: MovieLens ends in 2023 and the app's ratings start in 2026, so with a
half-life of days the prior would otherwise be decided by the app's few users alone.

`models/final/` is left as it is: its test results are the offline comparison, and they
would no longer be clean if the models had seen ratings from after the test period.
"""

import json
import shutil
import time

import mlflow
import pandas as pd

from recsys.evaluation.evaluate import load_split
from recsys.models import base_name, build_model
from recsys.models.recency import Recency
from recsys.models.train import dir_size_bytes
from recsys.utils.config import load_params
from recsys.utils.mlflow_utils import tracked_run
from recsys.utils.paths import APP_FEEDBACK_PROCESSED_DIR, METRICS_DIR, PROCESSED_DIR, RETRAINED_DIR
from recsys.utils.seed import make_deterministic


def main() -> None:
    params = load_params()
    make_deterministic(params["seed"], params["runtime"]["num_threads"])
    selection = json.loads((METRICS_DIR / "selection.json").read_text(encoding="utf-8"))
    name: str = selection["selected"]
    base = base_name(name)
    movielens, _ = load_split("test")  # training + validation
    app = pd.read_parquet(APP_FEEDBACK_PROCESSED_DIR / "train.parquet")
    n_items = len(pd.read_parquet(PROCESSED_DIR / "item_map.parquet"))

    tags = {"model": name, "stage": "retrain", "fitted_on": "train+val+app_feedback"}
    with tracked_run(f"retrain-{name}", tags=tags):
        start = time.perf_counter()
        model = build_model(base, params["models"][base], params["seed"])
        model.fit(pd.concat([movielens, app], ignore_index=True), n_items)
        if base != name:
            variant = build_model(name, params["models"][name]).fit(movielens, n_items)
            assert isinstance(variant, Recency)
            model = variant.wrap(model)
        seconds = time.perf_counter() - start

        if RETRAINED_DIR.exists():
            shutil.rmtree(RETRAINED_DIR)
        model.save(RETRAINED_DIR)
        size_mb = dir_size_bytes(RETRAINED_DIR) / 1e6
        mlflow.log_params(model.params)
        mlflow.log_metrics(
            {"train_seconds": seconds, "model_size_mb": size_mb, "app_likes": len(app)}
        )
    print(f"{name}: fitted on train+val and {len(app)} app likes in {seconds:.1f}s")


if __name__ == "__main__":
    main()
