"""Stages `train@<model>` and `train_final@<model>`: fit one model and save it.

- `train` fits on the training split and writes `models/<model>`, which is what the
  validation split is scored with and what tuning compares.
- `train_final` (`--final`) refits the same hyperparameters on training + validation and
  writes `models/final/<model>`: the model the test split is scored with, once, and the one
  that is exported for serving.

Only the five base models are trained. A recency variant reuses its base model's weights.

Training time goes to MLflow only. It changes from run to run, so it must stay out of the
DVC-tracked files, which have to be identical on every run.
"""

import argparse
import logging
import time
from pathlib import Path

import mlflow
import pandas as pd

from recsys.models import BASE_MODELS, build_model
from recsys.utils.config import load_params
from recsys.utils.mlflow_utils import tracked_run
from recsys.utils.paths import FINAL_MODELS_DIR, MODELS_DIR, PROCESSED_DIR, SPLITS_DIR
from recsys.utils.seed import make_deterministic


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(BASE_MODELS))
    parser.add_argument("--final", action="store_true", help="fit on training + validation")
    args = parser.parse_args()
    name: str = args.model
    splits = ("train", "val") if args.final else ("train",)
    out_dir = (FINAL_MODELS_DIR if args.final else MODELS_DIR) / name
    run_name = f"train-{name}-final" if args.final else f"train-{name}"

    logging.basicConfig(format="%(message)s")
    logging.getLogger("recsys").setLevel(logging.INFO)  # per-epoch progress
    params = load_params()
    make_deterministic(params["seed"], params["runtime"]["num_threads"])
    train = pd.concat(
        [pd.read_parquet(SPLITS_DIR / f"{split}.parquet") for split in splits], ignore_index=True
    )
    n_items = len(pd.read_parquet(PROCESSED_DIR / "item_map.parquet"))

    tags = {"model": name, "stage": "train", "fitted_on": "+".join(splits)}
    with tracked_run(run_name, tags=tags):
        model = build_model(name, params["models"][name], params["seed"])
        mlflow.log_params(model.params)
        start = time.perf_counter()
        model.fit(train, n_items)
        seconds = time.perf_counter() - start
        model.save(out_dir)
        size_mb = dir_size_bytes(out_dir) / 1e6
        mlflow.log_metrics({"train_seconds": seconds, "model_size_mb": size_mb})
    print(f"{name}: trained on {'+'.join(splits)} in {seconds:.1f}s, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
