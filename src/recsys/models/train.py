"""Stage `train@<model>`: fit one model on the training split and save it to `models/<model>`.

Training time goes to MLflow only. It changes from run to run, so it must stay out of the
DVC-tracked files, which have to be identical on every run.
"""

import argparse
import logging
import time
from pathlib import Path

import mlflow
import pandas as pd

from recsys.models import MODELS, build_model
from recsys.utils.config import load_params
from recsys.utils.mlflow_utils import tracked_run
from recsys.utils.paths import MODELS_DIR, PROCESSED_DIR, SPLITS_DIR
from recsys.utils.seed import make_deterministic


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODELS))
    name = parser.parse_args().model

    logging.basicConfig(format="%(message)s")
    logging.getLogger("recsys").setLevel(logging.INFO)  # per-epoch progress
    params = load_params()
    make_deterministic(params["seed"], params["runtime"]["num_threads"])
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    n_items = len(pd.read_parquet(PROCESSED_DIR / "item_map.parquet"))

    with tracked_run(f"train-{name}", tags={"model": name, "stage": "train"}):
        model = build_model(name, params["models"][name], params["seed"])
        mlflow.log_params(model.params)
        start = time.perf_counter()
        model.fit(train, n_items)
        seconds = time.perf_counter() - start
        model.save(MODELS_DIR / name)
        size_mb = dir_size_bytes(MODELS_DIR / name) / 1e6
        mlflow.log_metrics({"train_seconds": seconds, "model_size_mb": size_mb})
    print(f"{name}: trained in {seconds:.1f}s, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
