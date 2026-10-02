"""Filesystem layout of pipeline artifacts. Keep in sync with the outs/deps in dvc.yaml."""

from recsys.utils.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "ml-32m"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
METRICS_DIR = PROJECT_ROOT / "metrics"
