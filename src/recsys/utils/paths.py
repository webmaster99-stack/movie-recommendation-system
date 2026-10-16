"""Filesystem layout of pipeline artifacts. Keep in sync with the outs/deps in dvc.yaml."""

from recsys.utils.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "ml-32m"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
EVAL_DIR = DATA_DIR / "evaluation"
MONITORING_REFERENCE_DIR = DATA_DIR / "monitoring" / "reference"
APP_FEEDBACK_DIR = DATA_DIR / "app_feedback"  # snapshot of the app's ratings (`dvc add`ed)
APP_FEEDBACK_PROCESSED_DIR = PROCESSED_DIR / "app_feedback"  # as likes: train + holdout
MODELS_DIR = PROJECT_ROOT / "models"  # trained on the training split
FINAL_MODELS_DIR = MODELS_DIR / "final"  # refitted on training + validation
RETRAINED_DIR = MODELS_DIR / "retrained"  # the selected model, also fitted on app feedback
METRICS_DIR = PROJECT_ROOT / "metrics"
TUNING_DIR = PROJECT_ROOT / "tuning"
BUNDLE_DIR = PROJECT_ROOT / "bundle"
