"""MLflow setup shared by every training/evaluation stage.

Tracking goes to DagsHub when MLFLOW_TRACKING_URI (plus MLFLOW_TRACKING_USERNAME/PASSWORD) is
set, otherwise to a local SQLite file, so a stranger can run everything with no accounts.
Every run is tagged with the exact code and config it came from.
"""

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow
from dotenv import load_dotenv
from mlflow.entities import Run

from recsys.utils.config import PARAMS_PATH, PROJECT_ROOT, load_params


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.stdout.strip()


def configure_mlflow() -> str:
    # Variables already set in the environment (e.g. CI secrets) win over the .env file.
    load_dotenv(PROJECT_ROOT / ".env")
    params = load_params()["mlflow"]
    name = params["experiment_name"]
    remote_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if remote_uri:
        mlflow.set_tracking_uri(remote_uri)
        mlflow.set_experiment(name)
        return remote_uri

    local_uri = f"sqlite:///{(PROJECT_ROOT / params['local_db']).as_posix()}"
    mlflow.set_tracking_uri(local_uri)
    if mlflow.get_experiment_by_name(name) is None:
        # Pin artifacts to the project root instead of wherever the process was started from.
        artifacts = (PROJECT_ROOT / params["local_artifacts"]).as_uri()
        mlflow.create_experiment(name, artifact_location=artifacts)
    mlflow.set_experiment(name)
    return local_uri


@contextmanager
def tracked_run(run_name: str, tags: dict[str, Any] | None = None) -> Iterator[Run]:
    """Start an MLflow run tagged with git commit and config files for traceability."""
    configure_mlflow()
    provenance = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": str(bool(_git("status", "--porcelain"))),
    }
    with mlflow.start_run(run_name=run_name, tags={**provenance, **(tags or {})}) as run:
        for path in (PARAMS_PATH, PROJECT_ROOT / "uv.lock", PROJECT_ROOT / "dvc.lock"):
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="provenance")
        yield run
