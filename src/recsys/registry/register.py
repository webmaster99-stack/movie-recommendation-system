"""Stage `register`: put the exported bundle in the MLflow model registry as the challenger.

The bundle is uploaded as the artifacts of an MLflow run and registered as a new version of
`mlflow.registered_model`, with the `challenger` alias pointing at it. The version is tagged
with the git commit and a SHA256 of the bundle, so a deployed API can say exactly which model
it is serving.

A new challenger does not replace the model in production. Moving the `champion` alias is a
manual step, taken after reading the model card: `python -m recsys.registry.promote`.
"""

import contextlib
import hashlib
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from recsys.evaluation.evaluate import flatten
from recsys.registry.export import read_json
from recsys.utils.config import load_params
from recsys.utils.mlflow_utils import tracked_run
from recsys.utils.paths import BUNDLE_DIR

CHALLENGER = "challenger"
CHAMPION = "champion"


def dir_sha256(path: Path) -> str:
    """One hash for a directory: every file's relative path and bytes, in sorted order."""
    digest = hashlib.sha256()
    for file in sorted(f for f in path.rglob("*") if f.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def main() -> None:
    registered_model = load_params()["mlflow"]["registered_model"]
    metadata = read_json(BUNDLE_DIR / "metadata.json")
    name = metadata["model"]

    with tracked_run(f"register-{name}", tags={"model": name, "stage": "register"}) as run:
        mlflow.log_params({**metadata["base_params"], **metadata["params"]})
        for split, metrics in metadata["metrics"].items():
            mlflow.log_metrics(flatten(metrics, split))
        mlflow.log_artifacts(str(BUNDLE_DIR), artifact_path="bundle")

        client = MlflowClient()
        with contextlib.suppress(MlflowException):  # it already exists
            client.create_registered_model(registered_model)
        version = client.create_model_version(
            name=registered_model,
            source=f"{run.info.artifact_uri}/bundle",
            run_id=run.info.run_id,
            tags={
                "model": name,
                "git_commit": run.data.tags["git_commit"],
                "git_dirty": run.data.tags["git_dirty"],
                "bundle_sha256": dir_sha256(BUNDLE_DIR),
            },
        )
        client.set_registered_model_alias(registered_model, CHALLENGER, version.version)
    print(f"registered {name} as {registered_model} v{version.version} ({CHALLENGER})")


if __name__ == "__main__":
    main()
