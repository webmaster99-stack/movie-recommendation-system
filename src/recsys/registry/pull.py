"""`python -m recsys.registry.pull`: download the bundle a registry alias points at.

The deploy workflow uses it to put the `champion` into the serving image, so what is served
is decided by the manual promotion and not by whatever `bundle/` the last pipeline run left.
The download is checked against the SHA256 recorded when the version was registered; a
bundle that does not match is deleted again and the command fails.

    uv run python -m recsys.registry.pull --out bundle

`--summary FILE` appends `version=`, `model=` and `bundle_sha256=` lines to FILE (in CI,
`$GITHUB_OUTPUT`), for the steps that build and label the image.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from recsys.registry.register import CHAMPION, dir_sha256
from recsys.utils.config import load_params
from recsys.utils.mlflow_utils import configure_mlflow


class BundleMismatchError(RuntimeError):
    """The downloaded files are not the ones that were registered."""


def pull(client: MlflowClient, registered_model: str, alias: str, out: Path) -> ModelVersion:
    """Download the version behind `alias` to `out`, replacing whatever is there."""
    version = client.get_model_version_by_alias(registered_model, alias)
    expected = version.tags["bundle_sha256"]
    out.parent.mkdir(parents=True, exist_ok=True)
    # Downloaded next to `out` and moved into place only once it has been checked, so a
    # failed or tampered download never leaves a bundle behind for the image build to copy.
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=version.source,
                dst_path=tmp,
                tracking_uri=client.tracking_uri,
            )
        )
        actual = dir_sha256(downloaded)
        if actual != expected:
            raise BundleMismatchError(
                f"{registered_model} v{version.version}: downloaded bundle hashes to {actual}, "
                f"but {expected} was registered"
            )
        if out.exists():
            shutil.rmtree(out)
        shutil.move(downloaded, out)
    return version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default=CHAMPION)
    parser.add_argument("--out", type=Path, required=True, help="directory to write the bundle to")
    parser.add_argument("--summary", type=Path, help="file to append key=value lines to")
    args = parser.parse_args()

    configure_mlflow()
    registered_model = load_params()["mlflow"]["registered_model"]
    version = pull(MlflowClient(), registered_model, args.alias, args.out)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as file:
            file.write(f"version={version.version}\n")
            file.write(f"model={version.tags.get('model', '')}\n")
            file.write(f"bundle_sha256={version.tags['bundle_sha256']}\n")
    print(f"{registered_model} v{version.version} ({args.alias}) -> {args.out}")


if __name__ == "__main__":
    main()
