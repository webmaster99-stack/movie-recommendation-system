"""`python -m recsys.registry.promote`: make the challenger the champion.

The manual gate between "the pipeline produced a model" and "the API serves it". It moves
the `champion` alias of the registered model to the version `challenger` points at (or to
`--version N`). Nothing is retrained or uploaded, and the previous champion's version stays
in the registry, so a promotion can be undone by promoting the old version again.
"""

import argparse

from mlflow import MlflowClient

from recsys.registry.register import CHALLENGER, CHAMPION
from recsys.utils.config import load_params
from recsys.utils.mlflow_utils import configure_mlflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="version to promote (default: the challenger)")
    args = parser.parse_args()

    configure_mlflow()
    registered_model = load_params()["mlflow"]["registered_model"]
    client = MlflowClient()
    version = (
        args.version or client.get_model_version_by_alias(registered_model, CHALLENGER).version
    )
    client.set_registered_model_alias(registered_model, CHAMPION, version)
    tags = client.get_model_version(registered_model, version).tags
    print(f"{registered_model} v{version} ({tags.get('model', '?')}) is now the {CHAMPION}")


if __name__ == "__main__":
    main()
