"""`python -m recsys.registry.promote`: make the challenger the champion, if it is better.

The manual gate between "the pipeline produced a model" and "the API serves it". It
downloads the version `challenger` points at (or `--version N`) and the current champion,
compares them on the same users (`recsys.registry.gate`: no worse on the MovieLens test
split, and better on the app's held-out likes once there are enough app users), and moves
the `champion` alias only if the challenger passes. The result is recorded as `gate_*` tags
on the version either way.

The comparison reads the test split and the app feedback from the workspace, so run it in
the checkout the challenger was built from, after `dvc pull`.

`--skip-gate` promotes without comparing, for a rollback to an earlier version or a change
the gate can't judge (a new catalog). Nothing is retrained or uploaded, and the previous
champion stays in the registry, so a promotion can be undone by promoting it again.
"""

import argparse
import sys
import tempfile
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from recsys.registry.gate import load_bundle, run_gate
from recsys.registry.pull import download
from recsys.registry.register import CHALLENGER, CHAMPION
from recsys.utils.config import load_params
from recsys.utils.mlflow_utils import configure_mlflow
from recsys.utils.seed import make_deterministic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="version to promote (default: the challenger)")
    parser.add_argument("--skip-gate", action="store_true", help="promote without comparing")
    args = parser.parse_args()

    configure_mlflow()
    params = load_params()
    registered_model = params["mlflow"]["registered_model"]
    client = MlflowClient()
    candidate = (
        client.get_model_version(registered_model, args.version)
        if args.version
        else client.get_model_version_by_alias(registered_model, CHALLENGER)
    )
    try:
        champion = client.get_model_version_by_alias(registered_model, CHAMPION)
    except MlflowException:
        champion = None  # the first promotion: nothing to compare with
    if champion is not None and champion.version == candidate.version:
        print(f"{registered_model} v{candidate.version} is already the {CHAMPION}")
        return 0

    if champion is not None and not args.skip_gate:
        rng = make_deterministic(params["seed"], params["runtime"]["num_threads"])
        with tempfile.TemporaryDirectory() as tmp:
            for version, name in ((candidate, "candidate"), (champion, "champion")):
                download(client, version, Path(tmp) / name)
            report = run_gate(
                load_bundle(Path(tmp) / "candidate"),
                load_bundle(Path(tmp) / "champion"),
                params,
                rng,
            )
        against = f"the {CHAMPION}, v{champion.version}"
        print(f"v{candidate.version} against {against}, on {report['metric']}:")
        print("\n".join(f"  {reason}" for reason in report["reasons"]))
        tags = {
            "gate_passed": str(report["passed"]).lower(),
            "gate_against": champion.version,
            "gate_test_difference": str(report["test"]["difference"]),
            "gate_app_difference": str(report["app"]["difference"] if report["app"] else ""),
        }
        for key, value in tags.items():
            client.set_model_version_tag(registered_model, candidate.version, key, value)
        if not report["passed"]:
            print(f"not promoted: v{champion.version} stays the {CHAMPION}")
            return 1

    client.set_registered_model_alias(registered_model, CHAMPION, candidate.version)
    tags = client.get_model_version(registered_model, candidate.version).tags
    print(
        f"{registered_model} v{candidate.version} ({tags.get('model', '?')}) is now the {CHAMPION}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
