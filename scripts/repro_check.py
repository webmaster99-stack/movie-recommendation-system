"""Reproducibility check: run the whole pipeline twice and require byte-identical results.

    uv run python scripts/repro_check.py

The real pipeline takes hours and gigabytes, so this runs it on a small synthetic dataset
in the shape of MovieLens (generated here from a fixed seed, no download) with the
hyperparameters shrunk by `scripts/repro_overrides.yaml`. Everything else is the real thing:
the same `dvc.yaml`, the same stage commands and the same code, from `validate` to
`register`. Only the `download` stage is skipped.

It works in a throwaway copy of the project, so the real `data/`, `models/` and `params.yaml`
are never touched and no credentials are needed (MLflow logs to a local file there).

The check fails if any stage fails, or if any file under the pipeline's output directories
differs between the two runs. Both would be bugs: a broken pipeline, or a source of
randomness that is not controlled by the seed.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OVERRIDES = Path(__file__).with_name("repro_overrides.yaml")
# What the pipeline needs to run; everything else in the repository stays behind.
COPIED = ("src", "dvc.yaml", "params.yaml", ".dvc/config", ".dvc/.gitignore", ".dvcignore")
OUTPUT_DIRS = ("data/interim", "data/processed", "data/evaluation", "models", "metrics", "bundle")

DAY = 86_400
GENRES = ("Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Western")


def unix(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp())


def synthetic_movielens(seed: int, n_users: int = 1500, n_movies: int = 400) -> dict[str, str]:
    """CSV text of a small dataset with the structure the models rely on.

    Movies belong to one of eight taste groups and are released over time; a user mostly likes
    their own group, prefers popular and recently released movies, and rates over several
    months. That gives co-liked movies, a recency effect, and both returning and new users on
    either side of the split dates.
    """
    rng = np.random.default_rng(seed)
    start, end = unix("2021-01-01"), unix("2023-10-01")
    movie_ids = np.arange(1, n_movies + 1)
    group = rng.integers(0, len(GENRES), n_movies)
    released = rng.integers(start - 720 * DAY, end - 90 * DAY, n_movies)
    appeal = rng.lognormal(0.0, 1.0, n_movies)

    rows = []
    for user in range(1, n_users + 1):
        taste = rng.integers(0, len(GENRES))
        joined = int(rng.integers(start, end - 30 * DAY))
        n_ratings = int(min(8 + rng.geometric(1 / 30), 150))
        when = np.minimum(joined + rng.exponential(120 * DAY, n_ratings).astype(np.int64), end)
        age_days = np.maximum(joined - released, 0) / DAY
        weight = appeal * np.where(group == taste, 6.0, 1.0) * np.exp(-age_days / 300)
        weight[released > joined + 60 * DAY] = 0.0  # not out yet
        n_ratings = min(n_ratings, int((weight > 0).sum()))
        movies = rng.choice(movie_ids, n_ratings, replace=False, p=weight / weight.sum())
        # Users like most of what they pick from their own group and some of the rest.
        liked = rng.random(n_ratings) < np.where(group[movies - 1] == taste, 0.9, 0.4)
        stars = np.where(
            liked,
            rng.choice([3.5, 4.0, 4.5, 5.0], n_ratings),
            rng.choice([1.0, 2.0, 3.0], n_ratings),
        )
        rows.append(
            pd.DataFrame(
                {"userId": user, "movieId": movies, "rating": stars, "timestamp": when[:n_ratings]}
            )
        )
    ratings = pd.concat(rows, ignore_index=True)

    year = pd.to_datetime(released, unit="s").year
    movies_table = pd.DataFrame(
        {
            "movieId": movie_ids,
            "title": [f"Movie {i} ({y})" for i, y in zip(movie_ids, year, strict=True)],
            "genres": [GENRES[g] for g in group],
        }
    )
    links = pd.DataFrame(
        {
            "movieId": movie_ids,
            "imdbId": [f"{i:07d}" for i in movie_ids],
            # Every tenth movie has no TMDB id, as some do in the real data.
            "tmdbId": [None if i % 10 == 0 else 1000 + i for i in movie_ids],
        }
    ).astype({"tmdbId": "Int64"})
    tags = ratings.head(50).assign(tag="synthetic")[["userId", "movieId", "tag", "timestamp"]]
    tables = {"ratings": ratings, "movies": movies_table, "links": links, "tags": tags}
    return {name: table.to_csv(index=False, lineterminator="\n") for name, table in tables.items()}


def merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """`overrides` laid over `base`, descending into nested mappings."""
    merged = dict(base)
    for key, value in overrides.items():
        both_maps = isinstance(value, dict) and isinstance(merged.get(key), dict)
        merged[key] = merge(merged[key], value) if both_maps else value
    return merged


def prepare(workdir: Path) -> None:
    for name in COPIED:
        source, target = ROOT / name, workdir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, target)
    subprocess.run(["git", "init", "--quiet"], cwd=workdir, check=True)  # DVC wants a repository

    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    params = merge(params, yaml.safe_load(OVERRIDES.read_text(encoding="utf-8")))
    (workdir / "params.yaml").write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")

    raw = workdir / "data" / "raw" / "ml-32m"
    raw.mkdir(parents=True)
    for name, text in synthetic_movielens(params["seed"]).items():
        (raw / f"{name}.csv").write_text(text, encoding="utf-8", newline="\n")
    (raw / "README.txt").write_text("Synthetic data for the reproducibility check.\n")


def run_pipeline(workdir: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_TRACKING_")}
    env["PYTHONPATH"] = str(workdir / "src")  # the copy's code, not the installed project's
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    # --downstream validate: everything except `download`. --force and --no-run-cache: really
    # run every stage, instead of restoring the outputs of an identical earlier run.
    command = [sys.executable, "-m", "dvc", "repro", "--downstream", "validate"]
    subprocess.run([*command, "--force", "--no-run-cache"], cwd=workdir, env=env, check=True)


def snapshot(workdir: Path) -> dict[str, str]:
    """SHA256 of every file the pipeline wrote."""
    return {
        file.relative_to(workdir).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for directory in OUTPUT_DIRS
        for file in sorted((workdir / directory).rglob("*"))
        if file.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--keep", action="store_true", help="keep the working copy afterwards")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="recsys-repro-"))
    print(f"working copy: {workdir}", flush=True)
    try:
        prepare(workdir)
        snapshots = []
        for attempt in (1, 2):
            start = time.perf_counter()
            run_pipeline(workdir)
            snapshots.append(snapshot(workdir))
            print(
                f"run {attempt}: {len(snapshots[-1])} files in {time.perf_counter() - start:.0f}s",
                flush=True,
            )
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)

    first, second = snapshots
    different = sorted(f for f in first.keys() | second.keys() if first.get(f) != second.get(f))
    if different:
        print(f"NOT REPRODUCIBLE: {len(different)} of {len(first)} files differ between runs:")
        print("\n".join(f"  {file}" for file in different))
        return 1
    print(f"reproducible: all {len(first)} output files are byte-identical across two runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
