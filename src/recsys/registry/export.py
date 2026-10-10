"""Stage `export`: write the selected model as a self-contained serving bundle in `bundle/`.

The bundle is everything the API needs and nothing else, readable with numpy, scipy and the
standard library (the API image has no pandas, pyarrow or torch):

- `model/`: the selected model refitted on training + validation, as written by its `save`.
- `items.json`: the catalog, one movie per line, in `item_idx` order (the model's columns).
- `metadata.json`: hyperparameters, the data the model was trained on (with SHA256 hashes)
  and its validation and test metrics.
- `model_card.md`: the same in prose, with how the model was chosen and what it can't do.

Everything here is deterministic, so the bundle is a DVC output like any other. The git
commit is deliberately not written into it: a commit can't contain its own hash, so the
bundle would never match `dvc.lock`. The commit is recorded on the registered model version
instead (see `register`).
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from recsys.data.download import sha256_of
from recsys.evaluation.evaluate import load_split
from recsys.models import load_fitted
from recsys.models.train import dir_size_bytes
from recsys.utils.config import load_params
from recsys.utils.io import write_json
from recsys.utils.paths import BUNDLE_DIR, FINAL_MODELS_DIR, METRICS_DIR, PROCESSED_DIR, SPLITS_DIR

_SCOPES = {
    "all": "All users",
    "warm": "Returning users",
    "onboarding": "New users (first likes as history)",
}
_CARD_METRICS = ("ndcg_at_10", "recall_at_10", "mrr_at_10")
_DESCRIPTIONS = {
    "popularity": "Popularity: the most-liked movies, the same list for everyone.",
    "item_knn": "Item-kNN: movies similar to the ones in the user's history, where two "
    "movies are similar when the same users liked both.",
    "ials": "iALS (Hu, Koren & Volinsky 2008): matrix factorisation for implicit feedback. "
    "A user's vector is computed from their history in one least-squares step.",
    "ease": "EASE (Steck 2019): a linear item-to-item model with a closed-form solution, "
    "stored with only the strongest weights per movie.",
    "sasrec": "SASRec (Kang & McAuley 2018): a small transformer that reads the history as a "
    "sequence and predicts the next movie.",
}
_RECENCY_DESCRIPTION = (
    "On top of that, a recency prior: each movie's time-decayed like count is added to the "
    "personalised score, so movies people are watching now rank higher."
)


def read_json(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def catalog_records(items: pd.DataFrame) -> list[dict[str, Any]]:
    """Plain-Python rows of the catalog; missing years and TMDB ids become null."""
    rows: list[Any] = list(items.itertuples(index=False))
    return [
        {
            "item_idx": int(row.item_idx),
            "movie_id": int(row.movieId),
            "title": str(row.title),
            "year": None if pd.isna(row.year) else int(row.year),
            "genres": [str(genre) for genre in row.genres],
            "tmdb_id": None if pd.isna(row.tmdbId) else int(row.tmdbId),
            "imdb_id": None if pd.isna(row.imdbId) else str(row.imdbId),
        }
        for row in rows
    ]


def write_catalog(records: list[dict[str, Any]], path: Path) -> None:
    """A JSON array with one movie per line: valid JSON that still diffs line by line."""
    lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in records]
    path.write_text("[\n" + ",\n".join(lines) + "\n]\n", encoding="utf-8", newline="\n")


def headline(metrics: dict[str, Any]) -> dict[str, Any]:
    """The card's metrics for each group of users, out of a `metrics/<split>_<model>.json`."""
    return {
        scope: {"n_users": metrics[scope]["n_users"]}
        | {name: metrics[scope][name] for name in _CARD_METRICS}
        for scope in _SCOPES
    }


def training_data(interactions: pd.DataFrame, splits: tuple[str, ...]) -> dict[str, Any]:
    timestamps = pd.to_datetime(interactions["timestamp"], unit="s", utc=True)
    return {
        "splits": list(splits),
        "n_interactions": len(interactions),
        "n_users": int(interactions["user_idx"].nunique()),
        "first_interaction": timestamps.min().date().isoformat(),
        "last_interaction": timestamps.max().date().isoformat(),
        "sha256": {f"{s}.parquet": sha256_of(SPLITS_DIR / f"{s}.parquet") for s in splits},
    }


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


def _with_ci(value: dict[str, float]) -> str:
    return f"{value['mean']:.3f} [{value['ci_low']:.3f}, {value['ci_high']:.3f}]"


def model_card(
    metadata: dict[str, Any], selection: dict[str, Any], comparison_test: dict[str, Any]
) -> str:
    name, base = metadata["model"], metadata["base_model"]
    data, preprocess = metadata["trained_on"], metadata["preprocess"]
    metric = selection["metric"]
    description = _DESCRIPTIONS[base] + ("" if base == name else " " + _RECENCY_DESCRIPTION)
    lead = selection["lead_over_runner_up"]
    verdict = (
        "statistically significant" if lead["significant"] else "not statistically significant"
    )

    performance = [
        [
            label,
            split,
            str(metadata["metrics"][split][scope]["n_users"]),
            *(_with_ci(metadata["metrics"][split][scope][m]) for m in _CARD_METRICS),
        ]
        for scope, label in _SCOPES.items()
        for split in ("val", "test")
    ]
    candidates = [
        [
            f"**{c['model']}**" if c["model"] == name else c["model"],
            *(f"{c[scope]:.3f}" for scope in _SCOPES),
            *(f"{comparison_test['scopes'][scope]['means'][c['model']]:.3f}" for scope in _SCOPES),
            f"{c['model_size_mb']:.1f}",
            "yes" if c["within_budget"] else "no",
        ]
        for c in selection["candidates"]
    ]
    header = ["Model", "Val all", "Val returning", "Val new"]
    header += ["Test all", "Test returning", "Test new", "Size (MB)", "Fits"]
    chosen = _table(header, candidates)
    hashes = "\n".join(f"- `{file}`: `{digest}`" for file, digest in data["sha256"].items())
    all_params = {**metadata["base_params"], **metadata["params"]}
    params = "\n".join(f"- `{key}`: {value}" for key, value in all_params.items())

    return f"""# Model card: {name}

## What it is

{description}

It takes the movies a user liked (oldest first) and returns a ranked list of movies they
have not seen. It never takes a user id, so a user who signed up a minute ago and rated ten
movies gets recommendations from the same code path as everyone else, with no retraining.

Hyperparameters:

{params}

## Intended use

Movie recommendations in a demo web app: a "recommended for you" list from the user's liked
movies. It is a portfolio project trained on public research data, not a production system,
and its scores are rankings, not predicted ratings.

## Training data

[MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) (GroupLens Research). Ratings
from {preprocess["start_date"]} on with at least {preprocess["positive_threshold"]} stars
count as "liked"; the catalog is the {metadata["n_items"]:,} most-liked movies.

This model was fitted on the {" + ".join(data["splits"])} splits: {data["n_interactions"]:,}
likes by {data["n_users"]:,} users, from {data["first_interaction"]} to
{data["last_interaction"]}. SHA256 of the exact files:

{hashes}

## How it was chosen

{len(candidates)} candidates (five algorithms, plus a recency variant of each personalised
one) were tuned and compared on the validation split. The winner is the best `{metric}` over all
users among the models of at most {selection["max_model_size_mb"]} MB, the limit set by the
API's 512 MB host. Its lead over the runner-up, {selection["runner_up"]}, is
{lead["difference"]:.4f} [{lead["ci_low"]:.4f}, {lead["ci_high"]:.4f}], which is {verdict}
(paired t-test, Holm-adjusted p = {lead["p_adjusted"]}).

The test columns were computed afterwards, once, and played no part in the choice.

{chosen}

## Performance

Full ranking over the whole catalog with seen movies excluded; no sampled negatives.
Brackets are 95% bootstrap confidence intervals over users. Validation numbers come from the
model fitted on the training split; test numbers from this model.

{_table(["Users", "Split", "n", "NDCG@10", "Recall@10", "MRR@10"], performance)}

## Limitations

- **Frozen in time.** The data ends on {data["last_interaction"]}. Movies released later
  can't be recommended, and "popular right now" means popular then.
- **No cold-start for movies.** A movie outside the catalog, or one with no likes in the
  training data, is never recommended.
- **Likes only.** Low ratings are dropped rather than used as negative signals.
- **Popularity bias.** Like most collaborative models it favours well-known movies; see the
  coverage and novelty numbers in `metrics/`.
- **Who is in the data.** MovieLens users are volunteers on a research site and skew
  towards enthusiastic raters; tastes of other audiences are under-represented.
- **Offline numbers.** The metrics measure how well held-out likes are predicted, not
  whether real users enjoy the recommendations.

## Provenance

Built by the `export` stage of the DVC pipeline; `dvc.lock` pins the hash of every input.
The git commit is recorded as a tag on the registered MLflow model version.
"""


def main() -> None:
    params = load_params()
    selection = read_json(METRICS_DIR / "selection.json")
    name: str = selection["selected"]
    splits = ("train", "val")
    interactions, _ = load_split("test")  # everything before the test split
    model = load_fitted(name, FINAL_MODELS_DIR, params["models"][name], interactions)

    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    model.save(BUNDLE_DIR / "model")
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet")
    write_catalog(catalog_records(items), BUNDLE_DIR / "items.json")

    metadata = {
        "model": name,
        "base_model": selection["base_model"],
        "params": model.params,
        # A recency variant's own parameters are above; these are its base model's.
        "base_params": getattr(model, "base", model).params,
        "n_items": model.n_items,
        "model_size_bytes": dir_size_bytes(BUNDLE_DIR / "model"),
        "preprocess": params["preprocess"],
        "trained_on": training_data(interactions, splits),
        "metrics": {
            split: headline(read_json(METRICS_DIR / f"{split}_{name}.json"))
            for split in ("val", "test")
        },
    }
    write_json(metadata, BUNDLE_DIR / "metadata.json")
    card = model_card(metadata, selection, read_json(METRICS_DIR / "comparison_test.json"))
    (BUNDLE_DIR / "model_card.md").write_text(card, encoding="utf-8", newline="\n")
    print(f"exported {name} to {BUNDLE_DIR} ({dir_size_bytes(BUNDLE_DIR) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
