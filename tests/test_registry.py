import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from recsys.evaluation.ease_pruning import recommend_keep, weights_size_mb
from recsys.models import BASE_MODELS, MODELS, base_name
from recsys.registry.export import catalog_records, headline, model_card, write_catalog
from recsys.registry.register import dir_sha256
from recsys.registry.select import candidate, lead, select
from recsys.utils.config import PROJECT_ROOT, load_params
from recsys.utils.paths import METRICS_DIR

# --- select ------------------------------------------------------------------------------


def val_metrics(model: str, score: float, size_mb: float) -> dict[str, Any]:
    scopes = {scope: {"ndcg_at_10": {"mean": score}} for scope in ("all", "warm", "onboarding")}
    return {"model": model, "model_size_bytes": int(size_mb * 1e6), **scopes}


def test_candidate_reads_scores_and_checks_the_budget() -> None:
    out = candidate(val_metrics("ease", 0.2, 151.0), "ndcg_at_10", max_size_mb=150)
    assert out == {
        "model": "ease",
        "all": 0.2,
        "warm": 0.2,
        "onboarding": 0.2,
        "model_size_mb": 151.0,
        "within_budget": False,
    }
    assert candidate(val_metrics("ease", 0.2, 150.0), "ndcg_at_10", 150)["within_budget"]


def test_select_prefers_the_best_model_that_fits_the_budget() -> None:
    candidates = [
        candidate(val_metrics(name, score, size), "ndcg_at_10", max_size_mb=150)
        for name, score, size in [
            ("small", 0.20, 10.0),
            ("too_big", 0.30, 500.0),
            ("best", 0.25, 100.0),
            ("a_tie", 0.20, 10.0),
        ]
    ]
    assert [c["model"] for c in select(candidates)] == ["best", "a_tie", "small", "too_big"]


def test_lead_finds_the_pair_in_either_order() -> None:
    pair = {"better": "a", "worse": "b", "difference": 0.01, "ci_low": 0.0, "ci_high": 0.02}
    pair |= {"p_value": 0.01, "p_adjusted": 0.02, "significant": True}
    comparison = {"scopes": {"all": {"pairs": [pair]}}}
    expected = {k: v for k, v in pair.items() if k not in ("better", "worse", "p_value")}
    assert lead(comparison, "a", "b") == lead(comparison, "b", "a") == expected
    with pytest.raises(KeyError):
        lead(comparison, "a", "c")


# --- export ------------------------------------------------------------------------------


def test_catalog_is_plain_json_with_nulls_for_missing_values(tmp_path: Path) -> None:
    items = pd.DataFrame(
        {
            "item_idx": pd.array([0, 1], dtype="int32"),
            "movieId": pd.array([10, 20], dtype="int32"),
            "title": ["Amélie", "Untitled"],
            "year": pd.array([2001, pd.NA], dtype="Int16"),
            "genres": [["Comedy", "Romance"], []],
            "tmdbId": pd.array([194, pd.NA], dtype="Int64"),
            "imdbId": ["0211915", "0000001"],
        }
    )
    write_catalog(catalog_records(items), tmp_path / "items.json")
    text = (tmp_path / "items.json").read_text(encoding="utf-8")
    assert len(text.splitlines()) == 4  # brackets plus one movie per line
    assert json.loads(text) == [
        {
            "item_idx": 0,
            "movie_id": 10,
            "title": "Amélie",
            "year": 2001,
            "genres": ["Comedy", "Romance"],
            "tmdb_id": 194,
            "imdb_id": "0211915",
        },
        {
            "item_idx": 1,
            "movie_id": 20,
            "title": "Untitled",
            "year": None,
            "genres": [],
            "tmdb_id": None,
            "imdb_id": "0000001",
        },
    ]


def scope_metrics(n_users: int, mean: float) -> dict[str, Any]:
    value = {"mean": mean, "ci_low": mean - 0.01, "ci_high": mean + 0.01}
    return {"n_users": n_users, "coverage_at_10": 0.1} | {
        name: value for name in ("ndcg_at_10", "recall_at_10", "mrr_at_10", "ndcg_at_20")
    }


def test_headline_keeps_the_card_metrics_per_scope() -> None:
    metrics = {scope: scope_metrics(100, 0.2) for scope in ("all", "warm", "onboarding")}
    out = headline({"model": "ease", **metrics})
    assert set(out) == {"all", "warm", "onboarding"}
    assert set(out["warm"]) == {"n_users", "ndcg_at_10", "recall_at_10", "mrr_at_10"}


def test_model_card_reports_the_choice_and_both_splits() -> None:
    scopes = ("all", "warm", "onboarding")
    metadata = {
        "model": "ease_recency",
        "base_model": "ease",
        "params": {"half_life_days": 30, "weight": 0.5},
        "base_params": {"l2": 1138.0, "keep_per_item": 500},
        "n_items": 19642,
        "preprocess": {"start_date": "2015-01-01", "positive_threshold": 3.5},
        "trained_on": {
            "splits": ["train", "val"],
            "n_interactions": 7959933,
            "n_users": 68334,
            "first_interaction": "2015-01-01",
            "last_interaction": "2022-12-31",
            "sha256": {"train.parquet": "abc123", "val.parquet": "def456"},
        },
        "metrics": {
            "val": {scope: scope_metrics(9689, 0.25) for scope in scopes},
            "test": {scope: scope_metrics(8650, 0.24) for scope in scopes},
        },
    }
    rows = [
        ("ease_recency", 0.25, 80.0, True),
        ("ease", 0.21, 80.0, True),
        ("big", 0.3, 900.0, False),
    ]
    selection = {
        "metric": "ndcg_at_10",
        "max_model_size_mb": 150,
        "runner_up": "ease",
        "lead_over_runner_up": {
            "difference": 0.04,
            "ci_low": 0.03,
            "ci_high": 0.05,
            "p_adjusted": 1e-12,
            "significant": True,
        },
        "candidates": [
            {"model": m, "model_size_mb": size, "within_budget": fits} | dict.fromkeys(scopes, s)
            for m, s, size, fits in rows
        ],
    }
    means = {"ease_recency": 0.24, "ease": 0.2, "big": 0.29}
    comparison_test = {"scopes": {scope: {"means": means} for scope in scopes}}

    card = model_card(metadata, selection, comparison_test)
    assert card.startswith("# Model card: ease_recency\n")
    assert "EASE (Steck 2019)" in card and "recency prior" in card
    assert "- `l2`: 1138.0" in card and "- `weight`: 0.5" in card
    assert "7,959,933" in card and "`train.parquet`: `abc123`" in card
    assert "which is statistically significant" in card
    assert (
        "| **ease_recency** | 0.250 | 0.250 | 0.250 | 0.240 | 0.240 | 0.240 | 80.0 | yes |" in card
    )
    assert "| big | 0.300 | 0.300 | 0.300 | 0.290 | 0.290 | 0.290 | 900.0 | no |" in card
    assert "| Returning users | test | 8650 | 0.240 [0.230, 0.250] |" in card


# --- register ----------------------------------------------------------------------------


def test_dir_sha256_depends_on_names_and_contents_only(tmp_path: Path) -> None:
    for root, order in (("a", ("x.txt", "sub/y.txt")), ("b", ("sub/y.txt", "x.txt"))):
        for name in order:
            file = tmp_path / root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(name.encode())
    assert dir_sha256(tmp_path / "a") == dir_sha256(tmp_path / "b")
    (tmp_path / "b" / "x.txt").write_bytes(b"changed")
    assert dir_sha256(tmp_path / "a") != dir_sha256(tmp_path / "b")


# --- EASE pruning ------------------------------------------------------------------------


def test_weights_size_counts_values_and_column_indices() -> None:
    assert weights_size_mb(1000, None) == 4.0  # dense float32
    assert weights_size_mb(1000, 100) == 0.8  # float32 value + int32 column per weight
    assert weights_size_mb(1000, 5000) == weights_size_mb(1000, 999)


def test_recommend_keep_is_the_smallest_within_the_tolerated_drop() -> None:
    variants = [
        {"keep_per_item": None, "size_mb": 1500.0, "relative_drop": 0.0},
        {"keep_per_item": 1000, "size_mb": 157.0, "relative_drop": 0.002},
        {"keep_per_item": 500, "size_mb": 79.0, "relative_drop": 0.008},
        {"keep_per_item": 100, "size_mb": 16.0, "relative_drop": 0.05},
    ]
    assert recommend_keep(variants, max_relative_drop=0.01) == 500
    assert recommend_keep(variants, max_relative_drop=0.001) is None


def test_params_use_the_recommended_pruning() -> None:
    """Once the pruning study has run, params.yaml must hold the value it recommends."""
    path = METRICS_DIR / "ease_pruning.json"
    if not path.exists():
        pytest.skip("the ease_pruning stage has not run yet")
    result = json.loads(path.read_text(encoding="utf-8"))
    params = load_params()
    assert params["models"]["ease"]["keep_per_item"] == result["recommended_keep_per_item"]
    assert params["models"]["ease"]["l2"] == result["l2"]


# --- dvc.yaml ----------------------------------------------------------------------------


def test_dvc_stages_cover_exactly_the_registered_models() -> None:
    pipeline = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    stages = pipeline["stages"]
    scored_with = pipeline["vars"][0]["scored_with"]
    assert scored_with == {name: base_name(name) for name in MODELS}
    for stage in ("train", "train_final"):
        assert stages[stage]["foreach"] == list(BASE_MODELS)
    for stage in ("evaluate", "test"):
        assert stages[stage]["foreach"] == "${scored_with}"
    for split, stage in (("val", "select"), ("val", "export"), ("test", "export")):
        deps = set(stages[stage]["deps"])
        assert {f"metrics/{split}_{name}.json" for name in MODELS} <= deps, (split, stage)
