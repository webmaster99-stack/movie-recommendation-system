"""The API's scorer must agree with the training code it was copied from.

`recsys_api.model` repeats the arithmetic of `recsys.models` so the API can run without
pandas and torch. These tests save a model with the training code, load it with the API's
loader, and require identical scores and identical recommendations.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys.models.base import Recommender
from recsys.models.ease import EASE
from recsys.models.ials import IALS
from recsys.models.recency import EASERecency
from recsys.utils.config import PROJECT_ROOT
from recsys_api.bundle import Bundle
from recsys_api.model import (
    Ease,
    Recency,
    Scorer,
    UnsupportedModelError,
    load_model,
    recommend,
)

N_USERS, N_ITEMS = 60, 30
REAL_BUNDLE = PROJECT_ROOT / "bundle"


def synthetic_interactions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    liked = rng.random((N_USERS, N_ITEMS)) < 0.2
    users, items = np.nonzero(liked)
    timestamps = rng.integers(0, 60 * 86_400, size=len(users))
    return pd.DataFrame({"user_idx": users, "item_idx": items, "timestamp": timestamps})


def histories() -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    sizes = [0, 1, 1, 2, 3, 5, 8, 13, 20, N_ITEMS - 1]
    return [rng.permutation(N_ITEMS)[:size].astype(np.int32) for size in sizes]


def assert_same_recommendations(served: Scorer, trained: Recommender, history: np.ndarray) -> None:
    # With fewer than k unseen movies left, the training code pads its answer with seen
    # ones (it always returns k); the API returns only the unseen.
    unseen = N_ITEMS - len(history)
    expected = trained.recommend(history, k=10).tolist()[:unseen]
    assert recommend(served, history, history, 10).tolist() == expected


@pytest.mark.parametrize("keep_per_item", [None, 8])
def test_ease_scores_and_recommendations_match_training_code(
    tmp_path: Path, keep_per_item: int | None
) -> None:
    trained = EASE(l2=5.0, keep_per_item=keep_per_item).fit(synthetic_interactions(), N_ITEMS)
    trained.save(tmp_path)
    served = load_model(tmp_path)
    assert isinstance(served, Ease)
    for history in histories():
        np.testing.assert_array_equal(served.score(history), trained.score([history]))
        assert_same_recommendations(served, trained, history)


def test_recency_scores_and_recommendations_match_training_code(tmp_path: Path) -> None:
    interactions = synthetic_interactions()
    base = EASE(l2=5.0, keep_per_item=8).fit(interactions, N_ITEMS)
    trained = EASERecency(half_life_days=5, weight=0.7).fit(interactions, N_ITEMS).wrap(base)
    trained.save(tmp_path)
    served = load_model(tmp_path)
    assert isinstance(served, Recency)
    for history in histories():
        np.testing.assert_array_equal(served.score(history), trained.score([history]))
        assert_same_recommendations(served, trained, history)


@pytest.mark.skipif(
    not (REAL_BUNDLE / "metadata.json").is_file(), reason="needs the exported bundle"
)
def test_the_exported_bundle_is_served_as_trained() -> None:
    from recsys.models import load_model as load_trained

    name = json.loads((REAL_BUNDLE / "metadata.json").read_text(encoding="utf-8"))["model"]
    if not name.startswith("ease"):
        pytest.skip(f"the API has no scorer for {name}")
    trained = load_trained(name, REAL_BUNDLE / "model")
    served = load_model(REAL_BUNDLE / "model")
    rng = np.random.default_rng(2)
    for size in (0, 1, 10, 200):
        history = rng.permutation(trained.n_items)[:size].astype(np.int32)
        expected = trained.recommend(history, k=20)
        assert recommend(served, history, history, 20).tolist() == expected.tolist()


def test_a_model_without_a_scorer_is_refused_with_an_explanation(tmp_path: Path) -> None:
    IALS(factors=4, iterations=1, seed=0).fit(synthetic_interactions(), N_ITEMS).save(tmp_path)
    with pytest.raises(UnsupportedModelError, match="cannot serve 'ials'"):
        load_model(tmp_path)


def test_excluded_movies_are_never_recommended(bundle: Bundle) -> None:
    history = np.array([0], dtype=np.int32)
    # Liking The Matrix (0) points at its sequel (1), then Heat (4).
    assert recommend(bundle.model, history, history, 2).tolist() == [1, 4]
    seen = np.array([0, 1], dtype=np.int32)  # the sequel was rated low: seen, not liked
    assert recommend(bundle.model, history, seen, 2).tolist() == [4, 3]
    assert len(recommend(bundle.model, history, seen, 100)) == 4  # all that is left


def test_similar_movies_are_the_positive_weights_best_first(bundle: Bundle) -> None:
    model = bundle.similar_model
    assert model is not None
    assert model.similar(0, k=5).tolist() == [1, 4]  # Amelie's weight is negative
    assert model.similar(0, k=1).tolist() == [1]
    assert model.similar(5, k=5).tolist() == []
