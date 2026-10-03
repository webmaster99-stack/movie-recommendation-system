import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys.models import MODELS, build_model, load_model
from recsys.models.base import histories_to_csr, top_k
from recsys.models.item_knn import ItemKNN, bm25_weight
from recsys.models.popularity import Popularity

DAY = 86_400


def interactions(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "timestamp"])


def history(*items: int) -> np.ndarray:
    return np.array(items, dtype=np.int32)


# Users 0 and 1 liked items {0, 1}; user 2 liked {0, 2}. Item 3 has no interactions.
TRAIN = interactions([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 2, 0)])
N_ITEMS = 4


# --- shared ranking ----------------------------------------------------------------------


def test_top_k_breaks_ties_by_lower_index() -> None:
    scores = np.array([[1.0, 3.0, 3.0, 2.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    assert top_k(scores, 3).tolist() == [[1, 2, 3], [0, 1, 2]]


def test_histories_to_csr_handles_empty_histories() -> None:
    m = histories_to_csr([history(0, 2), history()], n_items=3)
    assert m.toarray().tolist() == [[1, 0, 1], [0, 0, 0]]


# --- popularity --------------------------------------------------------------------------


def test_popularity_ranks_by_count_and_excludes_seen() -> None:
    model = Popularity().fit(TRAIN, N_ITEMS)
    assert model.item_scores.tolist() == [3, 2, 1, 0]
    assert model.recommend(history(), k=3).tolist() == [0, 1, 2]
    assert model.recommend(history(0), k=2).tolist() == [1, 2]
    assert model.recommend(history(0), k=2, exclude_seen=False).tolist() == [0, 1]


def test_popularity_time_decay_prefers_recent_items() -> None:
    # Item 0: three likes 100 days before the end. Item 1: two likes on the last day.
    rows = [(u, 0, 0) for u in range(3)] + [(u, 1, 100 * DAY) for u in range(2)]
    assert Popularity().fit(interactions(rows), 2).recommend(history(), k=1).tolist() == [0]
    decayed = Popularity(half_life_days=50).fit(interactions(rows), 2)
    # 100 days = two half-lives, so item 0 counts 3 * 0.25 = 0.75 against item 1's 2.
    assert decayed.item_scores.tolist() == pytest.approx([0.75, 2.0])
    assert decayed.recommend(history(), k=1).tolist() == [1]


# --- item-kNN ----------------------------------------------------------------------------


def test_item_knn_cosine_similarities() -> None:
    model = ItemKNN(k=2, similarity="cosine").fit(TRAIN, N_ITEMS)
    sim = model.sim.toarray()
    # |item0| = sqrt(3), |item1| = sqrt(2), |item2| = 1; co-liked: (0,1) twice, (0,2) once.
    assert sim[0, 1] == pytest.approx(2 / np.sqrt(6))
    assert sim[0, 2] == pytest.approx(1 / np.sqrt(3))
    assert sim[1, 2] == 0.0
    np.testing.assert_allclose(sim, sim.T)
    assert np.diag(sim).tolist() == [0, 0, 0, 0]
    assert sim[3].tolist() == [0, 0, 0, 0]  # the item nobody liked has no neighbours


def test_item_knn_keeps_only_k_neighbours() -> None:
    model = ItemKNN(k=1, similarity="cosine").fit(TRAIN, N_ITEMS)
    # Item 0's best neighbour is 1; items 1 and 2 only have item 0.
    assert [row.nonzero()[0].tolist() for row in model.sim.toarray()] == [[1], [0], [0], []]


def test_item_knn_shrinkage_lowers_similarity() -> None:
    model = ItemKNN(k=2, similarity="cosine", shrinkage=2.0).fit(TRAIN, N_ITEMS)
    assert model.sim[0, 1] == pytest.approx(2 / (np.sqrt(6) + 2.0))


def test_item_knn_recommends_neighbours_of_the_history() -> None:
    model = ItemKNN(k=2, similarity="cosine").fit(TRAIN, N_ITEMS)
    assert model.recommend(history(2), k=1).tolist() == [0]
    # History {1}: item 0 scores 0.816; items 2 and 3 tie at 0, so the lower index comes next.
    assert model.recommend(history(1), k=3).tolist() == [0, 2, 3]
    # Scores add up over the history: item 0 gets sim(1,0) + sim(2,0).
    scores = model.score([history(1, 2)])
    assert scores[0, 0] == pytest.approx(2 / np.sqrt(6) + 1 / np.sqrt(3))


def test_bm25_weight_matches_formula() -> None:
    k1, b = 1.2, 0.75
    x = histories_to_csr([history(0, 1), history(0, 1), history(0, 2)], n_items=N_ITEMS)
    w = bm25_weight(x, k1, b).toarray()
    idf = np.log(4) - np.log1p(2)  # every user liked 2 of the 4 items
    item_len = np.array([3, 2, 1, 0])
    norm = (1 - b) + b * item_len / item_len.mean()
    expected = idf * (k1 + 1) / (k1 * norm + 1)
    np.testing.assert_allclose(w[0, :2], expected[:2], rtol=1e-6)
    np.testing.assert_allclose(w[2, 2], expected[2], rtol=1e-6)
    assert w[0, 2] == 0.0
    # The popular item gets the smallest weight.
    assert w[0, 0] < w[0, 1] < w[2, 2]


def test_item_knn_bm25_recommends_co_liked_items() -> None:
    model = ItemKNN(k=2, similarity="bm25").fit(TRAIN, N_ITEMS)
    assert model.recommend(history(2), k=1).tolist() == [0]
    assert model.recommend(history(1), k=1).tolist() == [0]


def test_item_knn_rejects_unknown_similarity() -> None:
    with pytest.raises(ValueError, match="similarity"):
        ItemKNN(similarity="jaccard")


# --- save / load -------------------------------------------------------------------------


def dir_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for f in sorted(path.iterdir()):
        digest.update(f.name.encode())
        digest.update(f.read_bytes())
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("popularity", {"half_life_days": 30.0}),
        ("item_knn", {"k": 2, "similarity": "cosine", "shrinkage": 1.0}),
        ("item_knn", {"k": 2, "similarity": "bm25"}),
    ],
)
def test_save_load_round_trip_is_byte_identical(
    name: str, params: dict[str, object], tmp_path: Path
) -> None:
    model = build_model(name, params).fit(TRAIN, N_ITEMS)
    model.save(tmp_path / "a")
    loaded = load_model(name, tmp_path / "a")

    histories = [history(0), history(1, 2), history()]
    np.testing.assert_array_equal(
        model.recommend_batch(histories, k=3), loaded.recommend_batch(histories, k=3)
    )
    assert loaded.params == model.params

    # A second fit + save must produce the same bytes, or DVC would see a changed model.
    build_model(name, params).fit(TRAIN, N_ITEMS).save(tmp_path / "b")
    assert dir_hash(tmp_path / "a") == dir_hash(tmp_path / "b")


def test_registry_names_match_classes() -> None:
    assert {"popularity": Popularity, "item_knn": ItemKNN} == MODELS
