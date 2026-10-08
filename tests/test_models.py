import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys.models import MODELS, build_model, load_model
from recsys.models.base import histories_to_csr, top_k
from recsys.models.ease import EASE
from recsys.models.ials import IALS
from recsys.models.item_knn import ItemKNN, bm25_weight
from recsys.models.popularity import Popularity
from recsys.models.sasrec import SASRec, _pad_left

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


# --- EASE --------------------------------------------------------------------------------


def test_ease_matches_closed_form() -> None:
    model = EASE(l2=0.5).fit(TRAIN, N_ITEMS)
    x = histories_to_csr([history(0, 1), history(0, 1), history(0, 2)], N_ITEMS).toarray()
    p = np.linalg.inv(x.T @ x + 0.5 * np.eye(N_ITEMS))
    expected = p / -np.diag(p)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(model.weights, expected, rtol=1e-5, atol=1e-7)
    # Scores are the summed rows of B for the history.
    np.testing.assert_allclose(
        model.score([history(1, 2)])[0], expected[1] + expected[2], rtol=1e-5, atol=1e-7
    )
    assert model.recommend(history(1), k=1).tolist() == [0]


def test_ease_keep_per_item_keeps_the_largest_absolute_weights() -> None:
    dense = EASE(l2=0.5).fit(TRAIN, N_ITEMS).weights
    sparse = EASE(l2=0.5, keep_per_item=1).fit(TRAIN, N_ITEMS).weights.toarray()
    for row_dense, row_sparse in zip(dense, sparse, strict=True):
        kept = row_sparse.nonzero()[0]
        assert len(kept) <= 1
        if len(kept):
            assert abs(row_dense[kept[0]]) == np.abs(row_dense).max()
            assert row_sparse[kept[0]] == row_dense[kept[0]]
    # Keeping every weight gives the dense model back.
    full = EASE(l2=0.5, keep_per_item=N_ITEMS).fit(TRAIN, N_ITEMS).weights.toarray()
    np.testing.assert_array_equal(full, dense)


# --- iALS --------------------------------------------------------------------------------

# Two groups of users with disjoint tastes: items 0-2 and items 3-5.
TWO_TASTES = interactions(
    [(u, i, 0) for u in range(6) for i in range(3) if i != u % 3]
    + [(u, i, 0) for u in range(6, 12) for i in range(3, 6) if i != u % 3 + 3]
)


def test_ials_recommends_within_the_users_taste_group() -> None:
    model = IALS(factors=2, regularization=0.1, alpha=10.0, iterations=20).fit(TWO_TASTES, 6)
    assert model.recommend(history(0, 1), k=1).tolist() == [2]
    assert model.recommend(history(3), k=2).tolist() in ([4, 5], [5, 4])
    assert not model.score([history()]).any()  # no history, no signal


def test_ials_fold_in_solves_the_weighted_least_squares_problem() -> None:
    model = IALS(factors=3, regularization=0.5, alpha=4.0, iterations=3).fit(TWO_TASTES, 6)
    y = model.item_factors.astype(np.float64)
    liked = history(0, 4)
    # Minimise sum_i c_i * (p_i - x.y_i)^2 + reg * |x|^2 with c = alpha, p = 1 for liked items
    # and c = 1, p = 0 for the rest, written as an ordinary least-squares problem.
    target = np.zeros(6)
    target[liked] = 1.0
    weight = np.sqrt(np.where(target > 0, 4.0, 1.0))
    a = np.vstack([y * weight[:, None], np.sqrt(0.5) * np.eye(3)])
    b = np.concatenate([target * weight, np.zeros(3)])
    user = np.linalg.lstsq(a, b, rcond=None)[0]
    np.testing.assert_allclose(model.score([liked])[0], y @ user, rtol=1e-4, atol=1e-6)


# --- SASRec ------------------------------------------------------------------------------

SASREC_SMALL = {"hidden_dim": 16, "max_len": 4, "dropout": 0.0, "batch_size": 8}


def test_pad_left_keeps_the_most_recent_items() -> None:
    out = _pad_left([history(1, 2, 3, 4), history(5), history()], length=3, pad=9)
    assert out.tolist() == [[2, 3, 4], [9, 9, 5], [9, 9, 9]]


def test_sasrec_learns_the_order_of_a_sequence() -> None:
    # Every user walks the cycle 0 -> 1 -> 2 -> 3 -> 4 -> 0, starting at a different item.
    rows = [(u, (u + t) % 5, t) for u in range(20) for t in range(5)]
    model = SASRec(**SASREC_SMALL, epochs=60, learning_rate=0.01, n_negatives=None)
    model.fit(interactions(rows), n_items=5)
    for item in range(5):
        previous = (item - 1) % 5
        assert model.recommend(history(previous, item), k=1).tolist() == [(item + 1) % 5]


def test_sasrec_sampled_negatives_and_user_subsample_train() -> None:
    rows = [(u, (u + t) % 5, t) for u in range(20) for t in range(5)]
    model = SASRec(**SASREC_SMALL, epochs=2, n_negatives=3, max_users=10)
    scores = model.fit(interactions(rows), n_items=5).score([history(0, 1), history()])
    assert scores.shape == (2, 5) and np.isfinite(scores).all()


def test_sasrec_ignores_history_beyond_max_len() -> None:
    model = SASRec(**SASREC_SMALL, epochs=1).fit(TRAIN, N_ITEMS)
    long, recent = history(3, 2, 0, 1, 2, 3), history(0, 1, 2, 3)
    np.testing.assert_array_equal(model.score([long]), model.score([recent]))


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
        ("ials", {"factors": 2, "iterations": 3}),
        ("ease", {"l2": 1.0}),
        ("ease", {"l2": 1.0, "keep_per_item": 2}),
        ("sasrec", {**SASREC_SMALL, "epochs": 2}),
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
    assert {
        "popularity": Popularity,
        "item_knn": ItemKNN,
        "ials": IALS,
        "ease": EASE,
        "sasrec": SASRec,
    } == MODELS


def test_build_model_passes_the_seed_only_to_models_that_use_it() -> None:
    assert build_model("ials", {}, seed=7).params["seed"] == 7
    assert "seed" not in build_model("ease", {}, seed=7).params
