"""Metrics checked against values worked out by hand."""

import numpy as np
import pytest

from recsys.evaluation import metrics

# Two users, lists of length 3.
# User 0: targets {3, 9}; the only hit is item 3 at rank 3.
# User 1: target {7}; no hit.
RECS = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
TARGETS = [np.array([3, 9], dtype=np.int32), np.array([7], dtype=np.int32)]
N_TARGETS = np.array([2, 1])


def test_hit_matrix() -> None:
    hits = metrics.hit_matrix(RECS, TARGETS)
    assert hits.tolist() == [[False, False, True], [False, False, False]]


def test_recall() -> None:
    hits = metrics.hit_matrix(RECS, TARGETS)
    assert metrics.recall(hits, N_TARGETS).tolist() == [0.5, 0.0]
    # At k=2 the hit at rank 3 is cut off.
    assert metrics.recall(hits[:, :2], N_TARGETS).tolist() == [0.0, 0.0]


def test_ndcg() -> None:
    hits = metrics.hit_matrix(RECS, TARGETS)
    # DCG = 1/log2(4) = 0.5. Ideal with 2 targets: 1/log2(2) + 1/log2(3) = 1.63093.
    expected = 0.5 / (1 + 1 / np.log2(3))
    assert metrics.ndcg(hits, N_TARGETS) == pytest.approx([expected, 0.0])
    assert expected == pytest.approx(0.306574, abs=1e-6)


def test_ndcg_is_one_for_a_perfect_list_even_with_more_targets_than_k() -> None:
    hits = np.array([[True, True]])
    assert metrics.ndcg(hits, np.array([5])) == pytest.approx([1.0])


def test_ndcg_rewards_higher_ranks() -> None:
    first = metrics.ndcg(np.array([[True, False, False]]), np.array([1]))
    last = metrics.ndcg(np.array([[False, False, True]]), np.array([1]))
    assert first[0] == pytest.approx(1.0)
    assert last[0] == pytest.approx(0.5)  # 1/log2(4)


def test_mrr() -> None:
    hits = np.array([[False, False, True], [False, False, False], [True, True, False]])
    assert metrics.mrr(hits) == pytest.approx([1 / 3, 0.0, 1.0])


def test_coverage() -> None:
    assert metrics.coverage(RECS, n_items=10) == 0.6
    assert metrics.coverage(np.array([[1, 2], [1, 2]]), n_items=10) == 0.2


def test_novelty_and_mean_popularity() -> None:
    popularity = np.array([0.5, 0.25, 0.125])
    recs = np.array([[0, 1], [2, 2]], dtype=np.int32)
    # -log2(0.5) = 1, -log2(0.25) = 2, -log2(0.125) = 3
    assert metrics.novelty(recs, popularity) == pytest.approx([1.5, 3.0])
    assert metrics.mean_popularity(recs, popularity) == pytest.approx([0.375, 0.125])


def test_bootstrap_ci_is_seeded_and_brackets_the_mean() -> None:
    values = np.random.default_rng(0).random((500, 2))
    low, high = metrics.bootstrap_ci(values, np.random.default_rng(1), 500, 0.95)
    again = metrics.bootstrap_ci(values, np.random.default_rng(1), 500, 0.95)
    np.testing.assert_array_equal(low, again[0])
    np.testing.assert_array_equal(high, again[1])

    mean = values.mean(axis=0)
    assert (low < mean).all() and (mean < high).all()
    # Standard error of a uniform mean with n=500 is 0.0129, so the CI is about +-0.025.
    assert high - low == pytest.approx(0.05, abs=0.01)


def test_bootstrap_ci_of_a_constant_has_zero_width() -> None:
    low, high = metrics.bootstrap_ci(np.full((50, 1), 0.3), np.random.default_rng(0), 100, 0.95)
    assert low == pytest.approx([0.3]) and high == pytest.approx([0.3])
