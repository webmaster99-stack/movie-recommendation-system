"""Ranking metrics over full-catalog top-k lists.

Per-user metrics take `hits`, a boolean (n_users, k) matrix where `hits[u, r]` says whether
the movie at rank r+1 of user u's list is one of their targets. Slice it (`hits[:, :10]`) to
get a metric at a smaller k. Each returns one value per user, so the caller can average,
bootstrap or split by segment.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int32]


def hit_matrix(recs: Ints, targets: Sequence[Ints]) -> npt.NDArray[np.bool_]:
    return np.stack([np.isin(r, t) for r, t in zip(recs, targets, strict=True)])


def recall(hits: npt.NDArray[np.bool_], n_targets: npt.NDArray[np.int64]) -> Floats:
    """Share of the user's targets that made it into the list."""
    return np.asarray(hits.sum(axis=1) / n_targets, dtype=np.float64)


def ndcg(hits: npt.NDArray[np.bool_], n_targets: npt.NDArray[np.int64]) -> Floats:
    """Like recall, but a hit at rank r is worth 1/log2(r+1), so hits near the top count more.

    Normalised by the best achievable list (all hits first), so the range is 0..1.
    """
    k = hits.shape[1]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(axis=1)
    ideal = np.cumsum(discounts)[np.minimum(n_targets, k) - 1]
    return np.asarray(dcg / ideal, dtype=np.float64)


def mrr(hits: npt.NDArray[np.bool_]) -> Floats:
    """1 / rank of the first hit, or 0 when the list has none."""
    first = hits.argmax(axis=1)
    return np.where(hits.any(axis=1), 1.0 / (first + 1), 0.0)


def coverage(recs: Ints, n_items: int) -> float:
    """Share of the catalog that appears in at least one user's list."""
    return len(np.unique(recs)) / n_items


def novelty(recs: Ints, item_popularity: Floats) -> Floats:
    """Mean self-information, -log2(popularity), of the recommended movies.

    `item_popularity` is the share of training users who liked each movie. A list of
    blockbusters scores low; higher means less obvious recommendations.
    """
    return np.asarray(-np.log2(item_popularity[recs]).mean(axis=1), dtype=np.float64)


def mean_popularity(recs: Ints, item_popularity: Floats) -> Floats:
    """Mean popularity of the recommended movies: the direct reading of popularity bias."""
    return np.asarray(item_popularity[recs].mean(axis=1), dtype=np.float64)


def bootstrap_ci(
    values: Floats, rng: np.random.Generator, n_resamples: int, confidence: float
) -> tuple[Floats, Floats]:
    """Percentile-bootstrap confidence interval for the mean of each column of `values`.

    Users are resampled with replacement. Drawing how many times each user appears
    (a multinomial) is equivalent to drawing the users themselves and needs no copies of
    the data.
    """
    n = len(values)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_resamples)
    means = counts @ values / n
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha], axis=0)
    return low, high
