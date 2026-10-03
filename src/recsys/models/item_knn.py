"""Item-kNN: "people who liked this also liked that".

Two movies are similar when the same users liked both. Each movie keeps only its `k` most
similar neighbours, and a user's score for a candidate movie is the sum of its similarities
to the movies in their history.

Two similarity measures:
- `cosine`: co-occurrence count divided by `|i| * |j| + shrinkage`. The shrinkage term pulls
  down similarities backed by only a handful of users, which are mostly noise.
- `bm25`: co-occurrence over BM25-weighted interactions (same weighting as the `implicit`
  library). Interactions from users who like everything count less, and very popular movies
  are damped.

The similarity matrix is built in blocks of rows: all 19,642 x 19,642 co-occurrences at once
would take gigabytes, while a block is dense only for a moment before being cut to top-k.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

from recsys.models.base import (
    History,
    Recommender,
    Scores,
    histories_to_csr,
    load_meta,
    save_arrays,
    top_k,
)

_BLOCK_ROWS = 1000  # memory/speed only; has no effect on the result
_SIMILARITIES = ("cosine", "bm25")


def bm25_weight(x: sp.csr_matrix, k1: float, b: float) -> sp.csr_matrix:
    """BM25-weight a binary user x item matrix."""
    coo = x.tocoo()
    n_users, n_items = coo.shape
    user_len = np.bincount(coo.row, minlength=n_users)
    item_len = np.bincount(coo.col, minlength=n_items)
    idf = np.log(n_items) - np.log1p(user_len)
    length_norm = (1.0 - b) + b * item_len / item_len.mean()
    data = coo.data * (k1 + 1.0) / (k1 * length_norm[coo.col] + coo.data) * idf[coo.row]
    weighted = sp.csr_matrix((data.astype(np.float32), (coo.row, coo.col)), shape=coo.shape)
    return weighted


class ItemKNN(Recommender):
    name = "item_knn"

    def __init__(
        self,
        k: int = 100,
        similarity: str = "cosine",
        shrinkage: float = 0.0,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ) -> None:
        if similarity not in _SIMILARITIES:
            raise ValueError(f"similarity must be one of {_SIMILARITIES}, got {similarity!r}")
        self.k = k
        self.similarity = similarity
        self.shrinkage = shrinkage
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

    @property
    def params(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "similarity": self.similarity,
            "shrinkage": self.shrinkage,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
        }

    def fit(self, interactions: pd.DataFrame, n_items: int) -> Self:
        users = interactions["user_idx"].to_numpy()
        items = interactions["item_idx"].to_numpy()
        ones = np.ones(len(interactions), dtype=np.float32)
        x = sp.csr_matrix((ones, (users, items)), shape=(int(users.max()) + 1, n_items))
        if self.similarity == "bm25":
            x = bm25_weight(x, self.bm25_k1, self.bm25_b)
        xt = x.T.tocsr()
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=0))).ravel()

        k = min(self.k, n_items - 1)
        neighbours: list[npt.NDArray[np.int32]] = []
        values: list[npt.NDArray[np.float32]] = []
        for start in range(0, n_items, _BLOCK_ROWS):
            block = np.asarray((xt[start : start + _BLOCK_ROWS] @ x).toarray(), dtype=np.float32)
            rows = np.arange(len(block))
            if self.similarity == "cosine":
                denom = np.outer(norms[start : start + _BLOCK_ROWS], norms) + self.shrinkage
                # Movies absent from the training data have norm 0; they get no neighbours.
                block = np.divide(block, denom, out=np.zeros_like(block), where=denom > 0)
            block[rows, start + rows] = 0.0  # a movie is not its own neighbour
            top = top_k(block, k)
            neighbours.append(top)
            values.append(np.take_along_axis(block, top, axis=1))

        sim = sp.csr_matrix(
            (
                np.concatenate(values).ravel(),
                np.concatenate(neighbours).ravel(),
                np.arange(0, n_items * k + 1, k),
            ),
            shape=(n_items, n_items),
        )
        sim.eliminate_zeros()  # movies with fewer than k co-liked movies
        sim.sort_indices()
        self.sim = sim
        self.n_items = n_items
        return self

    def score(self, histories: Sequence[History]) -> Scores:
        scores = (histories_to_csr(histories, self.n_items) @ self.sim).toarray()
        return np.asarray(scores, dtype=np.float32)

    def save(self, path: Path) -> None:
        meta = {"model": self.name, "params": self.params, "n_items": self.n_items}
        arrays = {
            "data": self.sim.data.astype(np.float32),
            "indices": self.sim.indices.astype(np.int32),
            "indptr": self.sim.indptr.astype(np.int64),
        }
        save_arrays(path, meta, arrays)

    @classmethod
    def load(cls, path: Path) -> Self:
        meta = load_meta(path)
        model = cls(**meta["params"])
        model.n_items = n = meta["n_items"]
        parts = tuple(np.load(path / f"{name}.npy") for name in ("data", "indices", "indptr"))
        model.sim = sp.csr_matrix(parts, shape=(n, n))
        return model
