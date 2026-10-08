"""EASE (Steck 2019): a linear item-to-item model with a closed-form solution.

It learns one weight per pair of movies, B[i, j] = "how much does liking i predict liking j",
by ridge regression of every movie on all the others, with a movie forbidden from predicting
itself. The solution needs a single matrix inversion:

    P = (X'X + l2 * I)^-1        B[i, j] = -P[i, j] / P[j, j]        B[j, j] = 0

A user's scores are the sum of the rows of B for the movies in their history. Unlike
Item-kNN's similarities, the weights are fitted jointly, so they can be negative ("liked i,
so probably not j") and two near-duplicate movies share their weight instead of counting
twice.

The full B is 19,642 x 19,642: 1.5 GB as float32. With `keep_per_item` set, each movie keeps
only its strongest weights (by absolute value) and B is stored sparse, which is what makes
the model small enough to serve.

The inversion runs in float64 and in place (Cholesky), so the peak memory is one 3.1 GB
matrix; the result is stored as float32.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import lapack

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


def _invert_spd_in_place(a: npt.NDArray[np.float64]) -> None:
    """Replace a symmetric positive-definite, Fortran-ordered matrix by its inverse."""
    _, info = lapack.dpotrf(a, lower=1, overwrite_a=1)
    if info != 0:
        raise np.linalg.LinAlgError(f"Cholesky factorisation failed (info={info})")
    _, info = lapack.dpotri(a, lower=1, overwrite_c=1)
    if info != 0:
        raise np.linalg.LinAlgError(f"Cholesky inversion failed (info={info})")
    # LAPACK fills only the lower triangle; mirror it into the upper one.
    n = len(a)
    for start in range(0, n, _BLOCK_ROWS):
        stop = min(start + _BLOCK_ROWS, n)
        diagonal_block = a[start:stop, start:stop]
        diagonal_block[:] = np.tril(diagonal_block) + np.tril(diagonal_block, -1).T
        a[start:stop, stop:] = a[stop:, start:stop].T


class EASE(Recommender):
    name = "ease"

    def __init__(self, l2: float = 500.0, keep_per_item: int | None = None) -> None:
        self.l2 = l2
        self.keep_per_item = keep_per_item

    @property
    def params(self) -> dict[str, Any]:
        return {"l2": self.l2, "keep_per_item": self.keep_per_item}

    def fit(self, interactions: pd.DataFrame, n_items: int) -> Self:
        users = interactions["user_idx"].to_numpy()
        items = interactions["item_idx"].to_numpy()
        ones = np.ones(len(interactions), dtype=np.float32)
        x = sp.csr_matrix((ones, (users, items)), shape=(int(users.max()) + 1, n_items))
        xt = x.T.tocsr()

        # Gram matrix X'X (co-occurrence counts), built in blocks of rows to stay in memory.
        p = np.empty((n_items, n_items), dtype=np.float64, order="F")
        for start in range(0, n_items, _BLOCK_ROWS):
            p[start : start + _BLOCK_ROWS] = (xt[start : start + _BLOCK_ROWS] @ x).toarray()
        p[np.diag_indices(n_items)] += self.l2
        _invert_spd_in_place(p)
        scale = -1.0 / np.diag(p)

        keep = None if self.keep_per_item is None else min(self.keep_per_item, n_items - 1)
        dense = np.empty((n_items, n_items), dtype=np.float32) if keep is None else None
        columns: list[npt.NDArray[np.int32]] = []
        values: list[npt.NDArray[np.float32]] = []
        for start in range(0, n_items, _BLOCK_ROWS):
            block = (p[start : start + _BLOCK_ROWS] * scale).astype(np.float32)
            rows = np.arange(len(block))
            block[rows, start + rows] = 0.0  # a movie never predicts itself
            if dense is not None:
                dense[start : start + _BLOCK_ROWS] = block
            elif keep is not None:
                top = top_k(np.abs(block), keep)
                columns.append(top)
                values.append(np.take_along_axis(block, top, axis=1))
        del p

        self.weights: npt.NDArray[np.float32] | sp.csr_matrix
        if dense is not None:
            self.weights = dense
        elif keep is not None:
            weights = sp.csr_matrix(
                (
                    np.concatenate(values).ravel(),
                    np.concatenate(columns).ravel(),
                    np.arange(0, n_items * keep + 1, keep),
                ),
                shape=(n_items, n_items),
            )
            weights.eliminate_zeros()
            weights.sort_indices()
            self.weights = weights
        self.n_items = n_items
        return self

    def score(self, histories: Sequence[History]) -> Scores:
        scores = histories_to_csr(histories, self.n_items) @ self.weights
        if sp.issparse(scores):
            scores = scores.toarray()
        return np.asarray(scores, dtype=np.float32)

    def save(self, path: Path) -> None:
        meta = {"model": self.name, "params": self.params, "n_items": self.n_items}
        if isinstance(self.weights, np.ndarray):
            arrays: dict[str, npt.NDArray[Any]] = {"weights": self.weights}
        else:
            arrays = {
                "data": self.weights.data.astype(np.float32),
                "indices": self.weights.indices.astype(np.int32),
                "indptr": self.weights.indptr.astype(np.int64),
            }
        save_arrays(path, meta, arrays)

    @classmethod
    def load(cls, path: Path) -> Self:
        meta = load_meta(path)
        model = cls(**meta["params"])
        model.n_items = n = meta["n_items"]
        if model.keep_per_item is None:
            model.weights = np.load(path / "weights.npy")
        else:
            parts = tuple(np.load(path / f"{name}.npy") for name in ("data", "indices", "indptr"))
            model.weights = sp.csr_matrix(parts, shape=(n, n))
        return model
