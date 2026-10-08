"""iALS (Hu, Koren & Volinsky 2008): matrix factorisation for implicit feedback.

Every movie gets a vector of `factors` numbers and so does every user; a user's score for a
movie is the dot product of the two. Movies a user liked are fitted towards 1 with weight
`alpha`, and every other movie towards 0 with weight 1, which is what lets the model learn
from "didn't watch" without treating it as "disliked".

Training uses the `implicit` library (alternating least squares: fix the movies and solve
for the users, then the other way round). Only the movie vectors are kept. A user's vector
is recomputed from their history with one least-squares solve ("fold-in"), the same
equation a training step uses:

    user = (Y'Y + (alpha - 1) * Yh'Yh + regularization * I)^-1 * alpha * sum(Yh)

where Y holds all movie vectors and Yh those in the history. New users are therefore
scored exactly like training users, and serving needs numpy only, not `implicit`.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from recsys.models.base import History, Recommender, Scores, load_meta, save_arrays


class IALS(Recommender):
    name = "ials"
    uses_seed = True

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        alpha: float = 1.0,
        iterations: int = 15,
        seed: int = 0,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    @property
    def params(self) -> dict[str, Any]:
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "seed": self.seed,
        }

    def _set_item_factors(self, item_factors: npt.NDArray[np.float32]) -> None:
        self.item_factors = item_factors
        y = item_factors.astype(np.float64)
        self._gram = y.T @ y + self.regularization * np.eye(y.shape[1])

    def fit(self, interactions: pd.DataFrame, n_items: int) -> Self:
        from implicit.cpu.als import AlternatingLeastSquares

        users = interactions["user_idx"].to_numpy()
        items = interactions["item_idx"].to_numpy()
        ones = np.ones(len(interactions), dtype=np.float32)
        x = sp.csr_matrix((ones, (users, items)), shape=(int(users.max()) + 1, n_items))
        # implicit runs one thread per user; BLAS threads inside those only slow it down.
        with threadpool_limits(limits=1, user_api="blas"):
            als = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                alpha=self.alpha,
                iterations=self.iterations,
                random_state=self.seed,
            )
            als.fit(x, show_progress=False)
        self._set_item_factors(np.ascontiguousarray(als.item_factors, dtype=np.float32))
        self.n_items = n_items
        return self

    def score(self, histories: Sequence[History]) -> Scores:
        y = self.item_factors
        lhs = np.empty((len(histories), self.factors, self.factors), dtype=np.float64)
        rhs = np.empty((len(histories), self.factors), dtype=np.float64)
        for u, history in enumerate(histories):
            liked = y[history].astype(np.float64)
            lhs[u] = self._gram + (self.alpha - 1.0) * (liked.T @ liked)
            rhs[u] = self.alpha * liked.sum(axis=0)
        user_factors = np.linalg.solve(lhs, rhs[:, :, None])[:, :, 0]
        return np.asarray(user_factors.astype(np.float32) @ y.T, dtype=np.float32)

    def save(self, path: Path) -> None:
        meta = {"model": self.name, "params": self.params, "n_items": self.n_items}
        save_arrays(path, meta, {"item_factors": self.item_factors})

    @classmethod
    def load(cls, path: Path) -> Self:
        meta = load_meta(path)
        model = cls(**meta["params"])
        model.n_items = meta["n_items"]
        model._set_item_factors(np.load(path / "item_factors.npy"))
        return model
