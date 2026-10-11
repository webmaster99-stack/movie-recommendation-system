"""Scoring with the exported model, on numpy and scipy alone.

This repeats the arithmetic of `recsys.models.ease` and `recsys.models.recency` instead of
importing them, because importing `recsys.models` pulls in pandas, implicit and torch, which
do not fit the API's 512 MB host. The operations are kept the same on purpose (same sparse
product, same standardisation, same stable sort), and `tests/test_model_parity.py` checks
that both sides return identical recommendations from the same saved model.

Only the models that have been selected for serving are implemented. Loading any other one
fails at start-up with a message that says so.
"""

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

ItemIds = npt.NDArray[np.int32]
Scores = npt.NDArray[np.float32]  # (1, n_items)


class UnsupportedModelError(RuntimeError):
    pass


class Scorer(Protocol):
    n_items: int

    def score(self, history: ItemIds) -> Scores:
        """A fresh (1, n_items) array for one history of item indices, oldest first."""


def _standardise(x: npt.NDArray[Any], axis: int | None = None) -> npt.NDArray[np.float32]:
    """Mean 0 and standard deviation 1 along `axis`; a constant input becomes all zeros."""
    centred = x - x.mean(axis=axis, keepdims=True)
    std = centred.std(axis=axis, keepdims=True)
    out = np.divide(centred, std, out=np.zeros_like(centred), where=std > 0)
    return np.asarray(out, dtype=np.float32)


class Ease:
    """Item-to-item weights B; a user's scores are the sum of B's rows for their history."""

    def __init__(self, weights: npt.NDArray[np.float32] | sp.csr_matrix) -> None:
        self.weights = weights
        self.n_items: int = weights.shape[0]

    @classmethod
    def load(cls, path: Path, meta: dict[str, Any]) -> "Ease":
        n = meta["n_items"]
        if meta["params"]["keep_per_item"] is None:
            return cls(np.load(path / "weights.npy"))
        parts = tuple(np.load(path / f"{name}.npy") for name in ("data", "indices", "indptr"))
        return cls(sp.csr_matrix(parts, shape=(n, n)))

    def score(self, history: ItemIds) -> Scores:
        ones = np.ones(len(history), dtype=np.float32)
        row = sp.csr_matrix((ones, history, [0, len(history)]), shape=(1, self.n_items))
        scores = row @ self.weights
        if sp.issparse(scores):
            scores = scores.toarray()
        return np.asarray(scores, dtype=np.float32)

    def similar(self, item: int, k: int) -> ItemIds:
        """The movies that liking `item` predicts most strongly, best first.

        Only positive weights count, so a movie with few of them returns fewer than `k`.
        """
        row = self.weights[item]
        weights = np.asarray(row.toarray() if sp.issparse(row) else row, dtype=np.float32).ravel()
        weights[item] = 0.0
        order = np.argsort(-weights, kind="stable")[:k]
        return order[weights[order] > 0].astype(np.int32)


class Recency:
    """A base model's standardised scores plus `weight` times a per-movie recency prior."""

    def __init__(self, base: Scorer, prior: npt.NDArray[np.float32], weight: float) -> None:
        self.base = base
        self.prior = prior
        self.weight = weight
        self.n_items = base.n_items

    def score(self, history: ItemIds) -> Scores:
        scores = _standardise(self.base.score(history), axis=1)
        scores += np.float32(self.weight) * self.prior
        return scores


_BASE_MODELS = {"ease": Ease}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_base(path: Path) -> Ease:
    meta = read_json(path / "meta.json")
    if meta["model"] not in _BASE_MODELS:
        raise UnsupportedModelError(
            f"the API cannot serve '{meta['model']}' yet (supported: {sorted(_BASE_MODELS)}, "
            "each with or without a recency prior); add a scorer to recsys_api/model.py"
        )
    return _BASE_MODELS[meta["model"]].load(path, meta)


def load_model(path: Path) -> Scorer:
    """The model saved in `path` by `Recommender.save` (the bundle's `model/` directory)."""
    if not (path / "base").is_dir():
        return _load_base(path)
    meta = read_json(path / "meta.json")
    return Recency(_load_base(path / "base"), np.load(path / "prior.npy"), meta["params"]["weight"])


def base_of(model: Scorer) -> Scorer:
    return model.base if isinstance(model, Recency) else model


def recommend(model: Scorer, history: ItemIds, exclude: ItemIds, k: int) -> ItemIds:
    """The `k` best items for `history`, never one of `exclude`; ties go to the lower index.

    Offline evaluation excludes exactly the history. The app also excludes movies the user
    rated low, which are not in the history but should not come back as recommendations.
    """
    scores = model.score(history)[0]
    scores[exclude] = -np.inf
    k = min(k, model.n_items - len(np.unique(exclude)))
    return np.argsort(-scores, kind="stable")[:k].astype(np.int32)
