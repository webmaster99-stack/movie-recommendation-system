"""The one interface all five models share.

`recommend` takes a raw history (the item indices a user liked, oldest first), never a user
id. A model therefore can't memorise users: someone who signed up a minute ago and rated ten
movies is scored exactly like a user from the training set, and offline evaluation and the
API go through the same code.

Subclasses implement `fit`, `score`, `save` and `load`. Ranking (masking seen items, picking
the top k) lives here so every model does it identically.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

from recsys.utils.io import write_json

History = npt.NDArray[np.int32]  # item_idx values, oldest first
Scores = npt.NDArray[np.float32]  # (n_users, n_items)
ItemIds = npt.NDArray[np.int32]


def histories_to_csr(histories: Sequence[History], n_items: int) -> sp.csr_matrix:
    """Binary user x item matrix with one row per history."""
    indptr = np.concatenate([[0], np.cumsum([len(h) for h in histories])])
    indices = np.concatenate([*histories, np.empty(0, dtype=np.int32)])
    data = np.ones(len(indices), dtype=np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=(len(histories), n_items))


def top_k(scores: Scores, k: int) -> ItemIds:
    """The k best-scoring columns per row, best first; ties go to the lower index.

    A stable full sort is slower than `argpartition`, but the result is fully defined by the
    scores. `argpartition` picks arbitrarily among tied scores, and NumPy uses different
    SIMD code paths per CPU, so the same model could recommend different movies on
    different machines.
    """
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return order.astype(np.int32)


def save_arrays(path: Path, meta: dict[str, Any], arrays: dict[str, npt.NDArray[Any]]) -> None:
    """Write a model as `meta.json` plus one `.npy` per array.

    Not `.npz`: a zip stores the time each entry was written, so the same model would hash
    differently on every run and DVC would see a change.
    """
    path.mkdir(parents=True, exist_ok=True)
    write_json(meta, path / "meta.json")
    for name, array in arrays.items():
        np.save(path / f"{name}.npy", array, allow_pickle=False)


def load_meta(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    return meta


class Recommender(ABC):
    name: ClassVar[str]
    n_items: int

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Hyperparameters, as passed to the constructor."""

    @abstractmethod
    def fit(self, interactions: pd.DataFrame, n_items: int) -> Self:
        """Learn from interactions with columns user_idx, item_idx, timestamp."""

    @abstractmethod
    def score(self, histories: Sequence[History]) -> Scores:
        """A fresh (len(histories), n_items) array; higher means more recommended."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write the fitted model into the directory `path`."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self:
        """Read a model written by `save`."""

    def recommend_batch(
        self, histories: Sequence[History], k: int, exclude_seen: bool = True
    ) -> ItemIds:
        """Top-k item indices for each history, shape (len(histories), k)."""
        scores = self.score(histories)
        if exclude_seen:
            rows = np.repeat(np.arange(len(histories)), [len(h) for h in histories])
            cols = np.concatenate([*histories, np.empty(0, dtype=np.int32)])
            scores[rows, cols] = -np.inf
        return top_k(scores, k)

    def recommend(self, user_history: History, k: int, exclude_seen: bool = True) -> ItemIds:
        history = np.asarray(user_history, dtype=np.int32)
        top: ItemIds = self.recommend_batch([history], k, exclude_seen)[0]
        return top
