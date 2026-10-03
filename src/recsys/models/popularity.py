"""Popularity: recommend the most-liked movies to everyone.

The baseline every other model must beat. It ignores the user's history apart from not
recommending movies they have already seen.

With `half_life_days` set, an interaction counts for half as much every `half_life_days`
before the end of the training data, so the ranking follows what is popular *now* rather
than over the whole training window.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd

from recsys.models.base import History, Recommender, Scores, load_meta, save_arrays

_SECONDS_PER_DAY = 86_400


class Popularity(Recommender):
    name = "popularity"

    def __init__(self, half_life_days: float | None = None) -> None:
        self.half_life_days = half_life_days

    @property
    def params(self) -> dict[str, Any]:
        return {"half_life_days": self.half_life_days}

    def fit(self, interactions: pd.DataFrame, n_items: int) -> Self:
        weights = None
        if self.half_life_days is not None:
            ts = interactions["timestamp"].to_numpy()
            age_days = (ts.max() - ts) / _SECONDS_PER_DAY
            weights = 0.5 ** (age_days / self.half_life_days)
        counts = np.bincount(interactions["item_idx"].to_numpy(), weights, minlength=n_items)
        self.item_scores = counts.astype(np.float32)
        self.n_items = n_items
        return self

    def score(self, histories: Sequence[History]) -> Scores:
        return np.tile(self.item_scores, (len(histories), 1))

    def save(self, path: Path) -> None:
        meta = {"model": self.name, "params": self.params, "n_items": self.n_items}
        save_arrays(path, meta, {"item_scores": self.item_scores})

    @classmethod
    def load(cls, path: Path) -> Self:
        meta = load_meta(path)
        model = cls(**meta["params"])
        model.n_items = meta["n_items"]
        model.item_scores = np.load(path / "item_scores.npy")
        return model
