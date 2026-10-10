"""Recency variants: a personalised model plus a "what is being watched right now" prior.

The personalised models see a history as a bag of liked movies with no dates, so they rank a
ten-year-old favourite and last month's release alike. Time-decayed Popularity showed how
much that costs for returning users. A recency variant adds the two signals:

    score = standardise(base model's scores) + weight * prior

- The base scores are standardised per user (mean 0, standard deviation 1 over the catalog),
  because every model scores on its own scale.
- `prior` is the log of the time-decayed like counts, the same counts Popularity ranks by
  (`half_life_days`), standardised over the catalog.
- `weight` sets how much the prior counts; 0 gives the base model's ranking back.

The base model is not retrained: a variant only adds one number per movie to an already
fitted model, attached with `wrap`. That keeps a variant cheap to tune (no refit per trial)
and to serve (the base model's weights plus one small vector).
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.models.base import History, Recommender, Scores, load_meta, save_arrays
from recsys.models.ease import EASE
from recsys.models.ials import IALS
from recsys.models.item_knn import ItemKNN
from recsys.models.popularity import Popularity
from recsys.models.sasrec import SASRec


def _standardise(x: npt.NDArray[Any], axis: int | None = None) -> npt.NDArray[np.float32]:
    """Mean 0 and standard deviation 1 along `axis`; a constant input becomes all zeros."""
    centred = x - x.mean(axis=axis, keepdims=True)
    std = centred.std(axis=axis, keepdims=True)
    out = np.divide(centred, std, out=np.zeros_like(centred), where=std > 0)
    return np.asarray(out, dtype=np.float32)


class Recency(Recommender):
    base_cls: ClassVar[type[Recommender]]
    base: Recommender

    def __init__(self, half_life_days: float = 30.0, weight: float = 1.0) -> None:
        self.half_life_days = half_life_days
        self.weight = weight

    @property
    def params(self) -> dict[str, Any]:
        return {"half_life_days": self.half_life_days, "weight": self.weight}

    def fit(self, interactions: pd.DataFrame, n_items: int) -> Self:
        """Compute the prior. The base model is fitted separately and attached with `wrap`."""
        counts = Popularity(self.half_life_days).fit(interactions, n_items).item_scores
        self.prior = _standardise(np.log1p(counts.astype(np.float64)))
        self.n_items = n_items
        return self

    def wrap(self, base: Recommender) -> Self:
        """Attach the fitted base model, trained on the same interactions as the prior."""
        if not isinstance(base, self.base_cls):
            raise TypeError(f"{self.name} wraps {self.base_cls.name}, got {base.name}")
        self.base = base
        return self

    def score(self, histories: Sequence[History]) -> Scores:
        scores = _standardise(self.base.score(histories), axis=1)
        scores += np.float32(self.weight) * self.prior
        return scores

    def save(self, path: Path) -> None:
        """Self-contained: the prior here and the base model under `base/`."""
        meta = {"model": self.name, "params": self.params, "n_items": self.n_items}
        save_arrays(path, meta, {"prior": self.prior})
        self.base.save(path / "base")

    @classmethod
    def load(cls, path: Path) -> Self:
        meta = load_meta(path)
        model = cls(**meta["params"])
        model.n_items = meta["n_items"]
        model.prior = np.load(path / "prior.npy")
        return model.wrap(cls.base_cls.load(path / "base"))


class ItemKNNRecency(Recency):
    name = "item_knn_recency"
    base_cls = ItemKNN


class IALSRecency(Recency):
    name = "ials_recency"
    base_cls = IALS


class EASERecency(Recency):
    name = "ease_recency"
    base_cls = EASE


class SASRecRecency(Recency):
    name = "sasrec_recency"
    base_cls = SASRec


RECENCY_MODELS: tuple[type[Recency], ...] = (
    ItemKNNRecency,
    IALSRecency,
    EASERecency,
    SASRecRecency,
)
