"""Model registry: maps the names used in params.yaml and dvc.yaml to classes."""

from pathlib import Path
from typing import Any

from recsys.models.base import Recommender
from recsys.models.ease import EASE
from recsys.models.ials import IALS
from recsys.models.item_knn import ItemKNN
from recsys.models.popularity import Popularity
from recsys.models.sasrec import SASRec

MODELS: dict[str, type[Recommender]] = {
    cls.name: cls for cls in (Popularity, ItemKNN, IALS, EASE, SASRec)
}


def build_model(name: str, params: dict[str, Any], seed: int = 0) -> Recommender:
    """`seed` reaches only the models with random steps (`uses_seed`); the rest ignore it."""
    cls: Any = MODELS[name]
    if cls.uses_seed:
        params = {**params, "seed": seed}
    model: Recommender = cls(**params)
    return model


def load_model(name: str, path: Path) -> Recommender:
    return MODELS[name].load(path)
