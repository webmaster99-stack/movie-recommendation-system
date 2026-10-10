"""Model registry: maps the names used in params.yaml and dvc.yaml to classes."""

from pathlib import Path
from typing import Any

import pandas as pd

from recsys.models.base import Recommender
from recsys.models.ease import EASE
from recsys.models.ials import IALS
from recsys.models.item_knn import ItemKNN
from recsys.models.popularity import Popularity
from recsys.models.recency import RECENCY_MODELS, Recency
from recsys.models.sasrec import SASRec

# The five algorithms, each with a `train` stage of its own.
BASE_MODELS: dict[str, type[Recommender]] = {
    cls.name: cls for cls in (Popularity, ItemKNN, IALS, EASE, SASRec)
}
# Every model that is evaluated: the five plus the recency variants built on top of them.
MODELS: dict[str, type[Recommender]] = {
    **BASE_MODELS,
    **{cls.name: cls for cls in RECENCY_MODELS},
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


def base_name(name: str) -> str:
    """The trained model `name` scores with: itself, or the model a recency variant wraps."""
    cls = MODELS[name]
    return cls.base_cls.name if issubclass(cls, Recency) else name


def load_fitted(
    name: str, models_dir: Path, params: dict[str, Any], interactions: pd.DataFrame
) -> Recommender:
    """Model `name` ready to score, from the trained models saved under `models_dir`.

    A recency variant has no saved model of its own: its prior is computed here from
    `interactions`, which must be the data its base model was trained on, with the variant's
    `params`.
    """
    base = load_model(base_name(name), models_dir / base_name(name))
    if base.name == name:
        return base
    variant = build_model(name, params).fit(interactions, base.n_items)
    assert isinstance(variant, Recency)
    return variant.wrap(base)
