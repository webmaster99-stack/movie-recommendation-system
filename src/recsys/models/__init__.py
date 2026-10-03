"""Model registry: maps the names used in params.yaml and dvc.yaml to classes."""

from pathlib import Path
from typing import Any

from recsys.models.base import Recommender
from recsys.models.item_knn import ItemKNN
from recsys.models.popularity import Popularity

MODELS: dict[str, type[Recommender]] = {cls.name: cls for cls in (Popularity, ItemKNN)}


def build_model(name: str, params: dict[str, Any]) -> Recommender:
    cls: Any = MODELS[name]
    model: Recommender = cls(**params)
    return model


def load_model(name: str, path: Path) -> Recommender:
    return MODELS[name].load(path)
