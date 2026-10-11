"""Everything read from the bundle directory at start-up: model, catalog and metadata."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recsys_api.catalog import Catalog
from recsys_api.model import Ease, Recency, Scorer, base_of, load_model, read_json


@dataclass(frozen=True)
class Bundle:
    model: Scorer
    catalog: Catalog
    metadata: dict[str, Any]

    @property
    def name(self) -> str:
        name: str = self.metadata["model"]
        return name

    @property
    def like_threshold(self) -> float:
        """The lowest rating that counted as a like when the model was trained."""
        return float(self.metadata["preprocess"]["positive_threshold"])

    @property
    def similar_model(self) -> Ease | None:
        base = base_of(self.model)
        return base if isinstance(base, Ease) else None


def load_bundle(path: Path) -> Bundle:
    if not (path / "metadata.json").is_file():
        raise FileNotFoundError(
            f"no model bundle in '{path}': run `uv run dvc pull bundle` (or `dvc repro export`) "
            "or set BUNDLE_DIR"
        )
    metadata = read_json(path / "metadata.json")
    model = load_model(path / "model")
    # The recency prior doubles as "how popular is this movie right now", which orders
    # search results and the onboarding list. A model without one falls back to catalog order.
    popularity = model.prior if isinstance(model, Recency) else np.zeros(model.n_items)
    catalog = Catalog(read_json(path / "items.json"), popularity)
    if len(catalog) != model.n_items:
        raise ValueError(f"{len(catalog)} movies in items.json but the model has {model.n_items}")
    return Bundle(model, catalog, metadata)
