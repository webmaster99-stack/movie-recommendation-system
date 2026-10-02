"""Deterministic writers: same data + same pinned library versions -> byte-identical files,
so DVC hashes match across runs and machines."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": on Windows, text mode would write \r\n and change the bytes/hash per OS.
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
