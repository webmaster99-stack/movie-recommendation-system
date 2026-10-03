"""Turn the temporal splits into (history, targets) pairs: what the model is shown per user
and what it should have recommended.

Two scenarios, because about 40% of the users in the validation and test periods are new:

- **warm**: the user has interactions before the cut-off. History is everything they liked
  before it; targets are everything they liked in the evaluation period.
- **onboarding**: the user first appears in the evaluation period. Their first `n_history`
  likes play the role of the movies rated during sign-up, and the rest are targets. This
  measures what the app's onboarding flow will deliver.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from recsys.models.base import History


@dataclass(frozen=True)
class EvalSet:
    user_idx: npt.NDArray[np.int32]
    histories: list[History]
    targets: list[History]


def _per_user(df: pd.DataFrame) -> tuple[npt.NDArray[np.int32], list[History]]:
    """Each user's items, oldest first. Users come out in ascending order."""
    df = df.sort_values(["user_idx", "timestamp", "item_idx"], kind="mergesort")
    users, starts = np.unique(df["user_idx"].to_numpy(), return_index=True)
    items = df["item_idx"].to_numpy(dtype=np.int32)
    return users.astype(np.int32), np.split(items, starts[1:])


def warm_eval_set(history: pd.DataFrame, targets: pd.DataFrame) -> EvalSet:
    targets = targets[targets["user_idx"].isin(history["user_idx"])]
    history = history[history["user_idx"].isin(targets["user_idx"])]
    users, histories = _per_user(history)
    _, target_items = _per_user(targets)
    return EvalSet(users, histories, target_items)


def onboarding_eval_set(history: pd.DataFrame, targets: pd.DataFrame, n_history: int) -> EvalSet:
    """New users with more than `n_history` likes, so at least one target is left."""
    new = targets[~targets["user_idx"].isin(history["user_idx"])]
    users, items = _per_user(new)
    keep = np.array([len(seq) > n_history for seq in items], dtype=bool)
    kept = [seq for seq, k in zip(items, keep, strict=True) if k]
    return EvalSet(users[keep], [s[:n_history] for s in kept], [s[n_history:] for s in kept])
