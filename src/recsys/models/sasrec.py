"""SASRec (Kang & McAuley 2018): a small transformer that predicts the next movie.

The other models treat a history as a set. SASRec reads it as a sequence, oldest first:
each position looks back at the earlier ones (causal self-attention) and outputs a vector
that should be close to the embedding of the movie the user liked next. Scoring a user is
one forward pass over their last `max_len` likes; the output at the final position is
compared with every movie embedding.

Training details:
- Every epoch, a user with more than `max_len + 1` likes contributes a randomly placed
  window of that length, so long histories are used in full over the epochs.
- The loss is a softmax over the true next movie and `n_negatives` random movies shared by
  the whole batch (`n_negatives: null` = all movies; much slower on CPU).
- `max_users` trains on a random subset of users to bound CPU time.

Weights are saved as plain `.npy` arrays (see `save_arrays`), not with `torch.save`, whose
zip format is not byte-stable.
"""

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from recsys.models.base import History, Recommender, Scores, load_meta, save_arrays

log = logging.getLogger(__name__)


def _dropout(x: Tensor, rate: float, rng: np.random.Generator | None) -> Tensor:
    """Dropout with the mask drawn by numpy; `rng=None` (scoring) switches it off.

    Not `nn.Dropout`: torch's CPU sampler is several times slower than numpy's, and with
    seven dropout layers it took a third of the training time.
    """
    if rng is None or rate == 0.0:
        return x
    keep = torch.from_numpy(rng.random(x.shape, dtype=np.float32) >= rate)
    return x * keep / (1.0 - rate)


class _Block(nn.Module):
    """Pre-norm transformer block: self-attention, then a feed-forward layer."""

    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x: Tensor, blocked: Tensor, rng: np.random.Generator | None) -> Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(self.norm1(x)).view(batch, length, 3, self.n_heads, dim // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (batch, heads, length, head_dim)
        attention = q @ k.transpose(-1, -2) / math.sqrt(dim // self.n_heads)
        attention = attention.masked_fill(blocked[:, None], float("-inf")).softmax(dim=-1)
        attended = (
            (_dropout(attention, self.dropout, rng) @ v).transpose(1, 2).reshape(batch, length, dim)
        )
        x = x + _dropout(self.out(attended), self.dropout, rng)
        out: Tensor = x + _dropout(self.ff(self.norm2(x)), self.dropout, rng)
        return out


class SASRecNet(nn.Module):
    def __init__(
        self,
        n_items: int,
        hidden_dim: int,
        n_blocks: int,
        n_heads: int,
        max_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pad = n_items  # one extra embedding row stands for "no movie"
        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=self.pad)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.blocks = nn.ModuleList(_Block(hidden_dim, n_heads, dropout) for _ in range(n_blocks))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_normal_(parameter)
        with torch.no_grad():
            self.item_emb.weight[self.pad].zero_()

    def forward(self, seq: Tensor, rng: np.random.Generator | None = None) -> Tensor:
        """(batch, length) item ids, left-padded -> (batch, length, hidden_dim).

        `rng` draws the dropout masks during training; leave it out when scoring.
        """
        length = seq.shape[1]
        is_pad = seq == self.pad
        x = self.item_emb(seq) * math.sqrt(self.item_emb.embedding_dim)
        x = _dropout(x + self.pos_emb(torch.arange(length)), self.dropout, rng)
        # A position may not look at later positions or at padding. It may always look at
        # itself, so that a padded position never ends up with nothing to attend to.
        future = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        blocked = (future | is_pad[:, None, :]) & ~torch.eye(length, dtype=torch.bool)
        for block in self.blocks:
            x = block(x, blocked, rng)
        out: Tensor = self.norm(x)
        return out

    def item_vectors(self) -> Tensor:
        return self.item_emb.weight[: self.pad]


def _pad_left(histories: Sequence[History], length: int, pad: int) -> npt.NDArray[np.int64]:
    """The last `length` items of each history, right-aligned in a `pad`-filled matrix."""
    out = np.full((len(histories), length), pad, dtype=np.int64)
    for row, history in zip(out, histories, strict=True):
        if len(history):
            tail = history[-length:]
            row[-len(tail) :] = tail
    return out


class SASRec(Recommender):
    name = "sasrec"
    uses_seed = True

    def __init__(
        self,
        hidden_dim: int = 64,
        n_blocks: int = 2,
        n_heads: int = 1,
        max_len: int = 50,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 20,
        batch_size: int = 256,
        n_negatives: int | None = 1024,
        max_users: int | None = None,
        seed: int = 0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.max_len = max_len
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_negatives = n_negatives
        self.max_users = max_users
        self.seed = seed

    @property
    def params(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "max_len": self.max_len,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "n_negatives": self.n_negatives,
            "max_users": self.max_users,
            "seed": self.seed,
        }

    def _build_net(self, n_items: int) -> SASRecNet:
        return SASRecNet(
            n_items, self.hidden_dim, self.n_blocks, self.n_heads, self.max_len, self.dropout
        )

    def _loss(self, net: SASRecNet, batch: Tensor, rng: np.random.Generator) -> Tensor:
        """Predict item t+1 from items 1..t, at every non-padded position of the batch."""
        inputs, targets = batch[:, :-1], batch[:, 1:]
        known = targets != net.pad
        hidden, target = net(inputs, rng)[known], targets[known]
        items = net.item_vectors()
        if self.n_negatives is None:
            return F.cross_entropy(hidden @ items.T, target)
        negatives = torch.from_numpy(rng.integers(0, len(items), self.n_negatives))
        positive_logit = (hidden * items[target]).sum(dim=-1, keepdim=True)
        negative_logits = hidden @ items[negatives].T
        # A sampled "negative" that is this position's true next movie must not count.
        negative_logits = negative_logits.masked_fill(
            negatives[None, :] == target[:, None], float("-inf")
        )
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long))

    def fit(self, interactions: pd.DataFrame, n_items: int) -> Self:
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        # One sequence per user, oldest first (ties broken by item for a fixed order).
        df = interactions.sort_values(["user_idx", "timestamp", "item_idx"], kind="mergesort")
        items = df["item_idx"].to_numpy(dtype=np.int64)
        _, starts, lengths = np.unique(
            df["user_idx"].to_numpy(), return_index=True, return_counts=True
        )
        starts, lengths = starts[lengths >= 2], lengths[lengths >= 2]  # need a next item
        if self.max_users is not None and self.max_users < len(starts):
            chosen = np.sort(rng.choice(len(starts), self.max_users, replace=False))
            starts, lengths = starts[chosen], lengths[chosen]

        window = self.max_len + 1  # max_len inputs plus the final target
        used = np.minimum(lengths, window)
        column = np.arange(window)
        filled = column >= (window - used)[:, None]

        net = self._build_net(n_items)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        for epoch in range(self.epochs):
            offset = rng.integers(0, lengths - used + 1)
            source = (starts + offset - (window - used))[:, None] + column
            sequences = np.full((len(starts), window), net.pad, dtype=np.int64)
            sequences[filled] = items[source[filled]]
            order = rng.permutation(len(sequences))
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                batch = torch.from_numpy(sequences[order[i : i + self.batch_size]])
                loss = self._loss(net, batch, rng)
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                total += loss.item() * len(batch)
            log.info("epoch %d/%d: loss %.4f", epoch + 1, self.epochs, total / len(sequences))

        self.net = net
        self.n_items = n_items
        return self

    @torch.no_grad()
    def score(self, histories: Sequence[History]) -> Scores:
        seq = torch.from_numpy(_pad_left(histories, self.max_len, self.net.pad))
        last = self.net(seq)[:, -1]
        return np.asarray((last @ self.net.item_vectors().T).numpy(), dtype=np.float32)

    def save(self, path: Path) -> None:
        meta = {"model": self.name, "params": self.params, "n_items": self.n_items}
        save_arrays(path, meta, {k: v.numpy() for k, v in self.net.state_dict().items()})

    @classmethod
    def load(cls, path: Path) -> Self:
        meta = load_meta(path)
        model = cls(**meta["params"])
        model.n_items = meta["n_items"]
        net = model._build_net(model.n_items)
        net.load_state_dict(
            {name: torch.from_numpy(np.load(path / f"{name}.npy")) for name in net.state_dict()}
        )
        model.net = net
        return model
