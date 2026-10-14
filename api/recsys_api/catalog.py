"""The movie catalog from the bundle's `items.json`, with title search held in memory.

Movies are addressed by their MovieLens `movie_id` everywhere outside this package. The
model's column index (`item_idx`) changes whenever the catalog is rebuilt, so it must never
be stored in the database or shown to a client.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

# MovieLens files a title under its first real word: "Matrix, The", and with an alternative
# title "City of Lost Children, The (Cité des enfants perdus, La)".
_TRAILING_ARTICLE = re.compile(
    r"^(?P<rest>[^()]+), (?P<article>The|A|An|Le|La|Les|L'|El|Il|Der|Die|Das)(?P<aka> \(.*\))?$"
)
_NOT_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_NO_GENRES = "(no genres listed)"


def display_title(title: str) -> str:
    """'Matrix, The' -> 'The Matrix'."""
    match = _TRAILING_ARTICLE.match(title)
    if match is None:
        return title
    article = match["article"]
    joined = article + match["rest"] if article.endswith("'") else f"{article} {match['rest']}"
    return joined + (match["aka"] or "")


def normalise(text: str) -> str:
    """Lower case, accents removed, punctuation as single spaces: 'Amélie!' -> 'amelie'."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NOT_ALPHANUMERIC.sub(" ", ascii_only).strip()


@dataclass(frozen=True, slots=True)
class Movie:
    item_idx: int
    movie_id: int
    title: str
    year: int | None
    genres: tuple[str, ...]
    tmdb_id: int | None
    imdb_id: str | None


class Catalog:
    def __init__(
        self, records: list[dict[str, Any]], popularity: npt.NDArray[np.floating[Any]]
    ) -> None:
        """`records` in `item_idx` order; `popularity` has one number per movie, higher first."""
        self.movies = [
            Movie(
                item_idx=r["item_idx"],
                movie_id=r["movie_id"],
                title=display_title(r["title"]),
                year=r["year"],
                genres=tuple(g for g in r["genres"] if g != _NO_GENRES),
                tmdb_id=r["tmdb_id"],
                imdb_id=r["imdb_id"],
            )
            for r in records
        ]
        if [m.item_idx for m in self.movies] != list(range(len(self.movies))):
            raise ValueError("items.json must list the movies in item_idx order")
        if len(popularity) != len(self.movies):
            raise ValueError("one popularity value per movie is required")
        self._by_movie_id = {m.movie_id: m for m in self.movies}
        self._search_keys = [normalise(m.title) for m in self.movies]
        # Most popular first; ties go to the lower index, as everywhere else.
        self.popular = np.argsort(-np.asarray(popularity), kind="stable").astype(np.int32)
        self._rank = np.empty(len(self.movies), dtype=np.int64)
        self._rank[self.popular] = np.arange(len(self.movies))
        self.onboarding = self._onboarding_order()

    def __len__(self) -> int:
        return len(self.movies)

    def get(self, movie_id: int) -> Movie | None:
        return self._by_movie_id.get(movie_id)

    def popularity_percentile(self, indices: npt.NDArray[np.integer[Any]]) -> float | None:
        """Where the given movies sit in the popularity order on average: 0 for the most
        popular movie, 1 for the least. None when there is nothing to average."""
        if len(indices) == 0 or len(self.movies) < 2:
            return None
        return float(self._rank[indices].mean() / (len(self.movies) - 1))

    def search(self, query: str, limit: int) -> list[Movie]:
        """Movies whose title contains every word of `query`.

        Titles that start with the query come first, then titles where it starts a word,
        then the rest; within each group the most popular come first.
        """
        q = normalise(query)
        if not q:
            return []
        # Runs on every keystroke of the search box, so the scan over all titles stays a
        # plain substring test per word; only the matches are ranked.
        keys = self._search_keys
        matches: list[int] | range = range(len(keys))
        for word in q.split():
            matches = [idx for idx in matches if word in keys[idx]]
        word_start = f" {q}"
        hits = [
            (0 if keys[idx].startswith(q) else 1 if word_start in keys[idx] else 2, rank, idx)
            for idx, rank in zip(matches, self._rank[matches].tolist(), strict=True)
        ]
        hits.sort()
        return [self.movies[idx] for _, _, idx in hits[:limit]]

    def _onboarding_order(self, pool: int = 600) -> list[int]:
        """Well-known movies for a new user to rate, taking turns between genres.

        The plain popularity ranking is mostly drama, action and sci-fi, so a new user who
        watches documentaries or musicals would find little to rate. Each genre in turn
        contributes its most popular movie that is not already listed.
        """
        candidates = [int(i) for i in self.popular[:pool]]
        genres = sorted({g for i in candidates for g in self.movies[i].genres})
        queues = {g: [i for i in candidates if g in self.movies[i].genres] for g in genres}
        picked: dict[int, None] = {}
        while any(queues.values()):
            for genre in genres:
                queue = queues[genre]
                while queue and queue[0] in picked:
                    queue.pop(0)
                if queue:
                    picked[queue.pop(0)] = None
        return list(picked) + [i for i in candidates if i not in picked]
