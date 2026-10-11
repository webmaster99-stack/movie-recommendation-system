"""Posters and plot summaries from TMDB, cached in Postgres.

TMDB's terms allow caching their data for at most 6 months and forbid redistributing it, so
it never enters the DVC pipeline or git: only this module fetches it, on demand, by the
`tmdb_id` that MovieLens provides. A cached row is refreshed once it is older than
`refresh_days` and is deleted, used or not, before it reaches `max_age_days`.

Without an API key nothing is fetched and movies are served without posters.
"""

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from recsys_api.catalog import Movie
from recsys_api.db import MovieMetadata, as_utc, now, upsert

logger = logging.getLogger(__name__)

API_URL = "https://api.themoviedb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p"
_FIELDS = ("title", "overview", "poster_path", "backdrop_path", "release_date", "runtime")
_PURGE_EVERY = timedelta(hours=6)


class TmdbClient:
    def __init__(self, api_key: str, http: httpx.Client | None = None) -> None:
        # TMDB issues two credentials: a short v3 key sent as a query parameter and a long
        # v4 token (a JWT) sent as a bearer header. Either works here.
        bearer = api_key.startswith("eyJ")
        self._params = {} if bearer else {"api_key": api_key}
        self._headers = {"Authorization": f"Bearer {api_key}"} if bearer else {}
        self._http = http or httpx.Client(timeout=5.0)

    def movie(self, tmdb_id: int) -> dict[str, Any] | None:
        """The movie's details; None when TMDB has no such id. Raises on any other failure."""
        response = self._http.get(
            f"{API_URL}/movie/{tmdb_id}", params=self._params, headers=self._headers
        )
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        response.raise_for_status()
        details: dict[str, Any] = response.json()
        return details


def _row(movie: Movie, details: dict[str, Any] | None, fetched_at: datetime) -> dict[str, Any]:
    details = details or {}
    return {
        "movie_id": movie.movie_id,
        "tmdb_id": movie.tmdb_id,
        "found": bool(details),
        **{field: details.get(field) or None for field in _FIELDS},
        "vote_average": details.get("vote_average"),
        "fetched_at": fetched_at,
    }


class MetadataCache:
    def __init__(
        self,
        client: TmdbClient | None,
        refresh_days: int = 150,
        max_age_days: int = 180,
        clock: Callable[[], datetime] = now,
    ) -> None:
        if refresh_days >= max_age_days:
            raise ValueError("rows must be refreshed before they are too old to keep")
        self.client = client
        self.refresh_after = timedelta(days=refresh_days)
        self.max_age = timedelta(days=max_age_days)
        self.clock = clock
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tmdb")
        self._last_purge: datetime | None = None

    def purge(self, session: Session) -> None:
        """Delete every row past the age limit, whether or not anyone asks for it again."""
        session.execute(
            delete(MovieMetadata).where(MovieMetadata.fetched_at < self.clock() - self.max_age)
        )
        session.commit()
        self._last_purge = self.clock()

    def _fetch(self, movie: Movie) -> tuple[Movie, dict[str, Any] | None, bool]:
        assert self.client is not None and movie.tmdb_id is not None
        try:
            return movie, self.client.movie(movie.tmdb_id), True
        except httpx.HTTPError as error:
            # Only the error's type: its message contains the request URL, and with it the key.
            logger.warning("TMDB lookup of movie %s failed: %s", movie.movie_id, type(error))
            return movie, None, False

    def get_many(self, session: Session, movies: Sequence[Movie]) -> dict[int, MovieMetadata]:
        """Metadata by `movie_id` for the movies TMDB knows; the rest are simply absent."""
        if self._last_purge is None or self.clock() - self._last_purge > _PURGE_EVERY:
            self.purge(session)
        wanted = {m.movie_id: m for m in movies if m.tmdb_id is not None}
        if not wanted:
            return {}
        rows = session.scalars(
            select(MovieMetadata).where(MovieMetadata.movie_id.in_(list(wanted)))
        ).all()
        cached = {row.movie_id: row for row in rows}

        if self.client is not None:
            current = self.clock()
            stale = [
                movie
                for movie_id, movie in wanted.items()
                if movie_id not in cached
                or cached[movie_id].tmdb_id != movie.tmdb_id
                or current - as_utc(cached[movie_id].fetched_at) > self.refresh_after
            ]
            fetched = [r for r in self._pool.map(self._fetch, stale) if r[2]]
            for movie, details, _ in fetched:
                columns = _row(movie, details, current)
                update = [name for name in columns if name != "movie_id"]
                upsert(session, MovieMetadata, columns, update)
            if fetched:
                session.commit()
                session.expire_all()
                refreshed = session.scalars(
                    select(MovieMetadata).where(
                        MovieMetadata.movie_id.in_([m.movie_id for m, _, _ in fetched])
                    )
                ).all()
                cached.update({row.movie_id: row for row in refreshed})
        # A refresh that failed leaves the old row in use, but never past the age limit.
        oldest = self.clock() - self.max_age
        return {
            movie_id: row
            for movie_id, row in cached.items()
            if row.found and as_utc(row.fetched_at) >= oldest
        }


def image_url(path: str | None, size: str) -> str | None:
    return f"{IMAGE_URL}/{size}{path}" if path else None
