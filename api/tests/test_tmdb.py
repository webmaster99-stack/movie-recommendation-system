from datetime import UTC, datetime, timedelta

import httpx
import pytest
from conftest import FakeTmdb
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from recsys_api.bundle import Bundle
from recsys_api.db import MovieMetadata
from recsys_api.tmdb import MetadataCache, TmdbClient, image_url

START = datetime(2026, 1, 1, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.time = START

    def __call__(self) -> datetime:
        return self.time

    def advance(self, days: float) -> None:
        self.time += timedelta(days=days)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def cache(tmdb: FakeTmdb, clock: Clock) -> MetadataCache:
    return MetadataCache(tmdb, refresh_days=150, max_age_days=180, clock=clock)  # type: ignore[arg-type]


@pytest.fixture
def session(engine: Engine) -> Session:
    return Session(engine)


def test_metadata_is_fetched_once_and_then_served_from_the_cache(
    cache: MetadataCache, tmdb: FakeTmdb, session: Session, bundle: Bundle
) -> None:
    movies = bundle.catalog.movies  # the last one has no tmdb_id
    first = cache.get_many(session, movies)
    assert sorted(tmdb.calls) == [194, 603, 604, 862, 949]
    assert set(first) == {10, 20, 30, 40, 50}
    assert first[10].poster_path == "/poster603.jpg"
    assert first[10].overview == "Plot of 603."
    assert first[10].runtime == 136

    again = cache.get_many(session, movies)
    assert len(tmdb.calls) == 5
    assert again[10].poster_path == "/poster603.jpg"


def test_rows_are_refreshed_after_the_refresh_age(
    cache: MetadataCache, tmdb: FakeTmdb, session: Session, bundle: Bundle, clock: Clock
) -> None:
    matrix = [bundle.catalog.movies[0]]
    cache.get_many(session, matrix)
    clock.advance(149)
    cache.get_many(session, matrix)
    assert tmdb.calls == [603]
    clock.advance(2)
    refreshed = cache.get_many(session, matrix)
    assert tmdb.calls == [603, 603]
    assert refreshed[10].fetched_at.replace(tzinfo=UTC) == clock.time


def test_a_failed_refresh_keeps_the_old_row_but_never_past_the_age_limit(
    cache: MetadataCache, tmdb: FakeTmdb, session: Session, bundle: Bundle, clock: Clock
) -> None:
    matrix = [bundle.catalog.movies[0]]
    cache.get_many(session, matrix)
    tmdb.failing = True
    clock.advance(160)
    assert cache.get_many(session, matrix)[10].poster_path == "/poster603.jpg"
    clock.advance(21)  # 181 days old: TMDB's terms no longer allow keeping it
    assert cache.get_many(session, matrix) == {}
    assert session.scalars(select(MovieMetadata)).all() == []


def test_old_rows_are_deleted_even_if_nobody_asks_for_them(
    cache: MetadataCache, session: Session, bundle: Bundle, clock: Clock
) -> None:
    cache.get_many(session, bundle.catalog.movies[:2])
    clock.advance(181)
    cache.get_many(session, [bundle.catalog.movies[3]])  # a different movie
    assert [row.movie_id for row in session.scalars(select(MovieMetadata))] == [40]


def test_a_movie_tmdb_does_not_know_is_remembered_as_missing(
    cache: MetadataCache, tmdb: FakeTmdb, session: Session, bundle: Bundle
) -> None:
    tmdb.missing = {603}
    matrix = [bundle.catalog.movies[0]]
    assert cache.get_many(session, matrix) == {}
    assert cache.get_many(session, matrix) == {}
    assert tmdb.calls == [603]  # not asked again until the row is due for a refresh
    row = session.scalars(select(MovieMetadata)).one()
    assert (row.found, row.poster_path) == (False, None)


def test_without_an_api_key_nothing_is_fetched(session: Session, bundle: Bundle) -> None:
    assert MetadataCache(None).get_many(session, bundle.catalog.movies) == {}


def test_refresh_must_come_before_the_age_limit() -> None:
    with pytest.raises(ValueError, match="refreshed before"):
        MetadataCache(None, refresh_days=180, max_age_days=180)


def test_image_url() -> None:
    assert image_url("/abc.jpg", "w342") == "https://image.tmdb.org/t/p/w342/abc.jpg"
    assert image_url(None, "w342") is None


@pytest.mark.parametrize(
    ("key", "sent_as_parameter", "sent_as_header"),
    [("v3key", "v3key", None), ("eyJ.v4.token", None, "Bearer eyJ.v4.token")],
)
def test_client_sends_either_kind_of_tmdb_credential(
    key: str, sent_as_parameter: str | None, sent_as_header: str | None
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/3/movie/603"
        assert request.url.params.get("api_key") == sent_as_parameter
        assert request.headers.get("Authorization") == sent_as_header
        return httpx.Response(200, json={"title": "The Matrix"})

    client = TmdbClient(key, httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.movie(603) == {"title": "The Matrix"}


def test_client_returns_none_for_an_unknown_movie_and_raises_on_other_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404 if request.url.path.endswith("/1") else 429)

    client = TmdbClient("key", httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.movie(1) is None
    with pytest.raises(httpx.HTTPStatusError):
        client.movie(2)
