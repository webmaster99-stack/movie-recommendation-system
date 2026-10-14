"""The migrations on a real Postgres: they apply, the API works on the tables they create,
and the monitoring views show what happened to the one role that may read them.

SQLite, which the other tests use, has no views, roles or arrays, so none of this can be
checked there. These tests need a throwaway Postgres and are skipped without one:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=postgres -p 54329:5432 postgres:17
    export TEST_POSTGRES_URL=postgresql://postgres:postgres@127.0.0.1:54329/postgres
    uv run pytest api/tests

Everything in that database is dropped first; the roles the migrations create stay. CI runs
them against a service container.
"""

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from conftest import JWT_SECRET, FakeTmdb, make_token
from fastapi.testclient import TestClient
from sqlalchemy import Engine, make_url, text
from sqlalchemy.exc import ProgrammingError

from recsys.monitoring.drift import connect, load_production, online_quality
from recsys.utils.config import PROJECT_ROOT
from recsys_api.auth import TokenVerifier
from recsys_api.bundle import Bundle
from recsys_api.config import Settings
from recsys_api.db import make_engine
from recsys_api.main import create_app
from recsys_api.tmdb import MetadataCache

POSTGRES_URL = os.environ.get("TEST_POSTGRES_URL", "")
pytestmark = pytest.mark.skipif(not POSTGRES_URL, reason="TEST_POSTGRES_URL is not set")

SUPABASE = PROJECT_ROOT / "supabase"
ALICE = uuid.UUID("00000000-0000-4000-8000-00000000000a")
BOB = uuid.UUID("00000000-0000-4000-8000-00000000000b")
_RESET = """
drop schema if exists monitoring cascade;
drop schema if exists auth cascade;
drop schema public cascade;
create schema public;
"""


def run_script(engine: Engine, sql: str) -> None:
    """A whole SQL file in one go, as `psql -f` would run it."""
    connection = engine.raw_connection()
    try:
        connection.cursor().execute(sql)
        connection.commit()
    finally:
        connection.close()


@pytest.fixture(scope="module")
def postgres() -> Iterator[Engine]:
    engine = make_engine(POSTGRES_URL)
    run_script(engine, _RESET)
    run_script(engine, (SUPABASE / "local" / "auth_stub.sql").read_text(encoding="utf-8"))
    for migration in sorted((SUPABASE / "migrations").glob("*.sql")):
        run_script(engine, migration.read_text(encoding="utf-8"))
    with engine.begin() as connection:
        for user, name in ((ALICE, "Alice"), (BOB, None)):
            connection.execute(
                text("insert into auth.users values (:id, cast(:meta as jsonb))"),
                {"id": user, "meta": f'{{"display_name": "{name}"}}' if name else "{}"},
            )
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def used_app(postgres: Engine, bundle: Bundle) -> dict[str, Any]:
    """The API on Postgres after a short session: Alice rates, gets a list, clicks its first
    movie and rates it; Bob only asks for a list. Returns what the API answered."""
    settings = Settings(_env_file=None, model_version="7")
    app = create_app(
        settings,
        bundle=bundle,
        engine=postgres,
        verifier=TokenVerifier(jwt_secret=JWT_SECRET),
        metadata=MetadataCache(FakeTmdb()),  # type: ignore[arg-type]
    )
    client = TestClient(app)
    alice, bob = ({"Authorization": f"Bearer {make_token(user)}"} for user in (ALICE, BOB))

    assert client.get("/health").status_code == 200
    rated = client.post("/ratings", json={"movie_id": 10, "rating": 5.0}, headers=alice)
    assert rated.status_code == 200, rated.text
    recommended = client.get("/recommendations", params={"k": 3}, headers=alice).json()
    request_id = recommended["request_id"]
    first = recommended["items"][0]["movie_id"]
    click = {"event_type": "click", "movie_id": first, "request_id": request_id, "position": 0}
    assert client.post("/events", json=click, headers=alice).status_code == 201
    rating = {"movie_id": first, "rating": 4.5, "request_id": request_id}
    assert client.post("/ratings", json=rating, headers=alice).status_code == 200
    popular = client.get("/recommendations", params={"k": 2}, headers=bob).json()
    assert client.get("/ratings", headers=alice).json()["total"] == 2
    assert client.get("/nowhere").status_code == 404
    app.state.request_log.flush()  # the log is written by a thread of its own
    return {"alice": recommended, "bob": popular}


def as_reader(engine: Engine, sql: str) -> list[dict[str, Any]]:
    """Rows of a query run with the rights of `monitoring_reader`."""
    with engine.connect() as connection:
        connection.execute(text("set role monitoring_reader"))
        return [dict(row) for row in connection.execute(text(sql)).mappings()]


def test_signing_up_creates_the_profile(postgres: Engine) -> None:
    with postgres.connect() as connection:
        rows = connection.execute(text("select id, display_name from public.profiles order by id"))
        assert [tuple(row) for row in rows] == [(ALICE, "Alice"), (BOB, None)]


def test_requests_view_has_every_request_but_health(postgres: Engine, used_app: Any) -> None:
    rows = as_reader(
        postgres,
        "select method, route, status, count(*) as n from monitoring.requests group by 1, 2, 3",
    )
    assert sorted(tuple(row.values()) for row in rows) == [
        ("GET", "(unmatched)", 404, 1),
        ("GET", "/ratings", 200, 1),
        ("GET", "/recommendations", 200, 2),
        ("POST", "/events", 201, 1),
        ("POST", "/ratings", 200, 2),
    ]


def test_recommendations_view_shows_what_became_of_each_list(
    postgres: Engine, used_app: dict[str, Any]
) -> None:
    rows = as_reader(postgres, "select * from monitoring.recommendations order by created_at")
    alice, bob = rows
    assert (alice["strategy"], alice["n_liked"], alice["n_items"]) == ("personalised", 1, 3)
    assert (alice["n_clicks"], alice["n_ratings"], alice["first_click_position"]) == (1, 1, 0)
    assert float(alice["best_rating_after"]) == 4.5
    assert (alice["model"], alice["model_version"], alice["catalog_size"]) == (
        "ease_recency",
        "7",
        6,
    )
    assert 0 <= alice["popularity_percentile"] <= 1
    assert 0 <= alice["model_ms"] <= alice["latency_ms"]

    assert (bob["strategy"], bob["n_liked"], bob["n_items"]) == ("popular", 0, 2)
    assert (bob["n_clicks"], bob["n_ratings"]) == (0, 0)
    assert bob["first_click_position"] is None and bob["best_rating_after"] is None
    assert bob["popularity_percentile"] == 0.1  # the two most popular of six: places 0 and 1


def test_served_movies_view_lists_each_list_in_order(
    postgres: Engine, used_app: dict[str, Any]
) -> None:
    rows = as_reader(
        postgres,
        "select strategy, movie_id, position from monitoring.served_movies "
        "order by created_at, position",
    )
    shown = [item["movie_id"] for item in used_app["alice"]["items"]]
    popular = [item["movie_id"] for item in used_app["bob"]["items"]]
    assert [tuple(row.values()) for row in rows] == [
        *(("personalised", movie_id, place) for place, movie_id in enumerate(shown)),
        *(("popular", movie_id, place) for place, movie_id in enumerate(popular)),
    ]


def test_ratings_and_events_views_carry_no_user_id(postgres: Engine, used_app: Any) -> None:
    ratings = as_reader(postgres, "select * from monitoring.ratings order by updated_at")
    assert [(row["movie_id"], float(row["rating"])) for row in ratings] == [(10, 5.0), (20, 4.5)]
    keys = {row["user_key"] for row in ratings}
    assert len(keys) == 1 and str(ALICE) not in keys and len(keys.pop()) == 32
    assert set(ratings[0]) == {"user_key", "movie_id", "rating", "created_at", "updated_at"}

    events = as_reader(postgres, "select * from monitoring.events order by created_at")
    assert [(e["event_type"], e["movie_id"], e["from_recommendation"]) for e in events] == [
        ("click", 20, True),
        ("rating", 20, True),
    ]
    assert not {"user_id", "request_id"} & set(events[0])


@pytest.mark.parametrize(
    "sql",
    [
        "select * from public.ratings",
        "select * from public.request_log",
        "select * from auth.users",
        "delete from monitoring.requests",
    ],
)
def test_the_monitoring_role_can_do_nothing_but_read_the_views(
    postgres: Engine, used_app: Any, sql: str
) -> None:
    with pytest.raises(ProgrammingError, match="permission denied"):
        as_reader(postgres, sql)


def test_the_drift_report_reads_production_as_the_monitoring_role(
    postgres: Engine, used_app: dict[str, Any]
) -> None:
    """The weekly job's queries, over a real login as `monitoring_reader`."""
    with postgres.begin() as connection:
        connection.execute(text("alter role monitoring_reader with password 'reader'"))
    url = make_url(POSTGRES_URL).set(username="monitoring_reader", password="reader")
    reader = connect(url.render_as_string(hide_password=False))
    current = datetime.now(UTC)

    production = load_production(reader, since=current - timedelta(days=28))
    ratings = production.ratings.sort_values("movie_id")
    assert ratings[["movie_id", "rating", "in_window"]].to_numpy().tolist() == [
        [10, 5.0, True],
        [20, 4.5, True],
    ]
    assert production.served["position"].tolist() == [0, 1, 2, 0, 1]
    assert online_quality(production.recommendations, production.served, 3.5) == {
        "n_lists": 2,
        "n_users": 2,
        "personalised_share": 0.5,
        "click_through_rate": 0.5,
        "hit_rate": 0.5,
        "liked_hit_rate": 0.5,
        "coverage": 4 / 6,  # Alice's three movies and Bob's two share one
        "popularity_percentile": pytest.approx(
            production.recommendations["popularity_percentile"].mean()
        ),
    }

    nothing_yet = load_production(reader, since=current + timedelta(days=1))
    assert nothing_yet.ratings.empty and nothing_yet.served.empty
    assert online_quality(nothing_yet.recommendations, nothing_yet.served, 3.5) == {"n_lists": 0}
    reader.dispose()
