"""The request log: one row per answered request, written without getting in its way."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from recsys_api.db import RequestLog
from recsys_api.request_log import UNMATCHED, RequestLogMiddleware, RequestLogWriter

Headers = dict[str, str]


def logged(engine: Engine, app: Any = None) -> list[tuple[str, str, int]]:
    """The rows of the log, after waiting for `app`'s writer to have written what it holds."""
    if app is not None:
        app.state.request_log.flush()
    with Session(engine) as session:
        rows = session.scalars(select(RequestLog).order_by(RequestLog.id)).all()
        assert all(row.latency_ms >= 0 for row in rows)
        return [(row.method, row.route, row.status) for row in rows]


def test_requests_are_logged_by_route_template_with_their_status(
    client: TestClient, engine: Engine, auth: Headers
) -> None:
    client.get("/movies/10")
    client.get("/movies/999")
    client.get("/ratings")  # no token
    client.post("/ratings", json={"movie_id": 10, "rating": 9}, headers=auth)
    client.get("/nowhere")
    assert logged(engine, client.app) == [
        ("GET", "/movies/{movie_id}", 200),
        ("GET", "/movies/{movie_id}", 404),
        ("GET", "/ratings", 401),
        ("POST", "/ratings", 422),
        ("GET", UNMATCHED, 404),
    ]


def test_health_checks_and_preflights_are_not_logged(client: TestClient, engine: Engine) -> None:
    client.get("/health")
    preflight = {"Origin": "https://app.example", "Access-Control-Request-Method": "POST"}
    assert client.options("/ratings", headers=preflight).status_code == 200
    assert logged(engine, client.app) == []


def failing_app(sessions: sessionmaker[Session]) -> FastAPI:
    app = FastAPI()
    app.state.request_log = RequestLogWriter(sessions)
    app.add_middleware(RequestLogMiddleware, writer=app.state.request_log)

    @app.get("/ok")
    def ok() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/broken/{item}")
    def broken(item: int) -> None:
        raise RuntimeError("a bug")

    return app


def test_an_endpoint_that_raises_is_logged_as_a_500(engine: Engine) -> None:
    client = TestClient(failing_app(sessionmaker(engine)), raise_server_exceptions=False)
    assert client.get("/broken/3").status_code == 500
    assert logged(engine, client.app) == [("GET", "/broken/{item}", 500)]


def test_a_database_that_is_down_does_not_fail_the_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unreachable = create_engine("sqlite:////no/such/directory/api.db")
    app = failing_app(sessionmaker(unreachable))
    with caplog.at_level(logging.WARNING, logger="recsys_api.request_log"):
        assert TestClient(app).get("/ok").json() == {"ok": True}
        app.state.request_log.flush()
    assert "request log not written" in caplog.text
    assert "no/such/directory" not in caplog.text  # the address of the database stays out


def test_rows_past_the_retention_period_are_deleted(engine: Engine) -> None:
    current = datetime(2026, 6, 1, tzinfo=UTC)
    log = RequestLogWriter(sessionmaker(engine), keep_days=90, clock=lambda: current)
    log.add("GET", "/a", 200, 1.0)
    log.flush()
    current += timedelta(days=89)
    log.add("GET", "/b", 200, 1.0)
    log.flush()
    assert [route for _, route, _ in logged(engine)] == ["/a", "/b"]
    current += timedelta(days=2)  # /a is now 91 days old, /b two
    log.add("GET", "/c", 200, 1.0)
    log.flush()
    assert [route for _, route, _ in logged(engine)] == ["/b", "/c"]


def test_closing_the_writer_writes_what_is_still_queued(engine: Engine) -> None:
    log = RequestLogWriter(sessionmaker(engine))
    for route in ("/a", "/b", "/c"):
        log.add("GET", route, 200, 1.0)
    log.close()
    assert [route for _, route, _ in logged(engine)] == ["/a", "/b", "/c"]


def test_shutting_the_api_down_closes_the_writer(client: TestClient, engine: Engine) -> None:
    with client:  # runs start-up and shutdown
        client.get("/movies/10")
    assert logged(engine) == [("GET", "/movies/{movie_id}", 200)]
