import uuid
from collections.abc import Callable

from conftest import FakeTmdb
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from recsys_api.db import Event, Rating, RecRequest

OTHER_USER = uuid.UUID("00000000-0000-4000-8000-000000000002")
Headers = dict[str, str]


def rate(client: TestClient, auth: Headers, movie_id: int, rating: float) -> None:
    response = client.post("/ratings", json={"movie_id": movie_id, "rating": rating}, headers=auth)
    assert response.status_code == 200, response.text


def ids(movies: list[dict[str, object]]) -> list[object]:
    return [movie["movie_id"] for movie in movies]


# --- public endpoints --------------------------------------------------------------------


def test_health_reports_the_model_being_served(client: TestClient, engine: Engine) -> None:
    engine.dispose()
    assert client.get("/health").json() == {
        "status": "ok",
        "model": "ease_recency",
        "model_version": "7",
        "commit": "abc1234",
        "n_movies": 6,
        "trained_through": "2022-12-31",
    }


def test_search_returns_movies_with_posters(client: TestClient) -> None:
    response = client.get("/movies/search", params={"q": "matrix"})
    assert response.status_code == 200
    assert response.json() == [
        {
            "movie_id": 10,
            "title": "The Matrix",
            "year": 1999,
            "genres": ["Action", "Sci-Fi"],
            "tmdb_id": 603,
            "imdb_id": "0000010",
            "poster_url": "https://image.tmdb.org/t/p/w342/poster603.jpg",
        },
        {
            "movie_id": 20,
            "title": "The Matrix Reloaded",
            "year": 2003,
            "genres": ["Action", "Sci-Fi"],
            "tmdb_id": 604,
            "imdb_id": "0000020",
            "poster_url": "https://image.tmdb.org/t/p/w342/poster604.jpg",
        },
    ]
    assert ids(client.get("/movies/search", params={"q": "t", "limit": 1}).json()) == [10]
    assert client.get("/movies/search").status_code == 422
    assert client.get("/movies/search", params={"q": ""}).status_code == 422


def test_movie_detail_adds_the_tmdb_fields(client: TestClient) -> None:
    movie = client.get("/movies/10").json()
    assert movie["title"] == "The Matrix"
    assert movie["overview"] == "Plot of 603."
    assert movie["backdrop_url"] == "https://image.tmdb.org/t/p/w1280/backdrop603.jpg"
    assert (movie["release_date"], movie["runtime"], movie["vote_average"]) == (
        "1999-03-31",
        136,
        8.2,
    )
    assert client.get("/movies/999").status_code == 404


def test_a_movie_is_served_without_tmdb_data_when_tmdb_is_down(
    client: TestClient, tmdb: FakeTmdb
) -> None:
    tmdb.failing = True
    movie = client.get("/movies/10").json()
    assert (movie["title"], movie["poster_url"], movie["overview"]) == ("The Matrix", None, None)
    untitled = client.get("/movies/60").json()  # MovieLens has no TMDB id for it
    assert (untitled["tmdb_id"], untitled["poster_url"], untitled["year"]) == (None, None, None)


def test_similar_movies(client: TestClient) -> None:
    assert ids(client.get("/movies/10/similar").json()) == [20, 50]
    assert ids(client.get("/movies/10/similar", params={"k": 1}).json()) == [20]
    assert client.get("/movies/60/similar").json() == []
    assert client.get("/movies/999/similar").status_code == 404


def test_onboarding_movies_are_paged(client: TestClient) -> None:
    page = client.get("/onboarding/movies", params={"limit": 4}).json()
    assert page["min_ratings"] == 10
    assert ids(page["movies"]) == [10, 40, 30, 50]
    rest = client.get("/onboarding/movies", params={"limit": 4, "offset": 4}).json()
    assert ids(rest["movies"]) == [20, 60]


def test_the_frontend_origin_may_call_and_others_may_not(client: TestClient) -> None:
    allowed = client.get("/health", headers={"Origin": "https://app.example"})
    assert allowed.headers["access-control-allow-origin"] == "https://app.example"
    refused = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in refused.headers


# --- authentication ----------------------------------------------------------------------


def test_personal_endpoints_need_a_valid_token(client: TestClient) -> None:
    calls = [
        ("GET", "/ratings"),
        ("POST", "/ratings"),
        ("DELETE", "/ratings/10"),
        ("GET", "/recommendations"),
        ("POST", "/events"),
    ]
    for method, path in calls:
        assert client.request(method, path).status_code == 401, path
        bad = client.request(method, path, headers={"Authorization": "Bearer nonsense"})
        assert bad.status_code == 401, path
        assert bad.headers["www-authenticate"] == "Bearer"


# --- ratings -----------------------------------------------------------------------------


def test_rating_a_movie_twice_keeps_one_row_with_the_new_value(
    client: TestClient, auth: Headers, engine: Engine
) -> None:
    rate(client, auth, 10, 4.5)
    rate(client, auth, 10, 2.0)
    with Session(engine) as session:
        rows = session.scalars(select(Rating)).all()
        assert [(row.movie_id, row.rating) for row in rows] == [(10, 2.0)]
        assert rows[0].updated_at >= rows[0].created_at


def test_ratings_are_listed_newest_first_and_only_to_their_owner(
    client: TestClient, auth: Headers, as_user: Callable[[uuid.UUID], Headers]
) -> None:
    rate(client, auth, 10, 5.0)
    rate(client, auth, 30, 3.0)
    rate(client, auth, 40, 3.5)
    page = client.get("/ratings", headers=auth).json()
    assert (page["total"], page["n_liked"]) == (3, 2)  # 3.5 stars and up count as a like
    assert [(item["movie"]["movie_id"], item["rating"]) for item in page["items"]] == [
        (40, 3.5),
        (30, 3.0),
        (10, 5.0),
    ]
    assert page["items"][0]["movie"]["poster_url"].endswith("/poster862.jpg")
    second = client.get("/ratings", params={"limit": 1, "offset": 1}, headers=auth).json()
    assert (second["total"], ids([item["movie"] for item in second["items"]])) == (3, [30])

    assert client.get("/ratings", headers=as_user(OTHER_USER)).json() == {
        "total": 0,
        "n_liked": 0,
        "items": [],
    }


def test_invalid_ratings_are_refused(client: TestClient, auth: Headers) -> None:
    for rating in (0, 0.25, 5.5, 3.7, "five"):
        body = {"movie_id": 10, "rating": rating}
        assert client.post("/ratings", json=body, headers=auth).status_code == 422, rating
    unknown = client.post("/ratings", json={"movie_id": 999, "rating": 4}, headers=auth)
    assert unknown.status_code == 404
    assert client.get("/ratings", headers=auth).json()["total"] == 0


def test_a_rating_can_be_deleted(
    client: TestClient, auth: Headers, as_user: Callable[[uuid.UUID], Headers]
) -> None:
    rate(client, auth, 10, 4.0)
    # Someone else deleting "their" rating of the same movie leaves this one alone.
    assert client.delete("/ratings/10", headers=as_user(OTHER_USER)).status_code == 204
    assert client.get("/ratings", headers=auth).json()["total"] == 1
    assert client.delete("/ratings/10", headers=auth).status_code == 204
    assert client.get("/ratings", headers=auth).json()["total"] == 0
    assert client.delete("/ratings/10", headers=auth).status_code == 204  # already gone


# --- recommendations ---------------------------------------------------------------------


def test_a_user_with_no_likes_gets_the_popular_movies(client: TestClient, auth: Headers) -> None:
    rate(client, auth, 10, 2.0)  # rated, but not liked
    body = client.get("/recommendations", params={"k": 3}, headers=auth).json()
    assert (body["strategy"], body["n_liked"]) == ("popular", 0)
    assert ids(body["items"]) == [40, 30, 20]  # popularity order, without the rated movie
    assert (body["model"], body["model_version"]) == ("ease_recency", "7")


def test_recommendations_follow_the_likes_and_skip_everything_rated(
    client: TestClient, auth: Headers
) -> None:
    rate(client, auth, 10, 5.0)
    body = client.get("/recommendations", params={"k": 2}, headers=auth).json()
    assert (body["strategy"], body["n_liked"]) == ("personalised", 1)
    assert ids(body["items"]) == [20, 50]  # the sequel, then Heat
    assert body["items"][0]["poster_url"].endswith("/poster604.jpg")

    rate(client, auth, 20, 1.0)  # disliked: not part of the history, but never shown again
    body = client.get("/recommendations", params={"k": 2}, headers=auth).json()
    assert (ids(body["items"]), body["n_liked"]) == ([50, 40], 1)

    everything = client.get("/recommendations", params={"k": 100}, headers=auth).json()
    assert sorted(ids(everything["items"])) == [30, 40, 50, 60]
    assert client.get("/recommendations", params={"k": 0}, headers=auth).status_code == 422


def test_every_recommendation_request_is_logged(
    client: TestClient, auth: Headers, user_id: uuid.UUID, engine: Engine
) -> None:
    rate(client, auth, 10, 5.0)
    body = client.get("/recommendations", params={"k": 2}, headers=auth).json()
    with Session(engine) as session:
        logged = session.scalars(select(RecRequest)).one()
        assert str(logged.id) == body["request_id"]
        assert (logged.user_id, logged.model, logged.model_version) == (
            user_id,
            "ease_recency",
            "7",
        )
        assert (logged.strategy, logged.n_liked, logged.movie_ids) == ("personalised", 1, [20, 50])
        assert 0 <= logged.model_ms <= logged.latency_ms


# --- events ------------------------------------------------------------------------------


def test_clicks_and_ratings_are_credited_to_the_list_they_came_from(
    client: TestClient, auth: Headers, user_id: uuid.UUID, engine: Engine
) -> None:
    rate(client, auth, 10, 5.0)  # not from a recommendation: no event
    request_id = client.get("/recommendations", headers=auth).json()["request_id"]
    click = {"event_type": "click", "movie_id": 20, "request_id": request_id, "position": 0}
    assert client.post("/events", json=click, headers=auth).status_code == 201
    rating = {"movie_id": 20, "rating": 4.5, "request_id": request_id}
    assert client.post("/ratings", json=rating, headers=auth).status_code == 200
    plain_click = {"event_type": "click", "movie_id": 40}  # e.g. from search results
    assert client.post("/events", json=plain_click, headers=auth).status_code == 201

    with Session(engine) as session:
        events = session.scalars(select(Event).order_by(Event.id)).all()
        assert [(e.event_type, e.movie_id, e.position, e.rating) for e in events] == [
            ("click", 20, 0, None),
            ("rating", 20, None, 4.5),
            ("click", 40, None, None),
        ]
        assert [str(e.request_id) if e.request_id else None for e in events] == [
            request_id,
            request_id,
            None,
        ]
        assert {e.user_id for e in events} == {user_id}


def test_events_are_validated(
    client: TestClient, auth: Headers, as_user: Callable[[uuid.UUID], Headers], engine: Engine
) -> None:
    request_id = client.get("/recommendations", headers=auth).json()["request_id"]
    click = {"event_type": "click", "movie_id": 20, "request_id": request_id}
    # A list shown to one user can't be credited by another, nor can one that never existed.
    assert client.post("/events", json=click, headers=as_user(OTHER_USER)).status_code == 404
    stranger_rating = {"movie_id": 20, "rating": 4, "request_id": request_id}
    refused = client.post("/ratings", json=stranger_rating, headers=as_user(OTHER_USER))
    assert refused.status_code == 404
    unknown = click | {"request_id": str(uuid.uuid4())}
    assert client.post("/events", json=unknown, headers=auth).status_code == 404
    assert client.post("/events", json=click | {"movie_id": 999}, headers=auth).status_code == 404
    for bad in ({"event_type": "rating"}, {"position": -1}, {"request_id": "abc"}):
        assert client.post("/events", json=click | bad, headers=auth).status_code == 422, bad
    with Session(engine) as session:
        assert session.scalars(select(Event)).all() == []
        assert session.scalars(select(Rating)).all() == []
