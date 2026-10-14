"""The endpoints.

They are plain (non-async) functions, which FastAPI runs in a thread pool: the database
driver and the scoring are blocking calls, and scoring one user takes a few milliseconds.

Movies are public; ratings, recommendations and events need a signed-in user.
"""

import time
import uuid
from collections.abc import Sequence
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from recsys_api.catalog import Movie
from recsys_api.db import Event, MovieMetadata, Rating, RecRequest, as_utc, now, upsert
from recsys_api.deps import Services, ServicesDep, SessionDep, UserDep
from recsys_api.model import recommend
from recsys_api.schemas import (
    EventIn,
    Health,
    MovieDetail,
    MovieOut,
    OnboardingMovies,
    RatedMovie,
    RatingIn,
    RatingOut,
    RatingsPage,
    Recommendations,
)
from recsys_api.tmdb import image_url

router = APIRouter()

_POSTER_SIZE = "w342"
_BACKDROP_SIZE = "w1280"


def _movie_out(movie: Movie, meta: MovieMetadata | None) -> MovieOut:
    return MovieOut(
        movie_id=movie.movie_id,
        title=movie.title,
        year=movie.year,
        genres=list(movie.genres),
        tmdb_id=movie.tmdb_id,
        imdb_id=movie.imdb_id,
        poster_url=image_url(meta.poster_path, _POSTER_SIZE) if meta else None,
    )


def _movies_out(services: Services, session: Session, movies: Sequence[Movie]) -> list[MovieOut]:
    metadata = services.metadata.get_many(session, movies)
    return [_movie_out(movie, metadata.get(movie.movie_id)) for movie in movies]


def _movie_or_404(services: Services, movie_id: int) -> Movie:
    movie = services.bundle.catalog.get(movie_id)
    if movie is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no movie with id {movie_id}")
    return movie


def _check_request(session: Session, user_id: uuid.UUID, request_id: uuid.UUID) -> None:
    """A click or rating can only be credited to a list that was shown to the same user."""
    request = session.get(RecRequest, request_id)
    if request is None or request.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no recommendation request {request_id}")


@router.get("/health")
def health(services: ServicesDep) -> Health:
    """Cheap on purpose (no database): the frontend polls it while the free host wakes up."""
    bundle = services.bundle
    return Health(
        model=bundle.name,
        model_version=services.settings.model_version,
        commit=services.settings.git_commit,
        n_movies=len(bundle.catalog),
        trained_through=bundle.metadata["trained_on"]["last_interaction"],
    )


@router.get("/movies/search")
def search_movies(
    services: ServicesDep,
    session: SessionDep,
    q: Annotated[str, Query(min_length=1, max_length=100)],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> list[MovieOut]:
    return _movies_out(services, session, services.bundle.catalog.search(q, limit))


@router.get("/movies/{movie_id}")
def get_movie(movie_id: int, services: ServicesDep, session: SessionDep) -> MovieDetail:
    movie = _movie_or_404(services, movie_id)
    meta = services.metadata.get_many(session, [movie]).get(movie_id)
    return MovieDetail(
        **_movie_out(movie, meta).model_dump(),
        overview=meta.overview if meta else None,
        backdrop_url=image_url(meta.backdrop_path, _BACKDROP_SIZE) if meta else None,
        release_date=meta.release_date if meta else None,
        runtime=meta.runtime if meta else None,
        vote_average=meta.vote_average if meta else None,
    )


@router.get("/movies/{movie_id}/similar")
def similar_movies(
    movie_id: int,
    services: ServicesDep,
    session: SessionDep,
    k: Annotated[int, Query(ge=1, le=50)] = 12,
) -> list[MovieOut]:
    """Movies liked by the people who liked this one, from the model's item-to-item weights."""
    movie = _movie_or_404(services, movie_id)
    model = services.bundle.similar_model
    if model is None:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED, f"'{services.bundle.name}' has no item-to-item weights"
        )
    catalog = services.bundle.catalog
    similar = [catalog.movies[i] for i in model.similar(movie.item_idx, k)]
    return _movies_out(services, session, similar)


@router.get("/onboarding/movies")
def onboarding_movies(
    services: ServicesDep,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 40,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> OnboardingMovies:
    """Well-known movies across genres for a new user to rate; `offset` pages through more."""
    catalog = services.bundle.catalog
    movies = [catalog.movies[i] for i in catalog.onboarding[offset : offset + limit]]
    return OnboardingMovies(
        min_ratings=services.settings.onboarding_min_ratings,
        movies=_movies_out(services, session, movies),
    )


@router.get("/ratings")
def list_ratings(
    services: ServicesDep,
    session: SessionDep,
    user_id: UserDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RatingsPage:
    """The caller's ratings, most recent first."""
    mine = Rating.user_id == user_id
    total = session.scalar(select(func.count()).select_from(Rating).where(mine)) or 0
    liked = Rating.rating >= services.bundle.like_threshold
    n_liked = session.scalar(select(func.count()).select_from(Rating).where(mine, liked)) or 0
    rows = session.scalars(
        select(Rating)
        .where(mine)
        .order_by(Rating.updated_at.desc(), Rating.movie_id)
        .limit(limit)
        .offset(offset)
    ).all()
    catalog = services.bundle.catalog
    # A movie can leave the catalog when the model is retrained; its rating is kept but hidden.
    known = [(row, movie) for row in rows if (movie := catalog.get(row.movie_id)) is not None]
    movies = _movies_out(services, session, [movie for _, movie in known])
    items = [
        RatedMovie(movie=out, rating=row.rating, updated_at=as_utc(row.updated_at))
        for (row, _), out in zip(known, movies, strict=True)
    ]
    return RatingsPage(total=total, n_liked=n_liked, items=items)


@router.post("/ratings")
def rate_movie(
    body: RatingIn, services: ServicesDep, session: SessionDep, user_id: UserDep
) -> RatingOut:
    """Rate a movie, or change an earlier rating of it."""
    _movie_or_404(services, body.movie_id)
    if body.request_id is not None:
        _check_request(session, user_id, body.request_id)
    timestamp = now()
    upsert(
        session,
        Rating,
        {
            "user_id": user_id,
            "movie_id": body.movie_id,
            "rating": body.rating,
            "created_at": timestamp,
            "updated_at": timestamp,
        },
        update=["rating", "updated_at"],
    )
    if body.request_id is not None:
        session.add(
            Event(
                user_id=user_id,
                request_id=body.request_id,
                event_type="rating",
                movie_id=body.movie_id,
                rating=body.rating,
            )
        )
    session.commit()
    return RatingOut(movie_id=body.movie_id, rating=body.rating, updated_at=timestamp)


@router.delete("/ratings/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_rating(movie_id: int, session: SessionDep, user_id: UserDep) -> Response:
    session.execute(delete(Rating).where(Rating.user_id == user_id, Rating.movie_id == movie_id))
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/recommendations")
def recommendations(
    services: ServicesDep,
    session: SessionDep,
    user_id: UserDep,
    k: Annotated[int, Query(ge=1, le=100)] = 20,
) -> Recommendations:
    """Movies for the caller, computed from their ratings at the time of the call.

    The model gets the liked movies, oldest first, exactly as in offline evaluation; nothing
    is retrained or stored per user, so a rating made a second ago already counts. Every
    rated movie is left out of the result, liked or not.
    """
    started = time.perf_counter()
    bundle = services.bundle
    catalog = bundle.catalog
    rows = session.execute(
        select(Rating.movie_id, Rating.rating)
        .where(Rating.user_id == user_id)
        .order_by(Rating.updated_at, Rating.movie_id)
    ).all()
    rated = [(movie, rating) for movie_id, rating in rows if (movie := catalog.get(movie_id))]
    seen = np.array([movie.item_idx for movie, _ in rated], dtype=np.int32)
    history = np.array(
        [movie.item_idx for movie, rating in rated if rating >= bundle.like_threshold],
        dtype=np.int32,
    )

    model_started = time.perf_counter()
    strategy: Literal["personalised", "popular"]
    if len(history):
        strategy = "personalised"
        top = recommend(bundle.model, history, seen, k)
    else:
        strategy = "popular"
        top = catalog.popular[~np.isin(catalog.popular, seen)][:k]
    model_ms = (time.perf_counter() - model_started) * 1000

    items = _movies_out(services, session, [catalog.movies[i] for i in top])
    request = RecRequest(
        id=uuid.uuid4(),
        user_id=user_id,
        model=bundle.name,
        model_version=services.settings.model_version,
        strategy=strategy,
        n_liked=len(history),
        movie_ids=[item.movie_id for item in items],
        model_ms=model_ms,
        latency_ms=(time.perf_counter() - started) * 1000,
        popularity_percentile=catalog.popularity_percentile(top),
        catalog_size=len(catalog),
    )
    session.add(request)
    session.commit()
    return Recommendations(
        request_id=request.id,
        model=bundle.name,
        model_version=services.settings.model_version,
        strategy=strategy,
        n_liked=len(history),
        items=items,
    )


@router.post("/events", status_code=status.HTTP_201_CREATED)
def log_event(
    body: EventIn, services: ServicesDep, session: SessionDep, user_id: UserDep
) -> Response:
    """Record a click on a movie, with the recommendation list it came from if any."""
    _movie_or_404(services, body.movie_id)
    if body.request_id is not None:
        _check_request(session, user_id, body.request_id)
    session.add(
        Event(
            user_id=user_id,
            request_id=body.request_id,
            event_type=body.event_type,
            movie_id=body.movie_id,
            position=body.position,
        )
    )
    session.commit()
    return Response(status_code=status.HTTP_201_CREATED)
