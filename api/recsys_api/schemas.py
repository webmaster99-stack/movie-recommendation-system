"""Request and response bodies. These are the contract with the frontend."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"] = "ok"
    model: str
    model_version: str
    commit: str  # the git commit the running image was built from
    n_movies: int
    trained_through: str  # date of the last interaction the model has seen


class MovieOut(BaseModel):
    movie_id: int  # MovieLens id: the id used in every other endpoint
    title: str
    year: int | None
    genres: list[str]
    tmdb_id: int | None
    imdb_id: str | None
    poster_url: str | None  # null until TMDB has been asked, or when it has no poster


class MovieDetail(MovieOut):
    overview: str | None
    backdrop_url: str | None
    release_date: str | None
    runtime: int | None  # minutes
    vote_average: float | None  # TMDB's, out of 10


class OnboardingMovies(BaseModel):
    min_ratings: int  # how many ratings the app asks for before showing recommendations
    movies: list[MovieOut]


class RatingIn(BaseModel):
    movie_id: int
    rating: float = Field(ge=0.5, le=5.0)
    # Set when the movie was rated from a recommendation list, to credit that list.
    request_id: uuid.UUID | None = None

    @field_validator("rating")
    @classmethod
    def half_stars(cls, value: float) -> float:
        if value * 2 != int(value * 2):
            raise ValueError("ratings go in steps of half a star")
        return value


class RatingOut(BaseModel):
    movie_id: int
    rating: float
    updated_at: datetime


class RatedMovie(BaseModel):
    movie: MovieOut
    rating: float
    updated_at: datetime


class RatingsPage(BaseModel):
    total: int  # all of the user's ratings, not just this page
    n_liked: int  # those high enough to count as a like for the model
    items: list[RatedMovie]


class Recommendations(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: uuid.UUID  # send it back with clicks and ratings that come from this list
    model: str
    model_version: str
    # `popular`: the user has no liked movies yet, so this is the same list for everyone.
    strategy: Literal["personalised", "popular"]
    n_liked: int
    items: list[MovieOut]


class EventIn(BaseModel):
    event_type: Literal["click"]
    movie_id: int
    request_id: uuid.UUID | None = None
    position: int | None = Field(default=None, ge=0)  # 0-based place in the list shown
