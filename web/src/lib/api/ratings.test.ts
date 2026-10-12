import { describe, expect, it } from "vitest";
import type { Movie, RatedMovie } from "./client";
import { applyRating, latestLiked } from "./ratings";

function movie(movieId: number): Movie {
  return {
    movie_id: movieId,
    title: `Movie ${movieId}`,
    year: 2000,
    genres: [],
    tmdb_id: null,
    imdb_id: null,
    poster_url: null,
  };
}

const rated = (movieId: number, rating: number): RatedMovie => ({
  movie: movie(movieId),
  rating,
  updated_at: "2026-01-01T00:00:00Z",
});

const ids = (ratings: RatedMovie[]) => ratings.map((item) => item.movie.movie_id);

describe("applyRating", () => {
  const before = [rated(1, 3), rated(2, 5)];

  it("puts a new rating first", () => {
    const after = applyRating(before, movie(3), 4.5);

    expect(ids(after)).toEqual([3, 1, 2]);
    expect(after[0].rating).toBe(4.5);
  });

  it("moves a changed rating to the front instead of listing the movie twice", () => {
    const after = applyRating(before, movie(2), 1);

    expect(ids(after)).toEqual([2, 1]);
    expect(after[0].rating).toBe(1);
  });

  it("removes a rating", () => {
    expect(ids(applyRating(before, movie(1), null))).toEqual([2]);
  });

  it("leaves the list it was given untouched", () => {
    applyRating(before, movie(1), null);

    expect(ids(before)).toEqual([1, 2]);
  });
});

describe("latestLiked", () => {
  it("is the newest movie rated four stars or more", () => {
    expect(latestLiked([rated(1, 3.5), rated(2, 4), rated(3, 5)])?.movie_id).toBe(2);
  });

  it("is nothing when no rating is that high", () => {
    expect(latestLiked([rated(1, 3.5), rated(2, 1)])).toBeUndefined();
  });
});
