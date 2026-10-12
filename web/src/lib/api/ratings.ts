import type { Movie, RatedMovie } from "./client";

/**
 * The ratings list as it will be once the server has accepted a change, so the stars move
 * the moment they are clicked. `rating: null` removes the movie's rating.
 */
export function applyRating(
  ratings: RatedMovie[],
  movie: Movie,
  rating: number | null,
  now: Date = new Date(),
): RatedMovie[] {
  const others = ratings.filter((item) => item.movie.movie_id !== movie.movie_id);
  if (rating === null) return others;
  // Newest first, as the API returns them.
  return [{ movie, rating, updated_at: now.toISOString() }, ...others];
}

/** What counts as "you liked this" in the interface: four stars and up. */
export const LIKED_FROM = 4;

/** The most recently rated movie the user liked, to hang a "because you liked" row on. */
export function latestLiked(ratings: RatedMovie[]): Movie | undefined {
  return ratings.find((item) => item.rating >= LIKED_FROM)?.movie;
}
