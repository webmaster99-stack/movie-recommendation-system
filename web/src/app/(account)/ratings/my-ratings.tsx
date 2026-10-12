"use client";

import { XIcon } from "lucide-react";
import Link from "next/link";
import { movieHref, Poster } from "@/components/movie-card";
import { StarRating } from "@/components/star-rating";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useRateMovie, useRatings } from "@/lib/api/hooks";

export function MyRatings() {
  const ratings = useRatings();
  const rate = useRateMovie();
  const items = ratings.data;

  return (
    <>
      <h1 className="text-5xl font-bold sm:text-6xl">My ratings</h1>
      {ratings.isError ? (
        <div className="mt-6">
          <p className="text-muted-foreground">Your ratings didn&apos;t load.</p>
          <Button variant="outline" size="lg" className="mt-4" onClick={() => ratings.refetch()}>
            Try again
          </Button>
        </div>
      ) : !items ? (
        <div className="mt-10 flex max-w-3xl flex-col gap-4" aria-hidden>
          {Array.from({ length: 6 }, (_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      ) : items.length === 0 ? (
        <div className="mt-6">
          <p className="text-muted-foreground">
            You haven&apos;t rated anything yet. Ratings are all the recommender knows about you.
          </p>
          <Button size="lg" className="mt-4" asChild>
            <Link href="/onboarding">Rate some movies</Link>
          </Button>
        </div>
      ) : (
        <>
          <p className="mt-3 text-muted-foreground">
            {items.length} {items.length === 1 ? "movie" : "movies"}, most recently rated first.
            Change the stars to change a rating.
          </p>
          <ul className="mt-8 max-w-3xl divide-y border-y">
            {items.map(({ movie, rating }) => (
              <li key={movie.movie_id} className="flex items-center gap-4 py-3">
                <Link href={movieHref(movie.movie_id)} className="w-14 shrink-0" tabIndex={-1}>
                  <Poster movie={{ ...movie, title: "" }} />
                </Link>
                <div className="flex min-w-0 flex-1 flex-col gap-x-6 gap-y-1 sm:flex-row sm:items-center">
                  <Link href={movieHref(movie.movie_id)} className="min-w-0 flex-1 hover:underline">
                    <span className="font-medium">{movie.title}</span>
                    {movie.year !== null && (
                      <span className="text-muted-foreground"> {movie.year}</span>
                    )}
                  </Link>
                  <StarRating
                    label={`Your rating of ${movie.title}`}
                    size="sm"
                    value={rating}
                    onChange={(next) => rate.mutate({ movie, rating: next })}
                  />
                </div>
                <Button
                  variant="ghost"
                  size="icon-lg"
                  aria-label={`Remove your rating of ${movie.title}`}
                  onClick={() => rate.mutate({ movie, rating: null })}
                >
                  <XIcon />
                </Button>
              </li>
            ))}
          </ul>
        </>
      )}
    </>
  );
}
