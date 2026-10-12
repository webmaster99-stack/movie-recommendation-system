"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { GRID, MovieCaption, MovieGridSkeleton, Poster } from "@/components/movie-card";
import { StarRating } from "@/components/star-rating";
import { Button } from "@/components/ui/button";
import { useOnboardingMovies, useRateMovie, useRatings } from "@/lib/api/hooks";
import { cn } from "@/lib/utils";

/**
 * Progress towards the first recommendations, as an admission ticket that gets one hole
 * punched per rating. It stays at the bottom of the window while the posters scroll.
 */
function Ticket({ rated, needed }: { rated: number; needed: number }) {
  const router = useRouter();
  const remaining = Math.max(needed - rated, 0);

  return (
    <div className="sticky bottom-4 z-10 mt-12 drop-shadow-[0_8px_24px_rgb(0_0_0/45%)]">
      <div className="ticket mx-auto flex max-w-xl items-stretch rounded-md bg-primary px-7 text-primary-foreground">
        <div className="min-w-0 flex-1 py-3">
          <p role="status" className="font-heading text-xl leading-none font-bold sm:text-2xl">
            {remaining === 0 ? "Your list is ready" : `${rated} of ${needed} rated`}
          </p>
          <div className="mt-2.5 flex gap-1 sm:gap-1.5" aria-hidden>
            {Array.from({ length: needed }, (_, i) => (
              <span
                key={i}
                className={cn(
                  "size-2.5 shrink-0 rounded-full sm:size-3.5",
                  i < rated ? "bg-background" : "border-2 border-primary-foreground/35",
                )}
              />
            ))}
          </div>
        </div>
        <div className="flex items-center border-l-2 border-dashed border-primary-foreground/40 pl-3 sm:pl-5">
          <Button
            disabled={remaining > 0}
            onClick={() => router.push("/for-you")}
            className="h-10 bg-background px-3 sm:px-4 text-foreground hover:bg-background/85 disabled:border-primary-foreground/40 disabled:bg-transparent disabled:text-primary-foreground disabled:opacity-100"
          >
            {remaining === 0 ? "See my recommendations" : `Rate ${remaining} more`}
          </Button>
        </div>
      </div>
    </div>
  );
}

export function Onboarding() {
  const movies = useOnboardingMovies();
  const ratings = useRatings();
  const rate = useRateMovie();

  const pages = movies.data?.pages;
  const needed = pages?.[0].min_ratings ?? 10;
  const mine = new Map(ratings.data?.map((item) => [item.movie.movie_id, item.rating]));

  return (
    <>
      <h1 className="text-5xl font-bold sm:text-6xl">Rate {needed} movies you&apos;ve seen</h1>
      <p className="mt-3 max-w-2xl text-muted-foreground">
        Click a star&apos;s left half for half a star. Skip what you haven&apos;t watched: there are
        more at the bottom, and you can{" "}
        <Link href="/search" className="text-foreground underline">
          search for a title
        </Link>
        .
      </p>

      <div className="mt-10">
        {movies.isError ? (
          <>
            <p className="text-muted-foreground">The movies didn&apos;t load.</p>
            <Button variant="outline" size="lg" className="mt-4" onClick={() => movies.refetch()}>
              Try again
            </Button>
          </>
        ) : !pages ? (
          <MovieGridSkeleton count={20} />
        ) : (
          <>
            <ul className={GRID}>
              {pages
                .flatMap((page) => page.movies)
                .map((movie) => (
                  <li key={movie.movie_id}>
                    <Poster movie={movie} />
                    <MovieCaption movie={movie} />
                    <StarRating
                      label={`Rate ${movie.title}`}
                      size="sm"
                      value={mine.get(movie.movie_id)}
                      onChange={(rating) => rate.mutate({ movie, rating })}
                      className="mt-1.5"
                    />
                  </li>
                ))}
            </ul>
            {movies.hasNextPage && (
              <Button
                variant="outline"
                size="lg"
                className="mt-10"
                disabled={movies.isFetchingNextPage}
                onClick={() => movies.fetchNextPage()}
              >
                {movies.isFetchingNextPage ? "Loading more…" : "Show more movies"}
              </Button>
            )}
          </>
        )}
      </div>

      {ratings.data && <Ticket rated={ratings.data.length} needed={needed} />}
    </>
  );
}
