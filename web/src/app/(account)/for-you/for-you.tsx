"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import {
  GRID,
  MovieCard,
  MovieGridSkeleton,
  MovieRow,
  MovieRowSkeleton,
  movieHref,
} from "@/components/movie-card";
import { Button } from "@/components/ui/button";
import { logClick, type RatedMovie } from "@/lib/api/client";
import { useMinRatings, useRatings, useRecommendations, useSimilarMovies } from "@/lib/api/hooks";
import { latestLiked } from "@/lib/api/ratings";

function BecauseYouLiked({ ratings }: { ratings: RatedMovie[] }) {
  const liked = latestLiked(ratings);
  const similar = useSimilarMovies(liked?.movie_id, 18);
  if (!liked || similar.isError) return null;

  const rated = new Set(ratings.map((item) => item.movie.movie_id));
  const unseen = similar.data?.filter((movie) => !rated.has(movie.movie_id));
  if (unseen?.length === 0) return null;

  return (
    <section className="mt-16">
      <h2 className="text-3xl font-bold">Because you liked {liked.title}</h2>
      <div className="mt-5">
        {unseen ? (
          <MovieRow>
            {unseen.map((movie) => (
              <MovieCard
                key={movie.movie_id}
                movie={movie}
                // Not from a recommendation list, so the click is logged without one.
                onOpen={() => void logClick(movie.movie_id).catch(() => {})}
              />
            ))}
          </MovieRow>
        ) : (
          <MovieRowSkeleton />
        )}
      </div>
    </section>
  );
}

export function ForYou() {
  const router = useRouter();
  const ratings = useRatings();
  const minRatings = useMinRatings();
  const ready = ratings.data !== undefined && minRatings.data !== undefined;
  const needsOnboarding = ready && ratings.data.length < minRatings.data;
  const recommendations = useRecommendations(20, ready && !needsOnboarding);

  useEffect(() => {
    if (needsOnboarding) router.replace("/onboarding");
  }, [needsOnboarding, router]);

  const failed = ratings.isError || minRatings.isError || recommendations.isError;
  const list = recommendations.data;

  return (
    <>
      <h1 className="text-5xl font-bold sm:text-6xl">Recommended for you</h1>
      {failed ? (
        <div className="mt-6">
          <p className="text-muted-foreground">Your recommendations didn&apos;t load.</p>
          <Button
            variant="outline"
            size="lg"
            className="mt-4"
            onClick={() => {
              void ratings.refetch();
              void minRatings.refetch();
              void recommendations.refetch();
            }}
          >
            Try again
          </Button>
        </div>
      ) : !list ? (
        <div className="mt-10">
          <MovieGridSkeleton count={20} />
        </div>
      ) : (
        <>
          <p className="mt-3 max-w-2xl text-muted-foreground">
            {list.strategy === "personalised" ? (
              <>
                Built from the {list.n_liked} {list.n_liked === 1 ? "movie" : "movies"} you rated
                highly, best match first. Each new rating changes the list the next time you open
                it.
              </>
            ) : (
              <>
                None of your ratings is high enough to count as a like yet, so this is what is
                popular right now.{" "}
                <Link href="/onboarding" className="text-foreground underline">
                  Rate a movie you loved
                </Link>{" "}
                to make the list yours.
              </>
            )}
          </p>
          <ol className={`${GRID} mt-10`}>
            {list.items.map((movie, position) => (
              <li key={movie.movie_id}>
                <MovieCard
                  movie={movie}
                  badge={position + 1}
                  href={movieHref(movie.movie_id, list.request_id)}
                  onOpen={() =>
                    void logClick(movie.movie_id, list.request_id, position).catch(() => {})
                  }
                />
              </li>
            ))}
          </ol>
          {ratings.data && <BecauseYouLiked ratings={ratings.data} />}
        </>
      )}
    </>
  );
}
