"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { use, useEffect, useState } from "react";
import { BrowserOnly } from "@/components/browser-only";
import { useAuth } from "@/components/auth-provider";
import { MovieCard, MovieRow, MovieRowSkeleton, Poster } from "@/components/movie-card";
import { StarRating } from "@/components/star-rating";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, type MovieDetail } from "@/lib/api/client";
import { useMovie, useMyRating, useRateMovie, useSimilarMovies } from "@/lib/api/hooks";

const UUID = /^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i;

export function MovieDetailSkeleton() {
  return (
    <div className="container-page grid gap-8 py-10 sm:grid-cols-[14rem_1fr]" aria-hidden>
      <Skeleton className="aspect-2/3 w-40 sm:w-full" />
      <div>
        <Skeleton className="h-14 w-2/3" />
        <Skeleton className="mt-4 h-5 w-1/3" />
        <Skeleton className="mt-8 h-28 max-w-prose" />
      </div>
    </div>
  );
}

function NotFound() {
  return (
    <div className="container-page py-16">
      <h1 className="text-5xl font-bold">This movie isn&apos;t in the catalog</h1>
      <p className="mt-4 text-muted-foreground">
        The address may be mistyped, or the movie was dropped when the model was retrained.
      </p>
      <Button size="lg" className="mt-6" asChild>
        <Link href="/search">Search for a movie</Link>
      </Button>
    </div>
  );
}

function runtimeText(minutes: number): string {
  const hours = Math.floor(minutes / 60);
  return hours ? `${hours} h ${minutes % 60} min` : `${minutes} min`;
}

function YourRating({ movie, requestId }: { movie: MovieDetail; requestId?: string }) {
  const { status } = useAuth();
  const rating = useMyRating(movie.movie_id);
  const rate = useRateMovie();

  if (status === "loading") return <Skeleton className="h-10 w-52" />;
  if (status === "signedOut") {
    return (
      <p className="text-muted-foreground">
        <Link
          href={`/login?next=${encodeURIComponent(`/movies/${movie.movie_id}`)}`}
          className="text-foreground underline"
        >
          Log in
        </Link>{" "}
        to rate this movie and get recommendations from it.
      </p>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <StarRating
        label={rating === undefined ? `Rate ${movie.title}` : `Your rating of ${movie.title}`}
        size="lg"
        value={rating}
        onChange={(next) => rate.mutate({ movie, rating: next, requestId })}
      />
      {rating === undefined ? (
        <span className="text-sm text-muted-foreground">Seen it? Rate it.</span>
      ) : (
        <Button variant="ghost" onClick={() => rate.mutate({ movie, rating: null })}>
          Remove rating
        </Button>
      )}
    </div>
  );
}

function Similar({ movieId }: { movieId: number }) {
  const similar = useSimilarMovies(movieId, 14);
  if (similar.isError || similar.data?.length === 0) return null;
  return (
    <section className="container-page mt-16">
      <h2 className="text-3xl font-bold">People who liked this also liked</h2>
      <div className="mt-5">
        {similar.data ? (
          <MovieRow>
            {similar.data.map((movie) => (
              <MovieCard key={movie.movie_id} movie={movie} />
            ))}
          </MovieRow>
        ) : (
          <MovieRowSkeleton />
        )}
      </div>
    </section>
  );
}

function MovieView({ movieId }: { movieId: number }) {
  const rec = useSearchParams().get("rec");
  // The list this movie was opened from, read once and then taken out of the address: a
  // rating may only be credited to a list by the user it was shown to, so the id must not
  // travel when the link is shared.
  const [requestId] = useState(() => (rec && UUID.test(rec) ? rec : undefined));
  useEffect(() => {
    if (rec !== null) window.history.replaceState(null, "", window.location.pathname);
  }, [rec]);

  const { data: movie, error, refetch } = useMovie(movieId);

  if (error instanceof ApiError && error.status === 404) return <NotFound />;
  if (error) {
    return (
      <div className="container-page py-16">
        <p className="text-muted-foreground">This movie didn&apos;t load.</p>
        <Button variant="outline" size="lg" className="mt-4" onClick={() => refetch()}>
          Try again
        </Button>
      </div>
    );
  }
  if (!movie) return <MovieDetailSkeleton />;

  return (
    <>
      <title>{movie.title}</title>
      {movie.backdrop_url && (
        <div className="relative -mb-40 h-72 overflow-hidden sm:h-96" aria-hidden>
          <img src={movie.backdrop_url} alt="" className="size-full object-cover opacity-45" />
          <div className="absolute inset-0 bg-linear-to-t from-background to-transparent" />
        </div>
      )}
      <article className="relative container-page grid gap-x-10 gap-y-6 py-10 sm:grid-cols-[14rem_1fr]">
        <Poster movie={movie} className="w-40 shadow-2xl sm:w-full" />
        <div className="min-w-0">
          <h1 className="text-5xl leading-[0.95] font-bold sm:text-7xl">{movie.title}</h1>
          <ul className="mt-4 flex flex-wrap gap-x-5 gap-y-1 text-muted-foreground">
            {movie.year !== null && <li>{movie.year}</li>}
            {movie.runtime !== null && <li>{runtimeText(movie.runtime)}</li>}
            {movie.genres.length > 0 && <li>{movie.genres.join(", ")}</li>}
            {movie.vote_average !== null && movie.vote_average > 0 && (
              <li>{movie.vote_average.toFixed(1)} out of 10 on TMDB</li>
            )}
          </ul>
          <div className="mt-7">
            <YourRating movie={movie} requestId={requestId} />
          </div>
          {movie.overview && (
            <p className="mt-7 max-w-prose text-lg leading-relaxed">{movie.overview}</p>
          )}
          <p className="mt-6 flex gap-5 text-sm text-muted-foreground">
            {movie.tmdb_id !== null && (
              <a href={`https://www.themoviedb.org/movie/${movie.tmdb_id}`} className="underline">
                On TMDB
              </a>
            )}
            {movie.imdb_id !== null && (
              <a href={`https://www.imdb.com/title/tt${movie.imdb_id}/`} className="underline">
                On IMDb
              </a>
            )}
          </p>
        </div>
      </article>
      <Similar movieId={movieId} />
    </>
  );
}

export function MoviePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  if (!/^\d+$/.test(id)) return <NotFound />;
  // Keyed so that nothing from one movie's page (its list id, a hovered star) carries over.
  return (
    <BrowserOnly fallback={<MovieDetailSkeleton />}>
      <MovieView key={id} movieId={Number(id)} />
    </BrowserOnly>
  );
}
