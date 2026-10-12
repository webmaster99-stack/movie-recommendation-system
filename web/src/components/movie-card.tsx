import Link from "next/link";
import type { ReactNode } from "react";
import type { Movie } from "@/lib/api/client";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";

export function movieHref(movieId: number, requestId?: string): string {
  // `rec` carries the recommendation list a movie was opened from to its page, so that a
  // rating given there is credited to that list.
  return requestId ? `/movies/${movieId}?rec=${requestId}` : `/movies/${movieId}`;
}

/**
 * The poster straight from TMDB's image CDN, as their terms ask (no copies on our side),
 * or the title set on a plain tile when TMDB has none.
 */
export function Poster({ movie, className }: { movie: Movie; className?: string }) {
  const frame = cn("aspect-2/3 w-full overflow-hidden rounded-md bg-card", className);
  if (movie.poster_url) {
    return (
      <img
        src={movie.poster_url}
        alt=""
        loading="lazy"
        decoding="async"
        className={cn(frame, "object-cover")}
      />
    );
  }
  return (
    <div className={cn(frame, "flex items-end p-3")} aria-hidden>
      <span className="font-heading text-xl leading-tight font-semibold text-muted-foreground">
        {movie.title}
      </span>
    </div>
  );
}

export function MovieCaption({ movie }: { movie: Movie }) {
  return (
    <p className="mt-2 text-sm leading-snug">
      <span className="line-clamp-2 font-medium">{movie.title}</span>
      {movie.year !== null && <span className="text-muted-foreground">{movie.year}</span>}
    </p>
  );
}

type CardProps = {
  movie: Movie;
  href?: string;
  onOpen?: () => void;
  /** Shown over the poster's corner: a rank, or the user's rating. */
  badge?: ReactNode;
  className?: string;
};

export function MovieCard({ movie, href, onOpen, badge, className }: CardProps) {
  return (
    <Link
      href={href ?? movieHref(movie.movie_id)}
      onClick={onOpen}
      className={cn("group relative block rounded-md", className)}
    >
      <Poster
        movie={movie}
        className="transition-[filter] duration-150 group-hover:brightness-110"
      />
      {badge !== undefined && (
        <span className="absolute top-0 left-0 rounded-tl-md rounded-br-md bg-background/85 px-2 py-0.5 font-heading text-lg leading-tight font-semibold">
          {badge}
        </span>
      )}
      <MovieCaption movie={movie} />
    </Link>
  );
}

export const GRID = "grid grid-cols-2 gap-x-4 gap-y-7 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5";

export function MovieGridSkeleton({ count = 10 }: { count?: number }) {
  return (
    <div className={GRID} aria-hidden>
      {Array.from({ length: count }, (_, i) => (
        <div key={i}>
          <Skeleton className="aspect-2/3 w-full" />
          <Skeleton className="mt-2 h-4 w-3/4" />
        </div>
      ))}
    </div>
  );
}

/** One line of posters that scrolls sideways. */
export function MovieRow({ children }: { children: ReactNode }) {
  return (
    <div className="-mx-5 flex snap-x gap-4 overflow-x-auto px-5 pb-3 sm:-mx-8 sm:px-8 [&>*]:w-36 [&>*]:shrink-0 [&>*]:snap-start sm:[&>*]:w-40">
      {children}
    </div>
  );
}

export function MovieRowSkeleton() {
  return (
    <MovieRow>
      {Array.from({ length: 7 }, (_, i) => (
        <Skeleton key={i} className="aspect-2/3" />
      ))}
    </MovieRow>
  );
}
