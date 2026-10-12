"use client";

import { useHydrated } from "@/components/browser-only";
import { Skeleton } from "@/components/ui/skeleton";
import { useOnboardingMovies } from "@/lib/api/hooks";

const PER_ROW = 14;

/**
 * Two rows of the catalog's best-known posters running off both edges of the window. It is
 * scenery (the same movies are rated one by one after signing up), so it is hidden from
 * screen readers and can't be focused.
 */
export function PosterWall() {
  const { data } = useOnboardingMovies();
  const hydrated = useHydrated();
  const posters = ((hydrated && data?.pages[0].movies) || [])
    .filter((movie) => movie.poster_url)
    .slice(0, PER_ROW * 2);
  const rows = [posters.slice(0, PER_ROW), posters.slice(PER_ROW)];

  return (
    <div className="flex flex-col gap-3 overflow-hidden" aria-hidden>
      {rows.map((row, index) => (
        <div
          key={index}
          // The second row starts half a poster to the left, like bricks.
          className={`flex gap-3 ${index === 1 ? "-ml-28" : "-ml-10"}`}
        >
          {row.length > 0
            ? row.map((movie) => (
                <img
                  key={movie.movie_id}
                  src={movie.poster_url as string}
                  alt=""
                  className="aspect-2/3 w-32 shrink-0 rounded-md object-cover sm:w-36"
                />
              ))
            : Array.from({ length: PER_ROW }, (_, i) => (
                <Skeleton key={i} className="aspect-2/3 w-32 shrink-0 sm:w-36" />
              ))}
        </div>
      ))}
    </div>
  );
}
