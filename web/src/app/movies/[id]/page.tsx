import { Suspense } from "react";
import { MovieDetailSkeleton, MoviePage } from "./movie-page";

export default function Page({ params }: PageProps<"/movies/[id]">) {
  return (
    // The movie's id is only known per request, so everything that reads it streams in.
    <Suspense fallback={<MovieDetailSkeleton />}>
      <MoviePage params={params} />
    </Suspense>
  );
}
