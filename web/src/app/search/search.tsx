"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { GRID, MovieCard, MovieGridSkeleton } from "@/components/movie-card";
import { Input } from "@/components/ui/input";
import { useHealth, useSearch } from "@/lib/api/hooks";

const DEBOUNCE_MS = 250;

export function Search() {
  const fromUrl = useSearchParams().get("q") ?? "";
  const [text, setText] = useState(fromUrl);
  const [query, setQuery] = useState(fromUrl.trim());
  const results = useSearch(query);
  const health = useHealth();

  // Search as the visitor types, once they pause, and keep the address shareable.
  useEffect(() => {
    const timer = setTimeout(() => {
      const next = text.trim();
      setQuery(next);
      window.history.replaceState(
        null,
        "",
        next ? `/search?q=${encodeURIComponent(next)}` : "/search",
      );
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [text]);

  return (
    <>
      <h1 className="text-5xl font-bold sm:text-6xl">Search</h1>
      <Input
        type="search"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="A movie's title"
        aria-label="Movie title"
        autoFocus
        maxLength={100}
        className="mt-6 h-12 max-w-xl text-lg md:text-lg"
      />
      <div className="mt-10" aria-live="polite">
        {!query ? (
          <p className="text-muted-foreground">
            {health.data
              ? `${health.data.n_movies.toLocaleString("en")} movies, the ones with enough ratings to recommend from.`
              : "Type a title to start."}
          </p>
        ) : results.isError ? (
          <p className="text-muted-foreground">
            The search didn&apos;t go through. Type again to retry.
          </p>
        ) : !results.data ? (
          <MovieGridSkeleton />
        ) : results.data.length === 0 ? (
          <p className="text-muted-foreground">
            No title contains “{query}”. Check the spelling or use fewer words; movies with very few
            ratings aren&apos;t in the catalog.
          </p>
        ) : (
          <ul className={GRID}>
            {results.data.map((movie) => (
              <li key={movie.movie_id}>
                <MovieCard movie={movie} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </>
  );
}
