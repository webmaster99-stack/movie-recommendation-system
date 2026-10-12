import Link from "next/link";
import { REPO_URL } from "@/lib/config";

export function SiteFooter() {
  return (
    <footer className="mt-24 border-t">
      <div className="container-page flex flex-col gap-6 py-8 text-sm text-muted-foreground sm:flex-row sm:items-start sm:justify-between">
        <div className="flex max-w-md items-start gap-4">
          {/* TMDB's terms require their logo and this exact sentence wherever their data is shown. */}
          <a href="https://www.themoviedb.org/" className="mt-1 shrink-0">
            <img src="/tmdb.svg" alt="TMDB" width={64} height={46} />
          </a>
          <p>
            Posters and plot summaries come from TMDB. This product uses the TMDB API but is not
            endorsed or certified by TMDB.
          </p>
        </div>
        <p className="max-w-xs">
          Trained on the{" "}
          <a href="https://grouplens.org/datasets/movielens/32m/" className="underline">
            MovieLens 32M
          </a>{" "}
          ratings from GroupLens.{" "}
          <Link href="/about" className="underline">
            How it works
          </Link>
          {" and "}
          <a href={REPO_URL} className="underline">
            source code
          </a>
          .
        </p>
      </div>
    </footer>
  );
}
