import type { Metadata } from "next";
import type { ReactNode } from "react";
import { ModelStatus } from "@/components/model-status";
import { REPO_URL } from "@/lib/config";

export const metadata: Metadata = { title: "How it works" };

// NDCG@10 over all users, from `metrics/` (validation picks the winner, test is looked at
// once). Copied by hand from the results of the model-selection phase; `notebooks/02_results.ipynb`
// has the full table with confidence intervals.
const RESULTS = [
  { model: "EASE + recency", validation: 0.238, test: 0.234, serving: true },
  { model: "iALS + recency", validation: 0.235, test: 0.234 },
  { model: "SASRec + recency", validation: 0.217, test: 0.21 },
  { model: "EASE", validation: 0.215, test: 0.221 },
  { model: "iALS", validation: 0.204, test: 0.209 },
  { model: "Item-kNN", validation: 0.2, test: 0.207 },
  { model: "Popularity", validation: 0.192, test: 0.176 },
  { model: "SASRec", validation: 0.161, test: 0.169 },
];
const BEST = Math.max(...RESULTS.map((row) => row.test));

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mt-16">
      <h2 className="text-4xl font-bold">{title}</h2>
      <div className="mt-5 flex max-w-[68ch] flex-col gap-4 leading-relaxed">{children}</div>
    </section>
  );
}

export default function AboutPage() {
  return (
    <div className="container-page py-10">
      <h1 className="max-w-3xl text-5xl leading-[0.95] font-bold sm:text-7xl">
        How the recommendations are made
      </h1>
      <p className="mt-5 max-w-[68ch] text-lg leading-relaxed text-muted-foreground">
        This is a portfolio project: five recommendation algorithms compared on the same data under
        the same rules, and the winner put behind a small web app. Everything, from the data
        download to the numbers below, can be rebuilt from the{" "}
        <a href={REPO_URL} className="text-foreground underline">
          source code
        </a>{" "}
        with one command.
      </p>

      <Section title="From your ratings to a list">
        <p>
          The model is called EASE. It learned, from 8.5 million ratings, one weight for every pair
          of movies: how much liking the first says about liking the second. That table is computed
          once, in closed form, and saved.
        </p>
        <p>
          When you open your recommendations, the server reads the movies you rated 3.5 stars or
          higher and adds up their rows of the table. That gives every movie in the catalog a score.
          A small bonus goes to movies that many people liked in the last weeks of the training
          data, the movies you have already rated are removed, and the twenty with the highest
          scores are shown.
        </p>
        <p>
          No model is trained for you and nothing about you is stored except your ratings. That is
          why a new account gets recommendations after ten ratings, and why changing one rating
          changes the next list.
        </p>
        <p>
          Your ratings also go into the next version of the model. Every so often they are copied
          into the project&apos;s published training data, with a scrambled id in place of your
          account and nothing but the movie, the stars and the time, and the model is refitted with
          them. The new version replaces this one only if it does at least as well on MovieLens and,
          once the app has enough users, better on their latest ratings. Ratings that have been
          copied stay in the published data even if you change or remove them here.
        </p>
      </Section>

      <Section title="Why this model">
        <p>
          Each model was tuned on a validation period and then scored once on later ratings it had
          never seen. The score is NDCG@10: it is higher the nearer the movies a person went on to
          like are to the top of their list of ten. Both new users with ten ratings and users with a
          long history are counted.
        </p>
        <table className="mt-2 w-full max-w-2xl text-left">
          <thead className="text-sm text-muted-foreground">
            <tr className="border-b">
              <th scope="col" className="py-2 font-normal">
                Model
              </th>
              <th scope="col" className="py-2 text-right font-normal">
                Validation
              </th>
              <th scope="col" className="py-2 pl-6 font-normal" colSpan={2}>
                Test
              </th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {RESULTS.map((row) => (
              <tr key={row.model} className="border-b">
                <th scope="row" className="py-2 font-medium">
                  {row.model}
                  {row.serving && (
                    <span className="ml-2 rounded-sm bg-primary px-1.5 py-0.5 text-xs font-semibold text-primary-foreground">
                      in use
                    </span>
                  )}
                </th>
                <td className="py-2 text-right">{row.validation.toFixed(3)}</td>
                <td className="py-2 pl-6">{row.test.toFixed(3)}</td>
                <td className="w-2/5 py-2 pl-3" aria-hidden>
                  <div
                    className={`h-2 rounded-full ${row.serving ? "bg-primary" : "bg-foreground/30"}`}
                    style={{ width: `${(row.test / BEST) * 100}%` }}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p>
          The recency bonus helps every model it is added to. EASE with the bonus was best on
          validation, so it was selected; on the test period iALS with the bonus ties it. The
          sequence model, SASRec, was trained on a laptop CPU for only ten epochs and would likely
          do better with more.
        </p>
      </Section>

      <Section title="What is running now">
        <ModelStatus />
      </Section>

      <Section title="What it is built with">
        <p>
          The experiments run as a DVC pipeline with every setting in one file and every random step
          seeded; running it twice gives byte-identical models and metrics. Runs and model versions
          are tracked in MLflow. The model is served by a FastAPI service that needs about 210 MB of
          memory, accounts and ratings live in Supabase (Postgres), and this site is Next.js.
        </p>
        <p>
          Ratings for training come from the MovieLens 32M dataset by GroupLens and from this
          app&apos;s users. Posters and plot summaries come from TMDB and are only ever shown here,
          never used for training. This product uses the TMDB API but is not endorsed or certified
          by TMDB.
        </p>
      </Section>
    </div>
  );
}
