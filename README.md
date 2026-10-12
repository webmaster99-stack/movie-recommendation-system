# Movie Recommendation System

An end-to-end, fully reproducible movie recommender. Five algorithms (Popularity, Item-kNN,
iALS, EASE, SASRec) are compared on MovieLens 32M, and the winner is served in a web app
(Next.js on Vercel, FastAPI on Render) with experiment tracking, model registry, data versioning
and production monitoring.

> 🚧 Work in progress. See the build phases below.

## Quickstart

```bash
uv sync          # exact, locked environment (Python 3.12)
uv run pytest    # tests
```

### Data

```bash
uv run dvc pull    # fetch the processed data from DagsHub (fast)
uv run dvc repro   # or rebuild everything from GroupLens (checksum-verified)
```

Data: [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) by GroupLens Research, used under
its license (see `data/raw/ml-32m/README.txt`). F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens
Datasets: History and Context.* ACM TiiS 5, 4: 19:1–19:19.

Optional: copy `.env.example` to `.env` to log runs to DagsHub instead of the local `mlflow.db`.

## Results

Nine candidates are compared: the five algorithms, plus a recency variant of each personalised
one. Every model ranks the full catalog of 19,642 movies, with movies the user has already
seen excluded. Numbers are NDCG@10. "Returning" users have history before the split date;
"new" users don't, so their first 10 likes are the history and the rest are predicted.

| Model | Val, all users | Val, returning | Val, new | Test, all users | Test, returning | Test, new | Size |
|---|---|---|---|---|---|---|---|
| **EASE + recency** (selected) | **0.238** | **0.149** | 0.370 | 0.234 | **0.136** | 0.383 | 77 MB |
| iALS + recency | 0.235 | 0.142 | 0.373 | **0.234** | 0.135 | 0.384 | 9 MB |
| SASRec + recency | 0.217 | 0.145 | 0.325 | 0.210 | 0.127 | 0.336 | 5 MB |
| EASE | 0.215 | 0.106 | **0.376** | 0.221 | 0.108 | 0.393 | 77 MB |
| iALS | 0.204 | 0.096 | 0.365 | 0.209 | 0.096 | 0.382 | 9 MB |
| Item-kNN | 0.200 | 0.084 | 0.373 | 0.207 | 0.084 | **0.396** | 131 MB |
| Item-kNN + recency | 0.200 | 0.084 | 0.373 | 0.207 | 0.083 | **0.396** | 131 MB |
| Popularity (30-day half-life) | 0.192 | 0.124 | 0.293 | 0.176 | 0.114 | 0.270 | 0.1 MB |
| SASRec | 0.161 | 0.065 | 0.303 | 0.169 | 0.066 | 0.326 | 5 MB |

Validation: 9,689 users (5,795 returning, 3,894 new). Test: 8,686 users (5,250 returning,
3,436 new). Hyperparameters were tuned on validation, so those columns are optimistic. The
test columns come from models refitted on train + validation and evaluated once, after the
choice was made.

What the table says:

- **The selected model is EASE with a recency prior.** It has the best validation NDCG@10
  over all users among the models within the 150 MB serving budget. Its lead over iALS +
  recency is 0.0026 [0.0008, 0.0044], significant after Holm correction (p = 0.031).
- **On the test split the top two are tied.** iALS + recency is ahead by 0.0001
  [-0.0018, 0.0019] (p = 0.95), at a ninth of the size. The selection rule only looks at
  validation, so the choice stands, but iALS + recency is an equally good, much smaller
  alternative.
- **Recency is worth more than the choice of algorithm.** Adding the prior lifts EASE from
  0.215 to 0.238, iALS from 0.204 to 0.235 and SASRec from 0.161 to 0.217 on validation, all
  from returning users, who mostly rate what has just come out. For new users it changes
  little or costs a little.
- **Item-kNN gains nothing from recency**: its search settled on a weight close to zero.
- **SASRec is limited by compute, not by design.** It trains on a laptop CPU for 10 epochs
  and its score was still rising when training stopped.

Confidence intervals, @20, coverage, novelty, results by history length and every pairwise
test are in `metrics/` (`uv run dvc metrics show`) and in `notebooks/02_results.ipynb`.

### Recency variants

A recency variant leaves the base model as it is and changes only the score: each movie's
time-decayed like count (the same quantity the Popularity model ranks by) is standardised
and added to the user's standardised personalised scores, times a weight. The base model is
not retrained, and serving needs one extra number per movie.

| Model | Half-life | Weight | Val NDCG@10: base | With recency |
|---|---|---|---|---|
| EASE | 11 days | 9.07 | 0.215 | 0.238 |
| iALS | 7 days | 4.11 | 0.204 | 0.235 |
| SASRec | 11 days | 2.09 | 0.161 | 0.217 |
| Item-kNN | 246 days | 0.05 | 0.200 | 0.200 |

### EASE pruning

The full EASE weight matrix is 1.5 GB, ten times the serving budget, so only the
largest-magnitude weights per movie are kept. The `ease_pruning` stage measures the cost
(`metrics/ease_pruning.json`); `params.yaml` uses the smallest setting that loses at most
0.1% of the unpruned NDCG@10.

| Weights kept per movie | Size | Val NDCG@10, all users | Change |
|---|---|---|---|
| all 19,642 (unpruned) | 1,543 MB | 0.2148 | |
| 2,000 | 314 MB | 0.2147 | -0.04% |
| 1,000 | 157 MB | 0.2146 | -0.08% |
| **500** (used) | 79 MB | 0.2148 | -0.02% |
| 200 | 31 MB | 0.2140 | -0.39% |
| 100 | 16 MB | 0.2128 | -0.93% |

### Selection, bundle and registry

```bash
uv run dvc repro                              # data -> train -> evaluate -> compare -> select -> export -> register
uv run dvc repro evaluate@item_knn            # a single model on validation (`test@item_knn` for test)
uv run python -m recsys.registry.promote      # move the `champion` alias to the current challenger
```

- `compare` runs paired tests between all models on the same users: a bootstrap confidence
  interval of the mean difference and a paired t-test, Holm-corrected
  (`metrics/comparison_val.json`, `metrics/comparison_test.json`).
- `select` picks the best validation NDCG@10 over all users among the models within
  `selection.max_model_size_mb` (`metrics/selection.json`).
- `export` writes `bundle/`: the selected model refitted on train + validation, the movie
  catalog, metadata, and a model card (`bundle/model_card.md`).
- `register` uploads the bundle to MLflow as a new version of `movie-recsys` with the
  `challenger` alias. Promotion to `champion` is a deliberate manual step.

### Reproducibility check

```bash
uv run python scripts/repro_check.py
```

Runs the whole pipeline twice on a small synthetic dataset, in a throwaway copy of the
project, and fails if any output file differs between the two runs. CI runs it on every pull
request (`.github/workflows/repro-check.yml`).

### Tuning

Hyperparameters were searched with Optuna (seeded, one trial at a time, so a rerun repeats the
same trials), maximising NDCG@10 averaged over all validation users of both scenarios. Search
spaces and trial counts are in `params.yaml` under `tuning`; every trial is recorded in
`tuning/<model>.json`, and the winning values are the ones in `params.yaml` under `models`.

| Model | Trials | NDCG@10 before | NDCG@10 after | Tuned values |
|---|---|---|---|---|
| Popularity | 11 (grid) | 0.163 | 0.192 | half-life 30 days |
| Item-kNN | 30 | 0.197 | 0.200 | 853 neighbours, cosine, shrinkage 19.5 |
| iALS | 30 | 0.201 | 0.204 | 110 factors, regularization 19.25, alpha 2.9, 21 iterations |
| EASE | 8 | 0.214 | 0.215 | l2 1138 |
| SASRec | 6 | 0.156 | 0.161 | learning rate 0.0024, dropout 0.24 |

```bash
uv run python -m recsys.tuning.tune ease   # rerun one search (minutes to hours per model)
uv run python -m recsys.tuning.tune ease_recency   # a recency variant; needs the base model in models/
```

Tuning is not part of `dvc repro`: the pipeline reads the tuned values from `params.yaml`, so
rebuilding it never repeats a search.

## API

A FastAPI service (`api/recsys_api`) that serves the selected model from `bundle/`. It runs
on numpy and scipy alone, without the training code and its dependencies, so that it fits a
512 MB host: under a load test in a container limited to 512 MB it peaks at about 210 MB.

```bash
uv run dvc pull bundle     # the exported model (or `uv run dvc repro export`)
uv run uvicorn recsys_api.main:create_app --factory --app-dir api --reload
```

With no configuration it uses a local SQLite file and serves movies without posters; open
<http://localhost:8000/docs> for the interactive documentation. `.env.example` lists the
settings for Supabase (database and sign-in) and TMDB (posters).

| Endpoint | Signed in | What it returns |
|---|---|---|
| `GET /health` | | Model name, registry version and the date the training data ends |
| `GET /movies/search?q=` | | Movies by title, best match first, then most popular |
| `GET /movies/{id}` | | One movie, with TMDB's plot summary and images |
| `GET /movies/{id}/similar` | | Movies liked by the people who liked this one |
| `GET /onboarding/movies` | | Well-known movies across genres for a new user to rate |
| `GET /ratings` | yes | The user's ratings, newest first |
| `POST /ratings` | yes | Rate a movie or change a rating (half stars, 0.5 to 5) |
| `DELETE /ratings/{id}` | yes | Remove a rating |
| `GET /recommendations?k=` | yes | Recommendations from the user's ratings as they are right now |
| `POST /events` | yes | Record a click on a recommended movie |

How it works:

- **No retraining per user.** The model takes a list of liked movies, not a user id. A
  request reads the user's ratings of 3.5 stars and up, oldest first, and scores the whole
  catalog in about 5 ms, so a rating made a second ago already counts. Movies the user has
  rated, liked or not, are never recommended. A user with no likes yet gets the most popular
  movies.
- **Same results as offline.** The API has its own copy of the scoring code, because the
  training package needs pandas and PyTorch. A test loads the same saved model both ways and
  requires identical scores and recommendations.
- **Sign-in** is Supabase's. The frontend sends the user's access token and the API checks
  its signature against the project's public keys. Row-level security in Postgres
  (`supabase/migrations/`) is a second layer: the public key in the frontend can read
  nothing but a signed-in user's own profile and ratings.
- **Posters and plot summaries** come from TMDB, fetched on demand and cached in Postgres
  for at most 6 months, as TMDB's terms require. They are never stored in git or DVC. This
  product uses the TMDB API but is not endorsed or certified by TMDB.
- **Every recommendation list is logged** with the model version, the movies shown and the
  time it took, and clicks and ratings refer back to the list they came from. Phase 8 builds
  its monitoring on these tables.

```bash
uv run pytest api/tests                                              # in-memory database, fake TMDB
docker build -f docker/api.Dockerfile --build-arg MODEL_VERSION=1 -t recsys-api .
docker run --rm --memory=512m -p 8000:8000 --env-file .env recsys-api
```

`api/loadtest/locustfile.py` simulates users who sign up, rate movies and ask for
recommendations; its docstring has the commands.

## Web app

A Next.js app (`web/`) on top of the API: sign up, rate ten movies, get recommendations.

```bash
cd web
npm ci
cp .env.example .env.local     # the API's address and the Supabase project (public values)
npm run dev                    # http://localhost:3000, with the API running on :8000
```

| Page | What it does |
|---|---|
| `/` | What the app is, over a wall of the catalog's best-known posters |
| `/signup`, `/login` | Supabase email and password |
| `/onboarding` | Rate movies until there are enough for a first list |
| `/for-you` | The top 20 recommendations, and a row of movies similar to the last one you liked |
| `/movies/{id}` | Plot, your rating, and what people who liked it also liked |
| `/search` | Search by title as you type |
| `/ratings` | Everything you rated; change or remove a rating |
| `/about` | How a recommendation is made, the model comparison, and the model being served |

How it works:

- **The pages are static; the data is fetched in the browser.** The API sleeps on its free
  host, and a server-rendered page would hang with it. Instead the page appears at once,
  a notice says the server is waking up, and the requests are retried until it answers.
- **Sign-in happens in the browser** with Supabase's client. The app sends the access
  token to the API, which is the only thing that reads or writes ratings.
- **Stars move before the server answers.** A rating is shown immediately and rolled back
  with a message if saving it fails.
- **Clicks and ratings are credited to the list they came from**, so that Phase 8 can
  measure whether recommendations are used.
- **The API's types are generated, not retyped.** `scripts/export_openapi.py` writes the
  API's schema to `web/openapi.json` and `npm run gen:api` turns it into TypeScript. A test
  fails when the committed schema no longer matches the API, and CI fails when the types no
  longer match the schema.
- **Posters are loaded from TMDB's image servers by the visitor's browser**; no copy is
  kept. The footer carries TMDB's logo and notice.

```bash
npm run lint && npm run typecheck && npm test && npm run build
```

## Deployment

| Part | Where | How it gets there |
|---|---|---|
| API | Render, free plan (512 MB, sleeps when idle) | `.github/workflows/deploy.yml`, after CI passes on `main` |
| Web app | Vercel | Vercel builds `web/` on every push; pull requests get a preview |
| Database and sign-in | Supabase | `supabase/migrations/` |

What the Deploy workflow does:

1. **Downloads the `champion` from the model registry** and checks it against the SHA256
   recorded when it was registered (`python -m recsys.registry.pull`). The model in
   production is therefore the one that was promoted by hand, not the last one trained.
2. **Runs the parity test on that bundle**, so the image is never built around a model the
   API would score differently from the pipeline.
3. **Builds the image and pushes it to ghcr.io**, starts it with Render's memory limit and
   calls `/health`.
4. **Calls Render's deploy hook with the image's digest**, then waits until the public
   `/health` reports this commit and this model version. Render keeps the old instance
   answering until the new one is up, so an answer alone would prove nothing.

After promoting a new model (`python -m recsys.registry.promote`), run the workflow by hand
(Actions > Deploy > Run workflow): a promotion changes the registry, not the code.

Setting it up from a fork takes four things:

- **GitHub secrets** `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and
  `MLFLOW_TRACKING_PASSWORD` (to read the registry). The first run pushes the image; make
  the package public under the repository's Packages so Render can pull it.
- **Render:** New > Blueprint, pointed at this repository. `render.yaml` describes the
  service and asks for `DATABASE_URL`, `SUPABASE_URL`, `TMDB_API_KEY` and `CORS_ORIGINS`.
  Then copy the service's deploy hook into the GitHub secret `RENDER_DEPLOY_HOOK_URL` and its
  address into the GitHub variable `API_URL`.
- **Vercel:** import the repository with `web` as the root directory and the three
  variables from `web/.env.example`.
- **`CORS_ORIGINS`** on Render is the Vercel address, and the image name in `render.yaml`
  and `deploy.yml` is the fork's.

## Build phases

- [x] 0. Foundation: tooling, config, determinism, CI
- [x] 1. Data pipeline (DVC)
- [x] 2. Evaluation framework + baselines (Popularity, Item-kNN)
- [x] 3. iALS, EASE, SASRec + tuning
- [x] 4. Model selection + registry
- [x] 5. API
- [x] 6. Frontend
- [x] 7. Deployment + CI/CD
- [ ] 8. Monitoring
- [ ] 9. Retraining loop + polish
