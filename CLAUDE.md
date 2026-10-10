# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A portfolio movie recommender built in phases (see the checklist in `README.md`). Five algorithms are compared on MovieLens 32M: Popularity, Item-kNN, iALS, EASE and SASRec, plus a recency variant of each personalised one. TMDB (posters, plot summaries) is used only by the web app, never by experiments. The winner is served by FastAPI on Render's free tier (512MB RAM) behind a Next.js frontend on Vercel, with Supabase for auth and Postgres. Ilian reviews each phase before the next one starts. Explain design choices as you go, and when a tool or approach decision comes up, present a pros/cons table with a recommendation.

**Hard requirement: full reproducibility.** A stranger who clones the repo and runs the pipeline must get the same metrics and artifacts. Every change must preserve this.

## Commands

All Python commands run through uv. Everything is pinned in `uv.lock`; DVC is a dev dependency, so run `uv run dvc`, never a global `dvc`.

```bash
uv sync                                   # install the locked environment (Python 3.12)
uv run ruff check . && uv run ruff format --check .
uv run mypy                               # strict; checks src/ only
uv run pytest                             # all tests
uv run pytest tests/test_seed.py::test_different_seeds_differ   # single test
uv run dvc repro                          # run the pipeline (stages defined in dvc.yaml from Phase 1)
uv run dvc repro evaluate@item_knn        # one model only; train/evaluate/train_final/test are per-model stages (`stage@model`)
uv run python -m recsys.tuning.tune ials  # Optuna search for one model -> tuning/ials.json (hours for all five)
uv run python -m recsys.registry.promote  # manual gate: move the `champion` alias to the current challenger (`--version N` for another)
uv run python scripts/repro_check.py      # whole pipeline twice on synthetic data in a throwaway copy; fails on any differing file (~25 min)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db        # browse local runs
```

CI (`.github/workflows/ci.yml`) runs `uv sync --frozen`, ruff, mypy and pytest; `.github/workflows/repro-check.yml` runs `scripts/repro_check.py`.

**Windows performance caveat:** on this machine, Python imports are very slow because of Defender scanning and Docker Desktop CPU load (importing `mlflow` has taken 2–4 minutes). Use long timeouts or `run_in_background` for anything that imports mlflow/pandas, and don't misread slowness as a hang. A multi-line `python -c` fails through the Bash tool on Windows; write a script to the scratchpad instead. A `>` inside an argument can be treated as a redirect even when quoted (e.g. `uv add "pkg>=1.0"` created a stray empty file named `1.0`), so add packages without version specifiers and let uv pin them.

## Architecture and conventions

- **`params.yaml` is the single source of truth** for every tunable value: seed, thread counts, data URLs and checksums, filters, split dates, hyperparameters. Load it with `recsys.utils.config.load_params()`. Don't hardcode values in code. DVC tracks params per stage, so only the affected stages re-run.
- **Determinism:** every entry point calls `recsys.utils.seed.make_deterministic(seed, num_threads)` first. It seeds Python, NumPy and torch, pins BLAS/OpenMP threads, and returns a `np.random.Generator`; pass that generator around explicitly instead of using global NumPy random state.
- **MLflow:** wrap training/evaluation in `recsys.utils.mlflow_utils.tracked_run(name)`. It logs the git commit/dirty flag and `params.yaml`, `uv.lock` and `dvc.lock` as provenance. The tracking URI is `MLFLOW_TRACKING_URI` (DagsHub, set in `.env`) when present; otherwise a local `sqlite:///mlflow.db` with artifacts in `mlartifacts/`. MLflow 3.x rejects the file store, so don't reintroduce `./mlruns`. Mypy skips MLflow's source on purpose (see `pyproject.toml`).
- **Data:** data, models and metrics outputs are versioned by DVC (DagsHub remote) and git-ignored. The pipeline downloads MovieLens from GroupLens and verifies the SHA256 in `params.yaml`. The ML-32M license allows redistribution, including transformations, under the same license, so its data is pushed to the DVC remote. The dataset's `README.txt`, which contains the license, is kept in `data/raw/ml-32m/` and pushed with it.
- **TMDB terms:** no caching TMDB data for more than 6 months, and no sharing TMDB datasets. So TMDB data must never enter the DVC pipeline, the DVC remote or git. Only the API fetches it, caching it in Postgres with refresh before 6 months, and the app shows the TMDB logo and the required notice.
- **Models:** every model subclasses `recsys.models.base.Recommender` and is registered in `recsys/models/__init__.py` under the name used in `params.yaml` (`models.<name>`) and in the `foreach` lists in `dvc.yaml`. Subclasses implement `fit`, `score`, `save`, `load`; masking seen items and picking the top k live in the base class (`top_k` uses a stable sort so ties never depend on the CPU). Save models with `save_arrays` (`.npy` + `meta.json`), never `.npz`: zip entries carry timestamps, which breaks byte-identical outputs.
- **Seeds in models:** a model with random steps sets `uses_seed = True` and takes a `seed` constructor argument; `build_model(name, params, seed)` passes the top-level `seed` from `params.yaml`, so it is not repeated under `models.<name>`.
- **Tuning:** `recsys.tuning.tune` is a command, not a DVC stage. It reads the search space from `tuning.models.<name>` in `params.yaml` and writes `tuning/<name>.json` (committed to git). The best values are then copied by hand into `models.<name>`, and `tests/test_tuning.py` fails if the two disagree. `dvc repro` therefore never re-runs a search. The objective is `tuning.metric` averaged over all validation users, warm and onboarding pooled.
- **Recency variants:** `<base>_recency` models (`recsys/models/recency.py`) wrap a fitted base model and add `weight` times a standardised time-decayed like count to its standardised scores. They are never trained or saved by the pipeline: `load_fitted` loads the base model from `models/<base>` and fits the prior from the interactions. `dvc.yaml`'s `scored_with` map (variant -> base model) drives the `evaluate` and `test` stages; `train` and `train_final` cover only the five base models. Tuning a variant needs its base model in `models/`.
- **EASE is stored pruned:** `models.ease.keep_per_item` keeps the largest-magnitude weights per movie as a sparse matrix. The dense matrix is 1.5 GB, which this machine's disk can't hold twice (workspace + DVC cache). The `ease_pruning` stage fits the full matrix in memory and writes the accuracy/size trade-off to `metrics/ease_pruning.json`; `tests/test_registry.py` fails if `keep_per_item` is not the value it recommends (smallest within `ease_pruning.max_relative_drop`).
- **SASRec on CPU:** about 2 minutes per epoch on this laptop, so `epochs` and the number of tuning trials are deliberately small. Dropout masks come from numpy (`_dropout`), not `nn.Dropout`, which was a third of the training time. Weights are saved as `.npy` files, not with `torch.save`.
- **Evaluation:** `recsys.evaluation.evaluate` scores a saved model on the validation split in two scenarios (`protocol.py`): `warm` (users with history before the cut-off) and `onboarding` (new users; their first `evaluation.onboarding.n_history` likes are the history). It writes `metrics/val_<model>.json` and per-user metrics to `data/evaluation/val/<model>.parquet`; each file also has an `all` scope pooling both scenarios. With `--split test` it scores the models refitted on train + val (`train --final` -> `models/final/<model>`) on the test split.
- **Selection and registry:** `compare@val|test` runs paired tests on the per-user parquet files (bootstrap CI of the mean difference, paired t-test, Holm correction). `select` picks the best validation `tuning.metric` over all users among models within `selection.max_model_size_mb`; test results never influence it. `export` writes `bundle/` (model, `items.json`, `metadata.json`, `model_card.md`); `register` uploads it to MLflow as a new `movie-recsys` version with the `challenger` alias. The git commit is a tag on the model version, not in the bundle, because the bundle is a DVC output and must be deterministic.
- **Timings stay out of DVC files:** training time and latency vary run to run, so they go to MLflow only. Anything written to `metrics/`, `models/` or `data/` must be deterministic.
- Layout: `src/recsys/{data,models,evaluation,tuning,registry,utils}` for the pipeline. `api/`, `web/` and `monitoring/` are placeholders for later phases. `docker/train.Dockerfile` is the reference environment, where results should be byte-identical.

## Project plan

Approved with Ilian on 2026-10-01. Follow it; propose changes to Ilian rather than deviating silently.

### Decisions
| Area | Choice |
|---|---|
| Data | MovieLens 32M (ratings, movies, tags, links) for all experiments. TMDB is app-only (decided 2026-10-02 after reading its terms). |
| Data versioning / pipelines | DVC (`dvc.yaml` stages, `params.yaml`, `dvc.lock` committed), remote on DagsHub |
| Experiment tracking + registry | MLflow on DagsHub, registry aliases `champion` / `challenger` |
| Algorithms (5) | Popularity, Item-kNN, iALS, EASE, SASRec |
| New users | Onboarding: rate ~10 movies → recommendations computed right away without retraining; popularity/genre fallback |
| Accounts / DB | Supabase (Auth + Postgres) |
| Backend | FastAPI + SQLAlchemy, Docker on Render free tier (512MB RAM, sleeps after 15 min idle). Hugging Face Spaces was rejected because Docker Spaces need PRO. |
| Frontend | Next.js + TypeScript + Tailwind (shadcn/ui) on Vercel |
| Monitoring | Request/recommendation logs in Postgres → Grafana Cloud dashboards and alerts; Evidently drift reports run on a schedule |
| CI/CD | GitHub Actions: lint, tests, pipeline smoke test, reproducibility check, deploy |

### Core design: one model interface
Every model implements `fit(interactions)`, `recommend(user_history, k, exclude_seen=True)` and `save/load`. `recommend` takes a **raw history, not a user ID**, so all 5 models can score unseen users:
- Popularity ignores the history.
- Item-kNN sums the similarities of the items in the history.
- EASE multiplies the history vector by the item-item matrix B.
- iALS computes a user vector from the history in one least-squares step ("fold-in").
- SASRec feeds the history as a sequence.

Onboarding therefore works with whichever model wins, and offline evaluation and serving share one code path.

### Reproducibility strategy
1. Pinned environment: `uv.lock`; `docker/train.Dockerfile` (CPU PyTorch) is the reference environment.
2. Determinism: `make_deterministic`; `torch.use_deterministic_algorithms(True)`; pinned BLAS/OpenMP threads; Optuna with a seeded `TPESampler` and sequential trials.
3. Data provenance: MovieLens is downloaded from GroupLens and checked against the SHA256 in `params.yaml`. Every downstream output is deterministic, so its hash must match `dvc.lock`. Strangers can either `dvc pull` or rebuild from scratch, and experiments need no API keys.
4. Provenance on every MLflow run (via `tracked_run`); local SQLite fallback when there are no credentials.
5. One command: `dvc repro` builds data, trains all 5 models, evaluates them and selects the winner. `metrics/*.json` are DVC metrics, so `dvc metrics diff` shows mismatches.
6. CI `repro-check.yml` runs the pipeline twice on a small fixed sample and asserts identical hashes and metrics. Byte-identical results are guaranteed inside Docker; on bare metal across different CPUs, metrics match within a small documented tolerance.

### Data pipeline (DVC stages)
1. `download`: ML-32M zip → verify checksum → `data/raw/`
2. `validate`: pandera schemas (ID ranges, ratings 0.5–5, timestamps, no duplicate (user, item) pairs); fail loudly.
3. `preprocess` (all values in params):
   - Keep a recent time window (e.g. 2010+).
   - Ratings ≥ 3.5 become positive implicit interactions; keep the full ratings for analysis.
   - Iterative k-core filter (users ≥ 5, items ≥ 10).
   - Cap the catalog at the ~20k most popular items.
   - Re-index IDs to contiguous integers.
4. `split`: global temporal split. Train runs up to T1, validation covers T1–T2, test is everything after T2. Same split for all models; no future data leaks into training.
5. `items`: catalog table from MovieLens (title, year, genres, tmdbId for the app to look up).

### Evaluation protocol
- Full ranking over the whole catalog, with no sampled negatives (Krichene & Rendle 2020). Seen items are excluded.
- Metrics @10 and @20: Recall, NDCG, MRR, catalog coverage, novelty/popularity bias, per-segment results (short vs long histories).
- Bootstrap 95% confidence intervals plus paired significance tests between models.
- Optuna tuning on the validation split (≈30–50 trials per model, set in params), as nested MLflow runs. Refit on train+val; evaluate **once** on test.
- Also logged: training time, inference latency (p50/p95), model size.

### The 5 algorithms
| Model | Library | Notes |
|---|---|---|
| Popularity (optionally time-decayed) | numpy | Baseline every other model must beat |
| Item-kNN (cosine / BM25) | `implicit` or scipy sparse | Also powers "similar movies" |
| iALS | `implicit` | Fast on CPU; fold-in for new users |
| EASE | numpy (closed form) | One parameter, λ; often state of the art on MovieLens |
| SASRec | PyTorch (own ~200-line implementation) | 2 blocks, 64-dim, seq len 50; CPU-trained on a user subsample set in params |

### Model selection & registry
- Serving-memory budget for Render's 512MB: serving bundle ≤ ~150MB; API peak RSS under load ≤ ~400MB. Log bundle size and peak RSS for every model.
  - EASE: prune B to the top-k weights per row, stored as sparse float32, with k chosen so NDCG drops < ~1%. Unpruned, it's ~1.6GB.
  - SASRec: export to ONNX and serve with `onnxruntime`; the API image never imports torch.
  - iALS / Item-kNN: float32 factors / sparse top-k neighbor lists.
  - Report both pruned and unpruned results.
- `select_and_register` stage:
  - Picks the best validation NDCG@10 among the models that meet the budget.
  - Registers it in MLflow as `movie-recsys` with the `challenger` alias. Promotion to `champion` is a manual gate.
- `export` writes a self-contained serving bundle: weights, ID maps, metadata, and a `model_card.md` with metrics, data hash and git SHA.
- A results notebook produces the comparison table and plots for the README.

### Backend API (FastAPI on Render free)
- Slim `api` dependency group: fastapi, uvicorn, numpy, scipy, onnxruntime, sqlalchemy, psycopg. No torch, pandas or implicit.
- CI bakes the `champion` bundle into the image, pushes it to GHCR, and triggers a Render deploy hook. `/health` reports the model version.
- Render sleeps after 15 minutes idle: the frontend shows "waking up the server…" while it retries `/health`. An optional UptimeRobot ping also provides uptime monitoring.
- Endpoints: `GET /health`, `GET /movies/search?q=`, `GET /movies/{id}`, `GET /movies/{id}/similar`, `GET /onboarding/movies`, `POST /ratings`, `GET /recommendations?k=`, `POST /events`.
- Auth: verify the Supabase JWT in a FastAPI dependency, with Postgres row-level security as a second layer.
- TMDB: the API fetches movie details by tmdbId as needed (key stays server-side) and caches them in a `movie_metadata` table with `fetched_at`, refreshing anything older than ~5 months.
- Tables:
  - `profiles`
  - `movie_metadata` (TMDB cache)
  - `ratings`
  - `rec_requests` (request ID, user, model version, items shown, latency)
  - `events` (clicks, ratings given after a recommendation)
- Tests: pytest + httpx TestClient with a fake model bundle.

### Frontend (Next.js on Vercel)
- Pages:
  - Landing page
  - Sign up / log in (Supabase Auth)
  - Onboarding: rate ≥10 posters
  - Home: "Recommended for you" plus a "Because you liked X" row
  - Movie detail with "similar movies"
  - My ratings
  - About / How it works (architecture, model card, live metrics)
- Posters load from TMDB's image CDN by URL. The footer shows the TMDB logo and the notice: "This product uses the TMDB API but is not endorsed or certified by TMDB."

### Monitoring
- Operational: latency, error rate and request volume, logged per request to Postgres. Grafana Cloud reads from the Supabase Postgres data source; alerts fire on p95 latency and error rate.
- Online ML quality:
  - Click-through rate on recommendations
  - Hit rate (did the user later rate a movie we recommended?)
  - Coverage and popularity bias of what's actually served
- Drift: a weekly `monitoring.yml` workflow runs Evidently, comparing production behavior (ratings distribution, genre mix, history lengths) and served-recommendation stats against the training reference. Reports go to MLflow and GitHub Pages.
- Retraining loop: app ratings → new DVC-versioned dataset (`data/app_feedback`) → retrain → promote a challenger that beats the champion.

### Phases (one at a time; Ilian reviews after each)
0. ✅ Foundation: uv, ruff/mypy/pytest, pre-commit, `params.yaml`, DVC init, MLflow utilities, seeding, train Dockerfile, CI skeleton. Still open: connecting the DagsHub remote, which needs Ilian's account.
1. ✅ Data: download/validate/preprocess/split/items stages, EDA notebook (`notebooks/01_eda.ipynb` explains the `preprocess`/`split` values). Result: 8.48M positives, 71,841 users, 19,642 movies. ~40% of val/test users have no prior history; evaluate them separately as an onboarding scenario in Phase 2.
2. ✅ Evaluation framework (metrics unit-tested on hand-computed examples) + Popularity and Item-kNN, MLflow logging. Validation NDCG@10, untuned: Popularity 0.063 warm / 0.312 onboarding; Item-kNN 0.081 / 0.369. Left for later phases: tuning the `models.*` params (3), paired significance tests on the per-user parquet files and the test split (4).
3. ✅ iALS, EASE, SASRec + Optuna tuning of all five (`tuning/*.json`). Tuned validation NDCG@10, warm / onboarding: Popularity (30-day half-life) 0.124 / 0.293; Item-kNN 0.084 / 0.373; iALS 0.096 / 0.365; EASE 0.106 / 0.377; SASRec 0.065 / 0.303. Decided with Ilian on 2026-10-08: recency weighting (time decay, which lifted Popularity's warm score from 0.063 to 0.124) is added to the comparison in Phase 4; SASRec stays CPU-trained even though it is undertrained at 10 epochs. Still open: EASE is stored pruned to 1,000 weights per movie because of disk space, so the unpruned comparison is still to do.
4. ✅ Selection & registry: recency variants, EASE pruning study, refit on train+val, test evaluation, paired significance tests, selection, export bundle with model card, MLflow registration, reproducibility check (script + CI workflow), `notebooks/02_results.ipynb`. Selected: `ease_recency` (EASE pruned to 500 weights per movie, 77 MB, plus an 11-day recency prior). NDCG@10 over all users, val / test: EASE + recency 0.238 / 0.234; iALS + recency 0.235 / 0.234; SASRec + recency 0.217 / 0.210; EASE 0.215 / 0.221; iALS 0.204 / 0.209; Item-kNN 0.200 / 0.207 (recency adds nothing); Popularity 0.192 / 0.176; SASRec 0.161 / 0.169. The winner's lead over iALS + recency is significant on validation (p = 0.031) but the two are tied on test, and iALS + recency is 9 MB. Still open: SASRec -> ONNX export (only needed if SASRec is ever selected); peak-RSS measurement under load (Phase 5, with the real API process). Ilian promoted `movie-recsys` v1 to `champion` on 2026-10-10.
5. API: FastAPI + Supabase schema + auth + tests + Dockerfile
6. Frontend: Next.js pages, Supabase auth, onboarding flow
7. Deploy & CI/CD: API image → GHCR → Render deploy hook; Vercel project; secrets
8. Monitoring: logging tables/views, Grafana dashboards + alerts, Evidently job
9. Retraining loop + polish: README with architecture diagram, results table, demo GIF, "how to reproduce" section

Accounts and keys needed (in `.env` / GitHub Secrets): DagsHub token, TMDB API key, Supabase (URL, anon/service keys, JWT secret), Render deploy hook URL, Vercel, Grafana Cloud.

### Verification
- Per phase: `uv run pytest`, `uv run ruff check .`, `uv run mypy`.
- Pipeline: `dvc repro` finishes; `dvc metrics show` lists all 5 models; MLflow shows runs with params, metrics and artifacts; the registry has a `champion`.
- Reproducibility: fresh clone → `uv sync` → `dvc pull` → `dvc repro` → `dvc status` shows nothing changed and `dvc metrics diff` shows no differences. Also checked in CI on the sample, and in Docker on the full data.
- API: `docker run --memory=512m`, call every endpoint, short `locust` load test with peak memory < ~400MB; the deployed `/health` reports the expected model version.
- End to end: sign up on Vercel → rate 10 movies → recommendations appear → clicks land in `events` → the Grafana dashboard updates.
- Monitoring: trigger `monitoring.yml` manually → the Evidently report is published.
