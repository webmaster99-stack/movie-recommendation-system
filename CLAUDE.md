# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A portfolio movie recommender built in phases (see the checklist in `README.md`). Five algorithms are compared on MovieLens 32M plus a TMDB metadata snapshot: Popularity, Item-kNN, iALS, EASE and SASRec. The winner is served by FastAPI on Render's free tier (512MB RAM) behind a Next.js frontend on Vercel, with Supabase for auth and Postgres. Ilian reviews each phase before the next one starts. Explain design choices as you go, and when a tool or approach decision comes up, present a pros/cons table with a recommendation.

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
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db        # browse local runs
```

CI (`.github/workflows/ci.yml`) runs `uv sync --frozen`, ruff, mypy and pytest.

**Windows performance caveat:** on this machine, Python imports are very slow because of Defender scanning and Docker Desktop CPU load (importing `mlflow` has taken 2–4 minutes). Use long timeouts or `run_in_background` for anything that imports mlflow/pandas, and don't misread slowness as a hang. A multi-line `python -c` fails through the Bash tool on Windows; write a script to the scratchpad instead.

## Architecture and conventions

- **`params.yaml` is the single source of truth** for every tunable value: seed, thread counts, data URLs and checksums, filters, split dates, hyperparameters. Load it with `recsys.utils.config.load_params()`. Don't hardcode values in code. DVC tracks params per stage, so only the affected stages re-run.
- **Determinism:** every entry point calls `recsys.utils.seed.make_deterministic(seed, num_threads)` first. It seeds Python, NumPy and torch, pins BLAS/OpenMP threads, and returns a `np.random.Generator`; pass that generator around explicitly instead of using global NumPy random state.
- **MLflow:** wrap training/evaluation in `recsys.utils.mlflow_utils.tracked_run(name)`. It logs the git commit/dirty flag and `params.yaml`, `uv.lock` and `dvc.lock` as provenance. The tracking URI is `MLFLOW_TRACKING_URI` (DagsHub, set in `.env`) when present; otherwise a local `sqlite:///mlflow.db` with artifacts in `mlartifacts/`. MLflow 3.x rejects the file store, so don't reintroduce `./mlruns`. Mypy skips MLflow's source on purpose (see `pyproject.toml`).
- **Data:** data, models and metrics outputs are versioned by DVC (DagsHub remote) and git-ignored. MovieLens can't be redistributed, so the pipeline downloads it from GroupLens and verifies the SHA256 in `params.yaml`; never push raw MovieLens files to a public remote. The TMDB snapshot is pulled once and shared via `dvc pull`.
- Layout: `src/recsys/{data,models,evaluation,tuning,registry,utils}` for the pipeline. `api/`, `web/` and `monitoring/` are placeholders for later phases. `docker/train.Dockerfile` is the reference environment, where results should be byte-identical.

## Project plan

Approved with Ilian on 2026-10-01. Follow it; propose changes to Ilian rather than deviating silently.

### Decisions
| Area | Choice |
|---|---|
| Data | MovieLens 32M (ratings + tags) + a one-time TMDB metadata snapshot (posters, plot summaries, cast, keywords) |
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
3. Data provenance: MovieLens is downloaded from GroupLens and checked against the SHA256 in `params.yaml`; every downstream output is deterministic, so its hash must match `dvc.lock`. The TMDB snapshot goes to the public DagsHub DVC remote with attribution, so strangers `dvc pull` it and never need a key (re-check TMDB terms when implementing).
4. Provenance on every MLflow run (via `tracked_run`); local SQLite fallback when there are no credentials.
5. One command: `dvc repro` builds data, trains all 5 models, evaluates them and selects the winner. `metrics/*.json` are DVC metrics, so `dvc metrics diff` shows mismatches.
6. CI `repro-check.yml` runs the pipeline twice on a small fixed sample and asserts identical hashes and metrics. Byte-identical results are guaranteed inside Docker; on bare metal across different CPUs, metrics match within a small documented tolerance.

### Data pipeline (DVC stages)
1. `download`: ML-32M zip → verify checksum → `data/raw/`
2. `tmdb_snapshot`: frozen stage that reads `links.csv` and pulls TMDB metadata; normal runs use `dvc pull` instead.
3. `validate`: pandera schemas (ID ranges, ratings 0.5–5, timestamps, no duplicate (user, item) pairs); fail loudly.
4. `preprocess` (all values in params):
   - Keep a recent time window (e.g. 2010+).
   - Ratings ≥ 3.5 become positive implicit interactions; keep the full ratings for analysis.
   - Iterative k-core filter (users ≥ 5, items ≥ 10).
   - Cap the catalog at the ~20k most popular items.
   - Re-index IDs to contiguous integers.
5. `split`: global temporal split. Train runs up to T1, validation covers T1–T2, test is everything after T2. Same split for all models; no future data leaks into training.
6. `features`: TMDB item features (genres, year, posters) for the UI and analysis.

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
- Tables:
  - `profiles`
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
- Posters load from TMDB's image CDN by URL, with attribution in the footer.

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
1. Data: download/validate/preprocess/split stages, TMDB snapshot, EDA notebook
2. Evaluation framework (metrics unit-tested on hand-computed examples) + Popularity and Item-kNN, MLflow logging
3. iALS, EASE, SASRec + Optuna tuning
4. Selection & registry: comparison report, significance tests, model card, export bundle, `champion` alias, CI reproducibility check
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
