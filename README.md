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

## Build phases

- [x] 0. Foundation: tooling, config, determinism, CI
- [x] 1. Data pipeline (DVC)
- [ ] 2. Evaluation framework + baselines
- [ ] 3. iALS, EASE, SASRec + tuning
- [ ] 4. Model selection + registry
- [ ] 5. API
- [ ] 6. Frontend
- [ ] 7. Deployment + CI/CD
- [ ] 8. Monitoring
- [ ] 9. Retraining loop + polish
