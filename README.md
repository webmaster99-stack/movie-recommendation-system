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

Optional: copy `.env.example` to `.env` to log runs to DagsHub instead of the local `mlflow.db`.

## Build phases

- [x] 0. Foundation: tooling, config, determinism, CI
- [ ] 1. Data pipeline (DVC)
- [ ] 2. Evaluation framework + baselines
- [ ] 3. iALS, EASE, SASRec + tuning
- [ ] 4. Model selection + registry
- [ ] 5. API
- [ ] 6. Frontend
- [ ] 7. Deployment + CI/CD
- [ ] 8. Monitoring
- [ ] 9. Retraining loop + polish
