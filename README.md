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

## Results so far (validation split)

Every model ranks the full catalog of 19,642 movies; movies the user has already seen are
excluded. Brackets are 95% bootstrap confidence intervals over users.

| Scenario | Model | NDCG@10 | Recall@10 | MRR@10 | Coverage@10 |
|---|---|---|---|---|---|
| Warm (5,795 returning users) | Popularity | 0.063 [0.059, 0.066] | 0.026 | 0.128 | 0.9% |
| | Item-kNN | **0.081** [0.077, 0.085] | **0.034** | **0.159** | **3.1%** |
| Onboarding (3,894 new users, first 10 likes as history) | Popularity | 0.312 [0.304, 0.321] | 0.055 | 0.491 | 0.1% |
| | Item-kNN | **0.369** [0.361, 0.378] | **0.071** | **0.564** | **4.5%** |

Hyperparameters are untuned starting values; tuning and the test split come in later phases.
Full numbers, including @20, novelty and results by history length: `uv run dvc metrics show`.

```bash
uv run dvc repro                     # data -> train -> evaluate, all models
uv run dvc repro evaluate@item_knn   # a single model
```

## Build phases

- [x] 0. Foundation: tooling, config, determinism, CI
- [x] 1. Data pipeline (DVC)
- [x] 2. Evaluation framework + baselines (Popularity, Item-kNN)
- [ ] 3. iALS, EASE, SASRec + tuning
- [ ] 4. Model selection + registry
- [ ] 5. API
- [ ] 6. Frontend
- [ ] 7. Deployment + CI/CD
- [ ] 8. Monitoring
- [ ] 9. Retraining loop + polish
