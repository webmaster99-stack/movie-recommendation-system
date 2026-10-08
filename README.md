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
excluded. Brackets are 95% bootstrap confidence intervals over users. Hyperparameters are
tuned on this split (see below), so the numbers are optimistic; the untouched test split is
evaluated once, in Phase 4.

| Scenario | Model | NDCG@10 | Recall@10 | MRR@10 | Coverage@10 | Size |
|---|---|---|---|---|---|---|
| Warm (5,795 returning users) | Popularity (30-day half-life) | **0.124** [0.119, 0.128] | **0.063** | **0.278** | 0.6% | 0.1 MB |
| | Item-kNN | 0.084 [0.079, 0.088] | 0.035 | 0.166 | 3.4% | 131 MB |
| | iALS | 0.096 [0.092, 0.100] | 0.041 | 0.192 | 7.9% | 9 MB |
| | EASE | 0.106 [0.101, 0.110] | 0.046 | 0.206 | **8.2%** | 154 MB |
| | SASRec | 0.065 [0.062, 0.068] | 0.029 | 0.137 | 8.1% | 5 MB |
| Onboarding (3,894 new users, first 10 likes as history) | Popularity (30-day half-life) | 0.293 [0.285, 0.301] | 0.054 | 0.409 | 0.1% | |
| | Item-kNN | 0.373 [0.365, 0.383] | 0.072 | 0.568 | 4.3% | |
| | iALS | 0.365 [0.357, 0.374] | 0.072 | 0.579 | 4.3% | |
| | EASE | **0.377** [0.369, 0.385] | **0.077** | **0.586** | **6.6%** | |
| | SASRec | 0.303 [0.296, 0.312] | 0.058 | 0.491 | 5.6% | |

What the table says:

- **EASE is the best personalised model** in both scenarios, ahead of iALS and Item-kNN.
- **Recency beats personalisation for returning users.** Popularity that halves an
  interaction's weight every 30 days tops the warm scenario: in 2022 people mostly rated what
  had just come out, and none of the personalised models look at dates yet.
- **SASRec is limited by compute, not by design.** It trains on a laptop CPU at about two
  minutes per epoch, so it gets 10 epochs and its score was still rising when training stopped.
- EASE is stored with the 1,000 strongest weights per movie instead of the full 1.5 GB matrix.

Full numbers, including @20, novelty and results by history length: `uv run dvc metrics show`.

```bash
uv run dvc repro                     # data -> train -> evaluate, all models
uv run dvc repro evaluate@item_knn   # a single model
```

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
```

Tuning is not part of `dvc repro`: the pipeline reads the tuned values from `params.yaml`, so
rebuilding it never repeats a search.

## Build phases

- [x] 0. Foundation: tooling, config, determinism, CI
- [x] 1. Data pipeline (DVC)
- [x] 2. Evaluation framework + baselines (Popularity, Item-kNN)
- [x] 3. iALS, EASE, SASRec + tuning
- [ ] 4. Model selection + registry
- [ ] 5. API
- [ ] 6. Frontend
- [ ] 7. Deployment + CI/CD
- [ ] 8. Monitoring
- [ ] 9. Retraining loop + polish
