"""The retraining loop: the app's own ratings, versioned with DVC and added to the model's fit.

1. `snapshot` (a command: it reads the live database) copies the ratings to
   `data/app_feedback/`, which is then versioned with `dvc add`.
2. `prepare` (stage `app_feedback`) turns them into likes in the model's index and holds out
   each app user's latest likes for the promotion gate.
3. `retrain` (stage `retrain`) fits the selected model on MovieLens plus the app's likes;
   `export` and `register` then make it the new challenger.

Whether the challenger replaces the champion is decided by `recsys.registry.promote`.
"""
