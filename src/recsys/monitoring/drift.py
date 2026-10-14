"""The drift report: is the app being used the way the model was trained?

    uv run python -m recsys.monitoring.drift --out site

Production data from the last `monitoring.drift.window_days` days is read from the
`monitoring` views of the app's database (`MONITORING_DATABASE_URL`, the read-only
`monitoring_reader` role) and compared, with Evidently, against the reference written by the
`monitoring_reference` stage. Five comparisons, each a page of its own:

- `ratings`: the stars given, and the release year and popularity of the movies rated.
- `rating_genres`: the genres of the movies rated.
- `users`: per user, how many movies they rated and liked: the length of the history the
  model gets, which is the first thing to differ between a research site and a new app.
- `served`, `served_genres`: the movies recommended in the app against the movies the model
  recommends to the users it was trained on.

Next to those, the online quality of the lists served in the window: click-through rate, hit
rate, coverage and popularity. The same numbers are on the Grafana dashboard; here they are
kept week by week in MLflow.

This is a command, not a DVC stage: it reads live data, so its output is different every
week by design and belongs in MLflow and on the report page, never in `metrics/`.
"""

import argparse
import html
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sqlalchemy import Engine, create_engine, text

from recsys.utils.config import PROJECT_ROOT, load_params
from recsys.utils.paths import MONITORING_REFERENCE_DIR
from recsys.utils.seed import make_deterministic

DATABASE_URL_VAR = "MONITORING_DATABASE_URL"
NOT_ENOUGH, OK, DRIFT = "not enough data", "ok", "drift"

# name -> (title, what a row is, numerical columns, categorical columns)
SECTIONS: dict[str, tuple[str, str, list[str], list[str]]] = {
    "ratings": (
        "Ratings",
        "one row per rating",
        ["rating", "movie_year", "popularity_percentile"],
        [],
    ),
    "rating_genres": ("Genres rated", "one row per rating and genre of its movie", [], ["genre"]),
    "users": ("Users", "one row per user", ["n_ratings", "n_liked", "like_share"], []),
    "served": (
        "Recommendations served",
        "one row per recommended movie",
        ["movie_year", "popularity_percentile"],
        [],
    ),
    "served_genres": (
        "Genres served",
        "one row per recommended movie and genre",
        [],
        ["genre"],
    ),
}


@dataclass(frozen=True)
class Reference:
    """The tables written by the `monitoring_reference` stage."""

    catalog: pd.DataFrame  # movie_id, year, genres, popularity_percentile
    ratings: pd.DataFrame  # movie_id, rating
    users: pd.DataFrame  # n_ratings, n_liked
    served: pd.DataFrame  # movie_id, position
    meta: dict[str, Any]


@dataclass(frozen=True)
class Production:
    """What the monitoring views returned for the window."""

    ratings: pd.DataFrame  # user_key, movie_id, rating, in_window: all ratings of active users
    served: pd.DataFrame  # movie_id, position
    recommendations: pd.DataFrame  # one row per list: see `online_quality`


def load_reference(path: Path = MONITORING_REFERENCE_DIR) -> Reference:
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(
            f"no reference in '{path}': run `uv run dvc pull data/monitoring/reference` "
            "(or `uv run dvc repro monitoring_reference`)"
        )
    return Reference(
        catalog=pd.read_parquet(path / "catalog.parquet"),
        ratings=pd.read_parquet(path / "ratings.parquet"),
        users=pd.read_parquet(path / "users.parquet"),
        served=pd.read_parquet(path / "served.parquet"),
        meta=json.loads((path / "meta.json").read_text(encoding="utf-8")),
    )


def connect(database_url: str) -> Engine:
    # Supabase hands out `postgresql://` URLs; SQLAlchemy needs the driver named. Its pooler
    # can't keep prepared statements.
    for prefix in ("postgresql://", "postgres://"):
        if database_url.startswith(prefix):
            database_url = "postgresql+psycopg://" + database_url.removeprefix(prefix)
    return create_engine(database_url, connect_args={"prepare_threshold": None})


def load_production(engine: Engine, since: datetime) -> Production:
    window = {"since": since}
    with engine.connect() as connection:
        # Every rating of the users who rated something in the window: a user's history is
        # all they have rated, not just this month's part of it.
        ratings = pd.read_sql_query(
            text(
                "select user_key, movie_id, rating::float8 as rating, "
                "updated_at >= :since as in_window "
                "from monitoring.ratings where user_key in "
                "(select user_key from monitoring.ratings where updated_at >= :since)"
            ),
            connection,
            params=window,
        )
        served = pd.read_sql_query(
            text(
                "select movie_id, position from monitoring.served_movies where created_at >= :since"
            ),
            connection,
            params=window,
        )
        recommendations = pd.read_sql_query(
            text(
                "select user_key, strategy, n_items, n_clicks, "
                "best_rating_after::float8 as best_rating_after, popularity_percentile, "
                "catalog_size from monitoring.recommendations where created_at >= :since"
            ),
            connection,
            params=window,
        )
    return Production(ratings, served, recommendations)


def user_table(ratings: pd.DataFrame, positive_threshold: float) -> pd.DataFrame:
    """Per `user_key`: movies rated and movies liked, as in the reference's `users`."""
    liked = (ratings["rating"] >= positive_threshold).rename("liked")
    per_user = liked.groupby(ratings["user_key"], sort=True).agg(["size", "sum"])
    return pd.DataFrame(
        {"n_ratings": per_user["size"].to_numpy(), "n_liked": per_user["sum"].to_numpy()}
    )


def _movies(movie_ids: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """The rows whose movie is in the catalog, with the catalog's description of it."""
    described = movie_ids.merge(catalog, on="movie_id", how="inner")
    return described.assign(movie_year=described["year"].astype("float64"))


def _genres(movies: pd.DataFrame) -> pd.DataFrame:
    genres = movies["genres"].explode().dropna()  # a movie without genres has no rows
    return pd.DataFrame({"genre": genres.astype(str).to_numpy()})


def build_frames(
    ratings: pd.DataFrame, users: pd.DataFrame, served: pd.DataFrame, catalog: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """The five tables of `SECTIONS`, built the same way for training and for production."""
    rated, shown = _movies(ratings, catalog), _movies(served, catalog)
    return {
        "ratings": rated[["rating", "movie_year", "popularity_percentile"]],
        "rating_genres": _genres(rated),
        "users": users.assign(like_share=users["n_liked"] / users["n_ratings"])[
            ["n_ratings", "n_liked", "like_share"]
        ].astype("float64"),
        "served": shown[["movie_year", "popularity_percentile"]],
        "served_genres": _genres(shown),
    }


def drift_section(
    name: str, reference: pd.DataFrame, current: pd.DataFrame, cfg: dict[str, Any], out_dir: Path
) -> dict[str, Any]:
    """Compare one table; write its Evidently page when there is enough data to compare."""
    _, _, numerical, categorical = SECTIONS[name]
    result: dict[str, Any] = {"n_reference": len(reference), "n_current": len(current)}
    if len(current) < cfg["min_rows"]:
        return result | {"status": NOT_ENOUGH, "columns": {}}

    definition = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    threshold = cfg["threshold"]
    preset = DataDriftPreset(
        num_method=cfg["num_method"], cat_method=cfg["cat_method"], threshold=threshold
    )
    snapshot = Report([preset]).run(
        Dataset.from_pandas(current.reset_index(drop=True), data_definition=definition),
        Dataset.from_pandas(reference.reset_index(drop=True), data_definition=definition),
    )
    snapshot.save_html(str(out_dir / f"{name}.html"))
    distances = {
        metric["config"]["column"]: float(metric["value"])
        for metric in snapshot.dict()["metrics"]
        if metric["config"]["type"].endswith(":ValueDrift")
    }
    # Both methods are distances: the larger, the further apart the two distributions.
    columns = {
        column: {"distance": round(distance, 4), "drifted": distance >= threshold}
        for column, distance in distances.items()
    }
    drifted = any(column["drifted"] for column in columns.values())
    return result | {"status": DRIFT if drifted else OK, "columns": columns, "page": f"{name}.html"}


def online_quality(
    recommendations: pd.DataFrame, served: pd.DataFrame, positive_threshold: float
) -> dict[str, Any]:
    """What happened to the lists served in the window. Rates are shares of all lists."""
    n_lists = len(recommendations)
    if n_lists == 0:
        return {"n_lists": 0}
    rated_after = recommendations["best_rating_after"]
    catalog_size = recommendations["catalog_size"].max()
    popularity = recommendations["popularity_percentile"].mean()
    return {
        "n_lists": n_lists,
        "n_users": int(recommendations["user_key"].nunique()),
        "personalised_share": float((recommendations["strategy"] == "personalised").mean()),
        # A list counts once, however many of its movies were clicked.
        "click_through_rate": float((recommendations["n_clicks"] > 0).mean()),
        # The user later rated a movie from the list / rated one high enough to be a like.
        "hit_rate": float(rated_after.notna().mean()),
        "liked_hit_rate": float((rated_after >= positive_threshold).mean()),
        # Share of the catalog that appeared in at least one list.
        "coverage": None
        if pd.isna(catalog_size)
        else float(served["movie_id"].nunique() / catalog_size),
        "popularity_percentile": None if pd.isna(popularity) else float(popularity),
    }


def build_report(
    reference: Reference,
    production: Production,
    cfg: dict[str, Any],
    out_dir: Path,
    generated_at: datetime,
) -> dict[str, Any]:
    """Write the report into `out_dir` and return its summary (also `summary.json` there)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = reference.meta["positive_threshold"]
    recent = production.ratings[production.ratings["in_window"]]
    reference_frames = build_frames(
        reference.ratings, reference.users, reference.served, reference.catalog
    )
    current_frames = build_frames(
        recent[["movie_id", "rating"]],
        user_table(production.ratings, threshold),
        production.served,
        reference.catalog,
    )
    sections = {
        name: drift_section(name, reference_frames[name], current_frames[name], cfg, out_dir)
        for name in SECTIONS
    }
    statuses = {section["status"] for section in sections.values()}
    summary = {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "window_days": cfg["window_days"],
        "methods": {key: cfg[key] for key in ("num_method", "cat_method", "threshold")},
        "min_rows": cfg["min_rows"],
        "reference": reference.meta,
        "status": DRIFT if DRIFT in statuses else OK if OK in statuses else NOT_ENOUGH,
        "sections": sections,
        "online": online_quality(production.recommendations, production.served, threshold),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (out_dir / "index.html").write_text(index_page(summary), encoding="utf-8", newline="\n")
    return summary


def flat_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """The summary's numbers under MLflow-style names."""
    metrics = {
        f"online.{name}": float(value)
        for name, value in summary["online"].items()
        if value is not None
    }
    for name, section in summary["sections"].items():
        metrics[f"drift.{name}.n_current"] = float(section["n_current"])
        for column, result in section["columns"].items():
            metrics[f"drift.{name}.{column}"] = result["distance"]
    drifted = [c["drifted"] for s in summary["sections"].values() for c in s["columns"].values()]
    return metrics | {"drift.n_columns_drifted": float(sum(drifted))}


_ONLINE_LABELS = {
    "n_lists": "Lists served",
    "n_users": "Users served",
    "personalised_share": "Personalised (the rest: popular fallback)",
    "click_through_rate": "Click-through rate (lists with a click)",
    "hit_rate": "Hit rate (a listed movie rated afterwards)",
    "liked_hit_rate": "Liked hit rate (rated high enough to count as a like)",
    "coverage": "Catalog coverage (share of movies shown at least once)",
    "popularity_percentile": "Mean popularity place (0 = most popular movie, 1 = least)",
}


def _number(name: str, value: Any) -> str:
    if value is None:
        return "–"
    if name.startswith("n_"):
        return f"{value:,}"
    return f"{value:.3f}" if name == "popularity_percentile" else f"{value:.1%}"


def _columns_text(section: dict[str, Any]) -> str:
    if section["status"] == NOT_ENOUGH:
        return "–"
    return ", ".join(
        f"{column} {result['distance']:.3f}" + (" (drifted)" if result["drifted"] else "")
        for column, result in section["columns"].items()
    )


def markdown_summary(summary: dict[str, Any]) -> str:
    """For the workflow's run page."""
    lines = [
        f"### Drift report: {summary['status']}",
        "",
        f"Last {summary['window_days']} days against the training reference of "
        f"`{summary['reference']['model']}`.",
        "",
        "| Comparison | Reference rows | Production rows | Status | Distances |",
        "|---|---|---|---|---|",
    ]
    for name, section in summary["sections"].items():
        lines.append(
            f"| {SECTIONS[name][0]} | {section['n_reference']:,} | {section['n_current']:,} "
            f"| {section['status']} | {_columns_text(section)} |"
        )
    lines += ["", "| Online quality | |", "|---|---|"]
    lines += [
        f"| {_ONLINE_LABELS[name]} | {_number(name, value)} |"
        for name, value in summary["online"].items()
    ]
    return "\n".join(lines) + "\n"


_PAGE_STYLE = """
:root { --bg: #fbfaf8; --fg: #1d1a21; --muted: #6b6572; --line: #e2dde6; --ok: #1f7a4d;
  --drift: #b3401d; --none: #6b6572; --link: #6b3fa0; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #17131c; --fg: #ece8f1; --muted: #a59dae; --line: #332c3c; --ok: #5fcf96;
    --drift: #ff9670; --none: #a59dae; --link: #c7a6f5; } }
body { background: var(--bg); color: var(--fg); margin: 0 auto; max-width: 60rem;
  padding: 2rem 1rem 4rem; font: 16px/1.55 system-ui, sans-serif; }
h1 { font-size: 1.7rem; margin: 0 0 .25rem; } h2 { font-size: 1.15rem; margin: 2.2rem 0 .6rem; }
p { margin: .5rem 0; max-width: 46rem; } .muted { color: var(--muted); } a { color: var(--link); }
.scroll { overflow-x: auto; } table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: .5rem .75rem .5rem 0; border-bottom: 1px solid var(--line);
  vertical-align: top; } th { font-size: .8rem; color: var(--muted); font-weight: 600; }
td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
.status { font-weight: 600; white-space: nowrap; } .ok { color: var(--ok); }
.drift { color: var(--drift); } .none { color: var(--none); }
"""
_STATUS_CLASS = {OK: "ok", DRIFT: "drift", NOT_ENOUGH: "none"}


def index_page(summary: dict[str, Any]) -> str:
    """The report's front page: a table of the comparisons linking to Evidently's pages."""
    methods, reference = summary["methods"], summary["reference"]
    rows = []
    for name, section in summary["sections"].items():
        title, unit, _, _ = SECTIONS[name]
        label = html.escape(title)
        if "page" in section:
            label = f'<a href="{section["page"]}">{label}</a>'
        status = section["status"]
        rows.append(
            f'<tr><td>{label}<br><span class="muted">{html.escape(unit)}</span></td>'
            f'<td class="num">{section["n_reference"]:,}</td>'
            f'<td class="num">{section["n_current"]:,}</td>'
            f'<td class="status {_STATUS_CLASS[status]}">{html.escape(status)}</td>'
            f"<td>{html.escape(_columns_text(section))}</td></tr>"
        )
    online = "".join(
        f'<tr><td>{html.escape(_ONLINE_LABELS[name])}</td><td class="num">'
        f"{html.escape(_number(name, value))}</td></tr>"
        for name, value in summary["online"].items()
    )
    status = summary["status"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drift report</title>
<style>{_PAGE_STYLE}</style>
</head>
<body>
<h1>Drift report: <span class="{_STATUS_CLASS[status]}">{html.escape(status)}</span></h1>
<p class="muted">Generated {html.escape(summary["generated_at"])} from the last
{summary["window_days"]} days of the app's data.</p>
<p>Is the app being used the way the model was trained? Production is compared with
<code>{html.escape(reference["model"])}</code>'s training data: MovieLens ratings from
{html.escape(reference["period"]["start"])} up to {html.escape(reference["period"]["end"])},
and the lists the model recommends to those users.</p>

<h2>Comparisons</h2>
<div class="scroll"><table>
<tr><th>Comparison</th><th>Reference rows</th><th>Production rows</th><th>Status</th>
<th>Distance per column</th></tr>
{"".join(rows)}
</table></div>
<p class="muted">A column has drifted when its distance is at least {methods["threshold"]}.
Numbers are compared by {html.escape(methods["num_method"])} distance (in standard deviations
of the reference), categories by {html.escape(methods["cat_method"])} distance (0 = the same
mix, 1 = nothing in common). A comparison with fewer than {summary["min_rows"]} production
rows is not made.</p>

<h2>Online quality in the window</h2>
<div class="scroll"><table>{online}</table></div>
<p class="muted">Rates are shares of the lists served. The numbers of every week are kept as
runs named <code>monitoring-drift</code> in MLflow.</p>
</body>
</html>
"""


def log_to_mlflow(summary: dict[str, Any], out_dir: Path) -> None:
    import mlflow

    from recsys.utils.mlflow_utils import tracked_run

    with tracked_run("monitoring-drift", tags={"stage": "monitoring", "status": summary["status"]}):
        mlflow.log_params(summary["methods"] | {"window_days": summary["window_days"]})
        mlflow.log_metrics(flat_metrics(summary))
        mlflow.log_artifacts(str(out_dir), artifact_path="drift_report")


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=Path("site"), help="directory for the report")
    parser.add_argument("--summary", type=Path, help="append a Markdown summary to this file")
    parser.add_argument("--no-mlflow", action="store_true", help="write the report only")
    args = parser.parse_args()

    params = load_params()
    cfg = params["monitoring"]["drift"]
    make_deterministic(params["seed"], params["runtime"]["num_threads"])
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.environ.get(DATABASE_URL_VAR)
    if not database_url:
        print(f"{DATABASE_URL_VAR} is not set: the report needs the app's database")
        return 1

    reference = load_reference()
    generated_at = datetime.now(UTC)
    since = generated_at - timedelta(days=cfg["window_days"])
    production = load_production(connect(database_url), since)
    summary = build_report(reference, production, cfg, args.out, generated_at)

    text_summary = markdown_summary(summary)
    print(text_summary)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as file:
            file.write(text_summary)
    if not args.no_mlflow:
        log_to_mlflow(summary, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
