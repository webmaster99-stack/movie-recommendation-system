"""Put the dashboards and alert rules in `monitoring/grafana/` into a Grafana.

    uv run --env-file .env python scripts/grafana_sync.py --check

The JSON files in git are the source of truth: running this again overwrites what is in
Grafana with them, so change a dashboard there, export its JSON into the file, and commit.
It creates or updates, in this order:

1. the Postgres data source `recsys-postgres`, when MONITORING_DATABASE_URL is set (the
   read-only `monitoring_reader` connection; Supabase's session pooler address);
2. the folder "Movie recommender";
3. every dashboard in `monitoring/grafana/dashboards/`;
4. the alert rules in `monitoring/grafana/alerts.json`. Where alerts are sent (contact
   points, notification policy) is set in Grafana and left alone.

`--check` then runs every query of every panel and alert through Grafana and fails if one
of them is rejected, which catches a view that changed under a dashboard.

Environment: GRAFANA_URL (https://<stack>.grafana.net) and GRAFANA_TOKEN (a service account
token with the Admin role: Editors may not create data sources or provision alert rules).
The local stack in `monitoring/docker-compose.yml` has no tokens; give GRAFANA_USER and
GRAFANA_PASSWORD instead.

Only the standard library is used, so it runs anywhere without the project installed.
"""

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

GRAFANA_DIR = Path(__file__).resolve().parents[1] / "monitoring" / "grafana"
DATASOURCE_UID = "recsys-postgres"
DATASOURCE_TYPE = "grafana-postgresql-datasource"
FOLDER_UID = "movie-recsys"
FOLDER_TITLE = "Movie recommender"


class GrafanaError(Exception):
    pass


class Grafana:
    def __init__(self, url: str, authorization: str) -> None:
        self.url = url.rstrip("/")
        self.authorization = authorization

    def call(
        self,
        method: str,
        path: str,
        body: Any = None,
        allowed: tuple[int, ...] = (),
        headers: dict[str, str] | None = None,
    ) -> tuple[int, Any]:
        """Status and decoded body. A status of 400 or more raises unless it is `allowed`."""
        request = urllib.request.Request(
            self.url + path,
            method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={
                "Authorization": self.authorization,
                "Content-Type": "application/json",
                "Accept": "application/json",
                **(headers or {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read()
        try:
            decoded = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            decoded = raw.decode(errors="replace")
        if status >= 400 and status not in allowed:
            raise GrafanaError(f"{method} {path} -> {status}: {decoded}")
        return status, decoded


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def datasource_from_url(database_url: str) -> dict[str, Any]:
    """Grafana's description of a Postgres connection given as `postgresql://user:pw@host/db`."""
    parts = urllib.parse.urlsplit(database_url)
    if not (parts.hostname and parts.username and parts.password):
        raise ValueError("the database URL needs a host, a user and a password")
    query = urllib.parse.parse_qs(parts.query)
    return {
        "uid": DATASOURCE_UID,
        "name": "Movie recommender (Postgres, read-only)",
        "type": DATASOURCE_TYPE,
        "access": "proxy",
        "url": f"{parts.hostname}:{parts.port or 5432}",
        "user": urllib.parse.unquote(parts.username),
        "jsonData": {
            "database": parts.path.lstrip("/") or "postgres",
            # Supabase accepts only encrypted connections; the local stack says ?sslmode=disable.
            "sslmode": query.get("sslmode", ["require"])[0],
            "postgresVersion": 1700,
            "maxOpenConns": 3,  # the free database has few connections to spare
            "maxIdleConns": 1,
            "connMaxLifetime": 600,
        },
        "secureJsonData": {"password": urllib.parse.unquote(parts.password)},
    }


def sync_datasource(grafana: Grafana, database_url: str) -> str:
    datasource = datasource_from_url(database_url)
    status, _ = grafana.call("GET", f"/api/datasources/uid/{DATASOURCE_UID}", allowed=(404,))
    if status == 404:
        grafana.call("POST", "/api/datasources", datasource)
        return "created"
    grafana.call("PUT", f"/api/datasources/uid/{DATASOURCE_UID}", datasource)
    return "updated"


def sync_folder(grafana: Grafana) -> str:
    status, _ = grafana.call("GET", f"/api/folders/{FOLDER_UID}", allowed=(404,))
    if status != 404:
        return "exists"
    grafana.call("POST", "/api/folders", {"uid": FOLDER_UID, "title": FOLDER_TITLE})
    return "created"


def sync_dashboard(grafana: Grafana, dashboard: dict[str, Any]) -> str:
    body = {
        "dashboard": dashboard | {"id": None},
        "folderUid": FOLDER_UID,
        "overwrite": True,
        "message": "scripts/grafana_sync.py",
    }
    _, saved = grafana.call("POST", "/api/dashboards/db", body)
    return str(saved["url"])


def sync_alerts(grafana: Grafana, group: dict[str, Any]) -> None:
    """Replace the whole rule group, so a rule removed from the file disappears from Grafana."""
    rules = [
        rule | {"orgID": 1, "folderUID": FOLDER_UID, "ruleGroup": group["title"]}
        for rule in group["rules"]
    ]
    grafana.call(
        "PUT",
        f"/api/v1/provisioning/folder/{FOLDER_UID}/rule-groups/{group['title']}",
        group | {"folderUid": FOLDER_UID, "rules": rules},
        # Without this header Grafana marks the rules as provisioned and locks them in its UI.
        headers={"X-Disable-Provenance": "true"},
    )


def queries(dashboards: list[dict[str, Any]], alerts: dict[str, Any]) -> list[tuple[str, Any]]:
    """(where it is used, query) for every SQL query in the dashboards and the alert rules."""
    found = []
    for dashboard in dashboards:
        for panel in dashboard["panels"]:
            for target in panel["targets"]:
                found.append((f"{dashboard['title']} / {panel['title']}", target))
    for rule in alerts["rules"]:
        for step in rule["data"]:
            if step["datasourceUid"] == DATASOURCE_UID:
                found.append((f"alert / {rule['title']}", step["model"]))
    return found


def check_query(grafana: Grafana, query: dict[str, Any]) -> str | None:
    """Run one query over the last 30 days; the error, or None when Grafana accepts it."""
    body = {
        "from": "now-30d",
        "to": "now",
        "queries": [
            query
            | {
                "datasource": {"type": DATASOURCE_TYPE, "uid": DATASOURCE_UID},
                "intervalMs": 3_600_000,
                "maxDataPoints": 1000,
            }
        ],
    }
    status, answer = grafana.call("POST", "/api/ds/query", body, allowed=(400, 500))
    results = answer.get("results", {}) if isinstance(answer, dict) else {}
    errors = [str(r["error"]) for r in results.values() if r.get("error")]
    if errors or status >= 400:
        return "; ".join(errors) or str(answer)
    return None


def authorization() -> str:
    token = os.environ.get("GRAFANA_TOKEN")
    if token:
        return f"Bearer {token}"
    user, password = os.environ.get("GRAFANA_USER"), os.environ.get("GRAFANA_PASSWORD")
    if user and password:
        return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
    raise GrafanaError("set GRAFANA_TOKEN, or GRAFANA_USER and GRAFANA_PASSWORD")


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="run every query afterwards")
    args = parser.parse_args()

    dashboards = [read_json(path) for path in sorted((GRAFANA_DIR / "dashboards").glob("*.json"))]
    alerts = read_json(GRAFANA_DIR / "alerts.json")
    try:
        url = os.environ.get("GRAFANA_URL")
        if not url:
            raise GrafanaError("set GRAFANA_URL to the address of the Grafana instance")
        grafana = Grafana(url, authorization())

        database_url = os.environ.get("MONITORING_DATABASE_URL")
        if database_url:
            print(f"data source {DATASOURCE_UID}: {sync_datasource(grafana, database_url)}")
        else:
            print(f"data source {DATASOURCE_UID}: left as it is (MONITORING_DATABASE_URL not set)")
        print(f"folder '{FOLDER_TITLE}': {sync_folder(grafana)}")
        for dashboard in dashboards:
            address = url.rstrip("/") + sync_dashboard(grafana, dashboard)
            print(f"dashboard '{dashboard['title']}': {address}")
        sync_alerts(grafana, alerts)
        print(f"alert rules: {', '.join(rule['title'] for rule in alerts['rules'])}")

        if args.check:
            failed = 0
            for where, query in queries(dashboards, alerts):
                error = check_query(grafana, query)
                failed += error is not None
                print(
                    f"  {'FAILED' if error else 'ok':6} {where}" + (f": {error}" if error else "")
                )
            if failed:
                print(f"{failed} queries failed")
                return 1
    except (GrafanaError, urllib.error.URLError, ValueError) as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
