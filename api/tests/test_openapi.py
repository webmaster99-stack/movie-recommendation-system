"""The frontend's types are generated from `web/openapi.json`, a committed copy of the API's
schema. This test keeps the copy in step with the code."""

import json

from fastapi.testclient import TestClient

from recsys.utils.config import PROJECT_ROOT


def test_committed_schema_matches_the_api(client: TestClient) -> None:
    committed = json.loads((PROJECT_ROOT / "web" / "openapi.json").read_text(encoding="utf-8"))
    assert client.app.openapi() == committed, (  # type: ignore[attr-defined]
        "the API changed: run `uv run python scripts/export_openapi.py` "
        "and then `npm run gen:api` in web/"
    )
