"""Write the API's OpenAPI schema to `web/openapi.json`, from which the frontend's types are
generated.

    uv run python scripts/export_openapi.py
    cd web && npm run gen:api

Run both after changing an endpoint or `recsys_api/schemas.py`. The schema is committed so
that the frontend builds without the API running; `api/tests/test_openapi.py` fails when it
no longer matches the code.
"""

import json
import sys
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "api"))

from recsys_api.bundle import Bundle  # noqa: E402
from recsys_api.config import Settings  # noqa: E402
from recsys_api.db import make_engine  # noqa: E402
from recsys_api.main import create_app  # noqa: E402

OPENAPI_PATH = PROJECT_ROOT / "web" / "openapi.json"


def openapi_schema() -> dict[str, Any]:
    # The schema depends only on the routes, so no model is loaded: the bundle is read
    # when a request is served, never while the app is put together.
    app = create_app(
        Settings(_env_file=None),
        bundle=cast(Bundle, None),
        engine=make_engine("sqlite://"),
    )
    return app.openapi()


def main() -> None:
    text = json.dumps(openapi_schema(), indent=2, sort_keys=True) + "\n"
    OPENAPI_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OPENAPI_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
