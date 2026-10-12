"""Wait until an API's `/health` reports the expected build, or fail.

    uv run python scripts/wait_for_health.py https://<service>.onrender.com \
        --model-version 1 --commit <sha> --timeout 900

The deploy workflow runs it twice: against the image it has just built, and against the
deployed service. Render keeps the previous instance answering until the new one is healthy,
so "it answers" proves nothing; the commit and the model version in the answer do.

Only the standard library is used, so it runs anywhere without the project installed.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def fetch_health(base_url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout) as response:
        body: dict[str, Any] = json.load(response)
    return body


def mismatches(health: dict[str, Any], expected: dict[str, str]) -> list[str]:
    """One line per expected value that `/health` does not report."""
    return [
        f"{key} is {health.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if str(health.get(key)) != value
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="the API's address, without /health")
    parser.add_argument("--model-version")
    parser.add_argument("--commit")
    parser.add_argument("--timeout", type=float, default=300, help="seconds to keep trying")
    parser.add_argument("--interval", type=float, default=10, help="seconds between attempts")
    args = parser.parse_args()
    expected = {
        key: value
        for key, value in (("model_version", args.model_version), ("commit", args.commit))
        if value is not None
    }

    deadline = time.monotonic() + args.timeout
    while True:
        try:
            health = fetch_health(args.url, timeout=30)
            problems = mismatches(health, expected)
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as error:
            problems = [f"no answer ({type(error).__name__}: {error})"]
        if not problems:
            print(f"healthy: {json.dumps(health)}")
            return 0
        print("; ".join(problems), flush=True)
        if time.monotonic() >= deadline:
            print(f"gave up after {args.timeout:.0f} s", file=sys.stderr)
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
