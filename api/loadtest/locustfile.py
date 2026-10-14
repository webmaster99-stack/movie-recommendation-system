"""Load test: simulated users who sign up, rate movies and ask for recommendations.

Run the API with a shared-secret (HS256) verifier and a throwaway database, so the test can
mint its own users and never touches Supabase or TMDB:

    DATABASE_URL=sqlite:///loadtest.db SUPABASE_JWT_SECRET=<32+ characters> TMDB_API_KEY= \
        SUPABASE_URL= uv run uvicorn recsys_api.main:create_app --factory --app-dir api

    LOADTEST_JWT_SECRET=<the same secret> uvx --with pyjwt locust \
        -f api/loadtest/locustfile.py \
        --headless --host http://127.0.0.1:8000 --users 20 --spawn-rate 5 --run-time 2m

`SUPABASE_URL=` (empty) matters when `.env` names a project: with one set, tokens must carry
its issuer, which the made-up ones here do not.

Locust is run with `uvx` instead of being a project dependency: it is only needed here, and
would add gevent and a web UI to the lock file. `--with pyjwt` gives it the one library this
file needs.
"""

import os
import random
import time
import uuid

import jwt
from locust import HttpUser, between, task

SECRET = os.environ["LOADTEST_JWT_SECRET"]
QUERIES = ["matrix", "star", "the", "love", "godfather", "toy story", "night", "a", "amelie"]
RATINGS = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.0, 4.5, 5.0, 5.0]


class Viewer(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        claims = {"sub": str(uuid.uuid4()), "aud": "authenticated", "exp": int(time.time()) + 3600}
        token = jwt.encode(claims, SECRET, algorithm="HS256")
        self.auth = {"Authorization": f"Bearer {token}"}
        self.last_request: str | None = None
        self.shown: list[int] = []
        # Sign-up: rate ten of the onboarding movies.
        page = self.client.get("/onboarding/movies", params={"limit": 60}).json()
        self.known = [movie["movie_id"] for movie in page["movies"]]
        for movie_id in random.sample(self.known, page["min_ratings"]):
            self.rate(movie_id)

    def rate(self, movie_id: int, request_id: str | None = None) -> None:
        body = {"movie_id": movie_id, "rating": random.choice(RATINGS), "request_id": request_id}
        self.client.post("/ratings", json=body, headers=self.auth)

    @task(5)
    def recommendations(self) -> None:
        body = self.client.get("/recommendations", params={"k": 20}, headers=self.auth).json()
        self.last_request = body["request_id"]
        self.shown = [movie["movie_id"] for movie in body["items"]]

    @task(3)
    def click_and_rate_a_recommendation(self) -> None:
        if not self.shown:
            return
        position = random.randrange(len(self.shown))
        movie_id = self.shown[position]
        event = {"event_type": "click", "movie_id": movie_id, "position": position}
        event["request_id"] = self.last_request
        self.client.post("/events", json=event, headers=self.auth)
        self.client.get(f"/movies/{movie_id}", name="/movies/[id]")
        self.client.get(f"/movies/{movie_id}/similar", name="/movies/[id]/similar")
        self.rate(movie_id, self.last_request)

    @task(3)
    def search(self) -> None:
        self.client.get("/movies/search", params={"q": random.choice(QUERIES)})

    @task(1)
    def my_ratings(self) -> None:
        self.client.get("/ratings", headers=self.auth)

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
