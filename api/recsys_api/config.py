"""Runtime settings, read from environment variables (and `.env` when present).

Unlike the pipeline, the API has no `params.yaml`: these are deployment settings (where the
database is, which origins may call), not values that change a result. Anything that does
come from training, such as the rating that counts as a like, is read from the bundle.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # `model_version` would otherwise clash with pydantic's reserved `model_` prefix.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    bundle_dir: Path = Path("bundle")
    # The registry version of the bundle. It is not inside the bundle (which must hash the
    # same on every build), so the image build passes it in.
    model_version: str = "dev"
    # The commit the image was built from, passed in the same way. `/health` reports both,
    # which is how a deploy checks that the new image is the one answering.
    git_commit: str = "dev"

    # Postgres (`postgresql://...`, Supabase's session pooler on Render) or, for local
    # development, `sqlite:///api.db`, whose tables are created on start.
    database_url: str = "sqlite:///api.db"

    # Tokens are checked against the project's public signing keys at
    # `<supabase_url>/auth/v1/.well-known/jwks.json`. Projects that still sign with the
    # legacy shared secret (HS256) need `supabase_jwt_secret` instead.
    supabase_url: str = ""
    supabase_jwt_secret: str = ""

    tmdb_api_key: str = ""
    # TMDB's terms forbid keeping their data for more than 6 months. Rows are refreshed
    # after `tmdb_refresh_days` and deleted after `tmdb_max_age_days` at the latest.
    tmdb_refresh_days: int = 150
    tmdb_max_age_days: int = 180

    # Every answered request is logged for the dashboards; rows older than this are deleted.
    request_log_days: int = 90

    cors_origins: str = "http://localhost:3000"  # comma-separated
    # Ratings asked for at sign-up; the evaluation's onboarding scenario uses the same number.
    onboarding_min_ratings: int = 10

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
