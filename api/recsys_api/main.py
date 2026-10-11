"""Application factory.

    uv run uvicorn recsys_api.main:create_app --factory --app-dir api --reload

Everything slow (reading the 77 MB model, opening the database pool) happens once here, so
a request never pays for it. Tests pass their own pieces in place of the real ones.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from recsys_api.auth import TokenVerifier
from recsys_api.bundle import Bundle, load_bundle
from recsys_api.config import Settings
from recsys_api.db import make_engine
from recsys_api.deps import Services
from recsys_api.routes import router
from recsys_api.tmdb import MetadataCache, TmdbClient


def create_app(
    settings: Settings | None = None,
    *,
    bundle: Bundle | None = None,
    engine: Engine | None = None,
    verifier: TokenVerifier | None = None,
    metadata: MetadataCache | None = None,
) -> FastAPI:
    settings = settings or Settings()
    if metadata is None:
        client = TmdbClient(settings.tmdb_api_key) if settings.tmdb_api_key else None
        metadata = MetadataCache(client, settings.tmdb_refresh_days, settings.tmdb_max_age_days)

    app = FastAPI(
        title="Movie recommender API",
        description="This product uses the TMDB API but is not endorsed or certified by TMDB.",
    )
    app.state.services = Services(
        settings=settings,
        bundle=bundle or load_bundle(settings.bundle_dir),
        sessions=sessionmaker(engine or make_engine(settings.database_url)),
        verifier=verifier or TokenVerifier(settings.supabase_url, settings.supabase_jwt_secret),
        metadata=metadata,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.include_router(router)
    return app
