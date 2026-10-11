"""What the endpoints depend on: the loaded services, a database session, the caller's id."""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session, sessionmaker

from recsys_api.auth import InvalidTokenError, TokenVerifier
from recsys_api.bundle import Bundle
from recsys_api.config import Settings
from recsys_api.tmdb import MetadataCache


@dataclass(frozen=True)
class Services:
    """Built once in `create_app` and shared by every request."""

    settings: Settings
    bundle: Bundle
    sessions: sessionmaker[Session]
    verifier: TokenVerifier
    metadata: MetadataCache


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def get_session(services: ServicesDep) -> Iterator[Session]:
    with services.sessions() as session:
        yield session


def get_user_id(
    services: ServicesDep,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(HTTPBearer(auto_error=False))
    ],
) -> uuid.UUID:
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "a valid Supabase access token is required",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise unauthorized
    try:
        return services.verifier.user_id(credentials.credentials)
    except InvalidTokenError as error:
        raise unauthorized from error


SessionDep = Annotated[Session, Depends(get_session)]
UserDep = Annotated[uuid.UUID, Depends(get_user_id)]
