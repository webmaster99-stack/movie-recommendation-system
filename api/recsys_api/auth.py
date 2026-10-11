"""Who is calling: verification of the Supabase access token sent as `Authorization: Bearer`.

The frontend signs users in with Supabase and forwards the access token, a JWT. The API
never sees a password; it checks the token's signature, expiry, audience and issuer, and
takes the user's id from the `sub` claim.

Supabase signs with an asymmetric key (ES256 or RS256) whose public half is published at
`<project>/auth/v1/.well-known/jwks.json`. Older projects sign with a shared secret (HS256).
The key is chosen by the token's algorithm, and each kind of key is only ever used with its
own kind of algorithm, so a token can't pick a weaker check than the project uses.
"""

import uuid
from typing import Any, Protocol

import jwt

_ASYMMETRIC = ("ES256", "RS256")
_AUDIENCE = "authenticated"


class InvalidTokenError(Exception):
    pass


class SigningKeys(Protocol):
    def get_signing_key_from_jwt(self, token: str) -> Any: ...


class TokenVerifier:
    def __init__(
        self,
        supabase_url: str = "",
        jwt_secret: str = "",
        signing_keys: SigningKeys | None = None,
    ) -> None:
        base = supabase_url.rstrip("/")
        self.issuer = f"{base}/auth/v1" if base else None
        self.secret = jwt_secret
        self.signing_keys = signing_keys
        if signing_keys is None and base:
            # Keys are fetched on first use and cached; a token signed with an unknown key
            # id triggers one refetch, which is how key rotation is picked up.
            self.signing_keys = jwt.PyJWKClient(f"{self.issuer}/.well-known/jwks.json")

    def user_id(self, token: str) -> uuid.UUID:
        try:
            algorithm = jwt.get_unverified_header(token).get("alg")
            key: Any
            if algorithm == "HS256" and self.secret:
                key = self.secret
            elif algorithm in _ASYMMETRIC and self.signing_keys is not None:
                key = self.signing_keys.get_signing_key_from_jwt(token).key
            else:
                raise InvalidTokenError(f"tokens signed with {algorithm} are not accepted")
            claims = jwt.decode(
                token,
                key,
                algorithms=[algorithm],
                audience=_AUDIENCE,
                issuer=self.issuer,
                options={"require": ["exp", "sub"]},
            )
            return uuid.UUID(claims["sub"])
        except (jwt.PyJWTError, ValueError) as error:
            raise InvalidTokenError(str(error)) from error
