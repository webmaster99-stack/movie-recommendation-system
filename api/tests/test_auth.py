import time
import uuid
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from recsys_api.auth import InvalidTokenError, TokenVerifier

SECRET = "a-test-secret-that-is-long-enough-for-hs256"
USER = uuid.UUID("00000000-0000-4000-8000-000000000001")
ISSUER = "https://project.supabase.co/auth/v1"


def claims(**overrides: Any) -> dict[str, Any]:
    base = {"sub": str(USER), "aud": "authenticated", "exp": int(time.time()) + 60, "iss": ISSUER}
    return base | overrides


class FakeSigningKeys:
    """Stands in for the JWKS endpoint: hands out one public key."""

    def __init__(self, public_key: Any) -> None:
        self.public_key = public_key

    def get_signing_key_from_jwt(self, token: str) -> Any:
        return SimpleNamespace(key=self.public_key)


@pytest.fixture(scope="module")
def private_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture
def asymmetric(private_key: ec.EllipticCurvePrivateKey) -> TokenVerifier:
    keys = FakeSigningKeys(private_key.public_key())
    return TokenVerifier("https://project.supabase.co/", signing_keys=keys)


def test_a_token_signed_by_the_project_key_gives_the_user_id(
    asymmetric: TokenVerifier, private_key: ec.EllipticCurvePrivateKey
) -> None:
    token = jwt.encode(claims(), private_key, algorithm="ES256")
    assert asymmetric.user_id(token) == USER


def test_a_token_signed_by_another_key_is_rejected(asymmetric: TokenVerifier) -> None:
    other = ec.generate_private_key(ec.SECP256R1())
    with pytest.raises(InvalidTokenError):
        asymmetric.user_id(jwt.encode(claims(), other, algorithm="ES256"))


def test_legacy_shared_secret_tokens_are_accepted_when_configured() -> None:
    verifier = TokenVerifier(jwt_secret=SECRET)
    assert verifier.user_id(jwt.encode(claims(), SECRET, algorithm="HS256")) == USER
    with pytest.raises(InvalidTokenError):
        verifier.user_id(jwt.encode(claims(), SECRET + "x", algorithm="HS256"))


@pytest.mark.parametrize(
    "bad",
    [
        {"exp": int(time.time()) - 10},  # expired
        {"aud": "anon"},  # not a signed-in user's token
        {"iss": "https://evil.example/auth/v1"},  # another project's token
        {"sub": "not-a-uuid"},
    ],
)
def test_wrong_claims_are_rejected(
    asymmetric: TokenVerifier, private_key: ec.EllipticCurvePrivateKey, bad: dict[str, Any]
) -> None:
    with pytest.raises(InvalidTokenError):
        asymmetric.user_id(jwt.encode(claims(**bad), private_key, algorithm="ES256"))


def test_missing_claims_are_rejected(
    asymmetric: TokenVerifier, private_key: ec.EllipticCurvePrivateKey
) -> None:
    for missing in ("sub", "exp"):
        partial = {k: v for k, v in claims().items() if k != missing}
        with pytest.raises(InvalidTokenError):
            asymmetric.user_id(jwt.encode(partial, private_key, algorithm="ES256"))


def test_a_token_cannot_choose_a_weaker_algorithm(asymmetric: TokenVerifier) -> None:
    # No shared secret is configured, so an HS256 token has nothing to be checked against,
    # whatever it was signed with; and an unsigned token is never accepted.
    with pytest.raises(InvalidTokenError, match="HS256"):
        asymmetric.user_id(jwt.encode(claims(), SECRET, algorithm="HS256"))
    with pytest.raises(InvalidTokenError):
        asymmetric.user_id(jwt.encode(claims(), None, algorithm="none"))  # type: ignore[arg-type]
    with pytest.raises(InvalidTokenError):
        asymmetric.user_id("not a token")


def test_without_any_key_every_token_is_rejected() -> None:
    with pytest.raises(InvalidTokenError):
        TokenVerifier().user_id(jwt.encode(claims(), SECRET, algorithm="HS256"))
