"""El token de Iniciar sesión con Apple solo vale si es de Apple y para nuestra app."""

from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from locus_v2.identity.infrastructure import apple_identity

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(**overrides) -> str:
    claims = {
        "iss": "https://appleid.apple.com",
        "aud": "com.carlos.locusia",
        "sub": "001234.apple",
        "email": "ana@example.com",
        "email_verified": "true",
        "exp": 4102444800,
        **overrides,
    }
    return jwt.encode(claims, _KEY, algorithm="RS256", headers={"kid": "test"})


@pytest.fixture(autouse=True)
def _apple_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        apple_identity._jwks,
        "get_signing_key_from_jwt",
        lambda token: SimpleNamespace(key=_KEY.public_key()),
    )


@pytest.mark.asyncio
async def test_a_token_for_our_app_is_accepted() -> None:
    claims = await apple_identity.AppleIdentityVerifier().verify(
        _token(), ["com.carlos.locusia"]
    )
    assert claims["sub"] == "001234.apple"


@pytest.mark.asyncio
async def test_a_token_for_another_app_is_rejected() -> None:
    with pytest.raises(ValueError):
        await apple_identity.AppleIdentityVerifier().verify(
            _token(aud="com.otra.app"), ["com.carlos.locusia"]
        )


@pytest.mark.asyncio
async def test_a_token_not_issued_by_apple_is_rejected() -> None:
    with pytest.raises(ValueError):
        await apple_identity.AppleIdentityVerifier().verify(
            _token(iss="https://evil.example"), ["com.carlos.locusia"]
        )
