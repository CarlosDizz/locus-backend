"""Verifica el identity token de «Iniciar sesión con Apple».

Es un JWT firmado por Apple con una de sus claves públicas (JWKS). Se comprueba
la firma, que lo emitió Apple y que va dirigido a nuestra app (aud = bundle id).
"""

import asyncio
from typing import Any

import jwt

APPLE_ISSUER = "https://appleid.apple.com"
APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"

# PyJWKClient guarda las claves en memoria; una sola instancia para todo el proceso.
_jwks = jwt.PyJWKClient(APPLE_KEYS_URL, cache_keys=True, lifespan=86400)


class AppleIdentityVerifier:
    async def verify(self, credential: str, audiences: list[str]) -> dict[str, Any]:
        if not audiences:
            raise ValueError("Apple Sign In is not configured")

        def verify_token() -> dict[str, Any]:
            try:
                key = _jwks.get_signing_key_from_jwt(credential)
                return jwt.decode(
                    credential,
                    key.key,
                    algorithms=["RS256"],
                    audience=audiences,
                    issuer=APPLE_ISSUER,
                )
            except jwt.PyJWTError as exc:
                raise ValueError(f"Invalid Apple token: {exc}") from exc

        return await asyncio.to_thread(verify_token)
