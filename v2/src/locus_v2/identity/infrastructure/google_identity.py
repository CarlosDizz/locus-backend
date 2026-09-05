import asyncio
from typing import Any

from google.auth.transport import requests
from google.oauth2 import id_token


class GoogleIdentityVerifier:
    async def verify(self, credential: str, audiences: list[str]) -> dict[str, Any]:
        if not audiences:
            raise ValueError("Google OAuth client IDs are not configured")

        def verify_token() -> dict[str, Any]:
            claims = id_token.verify_oauth2_token(credential, requests.Request())
            if claims.get("aud") not in audiences:
                raise ValueError("Unexpected Google token audience")
            return claims

        return await asyncio.to_thread(verify_token)
