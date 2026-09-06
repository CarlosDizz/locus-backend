"""Google Play purchase verification, ported from app/services/billing_service.py.

Async (httpx) instead of V1's sync `requests`, to match the rest of V2. Real
purchase verification needs a real service-account credential
(GOOGLE_PLAY_SERVICE_ACCOUNT_JSON/_FILE) which this local dev environment does
not have configured — see docs/testing-checklist.md Capitulo 4 for what could
and could not be exercised for real.
"""

import asyncio
from typing import Any

import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

from locus_v2.config import Settings


class GooglePlayVerificationError(RuntimeError):
    pass


class GooglePlayVerifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def verify_and_consume(
        self, *, product_id: str, purchase_token: str, package_name: str
    ) -> dict[str, Any]:
        if not self.settings.google_play_verify_purchases:
            return {"status": "skipped", "reason": "google_play_verify_purchases=false"}

        token = await asyncio.to_thread(self._access_token)
        headers = {"Authorization": f"Bearer {token}"}
        base = "https://androidpublisher.googleapis.com/androidpublisher/v3/applications"
        purchase_url = (
            f"{base}/{package_name}/purchases/products/{product_id}/tokens/{purchase_token}"
        )

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(purchase_url, headers=headers)
            if response.status_code >= 400:
                raise GooglePlayVerificationError(
                    f"Google Play no ha validado la compra ({response.status_code})"
                )
            payload = response.json()
            if payload.get("purchaseState") not in (None, 0):
                raise GooglePlayVerificationError("La compra de Google Play no está completada")

            consumed_by_server = False
            if payload.get("consumptionState") != 1:
                consume_response = await client.post(f"{purchase_url}:consume", headers=headers)
                if consume_response.status_code >= 400:
                    raise GooglePlayVerificationError(
                        "Google Play no ha permitido consumir la compra "
                        f"({consume_response.status_code})"
                    )
                consumed_by_server = True

        return {
            "status": "verified",
            "purchase_state": payload.get("purchaseState"),
            "consumption_state": payload.get("consumptionState"),
            "acknowledgement_state": payload.get("acknowledgementState"),
            "consumed_by_server": consumed_by_server,
            "order_id": payload.get("orderId", ""),
            "purchase_time_millis": payload.get("purchaseTimeMillis", ""),
        }

    def _access_token(self) -> str:
        credentials = self._credentials()
        credentials.refresh(GoogleAuthRequest())
        token = credentials.token
        if not token:
            raise GooglePlayVerificationError("No se ha podido obtener un token de acceso")
        return str(token)

    def _credentials(self) -> Any:
        import json

        scopes = ["https://www.googleapis.com/auth/androidpublisher"]
        if self.settings.google_play_service_account_json:
            info = json.loads(self.settings.google_play_service_account_json)
            return service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
                info, scopes=scopes
            )
        if self.settings.google_play_service_account_file:
            return service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
                self.settings.google_play_service_account_file, scopes=scopes
            )
        raise GooglePlayVerificationError(
            "Faltan credenciales de Google Play para verificar compras"
        )
