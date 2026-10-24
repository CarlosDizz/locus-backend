"""Envío a iPhone por APNs (HTTP/2 + token JWT firmado con la clave .p8 de Apple).

Portado de BACUL. Sin clave configurada no se envía nada: el aviso queda en el
log como «simulada», para poder probar en local.
"""

import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt

from locus_v2.config import Settings


@dataclass(frozen=True)
class SendResult:
    status: str  # enviada | simulada | fallida | caducado
    detail: str = ""


class ApnsSender:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._token: tuple[str, float] | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        s = self.settings
        return bool(s.apns_key and s.apns_key_id and s.apns_team_id)

    async def send(
        self, device_token: str, topic: str, title: str, body: str,
        data: dict[str, Any] | None = None,
    ) -> SendResult:
        if not self.configured:
            return SendResult("simulada", "Sin clave de APNs")
        host = (
            "https://api.sandbox.push.apple.com"
            if self.settings.apns_sandbox
            else "https://api.push.apple.com"
        )
        if self._client is None:
            self._client = httpx.AsyncClient(http2=True, timeout=10)
        response = await self._client.post(
            f"{host}/3/device/{device_token}",
            headers={
                "authorization": f"bearer {self._jwt()}",
                "apns-topic": topic,
                "apns-push-type": "alert",
                "apns-priority": "5",
            },
            json={
                "aps": {"alert": {"title": title, "body": body}, "sound": "default"},
                **(data or {}),
            },
        )
        if response.status_code == 200:
            return SendResult("enviada")
        if response.status_code == 410:
            return SendResult("caducado", "El móvil ya no acepta avisos")
        return SendResult("fallida", f"{response.status_code} {response.text[:200]}")

    def _jwt(self) -> str:
        # Apple pide renovarlo entre 20 y 60 minutos; se renueva a los 40.
        now = time.time()
        if self._token and now - self._token[1] < 40 * 60:
            return self._token[0]
        assert self.settings.apns_key is not None
        # En el .env la clave .p8 va en una línea, con los saltos escritos como \n.
        private_key = self.settings.apns_key.get_secret_value().replace("\\n", "\n")
        token = jwt.encode(
            {"iss": self.settings.apns_team_id, "iat": int(now)},
            private_key,
            algorithm="ES256",
            headers={"kid": self.settings.apns_key_id},
        )
        self._token = (token, now)
        return token

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
