"""El enlace /r/<token> que cuenta cada clic en un referido antes de redirigir.

Sin esto no se sabe qué programa rinde. El token lleva dentro el destino, quién
lo pulsa, de qué app y de qué viaje, firmado con la clave del servidor: así no
hay que guardar una fila por cada enlace que el planificador propone (casi
ninguno se pulsa) y nadie puede usar /r/ para redirigir a otra web.
"""

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass

from locus_v2.config import Settings


@dataclass(frozen=True)
class TrackedLink:
    url: str
    program: str
    app: str
    user_id: int | None
    trip_id: str


def _key(settings: Settings) -> bytes:
    return hashlib.sha256(
        b"referral-link:" + settings.jwt_secret.get_secret_value().encode()
    ).digest()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign(settings: Settings, link: TrackedLink) -> str:
    payload = _b64(json.dumps(
        {"u": link.url, "p": link.program, "a": link.app, "w": link.user_id, "t": link.trip_id},
        separators=(",", ":"), ensure_ascii=False,
    ).encode())
    mac = _b64(hmac.new(_key(settings), payload.encode(), hashlib.sha256).digest()[:16])
    return f"{payload}.{mac}"


def verify(settings: Settings, token: str) -> TrackedLink | None:
    payload, _, mac = token.partition(".")
    if not payload or not mac:
        return None
    expected = _b64(hmac.new(_key(settings), payload.encode(), hashlib.sha256).digest()[:16])
    if not hmac.compare_digest(expected, mac):
        return None
    try:
        data = json.loads(_unb64(payload))
    except ValueError:
        return None
    url = str(data.get("u") or "")
    if not url.startswith("https://"):
        return None
    return TrackedLink(
        url=url,
        program=str(data.get("p") or ""),
        app=str(data.get("a") or "guide"),
        user_id=int(data["w"]) if data.get("w") is not None else None,
        trip_id=str(data.get("t") or ""),
    )


def tracked_url(settings: Settings, link: TrackedLink) -> str:
    return f"{settings.public_api_base_url.rstrip('/')}/r/{sign(settings, link)}"
