"""El webhook no tiene autenticación. La firma es toda la credencial.

Ese endpoint suma dinero a monederos y está abierto a internet, así que lo único
que lo separa de un desconocido es esta comprobación. Y es fácil de romper sin
querer: la firma cubre los bytes exactos que mandó Paddle, así que cualquier capa
que interprete el JSON y lo vuelva a serializar hace fallar todos los eventos de
una forma que parece un secreto mal configurado.
"""

import hashlib
import hmac
import json
import time

import pytest
from pydantic import SecretStr

from locus_v2.billing.infrastructure.paddle_gateway import (
    PaddleError,
    PaddleGateway,
    PaddleNotConfigured,
)

SECRET = "pdl_ntfset_prueba_no_es_una_clave_real"


class _Settings:
    """Lo mínimo que mira la pasarela, sin cargar el entorno entero."""

    paddle_api_key = SecretStr("pdl_sdbx_apikey_falsa")
    paddle_webhook_secret: SecretStr | None = SecretStr(SECRET)
    paddle_sandbox = True
    web_app_base_url = "https://app.locusguide.es"
    openai_timeout_seconds = 30.0


def firmar(payload: bytes, secret: str = SECRET, timestamp: int | None = None) -> str:
    marca = timestamp if timestamp is not None else int(time.time())
    firma = hmac.new(
        secret.encode(), f"{marca}:".encode() + payload, hashlib.sha256
    ).hexdigest()
    return f"ts={marca};h1={firma}"


def evento() -> bytes:
    return json.dumps(
        {
            "event_id": "evt_1",
            "event_type": "transaction.completed",
            "data": {
                "id": "txn_1",
                "status": "completed",
                "custom_data": {"user_public_id": "abc", "product_id": "web_top_up_5"},
                "details": {"totals": {"grand_total": "499", "currency_code": "EUR"}},
            },
        }
    ).encode()


def gateway() -> PaddleGateway:
    return PaddleGateway(_Settings())  # type: ignore[arg-type]


def test_a_properly_signed_event_is_accepted() -> None:
    payload = evento()
    parsed = gateway().parse_webhook(payload, firmar(payload), now=time.time())

    assert parsed["event_type"] == "transaction.completed"
    assert parsed["data"]["id"] == "txn_1"


def test_a_forged_signature_is_refused() -> None:
    payload = evento()
    with pytest.raises(PaddleError):
        gateway().parse_webhook(
            payload, firmar(payload, secret="pdl_ntfset_otra_cosa"), now=time.time()
        )


def test_a_tampered_body_is_refused() -> None:
    # Firmamos un importe y enviamos otro: es el ataque que importa.
    original = evento()
    signature = firmar(original)
    alterado = original.replace(b'"499"', b'"99900"')

    with pytest.raises(PaddleError):
        gateway().parse_webhook(alterado, signature, now=time.time())


def test_an_old_event_is_refused() -> None:
    # Un evento capturado y reproducido dias despues no debe valer.
    payload = evento()
    ahora = time.time()
    viejo = firmar(payload, timestamp=int(ahora) - 86_400)

    with pytest.raises(PaddleError):
        gateway().parse_webhook(payload, viejo, now=ahora)


def test_a_malformed_signature_header_is_refused() -> None:
    with pytest.raises(PaddleError):
        gateway().parse_webhook(evento(), "esto-no-es-una-firma", now=time.time())


def test_without_a_secret_it_refuses_rather_than_trusting() -> None:
    settings = _Settings()
    settings.paddle_webhook_secret = None

    with pytest.raises(PaddleNotConfigured):
        PaddleGateway(settings).parse_webhook(  # type: ignore[arg-type]
            evento(), "ts=1;h1=x", now=time.time()
        )
