"""Paddle, como vendedor de cara al cliente.

Se eligió por una razón que no es técnica: Paddle es *merchant of record*, o sea
que el vendedor es Paddle y no nosotros, y se ocupa del IVA de servicios
digitales en todos los países. Eso permite cobrar sin estar dado de alta como
empresa, que es lo que bloqueaba la vía de Stripe. Se paga por ello: 5% + 0,50 $
todo incluido, frente al ~4% de Stripe — pero sigue por debajo del 15% de Google
Play en los tres tramos.

Dos diferencias con Stripe que conviene tener presentes al leer esto:

1. **El checkout no es una página suya.** Paddle devuelve `tu-dominio?_ptxn=<id>`
   y es la app quien abre la capa de pago con Paddle.js. Por eso aquí se
   devuelve una URL que apunta a la PWA, no a Paddle.
2. **Los precios pueden ir sueltos**, sin dar de alta productos en su catálogo,
   así que el importe libre no necesita nada montado por adelantado.

La firma del webhook sí es casi igual: HMAC-SHA256 sobre `{ts}:{cuerpo crudo}`.
"""

import hashlib
import hmac
import json
from typing import Any

import httpx

from locus_v2.billing.topup_catalogue import CURRENCY
from locus_v2.config import Settings

LIVE_API = "https://api.paddle.com"
SANDBOX_API = "https://sandbox-api.paddle.com"

# Paddle firma con una tolerancia corta para que un evento capturado no se pueda
# reproducir dias despues. Cinco segundos es lo que usan sus propios SDK; aqui se
# da algo mas de aire porque entre su reloj y el nuestro hay una red de por medio.
SIGNATURE_TOLERANCE_SECONDS = 300


class PaddleError(RuntimeError):
    pass


class PaddleNotConfigured(PaddleError):
    pass


class PaddleGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.paddle_api_key is not None

    @property
    def base_url(self) -> str:
        return SANDBOX_API if self.settings.paddle_sandbox else LIVE_API

    def _headers(self) -> dict[str, str]:
        if self.settings.paddle_api_key is None:
            raise PaddleNotConfigured("Los pagos web no están configurados")
        return {
            "Authorization": f"Bearer {self.settings.paddle_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    async def create_transaction(
        self, *, amount_cents: int, user_public_id: str, product_id: str, description: str
    ) -> dict[str, Any]:
        """Crea la transacción y devuelve a dónde mandar el navegador.

        El precio va suelto (`price` en línea en vez de `price_id`) para no tener
        que mantener un catálogo duplicado en el panel de Paddle: los importes ya
        viven en topup_catalogue.py, y tenerlos en dos sitios es pedir que se
        desincronicen.
        """
        payload = {
            "items": [
                {
                    "quantity": 1,
                    "price": {
                        "description": description,
                        "name": description,
                        "unit_price": {
                            "amount": str(amount_cents),
                            "currency_code": CURRENCY.upper(),
                        },
                        # "internal" = el IVA va DENTRO del precio anunciado. Sin
                        # esto Paddle lo sumaria encima y quien pulsa "4,99 €"
                        # acabaria pagando 6,04: el importe cobrado dejaria de
                        # cuadrar con el catalogo y confirm_web_topup lo
                        # rechazaria, con el dinero ya cobrado. Ademas es la
                        # promesa que hace la pantalla: lo que ves es lo que pagas.
                        "tax_mode": "internal",
                        "product": {
                            "name": "Saldo Locus",
                            "tax_category": "standard",
                        },
                    },
                }
            ],
            # Lo que permite saber a quién abonar cuando vuelva el webhook. Viaja
            # con la transacción y Paddle lo devuelve tal cual.
            "custom_data": {
                "user_public_id": user_public_id,
                "product_id": product_id,
                "amount_cents": str(amount_cents),
            },
            "collection_mode": "automatic",
        }
        async with httpx.AsyncClient(timeout=self.settings.openai_timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/transactions", headers=self._headers(), json=payload
            )
        if response.status_code >= 400:
            raise PaddleError(f"Paddle rechazó la transacción: {response.text[:300]}")
        return dict(response.json().get("data") or {})

    async def retrieve_transaction(self, transaction_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.settings.openai_timeout_seconds) as client:
            response = await client.get(
                f"{self.base_url}/transactions/{transaction_id}", headers=self._headers()
            )
        if response.status_code >= 400:
            raise PaddleError(f"Paddle no devolvió la transacción: {response.text[:300]}")
        return dict(response.json().get("data") or {})

    def parse_webhook(self, payload: bytes, signature_header: str, *, now: float) -> dict[str, Any]:
        """Valida la firma y devuelve el evento.

        La firma cubre los bytes exactos que mandó Paddle, así que hay que
        pasarle el cuerpo crudo: cualquier capa que interprete el JSON y lo
        vuelva a serializar romperá la validación siempre, y el fallo se parece
        a un secreto mal puesto sin serlo.
        """
        if self.settings.paddle_webhook_secret is None:
            raise PaddleNotConfigured("Falta el secreto del webhook de Paddle")

        timestamp, firma = _split_signature(signature_header)
        if abs(now - timestamp) > SIGNATURE_TOLERANCE_SECONDS:
            raise PaddleError("Firma del webhook caducada")

        esperado = hmac.new(
            self.settings.paddle_webhook_secret.get_secret_value().encode(),
            f"{timestamp}:".encode() + payload,
            hashlib.sha256,
        ).hexdigest()
        # compare_digest y no ==: comparar cadenas termina en cuanto encuentra una
        # diferencia, y ese tiempo se puede medir para adivinar la firma byte a byte.
        if not hmac.compare_digest(esperado, firma):
            raise PaddleError("Firma del webhook inválida")

        try:
            return dict(json.loads(payload))
        except ValueError as error:
            raise PaddleError("Cuerpo del webhook ilegible") from error


def _split_signature(header: str) -> tuple[int, str]:
    """`ts=1234567890;h1=abc...` en sus dos partes."""
    partes = dict(
        trozo.split("=", 1) for trozo in header.split(";") if "=" in trozo
    )
    marca, firma = partes.get("ts"), partes.get("h1")
    if not marca or not firma or not marca.isdigit():
        raise PaddleError("Cabecera de firma mal formada")
    return int(marca), firma
