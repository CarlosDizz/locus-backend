"""Stripe Checkout, for topping up the wallet from the PWA.

Hosted Checkout rather than a form of our own: the card never touches this
server, so the PCI surface stays as small as it can be, and Bizum, Apple Pay and
Google Pay come along without extra work. Bizum matters here — in Spain it
converts considerably better than a card.

Everything money-related is decided on this side. The browser proposes an
amount; what Stripe is told to charge is what this module validates, and what
gets credited later is what Stripe reports as actually paid.

Mirrors billing/infrastructure/google_play.py: a thin, testable wrapper with no
knowledge of wallets or ledgers.
"""

from typing import Any

import stripe

from locus_v2.billing.topup_catalogue import CURRENCY
from locus_v2.config import Settings


class StripeError(RuntimeError):
    pass


class StripeNotConfigured(StripeError):
    pass


class StripeGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.stripe_secret_key is not None

    def _client(self) -> Any:
        if self.settings.stripe_secret_key is None:
            raise StripeNotConfigured("Los pagos web no están configurados")
        return stripe.StripeClient(self.settings.stripe_secret_key.get_secret_value())

    async def create_checkout_session(
        self,
        *,
        amount_cents: int,
        user_public_id: str,
        product_id: str,
        description: str,
    ) -> dict[str, Any]:
        base = self.settings.web_app_base_url.rstrip("/")
        session = await self._client().checkout.sessions.create_async(
            {
                "mode": "payment",
                # Sin `{CHECKOUT_SESSION_ID}` el regreso no puede confirmarse a
                # mano cuando el webhook llega tarde o no esta dado de alta.
                "success_url": f"{base}/billing?pago=ok&sesion={{CHECKOUT_SESSION_ID}}",
                "cancel_url": f"{base}/billing?pago=cancelado",
                "line_items": [
                    {
                        "quantity": 1,
                        "price_data": {
                            "currency": CURRENCY,
                            "unit_amount": amount_cents,
                            "product_data": {"name": description},
                        },
                    }
                ],
                "metadata": {
                    "user_public_id": user_public_id,
                    "product_id": product_id,
                    "amount_cents": str(amount_cents),
                },
            }
        )
        return dict(session)

    async def retrieve_checkout_session(self, session_id: str) -> dict[str, Any]:
        return dict(await self._client().checkout.sessions.retrieve_async(session_id))

    def parse_webhook(self, payload: bytes, signature: str) -> dict[str, Any]:
        """Validate the signature and return the event.

        The signature is computed over the exact bytes Stripe sent, which is why
        the caller has to hand over the raw body: anything that parses and
        re-serialises the JSON first will fail verification every time, for
        reasons that look nothing like the cause.
        """
        if self.settings.stripe_webhook_secret is None:
            raise StripeNotConfigured("Falta el secreto del webhook de Stripe")
        try:
            event = stripe.Webhook.construct_event(
                payload,
                signature,
                self.settings.stripe_webhook_secret.get_secret_value(),
            )
        except ValueError as error:
            raise StripeError("Cuerpo del webhook ilegible") from error
        except stripe.SignatureVerificationError as error:
            raise StripeError("Firma del webhook inválida") from error
        return dict(event)
