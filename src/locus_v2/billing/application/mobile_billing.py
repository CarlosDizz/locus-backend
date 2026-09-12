"""V1-compatible mobile billing facade: wallet, ledger, usage events, top-ups.

Ported from app/services/billing_service.py. The internal pricing/ledger
engine (billing/application/processor.py, run by the worker) already existed
in V2 and needed no changes — this is the missing public read/write surface
on top of it.

Public views preserve Ionic's V1 field names while persistence stays V2-native.
"""

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from locus_v2.ai.models import AIModel, AIProvider
from locus_v2.billing.infrastructure.google_play import (
    GooglePlayVerificationError,
    GooglePlayVerifier,
)
from locus_v2.billing.infrastructure.paddle_gateway import (
    PaddleGateway,
    PaddleNotConfigured,
)
from locus_v2.billing.models import LedgerEntry, LedgerEntryKind, TopUp, UsageEvent, Wallet
from locus_v2.billing.topup_catalogue import (
    CURRENCY,
    WEB_TOPUP_MAX_CENTS,
    WEB_TOPUP_MIN_CENTS,
    WEB_TOPUP_PRODUCTS,
    resolve_amount_cents,
)
from locus_v2.config import Settings
from locus_v2.identity.models import User
from locus_v2.infrastructure.database.session import get_database
from locus_v2.shared.clock import utc_now
from locus_v2.shared.mobile_ids import mobile_id
from locus_v2.voice.models import VoiceSession

GOOGLE_PLAY_TOPUP_PRODUCTS: dict[str, int] = {
    "locus_top_up_1": 100,
    "locus_top_up_3": 300,
    "locus_top_up_5": 500,
    "locus_top_up_10": 1000,
}


class BillingError(RuntimeError):
    pass


@dataclass(frozen=True)
class WalletView:
    user_id: int
    currency: str
    balance_cents: int


@dataclass(frozen=True)
class LedgerEntryView:
    id: int
    entry_type: str
    amount_cents: int
    balance_after_cents: int
    description: str
    reference_type: str
    reference_id: str
    usage_interaction_type: str | None
    usage_source: str | None
    usage_endpoint: str | None
    usage_call_id: str | None
    usage_call_started_at: datetime | None
    usage_call_ended_at: datetime | None
    usage_audio_input_tokens: int | None
    usage_audio_output_tokens: int | None
    usage_image_input_tokens: int | None
    created_at: datetime


@dataclass(frozen=True)
class UsageEventView:
    id: int
    session_id: str | None
    provider: str
    endpoint: str
    model: str
    interaction_type: str
    source: str
    response_id: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    audio_input_tokens: int
    audio_output_tokens: int
    image_input_tokens: int
    provider_cost_eur_cents: int
    charged_amount_cents: int
    gross_margin_cents: int
    currency: str
    status: str
    created_at: datetime


@dataclass(frozen=True)
class TopUpView:
    id: int
    amount_cents: int
    bonus_cents: int
    provider: str
    provider_reference: str
    status: str
    created_at: datetime


class MobileBillingService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    async def get_wallet(self, user_id: int) -> WalletView:
        wallet = await self._wallet_or_raise(user_id)
        return WalletView(
            user_id=user_id, currency=wallet.currency, balance_cents=wallet.balance_cents
        )

    async def list_ledger(self, user_id: int, *, limit: int, offset: int) -> list[LedgerEntryView]:
        rows = (
            await self.session.scalars(
                select(LedgerEntry)
                .options(joinedload(LedgerEntry.usage_event))
                .where(LedgerEntry.user_id == user_id)
                .order_by(LedgerEntry.id.desc())
                .offset(offset)
                .limit(limit)
            )
        ).all()

        voice_session_ids = [
            row.usage_event.voice_session_id
            for row in rows
            if row.usage_event is not None and row.usage_event.voice_session_id is not None
        ]
        sessions: dict[int, VoiceSession] = {}
        if voice_session_ids:
            session_rows = (
                await self.session.scalars(
                    select(VoiceSession).where(VoiceSession.id.in_(voice_session_ids))
                )
            ).all()
            sessions = {row.id: row for row in session_rows}

        views: list[LedgerEntryView] = []
        for row in rows:
            usage_event = row.usage_event
            voice_session = (
                sessions.get(usage_event.voice_session_id)
                if usage_event is not None and usage_event.voice_session_id is not None
                else None
            )
            views.append(
                LedgerEntryView(
                    id=mobile_id(row),
                    entry_type=str(
                        (row.metadata_json or {}).get("legacy_entry_type")
                        or {
                            "charge": "usage_charge",
                            "credit": "top_up",
                            "refund": "refund",
                            "adjustment": "adjustment",
                        }.get(row.kind, row.kind)
                    ),
                    amount_cents=row.amount_cents,
                    balance_after_cents=row.balance_after_cents,
                    description=row.description,
                    reference_type=row.reference_type,
                    reference_id=row.reference_id,
                    usage_interaction_type=(
                        usage_event.interaction_type if usage_event is not None else None
                    ),
                    usage_source=_usage_source(usage_event),
                    usage_endpoint=_usage_endpoint(usage_event),
                    usage_call_id=(
                        str(
                            (voice_session.config_snapshot_json or {}).get("call_id")
                            or voice_session.public_id
                        )
                        if voice_session
                        else _raw_string(usage_event, "call_id")
                    ),
                    usage_call_started_at=voice_session.started_at if voice_session else None,
                    usage_call_ended_at=voice_session.ended_at if voice_session else None,
                    usage_audio_input_tokens=usage_event.audio_input_tokens
                    if usage_event
                    else None,
                    usage_audio_output_tokens=usage_event.audio_output_tokens
                    if usage_event
                    else None,
                    usage_image_input_tokens=usage_event.image_input_tokens
                    if usage_event
                    else None,
                    created_at=row.created_at,
                )
            )
        return views

    async def list_usage_events(self, user_id: int, *, limit: int) -> list[UsageEventView]:
        rows = (
            await self.session.execute(
                select(UsageEvent, AIModel.external_id, AIProvider.code)
                .join(AIModel, AIModel.id == UsageEvent.model_id)
                .join(AIProvider, AIProvider.id == UsageEvent.provider_id)
                .where(UsageEvent.user_id == user_id)
                .order_by(UsageEvent.id.desc())
                .limit(limit)
            )
        ).all()
        return [
            UsageEventView(
                id=mobile_id(row),
                session_id=_raw_string(row, "session_id"),
                provider=provider,
                endpoint=_usage_endpoint(row) or "",
                model=model,
                interaction_type=row.interaction_type,
                source=_usage_source(row) or "",
                response_id=str(row.raw_usage_json.get("response_id") or row.request_id or ""),
                input_tokens=int(row.raw_usage_json.get("input_tokens", _input_tokens(row))),
                cached_input_tokens=int(
                    row.raw_usage_json.get(
                        "cached_input_tokens",
                        row.cached_text_input_tokens
                        + row.cached_audio_input_tokens
                        + row.cached_image_input_tokens,
                    )
                ),
                output_tokens=int(
                    row.raw_usage_json.get(
                        "output_tokens", row.text_output_tokens + row.audio_output_tokens
                    )
                ),
                reasoning_tokens=int(
                    row.raw_usage_json.get(
                        "reasoning_tokens", row.raw_usage_json.get("_reasoning_tokens", 0)
                    )
                ),
                audio_input_tokens=row.audio_input_tokens,
                audio_output_tokens=row.audio_output_tokens,
                image_input_tokens=row.image_input_tokens,
                provider_cost_eur_cents=row.provider_cost_eur_cents,
                charged_amount_cents=row.charged_amount_cents,
                gross_margin_cents=row.gross_margin_cents,
                currency=row.currency,
                status=row.status,
                created_at=row.created_at,
            )
            for row, model, provider in rows
        ]

    async def create_topup(
        self,
        *,
        user_id: int,
        amount_cents: int,
        provider: str = "manual",
        provider_reference: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TopUpView:
        if not self.settings.billing_manual_topups_enabled:
            raise BillingError("Las recargas manuales no están habilitadas")
        if amount_cents <= 0:
            raise BillingError("El importe de la recarga debe ser positivo")
        topup = await self._apply_topup(
            user_id=user_id,
            amount_cents=amount_cents,
            provider=provider,
            provider_reference=provider_reference,
            metadata=metadata or {},
        )
        return _topup_view(topup)

    async def confirm_google_play_topup(
        self,
        *,
        user_id: int,
        product_id: str,
        purchase_token: str,
        order_id: str = "",
        package_name: str = "",
        raw_purchase: dict[str, Any] | None = None,
        verifier: GooglePlayVerifier | None = None,
    ) -> TopUpView:
        amount_cents = GOOGLE_PLAY_TOPUP_PRODUCTS.get(product_id)
        if amount_cents is None:
            raise BillingError("Producto de recarga no reconocido")
        if not purchase_token.strip():
            raise BillingError("Falta el token de compra de Google Play")

        purchase_token = purchase_token.strip()
        if package_name and package_name != self.settings.google_play_package_name:
            raise BillingError("El paquete de Google Play no corresponde a esta aplicación")
        dedupe_key = sha256(purchase_token.encode()).hexdigest()

        existing = await self.session.scalar(
            select(TopUp).where(TopUp.purchase_dedupe_key == dedupe_key)
        )
        if existing is not None:
            if existing.user_id != user_id:
                raise BillingError("Esta compra ya fue aplicada a otra cuenta")
            return _topup_view(existing)

        # Reserve the purchase before the network call. The unique key blocks other
        # transactions until this one commits or rolls back, including another account.
        wallet = await self._wallet_or_raise(user_id)
        topup = TopUp(
            user_id=user_id,
            wallet_id=wallet.id,
            amount_cents=amount_cents,
            bonus_cents=0,
            provider="google_play",
            provider_reference=purchase_token,
            purchase_dedupe_key=dedupe_key,
            status="pending",
            metadata_json={},
        )
        try:
            async with self.session.begin_nested():
                self.session.add(topup)
                await self.session.flush()
        except (IntegrityError, OperationalError):
            # Someone else got there first. The unique key did its job — the money
            # is safe either way — but a client that retried because its network
            # dropped deserves its receipt back, not a 500.
            existing = await self._recover_topup(dedupe_key)
            if existing is None:
                raise
            if existing.user_id != user_id:
                raise BillingError("Esta compra ya fue aplicada a otra cuenta") from None
            return _topup_view(existing)

        effective_package_name = self.settings.google_play_package_name
        active_verifier = verifier or GooglePlayVerifier(self.settings)
        try:
            verification = await active_verifier.verify_and_consume(
                product_id=product_id,
                purchase_token=purchase_token,
                package_name=effective_package_name,
            )
        except GooglePlayVerificationError as error:
            await self.session.rollback()
            raise BillingError(str(error)) from error

        topup = await self._apply_topup(
            user_id=user_id,
            amount_cents=amount_cents,
            provider="google_play",
            provider_reference=purchase_token,
            metadata={
                "product_id": product_id,
                "order_id": order_id,
                "package_name": effective_package_name,
                "raw_purchase": raw_purchase or {},
                "verification": verification,
            },
            reserved_topup=topup,
        )
        return _topup_view(topup)

    async def create_web_checkout(
        self,
        *,
        user: User,
        product_id: str = "",
        amount_cents: int | None = None,
        gateway: PaddleGateway | None = None,
    ) -> str:
        """Abre una transacción en Paddle y devuelve a dónde mandar el navegador.

        Lo que vuelve es una URL de la propia PWA con `?_ptxn=<id>`: Paddle no
        aloja la pantalla de pago como hacía Stripe, la abre en una capa sobre
        una página nuestra.
        """
        active = gateway or PaddleGateway(self.settings)
        if not active.enabled:
            raise PaddleNotConfigured("Los pagos web no están configurados")
        # Aqui se decide el importe, y solo aqui. Lo que llegue del navegador es
        # una propuesta.
        resolved = resolve_amount_cents(product_id, amount_cents)
        await self._wallet_or_raise(user.id)
        transaction = await active.create_transaction(
            amount_cents=resolved,
            user_public_id=user.public_id,
            product_id=product_id,
            description=f"Saldo Locus · {resolved / 100:.2f} €".replace(".", ","),
        )
        transaction_id = transaction.get("id")
        if not transaction_id:
            raise BillingError("Paddle no devolvió una transacción")
        base = self.settings.web_app_base_url.rstrip("/")
        return f"{base}/billing?_ptxn={transaction_id}"

    async def confirm_web_topup(self, transaction: dict[str, Any]) -> TopUpView | None:
        """Abona una transacción pagada de Paddle. Idempotente, y segura dos veces.

        Se llega por dos caminos a propósito: el webhook, que es la verdad porque
        el usuario puede cerrar la pestaña justo al pagar, y la vuelta a la app,
        que cubre un webhook que llegue tarde. Los dos acaban en la misma clave de
        deduplicación, así que el segundo encuentra el recibo y no cambia nada.

        Devuelve None si la transacción no está pagada — un checkout abandonado no
        es un error, es el caso común.
        """
        if transaction.get("status") != "completed":
            return None
        detalle = dict(transaction.get("details") or {})
        totales = dict(detalle.get("totals") or {})
        if (totales.get("currency_code") or "").lower() != CURRENCY:
            raise BillingError("Moneda de pago inesperada")

        transaction_id = transaction.get("id")
        if not transaction_id:
            raise BillingError("El pago no trae identificador de transacción")

        metadata = dict(transaction.get("custom_data") or {})
        user_public_id = str(metadata.get("user_public_id") or "")
        user_id = await self.session.scalar(
            select(User.id).where(User.public_id == user_public_id)
        )
        if user_id is None:
            raise BillingError("El pago no corresponde a ningún usuario")

        # Se abona lo que Paddle dice que se cobró de verdad — `grand_total`, ya
        # con impuestos — y no lo que pidiera nadie por el camino. El rango se
        # vuelve a comprobar aquí porque este número llega de fuera, aunque venga
        # firmado.
        amount_cents = int(totales.get("grand_total") or 0)
        product_id = str(metadata.get("product_id") or "")
        expected = WEB_TOPUP_PRODUCTS.get(product_id) if product_id else None
        if product_id and expected is None:
            raise BillingError("El pago menciona un producto que no existe")
        if expected is not None:
            # Un tramo del catalogo se comprueba contra su precio exacto, y ahi
            # acaba la validacion. Aplicarle ademas el rango del importe libre
            # rechazaba todos los pagos de 4,99 EUR, porque el suelo del campo
            # libre son 5,00 — con el dinero ya cobrado y el saldo sin subir.
            if amount_cents != expected:
                raise BillingError("El importe cobrado no coincide con el producto")
        elif not WEB_TOPUP_MIN_CENTS <= amount_cents <= WEB_TOPUP_MAX_CENTS:
            raise BillingError("Importe de recarga fuera de rango")

        dedupe_key = sha256(str(transaction_id).encode()).hexdigest()
        existing = await self.session.scalar(
            select(TopUp).where(TopUp.purchase_dedupe_key == dedupe_key)
        )
        if existing is not None:
            return _topup_view(existing)

        wallet = await self._wallet_or_raise(user_id)
        topup = TopUp(
            user_id=user_id,
            wallet_id=wallet.id,
            amount_cents=amount_cents,
            bonus_cents=0,
            provider="paddle",
            provider_reference=str(transaction_id),
            purchase_dedupe_key=dedupe_key,
            status="pending",
            metadata_json={},
        )
        try:
            async with self.session.begin_nested():
                self.session.add(topup)
                await self.session.flush()
        except (IntegrityError, OperationalError):
            # Paddle entrega el mismo evento varias veces sin avisar. La clave
            # unica ya ha protegido el dinero; aqui solo hay que devolver el
            # recibo para que deje de reintentar.
            recovered = await self._recover_topup(dedupe_key)
            if recovered is None:
                raise
            return _topup_view(recovered)

        topup = await self._apply_topup(
            user_id=user_id,
            amount_cents=amount_cents,
            provider="paddle",
            provider_reference=str(transaction_id),
            metadata={
                "transaction_id": str(transaction_id),
                "product_id": product_id,
            },
            reserved_topup=topup,
        )
        return _topup_view(topup)

    async def _recover_topup(self, dedupe_key: str) -> TopUp | None:
        """Read back the row that won the race, on a session that is still alive.

        The losing INSERT leaves this session unusable: MySQL discards the
        savepoint when it fails after a lock wait, so a read on the same session
        raises "SAVEPOINT ... does not exist" instead of returning the receipt
        (measured 2026-09-08 with eight simultaneous submits of one token). It
        was cosmetic while only Google Play used this — a phone that retried got
        an ugly error and the money was still safe. It stops being cosmetic with
        a payment webhook: Stripe reads a 500 as "try again" and keeps redelivering
        the same event for days, so a duplicate that cannot answer 200 turns into
        a loop.
        """
        await self.session.rollback()
        async with get_database().sessions() as session:
            return await session.scalar(
                select(TopUp).where(TopUp.purchase_dedupe_key == dedupe_key)
            )

    async def _apply_topup(
        self,
        *,
        user_id: int,
        amount_cents: int,
        provider: str,
        provider_reference: str,
        metadata: dict[str, Any],
        reserved_topup: TopUp | None = None,
    ) -> TopUp:
        wallet = await self._wallet_or_raise(user_id, lock=True)
        topup = reserved_topup or TopUp(
            user_id=user_id,
            wallet_id=wallet.id,
            amount_cents=amount_cents,
            bonus_cents=0,
            provider=provider,
            provider_reference=provider_reference,
            status="completed",
            metadata_json=metadata,
            completed_at=utc_now(),
        )
        topup.status = "completed"
        topup.completed_at = utc_now()
        topup.metadata_json = metadata
        self.session.add(topup)
        await self.session.flush()

        wallet.balance_cents += amount_cents
        self.session.add(
            LedgerEntry(
                user_id=user_id,
                wallet_id=wallet.id,
                kind=LedgerEntryKind.CREDIT,
                amount_cents=amount_cents,
                currency=wallet.currency,
                balance_after_cents=wallet.balance_cents,
                description="Recarga de saldo",
                reference_type="top_up",
                reference_id=str(topup.id),
                metadata_json=metadata,
                trace_id=f"topup:{topup.id}",
            )
        )
        await self.session.commit()
        await self.session.refresh(topup)
        return topup

    async def _wallet_or_raise(self, user_id: int, *, lock: bool = False) -> Wallet:
        query = select(Wallet).where(Wallet.user_id == user_id)
        if lock:
            query = query.with_for_update().execution_options(populate_existing=True)
        wallet = await self.session.scalar(query)
        if wallet is None:
            raise BillingError("Wallet no encontrada")
        return wallet


def _topup_view(topup: TopUp) -> TopUpView:
    return TopUpView(
        id=mobile_id(topup),
        amount_cents=topup.amount_cents,
        bonus_cents=topup.bonus_cents,
        provider=topup.provider,
        provider_reference=topup.provider_reference,
        status=topup.status,
        created_at=topup.created_at,
    )


def _raw_string(event: UsageEvent | None, key: str) -> str | None:
    if event is None:
        return None
    raw = event.raw_usage_json or {}
    metadata = raw.get("metadata_json") or {}
    value = raw.get(key) or (metadata.get(key) if isinstance(metadata, dict) else None)
    return str(value) if value else None


def _usage_endpoint(event: UsageEvent | None) -> str | None:
    if event is None:
        return None
    return _raw_string(event, "endpoint") or (
        "realtime" if event.voice_session_id is not None else "responses"
    )


def _usage_source(event: UsageEvent | None) -> str | None:
    if event is None:
        return None
    return _raw_string(event, "source") or (
        "call_room" if event.voice_session_id is not None else "chat"
    )


def _input_tokens(event: UsageEvent) -> int:
    return (
        event.text_input_tokens
        + event.cached_text_input_tokens
        + event.audio_input_tokens
        + event.cached_audio_input_tokens
        + event.image_input_tokens
        + event.cached_image_input_tokens
    )
