"""V1-compatible mobile billing facade.

Mounted at /api/billing, same prefix and shapes (adapted where V2's schema
genuinely differs — see mobile_billing.py's docstring) as app/routes/billing.py.
"""

import time
from dataclasses import asdict
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.api.auth import CurrentUserDep
from locus_v2.billing.application.mobile_billing import BillingError, MobileBillingService
from locus_v2.billing.infrastructure.paddle_gateway import (
    PaddleError,
    PaddleGateway,
    PaddleNotConfigured,
)
from locus_v2.billing.topup_catalogue import (
    WEB_TOPUP_MAX_CENTS,
    WEB_TOPUP_MIN_CENTS,
    TopUpError,
)
from locus_v2.config import Settings, get_settings
from locus_v2.infrastructure.database.session import get_session
from locus_v2.shared.clock import UtcDatetime
from locus_v2.shared.mobile_ids import mobile_id

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/billing", tags=["billing"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


class WebCheckoutRequest(BaseModel):
    # Uno de los dos: un producto con boton propio, o un importe libre. El
    # servidor valida y decide; esto es una propuesta del navegador.
    product_id: str = ""
    amount_cents: int | None = Field(
        default=None, ge=WEB_TOPUP_MIN_CENTS, le=WEB_TOPUP_MAX_CENTS
    )


class WebCheckoutResponse(BaseModel):
    url: str


class WalletResponse(BaseModel):
    user_id: int
    currency: str
    balance_cents: int
    # Si el servidor no tiene claves de la pasarela, la PWA no debe ofrecer
    # recargas: sin esto los botones saldrian activos y responderian 503 al
    # pulsarlos, que es peor que decir con claridad que todavia no se puede. Se
    # enciende solo el dia que se configuren las claves, sin desplegar la app.
    web_payments_enabled: bool = False
    # Publico por diseno: Paddle.js lo necesita en el navegador. Viaja desde el
    # servidor en vez de compilarse en la app para que no puedan quedar
    # desparejados — un token de sandbox con claves de produccion, o al reves,
    # falla de formas que no se parecen a su causa.
    paddle_client_token: str = ""
    paddle_sandbox: bool = True


class LedgerEntryResponse(BaseModel):
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
    usage_call_started_at: UtcDatetime | None
    usage_call_ended_at: UtcDatetime | None
    usage_audio_input_tokens: int | None
    usage_audio_output_tokens: int | None
    usage_image_input_tokens: int | None
    created_at: UtcDatetime


class UsageEventResponse(BaseModel):
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
    created_at: UtcDatetime


class TopUpResponse(BaseModel):
    id: int
    amount_cents: int
    bonus_cents: int
    provider: str
    provider_reference: str
    status: str
    created_at: UtcDatetime


class TopUpRequest(BaseModel):
    amount_cents: int = Field(gt=0)
    provider: str = "manual"
    provider_reference: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class GooglePlayTopUpRequest(BaseModel):
    product_id: str
    purchase_token: str
    order_id: str = ""
    package_name: str = ""
    raw_purchase: dict[str, Any] = Field(default_factory=dict)


def _service(session: AsyncSession, settings: Settings) -> MobileBillingService:
    return MobileBillingService(session, settings)


@router.get("/wallet", response_model=WalletResponse)
async def get_wallet(
    session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> WalletResponse:
    try:
        wallet = await _service(session, settings).get_wallet(current_user.id)
    except BillingError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return WalletResponse(
        user_id=mobile_id(current_user),
        currency=wallet.currency,
        balance_cents=wallet.balance_cents,
        web_payments_enabled=PaddleGateway(settings).enabled,
        paddle_client_token=settings.paddle_client_token,
        paddle_sandbox=settings.paddle_sandbox,
    )


@router.get("/ledger", response_model=list[LedgerEntryResponse])
async def get_ledger(
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[LedgerEntryResponse]:
    entries = await _service(session, settings).list_ledger(
        current_user.id, limit=limit, offset=offset
    )
    return [LedgerEntryResponse(**asdict(entry)) for entry in entries]


@router.get("/usage-events", response_model=list[UsageEventResponse])
async def get_usage_events(
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
    limit: int = Query(default=100, ge=1, le=250),
) -> list[UsageEventResponse]:
    events = await _service(session, settings).list_usage_events(current_user.id, limit=limit)
    return [UsageEventResponse(**asdict(event)) for event in events]


@router.post("/topups", response_model=TopUpResponse)
async def create_topup(
    payload: TopUpRequest, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> TopUpResponse:
    if not settings.billing_manual_topups_enabled:
        raise HTTPException(status_code=403, detail="Las recargas manuales no están habilitadas")
    try:
        topup = await _service(session, settings).create_topup(
            user_id=current_user.id,
            amount_cents=payload.amount_cents,
            provider=payload.provider,
            provider_reference=payload.provider_reference,
            metadata=payload.metadata,
        )
    except BillingError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _topup_response(topup)


@router.post("/google-play/topups/confirm", response_model=TopUpResponse)
async def confirm_google_play_topup(
    payload: GooglePlayTopUpRequest,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> TopUpResponse:
    try:
        topup = await _service(session, settings).confirm_google_play_topup(
            user_id=current_user.id,
            product_id=payload.product_id,
            purchase_token=payload.purchase_token,
            order_id=payload.order_id,
            package_name=payload.package_name,
            raw_purchase=payload.raw_purchase,
        )
    except BillingError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _topup_response(topup)


@router.post("/web/checkout", response_model=WebCheckoutResponse)
async def create_web_checkout(
    payload: WebCheckoutRequest,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> WebCheckoutResponse:
    try:
        url = await _service(session, settings).create_web_checkout(
            user=current_user,
            product_id=payload.product_id,
            amount_cents=payload.amount_cents,
        )
    except PaddleNotConfigured as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (BillingError, PaddleError, TopUpError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return WebCheckoutResponse(url=url)


@router.post("/web/paddle/webhook")
async def paddle_webhook(
    request: Request, session: SessionDep, settings: SettingsDep
) -> dict[str, str]:
    """Donde una recarga web se convierte de verdad en saldo.

    Sin autenticación a propósito: la firma es la credencial. Lo que se verifica
    son los bytes crudos, por eso se leen con `request.body()` — cualquier capa
    que interprete el JSON y lo vuelva a serializar rompe la validación de una
    forma que parece un secreto mal puesto y no lo es.

    Responde 200 incluso a lo que ya había aplicado: una pasarela lee un 500 como
    «vuelve a intentarlo» y reenvía el mismo evento durante días.
    """
    gateway = PaddleGateway(settings)
    try:
        event = gateway.parse_webhook(
            await request.body(),
            request.headers.get("paddle-signature", ""),
            now=time.time(),
        )
    except PaddleNotConfigured as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except PaddleError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    # `transaction.completed` es el que dice que el dinero está cobrado.
    # `transaction.paid` llega antes y puede no traer los totales definitivos.
    if event.get("event_type") != "transaction.completed":
        return {"status": "ignored"}

    transaction = dict(event.get("data") or {})
    try:
        topup = await _service(session, settings).confirm_web_topup(transaction)
    except BillingError as error:
        # Un evento que nunca vamos a poder aplicar (usuario borrado, importe
        # imposible) no debe quedarse reintentando para siempre: se registra y se
        # acepta. El dinero está cobrado en Paddle y se ve en su panel.
        logger.error(
            "paddle_webhook_rejected",
            reason=str(error),
            transaction_id=transaction.get("id", ""),
        )
        return {"status": "rejected"}
    return {"status": "applied" if topup is not None else "unpaid"}


@router.post("/web/checkout/{transaction_id}/confirm", response_model=TopUpResponse | None)
async def confirm_web_checkout(
    transaction_id: str,
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> TopUpResponse | None:
    """Red de seguridad para un webhook que llegue tarde, o que no esté dado de alta.

    Comparte la clave de deduplicación con el webhook, así que llamar a los dos
    abona una sola vez.
    """
    gateway = PaddleGateway(settings)
    try:
        transaction = await gateway.retrieve_transaction(transaction_id)
        topup = await _service(session, settings).confirm_web_topup(transaction)
    except PaddleNotConfigured as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (BillingError, PaddleError, TopUpError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _topup_response(topup) if topup is not None else None


def _topup_response(topup: Any) -> TopUpResponse:
    return TopUpResponse(
        id=topup.id,
        amount_cents=topup.amount_cents,
        bonus_cents=topup.bonus_cents,
        provider=topup.provider,
        provider_reference=topup.provider_reference,
        status=topup.status,
        created_at=topup.created_at.isoformat(),
    )
