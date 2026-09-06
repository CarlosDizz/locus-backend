"""V1-compatible mobile billing facade.

Mounted at /api/billing, same prefix and shapes (adapted where V2's schema
genuinely differs — see mobile_billing.py's docstring) as app/routes/billing.py.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.api.auth import CurrentUserDep
from locus_v2.billing.application.mobile_billing import BillingError, MobileBillingService
from locus_v2.config import Settings, get_settings
from locus_v2.infrastructure.database.session import get_session

router = APIRouter(prefix="/api/billing", tags=["billing"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


class WalletResponse(BaseModel):
    user_id: int
    currency: str
    balance_cents: int


class LedgerEntryResponse(BaseModel):
    id: int
    kind: str
    amount_cents: int
    balance_after_cents: int
    description: str
    reference_type: str
    reference_id: str
    usage_event_id: int | None
    usage_interaction_type: str | None
    voice_session_id: int | None
    voice_call_started_at: str | None
    voice_call_ended_at: str | None
    created_at: str


class UsageEventResponse(BaseModel):
    id: int
    interaction_type: str
    text_input_tokens: int
    cached_text_input_tokens: int
    text_output_tokens: int
    audio_input_tokens: int
    audio_output_tokens: int
    image_input_tokens: int
    provider_cost_eur_cents: int
    charged_amount_cents: int
    gross_margin_cents: int
    currency: str
    status: str
    created_at: str


class TopUpResponse(BaseModel):
    id: int
    amount_cents: int
    bonus_cents: int
    provider: str
    provider_reference: str
    status: str
    created_at: str


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
        user_id=wallet.user_id, currency=wallet.currency, balance_cents=wallet.balance_cents
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
    return [
        LedgerEntryResponse(
            id=entry.id,
            kind=entry.kind,
            amount_cents=entry.amount_cents,
            balance_after_cents=entry.balance_after_cents,
            description=entry.description,
            reference_type=entry.reference_type,
            reference_id=entry.reference_id,
            usage_event_id=entry.usage_event_id,
            usage_interaction_type=entry.usage_interaction_type,
            voice_session_id=entry.voice_session_id,
            voice_call_started_at=(
                entry.voice_call_started_at.isoformat() if entry.voice_call_started_at else None
            ),
            voice_call_ended_at=(
                entry.voice_call_ended_at.isoformat() if entry.voice_call_ended_at else None
            ),
            created_at=entry.created_at.isoformat(),
        )
        for entry in entries
    ]


@router.get("/usage-events", response_model=list[UsageEventResponse])
async def get_usage_events(
    session: SessionDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
    limit: int = Query(default=100, ge=1, le=250),
) -> list[UsageEventResponse]:
    events = await _service(session, settings).list_usage_events(current_user.id, limit=limit)
    return [
        UsageEventResponse(
            id=event.id,
            interaction_type=event.interaction_type,
            text_input_tokens=event.text_input_tokens,
            cached_text_input_tokens=event.cached_text_input_tokens,
            text_output_tokens=event.text_output_tokens,
            audio_input_tokens=event.audio_input_tokens,
            audio_output_tokens=event.audio_output_tokens,
            image_input_tokens=event.image_input_tokens,
            provider_cost_eur_cents=event.provider_cost_eur_cents,
            charged_amount_cents=event.charged_amount_cents,
            gross_margin_cents=event.gross_margin_cents,
            currency=event.currency,
            status=event.status,
            created_at=event.created_at.isoformat(),
        )
        for event in events
    ]


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
