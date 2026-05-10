from fastapi import APIRouter, Depends, HTTPException

from app.deps.auth import get_current_user_required
from app.schemas.auth import UserResponse
from app.schemas.billing import (
    LedgerEntryResponse,
    TopUpRequest,
    TopUpResponse,
    UsageEventResponse,
    UsageRecordRequest,
    UsageRecordResponse,
    WalletResponse,
)
from app.services.billing_service import BillingError, billing_service


router = APIRouter(prefix="/api/billing", tags=["billing"])


@router.get("/wallet", response_model=WalletResponse)
async def get_wallet(current_user: UserResponse = Depends(get_current_user_required)) -> WalletResponse:
    wallet = billing_service.get_wallet(current_user.id)
    if wallet is None:
        raise HTTPException(status_code=404, detail="Wallet no encontrada")
    return WalletResponse(user_id=current_user.id, currency=wallet.currency, balance_cents=wallet.balance_cents)


@router.get("/ledger", response_model=list[LedgerEntryResponse])
async def get_ledger(current_user: UserResponse = Depends(get_current_user_required)) -> list[LedgerEntryResponse]:
    entries = billing_service.list_ledger_entries(current_user.id)
    return [
        LedgerEntryResponse(
            id=entry.id,
            entry_type=entry.entry_type,
            amount_cents=entry.amount_cents,
            balance_after_cents=entry.balance_after_cents,
            description=entry.description,
            reference_type=entry.reference_type,
            reference_id=entry.reference_id,
            created_at=entry.created_at,
        )
        for entry in entries
    ]


@router.get("/usage-events", response_model=list[UsageEventResponse])
async def get_usage_events(current_user: UserResponse = Depends(get_current_user_required)) -> list[UsageEventResponse]:
    events = billing_service.list_usage_events(current_user.id)
    return [
        UsageEventResponse(
            id=event.id,
            session_id=event.session_id,
            provider=event.provider,
            endpoint=event.endpoint,
            model=event.model,
            interaction_type=event.interaction_type,
            source=event.source,
            response_id=event.response_id,
            input_tokens=event.input_tokens,
            cached_input_tokens=event.cached_input_tokens,
            output_tokens=event.output_tokens,
            reasoning_tokens=event.reasoning_tokens,
            audio_input_tokens=event.audio_input_tokens,
            audio_output_tokens=event.audio_output_tokens,
            image_input_tokens=event.image_input_tokens,
            provider_cost_eur_cents=event.provider_cost_eur_cents,
            charged_amount_cents=event.charged_amount_cents,
            gross_margin_cents=event.gross_margin_cents,
            currency=event.currency,
            status=event.status,
            created_at=event.created_at,
        )
        for event in events
    ]


@router.post("/topups", response_model=TopUpResponse)
async def create_topup(
    payload: TopUpRequest,
    current_user: UserResponse = Depends(get_current_user_required),
) -> TopUpResponse:
    try:
        topup = billing_service.create_topup(
            user_id=current_user.id,
            amount_cents=payload.amount_cents,
            provider=payload.provider,
            provider_reference=payload.provider_reference,
            metadata=payload.metadata,
        )
    except BillingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TopUpResponse(
        id=topup.id,
        amount_cents=topup.amount_cents,
        bonus_cents=topup.bonus_cents,
        provider=topup.provider,
        provider_reference=topup.provider_reference,
        status=topup.status,
        created_at=topup.created_at,
    )


@router.post("/usage-events", response_model=UsageRecordResponse)
async def record_usage(
    payload: UsageRecordRequest,
    current_user: UserResponse = Depends(get_current_user_required),
) -> UsageRecordResponse:
    try:
        usage_event, wallet = billing_service.record_usage(
            user_id=current_user.id,
            session_id=payload.session_id,
            provider=payload.provider,
            endpoint=payload.endpoint,
            model=payload.model,
            response_id=payload.response_id,
            usage=payload.usage,
            metadata=payload.metadata,
        )
    except BillingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UsageRecordResponse(
        id=usage_event.id,
        charged_amount_cents=usage_event.charged_amount_cents,
        balance_cents=wallet.balance_cents,
    )
