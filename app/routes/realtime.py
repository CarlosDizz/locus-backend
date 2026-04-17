from fastapi import APIRouter, Depends, HTTPException

from app.clients.openai_client import OpenAIClientError
from app.deps.auth import get_current_user_optional
from app.schemas.auth import UserResponse
from app.schemas.realtime import RealtimeClientSecretResponse, RealtimeSessionRequest, RealtimeSessionResponse
from app.services.billing_service import BillingError
from app.services.realtime_service import realtime_service


router = APIRouter(prefix="/api/realtime", tags=["realtime"])


@router.post("/session", response_model=RealtimeSessionResponse)
async def prepare_realtime_session(
    payload: RealtimeSessionRequest,
    current_user: UserResponse | None = Depends(get_current_user_optional),
) -> RealtimeSessionResponse:
    if current_user is not None:
        payload.user_id = current_user.id
    return realtime_service.prepare_session(payload)


@router.post("/client-secret", response_model=RealtimeClientSecretResponse)
async def create_realtime_client_secret(
    payload: RealtimeSessionRequest,
    current_user: UserResponse | None = Depends(get_current_user_optional),
) -> RealtimeClientSecretResponse:
    if current_user is not None:
        payload.user_id = current_user.id
    try:
        return realtime_service.create_client_secret(payload)
    except OpenAIClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BillingError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
