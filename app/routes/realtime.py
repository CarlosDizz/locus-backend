import json

from fastapi import APIRouter, Depends, HTTPException

from app.clients.openai_client import OpenAIClientError
from app.deps.auth import get_current_user_optional
from app.schemas.auth import UserResponse
from app.schemas.realtime import (
    RealtimeClientSecretResponse,
    RealtimePhotoInsightRequest,
    RealtimePhotoInsightResponse,
    RealtimeSessionRequest,
    RealtimeSessionResponse,
    RealtimeToolRequest,
)
from app.services.billing_service import BillingError
from app.services.realtime_service import realtime_service
from app.services.tool_runtime_service import tool_runtime_service


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


@router.post("/tool")
async def execute_realtime_tool(payload: RealtimeToolRequest) -> dict:
    result = tool_runtime_service.execute(payload.session_id, payload.tool_name, payload.arguments)
    return json.loads(result)


@router.post("/photo-insight", response_model=RealtimePhotoInsightResponse)
async def get_realtime_photo_insight(
    payload: RealtimePhotoInsightRequest,
    current_user: UserResponse | None = Depends(get_current_user_optional),
) -> RealtimePhotoInsightResponse:
    if current_user is not None:
        realtime_service.prepare_session(
            RealtimeSessionRequest(session_id=payload.session_id, user_id=current_user.id)
        )
    try:
        return realtime_service.analyze_photo(payload)
    except OpenAIClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
