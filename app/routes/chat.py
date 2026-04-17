import json

from fastapi import APIRouter, Depends

from app.deps.auth import get_current_user_optional
from app.schemas.auth import UserResponse

from app.schemas.chat import ChatMessageRequest, ChatResponse, ChatSetupRequest
from app.schemas.session import SessionUpdateRequest
from app.services.chat_service import chat_service
from app.services.session_service import session_service


router = APIRouter(prefix="/api/chat", tags=["chat"])
compat_router = APIRouter(tags=["compat"])


@router.post("/setup", response_model=ChatResponse)
async def setup_chat(
    payload: ChatSetupRequest,
    current_user: UserResponse | None = Depends(get_current_user_optional),
) -> ChatResponse:
    if current_user is not None:
        payload.user_id = current_user.id
    return chat_service.setup_chat(payload)


@router.post("/messages", response_model=ChatResponse)
async def send_message(
    payload: ChatMessageRequest,
    current_user: UserResponse | None = Depends(get_current_user_optional),
) -> ChatResponse:
    if current_user is not None:
        payload.user_id = current_user.id
    return chat_service.handle_message(payload)


@compat_router.post("/home_chat")
async def home_chat(payload: dict) -> dict:
    action = payload.get("action", "")
    session_id = payload.get("roomId", "")
    lat = payload.get("lat")
    lng = payload.get("lng")

    if action == "setup_profile":
        response = chat_service.setup_chat(
            ChatSetupRequest(
                session_id=session_id,
                user_id=None,
                profile_context=payload.get("context", ""),
                lat=lat,
                lng=lng,
            )
        )
    else:
        if payload.get("context"):
            session_service.update_session(
                session_id,
                SessionUpdateRequest(user_id=None, profile_context=payload["context"], lat=lat, lng=lng),
            )
        response = chat_service.handle_message(
            ChatMessageRequest(
                session_id=session_id,
                user_id=None,
                message=payload.get("text", ""),
                lat=lat,
                lng=lng,
            )
        )

    pois_payload = ""
    if response.pois:
        pois_payload = f"\n<POIS>{json.dumps([poi.model_dump() for poi in response.pois], ensure_ascii=False)}</POIS>"
    return {"reply": f"{response.reply}{pois_payload}".strip()}
