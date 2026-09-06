"""Authenticated Ionic map-chat facade."""

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from locus_v2.api.auth import CurrentUserDep, SessionDep, SettingsDep
from locus_v2.billing.models import Wallet
from locus_v2.chat.configuration import ChatConfigurationError
from locus_v2.chat.schemas import ChatMessageRequest, ChatResponse, ChatSetupRequest
from locus_v2.chat.service import ChatService, ChatServiceError
from locus_v2.sessions.application.service import MapSessionService

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/setup", response_model=ChatResponse)
async def setup_chat(
    payload: ChatSetupRequest, session: SessionDep, current_user: CurrentUserDep,
) -> ChatResponse:
    service = MapSessionService(session)
    existing = await service.get_session(payload.session_id)
    if existing and existing.user_id not in (None, current_user.id):
        raise HTTPException(403, "La conversacion pertenece a otro usuario")
    state = await service.create_session(
        session_id=payload.session_id, user_id=current_user.id,
        profile_context=payload.profile_context, lat=payload.lat, lng=payload.lng,
        metadata={},
    )
    return ChatResponse(
        session_id=state.session_id, reply="", pois=state.nearby_pois,
        ephemeral_pois=state.ephemeral_map_pois,
    )


@router.post("/messages", response_model=ChatResponse)
async def chat_message(
    payload: ChatMessageRequest, session: SessionDep, settings: SettingsDep,
    current_user: CurrentUserDep,
) -> ChatResponse:
    service = MapSessionService(session)
    state = await service.get_session(payload.session_id)
    if state is None:
        raise HTTPException(404, "Conversacion no encontrada")
    if state.user_id != current_user.id:
        raise HTTPException(403, "La conversacion pertenece a otro usuario")
    wallet = await session.scalar(select(Wallet).where(Wallet.user_id == current_user.id))
    if wallet is None or wallet.balance_cents <= 0:
        raise HTTPException(402, "Saldo insuficiente")
    state = await service.update_session(
        state.session_id, user_id=current_user.id, profile_context=None,
        profile_preferences=None, lat=payload.lat, lng=payload.lng,
        active_poi_name=None, metadata={},
    )
    try:
        result = await ChatService(session, settings).send_message(
            user_id=current_user.id, routing_profile="", context_type="map",
            context_id=None, locale=state.profile.language,
            message=payload.message.strip(), map_session=state,
        )
    except (ChatConfigurationError, ChatServiceError) as error:
        raise HTTPException(503, "El chat no esta disponible temporalmente") from error
    await service.append_memory(state.session_id, "user", payload.message.strip())
    await service.append_memory(state.session_id, "assistant", result.reply)
    return ChatResponse(
        session_id=state.session_id, reply=result.reply, pois=state.nearby_pois,
        ephemeral_pois=state.ephemeral_map_pois,
    )
