"""Authenticated Ionic map-chat facade."""

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from locus_v2.api.auth import CurrentUserDep, SessionDep, SettingsDep
from locus_v2.billing.models import Wallet
from locus_v2.chat.configuration import ChatConfigurationError
from locus_v2.chat.schemas import ChatMessageRequest, ChatResponse, ChatSetupRequest
from locus_v2.chat.service import ChatService, ChatServiceError
from locus_v2.config import Settings
from locus_v2.places.service import PlaceSearchService, distance_km
from locus_v2.sessions.application.service import MapSessionService
from locus_v2.sessions.models import SessionStateView

router = APIRouter(prefix="/api/chat", tags=["chat"])

BASE_MAP_QUERY = "lugares turisticos"
BASE_MAP_LIMIT = 8
# Below this the map the user is already looking at is still the right one;
# above it they have travelled and the base pins are stale. Same rule as V1.
BASE_MAP_REFRESH_KM = 10.0


@router.post("/setup", response_model=ChatResponse)
async def setup_chat(
    payload: ChatSetupRequest, session: SessionDep, settings: SettingsDep,
    current_user: CurrentUserDep,
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
    # V1 seeded the base map here; without it the very first message reached the
    # model with an empty `nearby_pois`, so it could only talk in generalities.
    state = await _refresh_base_map(service, session, settings, state, force=True)
    await service.set_ephemeral_map_pois(state.session_id, [])
    return ChatResponse(
        session_id=state.session_id, reply="", pois=state.nearby_pois,
        ephemeral_pois=[],
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
    state = await _refresh_base_map(service, session, settings, state)
    message = payload.message.strip()
    await service.append_memory(state.session_id, "user", message)
    try:
        result = await ChatService(session, settings).send_message(
            user_id=current_user.id, routing_profile="", context_type="map",
            context_id=None, locale=state.profile.language,
            message=message, map_session=state,
        )
    except (ChatConfigurationError, ChatServiceError) as error:
        raise HTTPException(503, "El chat no esta disponible temporalmente") from error
    final = await service.append_memory(state.session_id, "assistant", result.reply)
    # Read the map back AFTER the turn: mark_pois_on_map / promote_poi_to_catalog
    # run inside it, and the app renders whatever this response carries.
    return ChatResponse(
        session_id=final.session_id, reply=result.reply, pois=final.nearby_pois,
        ephemeral_pois=final.ephemeral_map_pois,
    )


async def _refresh_base_map(
    service: MapSessionService,
    session: SessionDep,
    settings: Settings,
    state: SessionStateView,
    *,
    force: bool = False,
) -> SessionStateView:
    lat, lng = state.location.lat, state.location.lng
    if lat is None or lng is None:
        return state
    if not force and state.nearby_pois:
        if _centroid_distance_km(lat, lng, state) < BASE_MAP_REFRESH_KM:
            return state
    pois = await PlaceSearchService(session, settings).search_catalog(
        query=BASE_MAP_QUERY, lat=lat, lng=lng,
        locale=state.profile.language, limit=BASE_MAP_LIMIT,
    )
    if not pois:
        return state
    return await service.set_nearby_pois(state.session_id, pois)


def _centroid_distance_km(lat: float, lng: float, state: SessionStateView) -> float:
    pois = state.nearby_pois
    return distance_km(
        lat, lng,
        sum(poi.lat for poi in pois) / len(pois),
        sum(poi.lng for poi in pois) / len(pois),
    )
