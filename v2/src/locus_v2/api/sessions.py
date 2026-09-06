"""V1-compatible map session facade.

Mounted at /api/sessions, same shape as app/routes/sessions.py +
app/schemas/session.py. See sessions/application/service.py for what this
does and does not cover (no call-room orchestration here).
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.api.auth import CurrentUserDep, OptionalUserDep
from locus_v2.infrastructure.database.session import get_session
from locus_v2.identity.models import User
from locus_v2.sessions.application.service import MapSessionService, SessionNotFoundError
from locus_v2.sessions.models import SessionStateView
from locus_v2.shared.mobile_ids import mobile_id

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]


class SessionCreateRequest(BaseModel):
    session_id: str | None = None
    user_id: int | None = None
    profile_context: str = ""
    lat: float | None = None
    lng: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionUpdateRequest(BaseModel):
    user_id: int | None = None
    profile_context: str | None = None
    profile_preferences: dict[str, Any] | None = None
    lat: float | None = None
    lng: float | None = None
    active_poi_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionParticipantTouchRequest(BaseModel):
    active_call: bool = False


class SessionCallStateRequest(BaseModel):
    status: str = "idle"


class SessionCallLogRequest(BaseModel):
    kind: str
    author: str
    text: str
    image_url: str | None = None


class SessionResponse(BaseModel):
    session: SessionStateView


def _service(session: AsyncSession) -> MapSessionService:
    return MapSessionService(session)


async def _response(state: SessionStateView, session: AsyncSession) -> SessionResponse:
    result = state.model_copy(deep=True)
    ids = {result.user_id, result.call_live.host_user_id}
    ids.update(item.user_id for item in result.participants)
    ids.update(item.user_id for item in result.call_log)
    users = await session.scalars(select(User).where(User.id.in_(ids - {None})))
    mapping = {user.id: mobile_id(user) for user in users}
    result.user_id = mapping.get(result.user_id)
    result.call_live.host_user_id = mapping.get(result.call_live.host_user_id)
    result.participants = [item for item in result.participants if item.user_id in mapping]
    for item in result.participants:
        item.user_id = mapping[item.user_id]
    for entry in result.call_log:
        entry.user_id = mapping.get(entry.user_id)
    for key in ("participants", "call_live", "call_log"):
        if key in result.metadata:
            value = getattr(result, key)
            result.metadata[key] = (
                [item.model_dump() for item in value]
                if isinstance(value, list) else value.model_dump()
            )
    return SessionResponse(session=result)


def _client_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items()
            if key not in {"participants", "call_live", "call_log"}}


@router.post("", response_model=SessionResponse)
async def create_session(
    payload: SessionCreateRequest, session: SessionDep, current_user: OptionalUserDep
) -> SessionResponse:
    user_id = current_user.id if current_user is not None else None
    state = await _service(session).create_session(
        session_id=payload.session_id,
        user_id=user_id,
        profile_context=payload.profile_context,
        lat=payload.lat,
        lng=payload.lng,
        metadata=_client_metadata(payload.metadata),
    )
    return await _response(state, session)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session_state(session_id: str, session: SessionDep) -> SessionResponse:
    state = await _service(session).get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return await _response(state, session)


@router.post("/{session_id}/reset", response_model=SessionResponse)
async def reset_session(session_id: str, session: SessionDep) -> SessionResponse:
    try:
        state = await _service(session).reset_conversation(session_id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Session not found") from error
    return await _response(state, session)


@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    payload: SessionUpdateRequest,
    session: SessionDep,
    current_user: OptionalUserDep,
) -> SessionResponse:
    user_id = current_user.id if current_user is not None else None
    state = await _service(session).update_session(
        session_id,
        user_id=user_id,
        profile_context=payload.profile_context,
        profile_preferences=payload.profile_preferences,
        lat=payload.lat,
        lng=payload.lng,
        active_poi_name=payload.active_poi_name,
        metadata=_client_metadata(payload.metadata),
    )
    return await _response(state, session)


@router.post("/{session_id}/presence", response_model=SessionResponse)
async def touch_participant_presence(
    session_id: str,
    payload: SessionParticipantTouchRequest,
    session: SessionDep,
    current_user: CurrentUserDep,
) -> SessionResponse:
    state = await _service(session).touch_participant(
        session_id, current_user, active_call=payload.active_call
    )
    return await _response(state, session)


@router.delete("/{session_id}/presence", response_model=SessionResponse)
async def leave_participant_presence(
    session_id: str, session: SessionDep, current_user: CurrentUserDep
) -> SessionResponse:
    state = await _service(session).leave_participant(session_id, current_user)
    return await _response(state, session)


@router.post("/{session_id}/call-state", response_model=SessionResponse)
async def set_call_state(
    session_id: str,
    payload: SessionCallStateRequest,
    session: SessionDep,
    current_user: CurrentUserDep,
) -> SessionResponse:
    state = await _service(session).set_call_state(session_id, current_user, payload.status)
    return await _response(state, session)


@router.post("/{session_id}/call-log", response_model=SessionResponse)
async def append_call_log(
    session_id: str,
    payload: SessionCallLogRequest,
    session: SessionDep,
    current_user: CurrentUserDep,
) -> SessionResponse:
    state = await _service(session).append_call_log(
        session_id,
        user=current_user,
        kind=payload.kind,
        author=payload.author,
        text=payload.text,
        image_url=payload.image_url,
    )
    return await _response(state, session)
