from fastapi import APIRouter, Depends, HTTPException

from app.deps.auth import get_current_user_optional, get_current_user_required
from app.schemas.auth import UserResponse
from app.schemas.session import (
    SessionCallLogRequest,
    SessionCallStateRequest,
    SessionCreateRequest,
    SessionParticipantTouchRequest,
    SessionResponse,
    SessionUpdateRequest,
)
from app.services.session_service import session_service


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse)
async def create_session(
    payload: SessionCreateRequest,
    current_user: UserResponse | None = Depends(get_current_user_optional),
) -> SessionResponse:
    if current_user is not None:
        payload.user_id = current_user.id
    session = session_service.create_session(payload)
    return SessionResponse(session=session)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    session = session_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(session=session)


@router.post("/{session_id}/reset", response_model=SessionResponse)
async def reset_session(session_id: str) -> SessionResponse:
    session = session_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session = session_service.reset_conversation(session_id)
    return SessionResponse(session=session)


@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    payload: SessionUpdateRequest,
    current_user: UserResponse | None = Depends(get_current_user_optional),
) -> SessionResponse:
    if current_user is not None:
        payload.user_id = current_user.id
    session = session_service.update_session(session_id, payload)
    return SessionResponse(session=session)


@router.post("/{session_id}/presence", response_model=SessionResponse)
async def touch_participant_presence(
    session_id: str,
    payload: SessionParticipantTouchRequest,
    current_user: UserResponse = Depends(get_current_user_required),
) -> SessionResponse:
    session = session_service.touch_participant(session_id, current_user, active_call=payload.active_call)
    return SessionResponse(session=session)


@router.delete("/{session_id}/presence", response_model=SessionResponse)
async def leave_participant_presence(
    session_id: str,
    current_user: UserResponse = Depends(get_current_user_required),
) -> SessionResponse:
    session = session_service.leave_participant(session_id, current_user)
    return SessionResponse(session=session)


@router.post("/{session_id}/call-state", response_model=SessionResponse)
async def set_call_state(
    session_id: str,
    payload: SessionCallStateRequest,
    current_user: UserResponse = Depends(get_current_user_required),
) -> SessionResponse:
    session = session_service.set_call_state(session_id, current_user, payload.status)
    return SessionResponse(session=session)


@router.post("/{session_id}/call-log", response_model=SessionResponse)
async def append_call_log(
    session_id: str,
    payload: SessionCallLogRequest,
    current_user: UserResponse = Depends(get_current_user_required),
) -> SessionResponse:
    session = session_service.append_call_log(
        session_id,
        user=current_user,
        kind=payload.kind,
        author=payload.author,
        text=payload.text,
        image_url=payload.image_url,
    )
    return SessionResponse(session=session)
