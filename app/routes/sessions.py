from fastapi import APIRouter, Depends, HTTPException

from app.deps.auth import get_current_user_optional
from app.schemas.auth import UserResponse
from app.schemas.session import SessionCreateRequest, SessionResponse, SessionUpdateRequest
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
