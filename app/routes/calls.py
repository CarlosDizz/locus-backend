from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketException, status

from app.deps.auth import get_current_user_required
from app.schemas.auth import UserResponse
from app.schemas.call import CallActionResponse, CallCreateRequest, CallCreateResponse, CallJoinTokenResponse
from app.services.auth_service import auth_service
from app.services.call_room_service import CallRoomError, call_room_service


router = APIRouter(prefix="/api/calls", tags=["calls"])
ws_router = APIRouter(tags=["calls"])


def _extract_ws_bearer_token(websocket: WebSocket) -> str:
    authorization = websocket.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return str(websocket.query_params.get("token") or "").strip()


def _resolve_ws_user(websocket: WebSocket) -> UserResponse:
    token = _extract_ws_bearer_token(websocket)
    user = auth_service.get_user_from_token(token)
    if user is None:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Autenticación requerida")
    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        auth_provider=user.auth_provider,
        avatar_url=user.avatar_url,
        is_active=user.is_active,
    )


@router.post("", response_model=CallCreateResponse)
async def create_call(
    payload: CallCreateRequest,
    current_user: UserResponse = Depends(get_current_user_required),
) -> CallCreateResponse:
    try:
        call, join_token = await call_room_service.create_call(
            user=current_user,
            session_id=payload.session_id,
            poi_id=payload.poi_id,
            poi_name=payload.poi_name,
        )
    except CallRoomError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CallCreateResponse(call=call, join_token=join_token)


@router.post("/{call_id}/join-token", response_model=CallJoinTokenResponse)
async def create_join_token(
    call_id: str,
    current_user: UserResponse = Depends(get_current_user_required),
) -> CallJoinTokenResponse:
    try:
        call, join_token = await call_room_service.get_join_token(call_id=call_id, user=current_user)
    except CallRoomError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CallJoinTokenResponse(call=call, join_token=join_token)


@router.post("/{call_id}/leave", response_model=CallActionResponse)
async def leave_call(
    call_id: str,
    current_user: UserResponse = Depends(get_current_user_required),
) -> CallActionResponse:
    try:
        call = await call_room_service.leave_call(call_id=call_id, user=current_user)
    except CallRoomError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CallActionResponse(call=call)


@router.post("/{call_id}/end", response_model=CallActionResponse)
async def end_call(
    call_id: str,
    current_user: UserResponse = Depends(get_current_user_required),
) -> CallActionResponse:
    try:
        call = await call_room_service.end_call(call_id=call_id, user=current_user)
    except CallRoomError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CallActionResponse(call=call)


@ws_router.websocket("/ws/calls/{call_id}")
async def call_socket(
    websocket: WebSocket,
    call_id: str,
    join_token: str = Query(default=""),
) -> None:
    current_user = _resolve_ws_user(websocket)
    try:
        token_call_id, token_user_id = call_room_service.verify_join_token(join_token)
    except CallRoomError as exc:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc)) from exc

    if token_call_id.upper() != call_id.upper() or token_user_id != current_user.id:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Join token inválido")

    await call_room_service.connect(websocket=websocket, call_id=call_id, user=current_user)
