"""V1-compatible real-time call rooms (app/routes/calls.py, app/schemas/call.py).

REST endpoints create/join/leave/end a call; the WebSocket carries the live
floor/audio/text/image protocol the real Ionic app's room-socket.service.ts
speaks. Call state itself lives in Redis (locus_v2.calls.store.RoomStore), not
in this process, so the socket and the REST handlers can be served from
different requests without sharing memory.
"""

import asyncio
import contextlib
import json
from functools import partial
from typing import Any
from uuid import uuid4

import structlog
from fastapi import (
    APIRouter,
    Query,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)
from fastapi.exceptions import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from locus_v2.api.auth import CurrentUserDep, SessionDep, SettingsDep
from locus_v2.calls.bridge import ensure_bridge
from locus_v2.calls.models import CallError, CreateCall
from locus_v2.calls.policy import ensure_host_can_consume, resolve_context
from locus_v2.calls.service import CallService, decode_image
from locus_v2.calls.store import RoomStore
from locus_v2.config import Settings, get_settings
from locus_v2.identity.application.mobile_auth import MobileAuthService
from locus_v2.identity.models import User
from locus_v2.infrastructure.database import get_database
from locus_v2.infrastructure.redis import get_redis

logger = structlog.get_logger()

router = APIRouter(prefix="/api/calls", tags=["calls"])
ws_router = APIRouter(tags=["calls"])


class CallResponse(BaseModel):
    call: dict[str, Any]
    join_token: str


class CallActionResponse(BaseModel):
    call: dict[str, Any]


def _service(settings: Settings) -> CallService:
    store = RoomStore(get_redis())
    consume = partial(ensure_host_can_consume, get_database(), settings)
    return CallService(store, settings, consume)


@router.post("", response_model=CallResponse)
async def create_call(
    payload: CreateCall, session: SessionDep, settings: SettingsDep, current_user: CurrentUserDep
) -> CallResponse:
    try:
        poi = await resolve_context(session, current_user, payload)
        service = _service(settings)
        room = await service.create(current_user, payload, poi, routing_profile="poi_guide")
    except CallError as error:
        raise HTTPException(error.status, str(error)) from error
    ensure_bridge(room.call_id, service.store, get_database(), settings)
    return CallResponse(call=room.snapshot(), join_token=service.join_token(room, current_user.id))


@router.post("/{call_id}/join-token", response_model=CallResponse)
async def create_join_token(
    call_id: str, settings: SettingsDep, current_user: CurrentUserDep
) -> CallResponse:
    service = _service(settings)
    try:
        room = await service.join(call_id, current_user)
    except CallError as error:
        raise HTTPException(error.status, str(error)) from error
    ensure_bridge(room.call_id, service.store, get_database(), settings)
    return CallResponse(call=room.snapshot(), join_token=service.join_token(room, current_user.id))


@router.post("/{call_id}/leave", response_model=CallActionResponse)
async def leave_call(
    call_id: str, settings: SettingsDep, current_user: CurrentUserDep
) -> CallActionResponse:
    service = _service(settings)
    try:
        room = await service.leave(call_id, current_user.id)
    except CallError as error:
        raise HTTPException(error.status, str(error)) from error
    return CallActionResponse(call=room.snapshot())


@router.post("/{call_id}/end", response_model=CallActionResponse)
async def end_call(
    call_id: str, settings: SettingsDep, current_user: CurrentUserDep
) -> CallActionResponse:
    service = _service(settings)
    try:
        room = await service.leave(call_id, current_user.id, end=True)
    except CallError as error:
        raise HTTPException(error.status, str(error)) from error
    return CallActionResponse(call=room.snapshot())


@router.get("/{call_id}/images/{image_id}")
async def get_call_image(
    call_id: str, image_id: str, settings: SettingsDep, t: str = Query(default="")
) -> Response:
    """Serve a photo shared in a call, addressed by the URL kept in the room log.

    Authorized by the signed `t` token rather than the session bearer: an
    <img src> cannot send an Authorization header, and this URL only ever
    reaches people who can already read the call's transcript.
    """
    service = _service(settings)
    try:
        service.verify_image_token(t, call_id, image_id)
        data_url = await service.store.get_image(call_id, image_id)
    except CallError as error:
        raise HTTPException(error.status, str(error)) from error
    if data_url is None:
        raise HTTPException(404, "Image not found or expired")
    try:
        mime_type, image_bytes = decode_image(data_url)
    except CallError as error:
        raise HTTPException(500, "Stored image is unreadable") from error
    return Response(
        content=image_bytes,
        media_type=mime_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


def _extract_ws_bearer_token(websocket: WebSocket) -> str:
    authorization = websocket.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return str(websocket.query_params.get("token") or "").strip()


async def _resolve_ws_user(websocket: WebSocket, settings: Settings) -> User:
    token = _extract_ws_bearer_token(websocket)
    if not token:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION, reason="Autenticacion requerida"
        )
    async with get_database().sessions() as session:
        user = await MobileAuthService(session, settings).authenticate(token)
    if user is None:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION, reason="Autenticacion requerida"
        )
    return user


@ws_router.websocket("/ws/calls/{call_id}")
async def call_socket(
    websocket: WebSocket, call_id: str, join_token: str = Query(default="")
) -> None:
    settings = get_settings()
    user = await _resolve_ws_user(websocket, settings)
    service = _service(settings)
    try:
        service.verify_token(join_token, call_id, user.id)
    except CallError as error:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason=str(error)) from error

    await websocket.accept()
    connection = uuid4().hex
    try:
        room = await service.connect(call_id, user.id, join_token, connection)
    except CallError as error:
        await websocket.send_json({"type": "call.error", "message": str(error)})
        await websocket.close()
        return
    ensure_bridge(room.call_id, service.store, get_database(), settings)
    await websocket.send_json(
        {"type": "call.snapshot", "call": room.snapshot(), "ui": room.ui(user.id)}
    )

    events_key = service.store.key(call_id, "events")
    pubsub = service.store.redis.pubsub()
    await pubsub.subscribe(events_key)

    async def forward_events() -> None:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            event = json.loads(message["data"])
            if event.get("type") == "state":
                try:
                    current = await service.store.get(call_id)
                except CallError:
                    return
                await websocket.send_json(
                    {"type": "call.snapshot", "call": current.snapshot(), "ui": current.ui(user.id)}
                )
                continue
            if "target" in event and event["target"] != user.id:
                continue
            if "exclude" in event and event["exclude"] == user.id:
                continue
            await websocket.send_json(
                {k: v for k, v in event.items() if k not in {"target", "exclude"}}
            )

    async def receive_client() -> None:
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                return
            try:
                event = json.loads(raw)
            except ValueError:
                await websocket.send_json({"type": "call.error", "message": "invalid_event"})
                continue
            logger.info(
                "call_client_event", call_id=call_id, user_id=user.id, kind=event.get("type")
            )
            try:
                await service.event(call_id, user.id, connection, event)
            except CallError as error:
                logger.warning(
                    "call_client_event_rejected",
                    call_id=call_id,
                    user_id=user.id,
                    kind=event.get("type"),
                    error=str(error),
                )
                await websocket.send_json({"type": "call.error", "message": str(error)})

    async def heartbeat_loop() -> None:
        # Room.expire() (run inside every RoomStore.change()) treats a member as
        # disconnected once 20s pass with no seen_at update — but neither this
        # gateway nor the Ionic client ever sent one after the initial connect,
        # so any pause longer than 20s (typing a message, thinking) silently
        # invalidated the connection and the next real event got rejected with
        # "Connection was replaced or membership revoked". A live socket proves
        # liveness on its own; keep seen_at fresh for as long as it stays open.
        while True:
            await asyncio.sleep(8)
            try:
                await service.heartbeat(call_id, user.id, connection)
            except CallError:
                return

    forward_task = asyncio.create_task(forward_events())
    receive_task = asyncio.create_task(receive_client())
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    try:
        done, pending = await asyncio.wait(
            {forward_task, receive_task, heartbeat_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError):
                await task
    finally:
        with contextlib.suppress(Exception):
            await pubsub.unsubscribe(events_key)
            await pubsub.aclose()
        with contextlib.suppress(CallError):
            await service.disconnect(call_id, user.id, connection)
