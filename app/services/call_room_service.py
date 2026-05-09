from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import queue
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

import websocket

from app.config import settings
from app.schemas.auth import UserResponse
from app.schemas.call import CallParticipant, CallSnapshot, UICapabilities
from app.schemas.session import SessionUpdateRequest
from app.services.billing_service import BillingError, billing_service
from app.services.realtime_service import realtime_service
from app.services.session_service import session_service
from app.services.tool_runtime_service import tool_runtime_service


@dataclass
class ParticipantRuntime:
    user: UserResponse
    joined_at: datetime
    websocket: WebSocket | None = None
    connected: bool = False

    def snapshot(self, *, is_host: bool) -> CallParticipant:
        return CallParticipant(
            user_id=self.user.id,
            display_name=self.user.display_name,
            avatar_url=self.user.avatar_url,
            is_host=is_host,
            connected=self.connected,
            joined_at=self.joined_at,
        )


@dataclass
class CallRuntime:
    call_id: str
    join_code: str
    host_user: UserResponse
    host_session_id: str
    poi_id: int | str | None
    poi_name: str
    status: str
    speaker_user_id: int | None
    last_turn_user_id: int | None = None
    participants: dict[int, ParticipantRuntime] = field(default_factory=dict)
    max_members: int = 10
    log: list[dict[str, Any]] = field(default_factory=list)
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC) + timedelta(hours=2))
    host_grace_deadline: datetime | None = None
    grace_task: asyncio.Task | None = None
    bridge: "RealtimeBridge | None" = None

    def snapshot(self) -> CallSnapshot:
        ordered = sorted(self.participants.values(), key=lambda item: item.joined_at)
        return CallSnapshot(
            call_id=self.call_id,
            join_code=self.join_code,
            host_user_id=self.host_user.id,
            host_display_name=self.host_user.display_name,
            host_session_id=self.host_session_id,
            poi_id=self.poi_id,
            poi_name=self.poi_name,
            status=self.status,
            speaker_user_id=self.speaker_user_id,
            max_members=self.max_members,
            member_count=len(self.participants),
            participants=[item.snapshot(is_host=item.user.id == self.host_user.id) for item in ordered],
            host_grace_deadline=self.host_grace_deadline,
            expires_at=self.expires_at,
            log=list(self.log[-80:]),
        )


class CallRoomError(RuntimeError):
    pass


class RealtimeBridge:
    def __init__(self, *, room: CallRuntime, loop: asyncio.AbstractEventLoop, service: "CallRoomService") -> None:
        self.room = room
        self.loop = loop
        self.service = service
        self.thread: threading.Thread | None = None
        self.outgoing: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.ws = None
        self.connected = threading.Event()
        self.closed = False
        self.session_update_event_id = f"session_update_{secrets.token_hex(6)}"
        self.initial_response_sent = False

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, name=f"call-bridge-{self.room.call_id}", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.closed = True
        self.stop_event.set()
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass

    def interrupt(self) -> None:
        self.send({"type": "response.cancel"})

    def append_audio(self, audio_b64: str) -> None:
        self.send({"type": "input_audio_buffer.append", "audio": audio_b64})

    def commit_audio(self) -> None:
        self.send({"type": "input_audio_buffer.commit"})
        self.send({"type": "response.create"})

    def send_text(self, text: str, author: str) -> None:
        self.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"{author}: {text}"}],
                },
            }
        )
        self.send({"type": "response.create"})

    def send_image(self, *, image_data_url: str, author: str) -> None:
        self.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"{author} ha enviado una foto para analizar."},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                },
            }
        )
        self.send({"type": "response.create"})

    def send_tool_output(self, call_id: str, output: str) -> None:
        self.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": call_id, "output": output},
            }
        )
        self.send({"type": "response.create"})

    def handle_session_updated(self, event: dict[str, Any]) -> None:
        if self.initial_response_sent:
            return
        if str(event.get("type") or "") != "session.updated":
            return
        if self.room.log:
            return
        self.initial_response_sent = True
        self.send({"type": "response.create"})

    def send(self, payload: dict[str, Any]) -> None:
        if self.closed:
            return
        self.outgoing.put(payload)

    def _run(self) -> None:
        url = f"wss://api.openai.com/v1/realtime?model={realtime_service.openai.realtime_model()}"
        headers = [
            f"Authorization: Bearer {settings.openai_api_key}",
        ]
        try:
            self.ws = websocket.create_connection(url, header=headers, timeout=5, enable_multithread=True)
            self.ws.settimeout(0.2)
            prepared = realtime_service.build_room_runtime(self.room.host_session_id, self.room.poi_name)
            self.ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": self.session_update_event_id,
                        "session": {
                            "type": "realtime",
                            "instructions": prepared["instructions"],
                            "tools": prepared["tools"],
                            "tool_choice": "auto",
                            "output_modalities": ["audio"],
                            "audio": {
                                "input": {
                                    "format": {
                                        "type": "audio/pcm",
                                        "rate": 24000,
                                    },
                                    "transcription": {
                                        "model": settings.openai_realtime_input_transcription_model,
                                        "language": settings.openai_realtime_input_transcription_language,
                                    },
                                    "turn_detection": None,
                                },
                                "output": {
                                    "format": {
                                        "type": "audio/pcm",
                                    },
                                    "voice": settings.openai_realtime_voice,
                                },
                            },
                        },
                    }
                )
            )
            self.connected.set()
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(self.service.handle_bridge_error(self.room.call_id, str(exc)), self.loop)
            return

        while not self.stop_event.is_set():
            while not self.outgoing.empty():
                payload = self.outgoing.get_nowait()
                try:
                    self.ws.send(json.dumps(payload))
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(self.service.handle_bridge_error(self.room.call_id, str(exc)), self.loop)
                    self.stop_event.set()
                    break

            if self.stop_event.is_set():
                break

            try:
                message = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as exc:
                if not self.stop_event.is_set():
                    asyncio.run_coroutine_threadsafe(self.service.handle_bridge_error(self.room.call_id, str(exc)), self.loop)
                break

            if not message:
                continue
            try:
                event = json.loads(message)
            except Exception:
                continue
            asyncio.run_coroutine_threadsafe(self.service.handle_realtime_event(self.room.call_id, event), self.loop)

        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass


class CallRoomService:
    MAX_MEMBERS = 10
    HOST_GRACE_SECONDS = 90
    JOIN_TOKEN_TTL_SECONDS = 600

    def __init__(self) -> None:
        self._calls: dict[str, CallRuntime] = {}
        self._lock = asyncio.Lock()

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _app_secret(self) -> bytes:
        return settings.app_secret.encode("utf-8")

    def _build_call_id(self) -> str:
        return f"CALL-{secrets.token_hex(3).upper()}"

    def _build_join_token(self, *, call_id: str, user_id: int) -> str:
        exp = int(time.time()) + self.JOIN_TOKEN_TTL_SECONDS
        payload = f"{call_id}:{user_id}:{exp}"
        signature = hmac.new(self._app_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return base64.urlsafe_b64encode(f"{payload}:{signature}".encode("utf-8")).decode("utf-8")

    def verify_join_token(self, join_token: str) -> tuple[str, int]:
        try:
            decoded = base64.urlsafe_b64decode(join_token.encode("utf-8")).decode("utf-8")
            call_id, user_id, exp, signature = decoded.split(":", 3)
        except Exception as exc:
            raise CallRoomError("Join token inválido") from exc

        payload = f"{call_id}:{user_id}:{exp}"
        expected = hmac.new(self._app_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise CallRoomError("Join token inválido")
        if int(exp) < int(time.time()):
            raise CallRoomError("Join token expirado")
        return call_id, int(user_id)

    def _capabilities_for(self, room: CallRuntime, user_id: int) -> UICapabilities:
        is_host = room.host_user.id == user_id
        if room.status == "ended":
            return UICapabilities(reason="call_ended")
        if room.status == "paused":
            return UICapabilities(reason="paused_waiting_host")
        if room.status == "idle":
            return UICapabilities(can_talk=True, can_text=True, can_image=True)
        if room.status == "user_speaking":
            if room.speaker_user_id == user_id:
                return UICapabilities(can_talk=True, reason="you_hold_floor")
            return UICapabilities(reason="another_user_speaking")
        if room.status == "assistant_speaking":
            if is_host:
                return UICapabilities(can_talk=True, can_interrupt=True, reason="host_interrupt_enabled")
            return UICapabilities(reason="assistant_speaking")
        return UICapabilities(reason="busy")

    def _snapshot_payload(self, room: CallRuntime, user_id: int) -> dict[str, Any]:
        return {
            "type": "call.snapshot",
            "call": room.snapshot().model_dump(mode="json"),
            "ui": self._capabilities_for(room, user_id).model_dump(mode="json"),
        }

    async def create_call(self, *, user: UserResponse, session_id: str, poi_id: int | str | None, poi_name: str) -> tuple[CallSnapshot, str]:
        async with self._lock:
            billing_service.ensure_user_can_consume(user.id)
            call_id = self._build_call_id()
            room = CallRuntime(
                call_id=call_id,
                join_code=call_id,
                host_user=user,
                host_session_id=session_id.upper(),
                poi_id=poi_id,
                poi_name=poi_name or "",
                status="idle",
                speaker_user_id=None,
                max_members=self.MAX_MEMBERS,
            )
            room.participants[user.id] = ParticipantRuntime(user=user, joined_at=self._now())
            self._calls[call_id] = room
            session_service.attach_user(session_id, user.id)
            session_service.touch_participant(session_id, user, active_call=False)
            if poi_name:
                session_service.update_session(
                    session_id,
                    SessionUpdateRequest(
                        user_id=user.id,
                        active_poi_name=poi_name,
                    ),
                )
            return room.snapshot(), self._build_join_token(call_id=call_id, user_id=user.id)

    async def get_join_token(self, *, call_id: str, user: UserResponse) -> tuple[CallSnapshot, str]:
        async with self._lock:
            room = self._calls.get(call_id.upper())
            if room is None or room.status == "ended":
                raise CallRoomError("La llamada ya no está disponible")
            if user.id not in room.participants and len(room.participants) >= room.max_members:
                raise CallRoomError("La llamada ya está llena")
            if user.id not in room.participants:
                room.participants[user.id] = ParticipantRuntime(user=user, joined_at=self._now())
            if room.status == "paused" and user.id == room.host_user.id:
                room.status = "idle"
                room.host_grace_deadline = None
                if room.grace_task is not None:
                    room.grace_task.cancel()
                    room.grace_task = None
            return room.snapshot(), self._build_join_token(call_id=room.call_id, user_id=user.id)

    async def leave_call(self, *, call_id: str, user: UserResponse) -> CallSnapshot:
        async with self._lock:
            room = self._get_room_or_raise(call_id)
            participant = room.participants.get(user.id)
            if participant:
                participant.connected = False
                participant.websocket = None
            if user.id == room.host_user.id:
                await self._pause_or_end_room(room, due_to_disconnect=False)
            else:
                room.participants.pop(user.id, None)
            return room.snapshot()

    async def end_call(self, *, call_id: str, user: UserResponse) -> CallSnapshot:
        async with self._lock:
            room = self._get_room_or_raise(call_id)
            if room.host_user.id != user.id:
                raise CallRoomError("Solo el anfitrión puede cerrar la llamada")
            await self._end_room(room)
            return room.snapshot()

    async def connect(self, *, websocket: WebSocket, call_id: str, user: UserResponse) -> None:
        await websocket.accept()
        async with self._lock:
            room = self._get_room_or_raise(call_id)
            if user.id not in room.participants:
                if len(room.participants) >= room.max_members:
                    await websocket.send_json({"type": "call.error", "message": "La llamada ya está llena"})
                    await websocket.close()
                    return
                room.participants[user.id] = ParticipantRuntime(user=user, joined_at=self._now())
            participant = room.participants[user.id]
            participant.websocket = websocket
            participant.connected = True
            if room.status == "paused" and user.id == room.host_user.id:
                room.status = "idle"
                room.host_grace_deadline = None
                if room.grace_task is not None:
                    room.grace_task.cancel()
                    room.grace_task = None
            await websocket.send_json(self._snapshot_payload(room, user.id))
            await self._broadcast_snapshot(room)

        try:
            while True:
                message = await websocket.receive_text()
                event = json.loads(message)
                await self.handle_client_event(call_id=call_id, user=user, event=event)
        except WebSocketDisconnect:
            await self.disconnect(call_id=call_id, user=user)

    async def disconnect(self, *, call_id: str, user: UserResponse) -> None:
        async with self._lock:
            room = self._calls.get(call_id.upper())
            if room is None:
                return
            participant = room.participants.get(user.id)
            if participant is None:
                return
            participant.connected = False
            participant.websocket = None
            if room.speaker_user_id == user.id and room.status == "user_speaking":
                room.status = "idle"
                room.last_turn_user_id = user.id
                room.speaker_user_id = None
            if user.id == room.host_user.id:
                await self._pause_or_end_room(room, due_to_disconnect=True)
            else:
                room.participants.pop(user.id, None)
                await self._broadcast_snapshot(room)

    async def handle_client_event(self, *, call_id: str, user: UserResponse, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        async with self._lock:
            room = self._get_room_or_raise(call_id)
            if event_type == "floor.request":
                await self._handle_floor_request(room, user)
                return
            if event_type == "floor.release":
                await self._handle_floor_release(room, user)
                return
            if event_type == "audio.chunk":
                await self._handle_audio_chunk(room, user, str(event.get("audio") or ""))
                return
            if event_type == "audio.commit":
                await self._handle_audio_commit(room, user)
                return
            if event_type == "text.submit":
                await self._handle_text_submit(room, user, str(event.get("text") or ""))
                return
            if event_type == "image.submit":
                await self._handle_image_submit(room, user, str(event.get("image_data_url") or ""))
                return
            if event_type == "call.leave":
                await self.leave_call(call_id=call_id, user=user)
                return
            participant = room.participants.get(user.id)
            if participant and participant.websocket is not None:
                await participant.websocket.send_json({"type": "call.error", "message": "Evento no soportado"})

    async def _handle_floor_request(self, room: CallRuntime, user: UserResponse) -> None:
        participant = room.participants.get(user.id)
        if participant is None or participant.websocket is None:
            return
        if room.status == "idle":
            room.status = "user_speaking"
            room.speaker_user_id = user.id
            room.last_turn_user_id = user.id
            await participant.websocket.send_json({"type": "floor.granted"})
            await self._broadcast_snapshot(room)
            return
        if room.status == "assistant_speaking" and user.id == room.host_user.id:
            if room.bridge is not None:
                room.bridge.interrupt()
            room.status = "user_speaking"
            room.speaker_user_id = user.id
            room.last_turn_user_id = user.id
            await self._broadcast(room, {"type": "assistant.interrupted", "by_user_id": user.id})
            await participant.websocket.send_json({"type": "floor.granted"})
            await self._broadcast_snapshot(room)
            return
        await participant.websocket.send_json({"type": "floor.denied", "reason": self._capabilities_for(room, user.id).reason})

    async def _ensure_bridge(self, room: CallRuntime) -> None:
        if room.bridge is not None:
            return
        room.bridge = RealtimeBridge(room=room, loop=asyncio.get_running_loop(), service=self)
        room.bridge.start()

    async def _handle_floor_release(self, room: CallRuntime, user: UserResponse) -> None:
        participant = room.participants.get(user.id)
        if participant is None or participant.websocket is None:
            return
        if room.status == "user_speaking" and room.speaker_user_id == user.id:
            room.status = "idle"
            room.last_turn_user_id = None
            room.speaker_user_id = None
            await participant.websocket.send_json({"type": "floor.released"})
            await self._broadcast_snapshot(room)

    async def _handle_audio_chunk(self, room: CallRuntime, user: UserResponse, audio_b64: str) -> None:
        if room.status != "user_speaking" or room.speaker_user_id != user.id:
            return
        await self._ensure_bridge(room)
        room.bridge.append_audio(audio_b64)
        await self._broadcast(room, {"type": "peer_audio.chunk", "user_id": user.id, "audio": audio_b64}, exclude_user_id=user.id)

    async def _handle_audio_commit(self, room: CallRuntime, user: UserResponse) -> None:
        if room.status != "user_speaking" or room.speaker_user_id != user.id:
            return
        await self._ensure_bridge(room)
        room.speaker_user_id = None
        room.status = "assistant_speaking"
        room.bridge.commit_audio()
        await self._broadcast(room, {"type": "assistant.started"})
        await self._broadcast_snapshot(room)

    async def _handle_text_submit(self, room: CallRuntime, user: UserResponse, text: str) -> None:
        clean = text.strip()
        participant = room.participants.get(user.id)
        if not participant or participant.websocket is None:
            return
        if not clean:
            return
        capabilities = self._capabilities_for(room, user.id)
        if not capabilities.can_text:
            await participant.websocket.send_json({"type": "message.rejected", "reason": capabilities.reason})
            return
        await self._ensure_bridge(room)
        room.log.append(self._log_entry("user-text", user.display_name, clean, user.id))
        room.status = "assistant_speaking"
        room.bridge.send_text(clean, user.display_name)
        await self._broadcast_snapshot(room)
        await self._broadcast(room, {"type": "assistant.started"})

    async def _handle_image_submit(self, room: CallRuntime, user: UserResponse, image_data_url: str) -> None:
        participant = room.participants.get(user.id)
        if not participant or participant.websocket is None:
            return
        capabilities = self._capabilities_for(room, user.id)
        if not capabilities.can_image:
            await participant.websocket.send_json({"type": "message.rejected", "reason": capabilities.reason})
            return
        if not image_data_url.startswith("data:image/"):
            await participant.websocket.send_json({"type": "message.rejected", "reason": "invalid_image"})
            return
        await self._ensure_bridge(room)
        room.log.append(self._log_entry("user-photo", user.display_name, "Ha enviado una foto.", user.id, image_data_url))
        room.status = "assistant_speaking"
        room.bridge.send_image(image_data_url=image_data_url, author=user.display_name)
        await self._broadcast_snapshot(room)
        await self._broadcast(room, {"type": "assistant.started"})

    async def handle_realtime_event(self, call_id: str, event: dict[str, Any]) -> None:
        async with self._lock:
            room = self._calls.get(call_id.upper())
            if room is None:
                return
            event_type = str(event.get("type") or "")
            if event_type == "session.updated":
                if room.bridge is not None:
                    room.bridge.handle_session_updated(event)
                return
            if event_type == "response.output_audio.delta":
                await self._broadcast(room, {"type": "assistant.audio_chunk", "audio": event.get("delta", "")})
                return
            if event_type == "response.output_audio_transcript.done":
                transcript = str(event.get("transcript") or "").strip()
                if transcript:
                    room.log.append(self._log_entry("ai", "Locus", transcript, None))
                    await self._broadcast_snapshot(room)
                return
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = str(event.get("transcript") or "").strip()
                item = event.get("item_id")
                if transcript:
                    speaker_user_id = room.last_turn_user_id
                    speaker_name = "Viajero"
                    if speaker_user_id is not None and speaker_user_id in room.participants:
                        speaker_name = room.participants[speaker_user_id].user.display_name or speaker_name
                    room.log.append(self._log_entry("user-voice", speaker_name, transcript, speaker_user_id))
                    await self._broadcast(
                        room,
                        {
                            "type": "peer_transcript",
                            "text": transcript,
                            "item_id": item,
                            "user_id": speaker_user_id,
                            "author": speaker_name,
                        },
                    )
                    await self._broadcast_snapshot(room)
                return
            if event_type == "response.function_call_arguments.done":
                await self._handle_function_call(room, event)
                return
            if event_type == "response.done":
                await self._handle_response_done(room, event)
                return
            if event_type == "error":
                error = event.get("error") or {}
                message = str(error.get("message") or "Ha fallado la sesión realtime")
                param = str(error.get("param") or "").strip()
                code = str(error.get("code") or "").strip()
                if param:
                    message = f"{message} ({param})"
                if code:
                    message = f"{message} [{code}]"
                room.log.append(self._log_entry("error", "Sistema", message, None))
                room.status = "idle"
                room.speaker_user_id = None
                await self._broadcast(room, {"type": "call.error", "message": message})
                await self._broadcast_snapshot(room)

    async def _handle_function_call(self, room: CallRuntime, event: dict[str, Any]) -> None:
        tool_name = str(event.get("name") or "")
        call_id = str(event.get("call_id") or "")
        arguments = str(event.get("arguments") or "{}")
        if not room.bridge:
            return
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = {}
        result = tool_runtime_service.execute(room.host_session_id, tool_name, parsed)
        room.bridge.send_tool_output(call_id, result)

    async def _handle_response_done(self, room: CallRuntime, event: dict[str, Any]) -> None:
        response = event.get("response") or {}
        usage = response.get("usage") or {}
        response_id = str(response.get("id") or "")
        room.status = "idle"
        room.speaker_user_id = None
        room.last_turn_user_id = None
        await self._broadcast(room, {"type": "assistant.done"})
        if usage and response_id:
            try:
                billing_service.record_usage(
                    user_id=room.host_user.id,
                    session_id=room.host_session_id,
                    provider="openai",
                    endpoint="realtime",
                    model=realtime_service.openai.realtime_model(),
                    response_id=response_id,
                    usage=usage,
                    metadata={"source": "call_room", "call_id": room.call_id},
                )
            except BillingError:
                room.log.append(self._log_entry("error", "Sistema", "Saldo insuficiente para continuar la llamada.", None))
        await self._broadcast_snapshot(room)

    async def handle_bridge_error(self, call_id: str, message: str) -> None:
        async with self._lock:
            room = self._calls.get(call_id.upper())
            if room is None:
                return
            room.status = "idle" if room.status != "ended" else room.status
            room.speaker_user_id = None
            room.last_turn_user_id = None
            room.log.append(self._log_entry("error", "Sistema", message, None))
            await self._broadcast(room, {"type": "call.error", "message": message})
            await self._broadcast_snapshot(room)

    async def _broadcast_snapshot(self, room: CallRuntime) -> None:
        for participant in room.participants.values():
            if participant.websocket is None or not participant.connected:
                continue
            await participant.websocket.send_json(self._snapshot_payload(room, participant.user.id))

    async def _broadcast(self, room: CallRuntime, payload: dict[str, Any], *, exclude_user_id: int | None = None) -> None:
        stale_ids: list[int] = []
        for participant in room.participants.values():
            if exclude_user_id is not None and participant.user.id == exclude_user_id:
                continue
            if participant.websocket is None or not participant.connected:
                continue
            try:
                await participant.websocket.send_json(payload)
            except RuntimeError:
                stale_ids.append(participant.user.id)
        for user_id in stale_ids:
            room.participants[user_id].connected = False
            room.participants[user_id].websocket = None

    def _get_room_or_raise(self, call_id: str) -> CallRuntime:
        room = self._calls.get(call_id.upper())
        if room is None:
            raise CallRoomError("La llamada ya no está disponible")
        return room

    async def _pause_or_end_room(self, room: CallRuntime, *, due_to_disconnect: bool) -> None:
        if due_to_disconnect:
            room.status = "paused"
            room.speaker_user_id = None
            room.last_turn_user_id = None
            room.host_grace_deadline = self._now() + timedelta(seconds=self.HOST_GRACE_SECONDS)
            if room.bridge is not None:
                room.bridge.interrupt()
            if room.grace_task is not None:
                room.grace_task.cancel()
            room.grace_task = asyncio.create_task(self._grace_timeout(room.call_id))
            await self._broadcast_snapshot(room)
            return
        await self._end_room(room)

    async def _grace_timeout(self, call_id: str) -> None:
        await asyncio.sleep(self.HOST_GRACE_SECONDS)
        async with self._lock:
            room = self._calls.get(call_id.upper())
            if room is None:
                return
            if room.status == "paused" and room.host_grace_deadline and room.host_grace_deadline <= self._now():
                await self._end_room(room)

    async def _end_room(self, room: CallRuntime) -> None:
        room.status = "ended"
        room.speaker_user_id = None
        room.last_turn_user_id = None
        room.host_grace_deadline = None
        if room.grace_task is not None:
            room.grace_task.cancel()
            room.grace_task = None
        if room.bridge is not None:
            room.bridge.close()
            room.bridge = None
        await self._broadcast(room, {"type": "call.ended"})
        for participant in room.participants.values():
            if participant.websocket is not None:
                try:
                    await participant.websocket.close()
                except RuntimeError:
                    pass
        self._calls.pop(room.call_id, None)

    def _log_entry(
        self,
        kind: str,
        author: str,
        text: str,
        user_id: int | None,
        image_url: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": f"log_{secrets.token_hex(8)}",
            "kind": kind,
            "author": author,
            "text": text,
            "timestamp": self._now().isoformat(),
            "user_id": user_id,
            "image_url": image_url,
        }


call_room_service = CallRoomService()
