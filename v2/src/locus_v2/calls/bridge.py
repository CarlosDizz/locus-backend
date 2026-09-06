"""Bridges a live call room to a real AI voice provider.

The REST/WS layer (api/calls.py) only moves floor/audio/text/image commands
through Redis (locus_v2.calls.store.RoomStore) — it never talks to a model.
One CallVoiceBridge task per active call_id drains that room's command stream
(the same stream a distinct realtime process could drain in a multi-instance
deployment) and drives a locus_v2.voice provider exactly like voice/gateway.py
does for the single-user POI guide, translating provider events back into the
room's own event vocabulary (assistant.audio_chunk / assistant.transcript /
assistant.done) so every connected participant sees the same thing.

Usage/billing reuses the exact voice_sessions/usage_events path voice/gateway.py
uses for the single-user POI guide: a VoiceSession row gives UsageEvent
somewhere to point its required FK, and the billing worker (entrypoints/worker.py)
picks up any usage_events row left PENDING here the same way it already
processes chat_call/realtime_call rows — no worker change needed.

Scope note: tool calling (document_poi, plan_poi_visit, find_activities) is
intentionally not wired here yet — the prompt still renders and the model
still answers from its own knowledge, but a call does not (yet) invoke
catalog tools. Image submissions are acknowledged but not analyzed:
LiveProvider has no image-input method today.
"""

import asyncio
import base64
import contextlib
from functools import partial
from uuid import uuid4

import structlog
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from locus_v2.ai.enums import PublicationStatus, ServiceKind
from locus_v2.ai.models import AIModel, RoutingProfile
from locus_v2.billing.models import UsageEvent, UsageStatus
from locus_v2.billing.pricing import NormalizedUsage
from locus_v2.calls.models import CallError
from locus_v2.calls.service import CallService
from locus_v2.calls.store import RoomStore
from locus_v2.config import Settings
from locus_v2.infrastructure.database.session import Database
from locus_v2.shared.clock import utc_now
from locus_v2.shared.prompting import PromptRenderingError, render_prompt
from locus_v2.voice.models import VoiceSession, VoiceSessionStatus
from locus_v2.voice.protocol import AudioFormat
from locus_v2.voice.providers.base import LiveProvider, LiveSessionConfig, ProviderEventType
from locus_v2.voice.providers.factory import build_provider_registry

logger = structlog.get_logger()

_ACTIVE: dict[str, asyncio.Task[None]] = {}


def _forget(call_id: str, done: asyncio.Task[None]) -> None:
    if _ACTIVE.get(call_id) is done:
        _ACTIVE.pop(call_id, None)


def ensure_bridge(call_id: str, store: RoomStore, database: Database, settings: Settings) -> None:
    """Start the AI bridge for this call if one is not already running.

    In-process tracking is enough for the single-worker deployment this runs
    in today (mirrors app/services/call_room_service.py's in-memory `_calls`
    dict) — a multi-worker deployment would need to move this to a Redis lock
    keyed the same way RoomStore keys everything else, but that's not this
    codebase's shape yet.
    """
    existing = _ACTIVE.get(call_id)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(_CallVoiceBridge(call_id, store, database, settings).run())
    _ACTIVE[call_id] = task
    task.add_done_callback(partial(_forget, call_id))


class _CallVoiceBridge:
    def __init__(
        self, call_id: str, store: RoomStore, database: Database, settings: Settings
    ) -> None:
        self.call_id = call_id
        self.store = store
        self.database = database
        self.settings = settings
        self.service = CallService(store, settings, consume=_ignore_consume)
        self.provider: LiveProvider | None = None
        self._assistant_text = ""
        self.trace_id = uuid4().hex
        self.host_id: int | None = None
        self.model_id: int | None = None
        self.voice_session_id: int | None = None

    async def run(self) -> None:
        final_status = VoiceSessionStatus.COMPLETED
        try:
            self.provider = await self._connect()
        except CallError as error:
            await self.store.publish(self.call_id, {"type": "call.error", "message": str(error)})
            return
        except Exception as error:  # noqa: BLE001 - a broken bridge must not crash the call room
            logger.exception("call_voice_bridge_connect_failed", call_id=self.call_id)
            await self.store.publish(
                self.call_id,
                {"type": "call.error", "message": f"No se pudo conectar la IA: {error}"},
            )
            return
        commands_task = asyncio.create_task(self._consume_commands())
        events_task = asyncio.create_task(self._pump_provider_events())
        watchdog_task = asyncio.create_task(self._watch_room_ended())
        try:
            done, pending = await asyncio.wait(
                {commands_task, events_task, watchdog_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            for task in done:
                if task is not watchdog_task:
                    with contextlib.suppress(Exception):
                        task.result()
        except Exception:  # noqa: BLE001 - still need to close out the session below
            final_status = VoiceSessionStatus.FAILED
            raise
        finally:
            with contextlib.suppress(Exception):
                await self.provider.close()
            with contextlib.suppress(Exception):
                await self._finish(final_status)

    async def _connect(self) -> LiveProvider:
        room = await self.store.get(self.call_id)
        async with self.database.sessions() as session:
            profile = await session.scalar(
                select(RoutingProfile)
                .options(
                    joinedload(RoutingProfile.primary_model).joinedload(AIModel.provider),
                    joinedload(RoutingProfile.prompt_version),
                )
                .where(
                    RoutingProfile.environment == self.settings.env,
                    RoutingProfile.service_kind == ServiceKind.VOICE,
                    RoutingProfile.status == PublicationStatus.PUBLISHED,
                    RoutingProfile.experience_code == "poi_guide",
                )
            )
            if profile is None:
                raise CallError("No hay un guia de voz publicado para llamadas", 503)
            try:
                prompt = render_prompt(
                    profile.prompt_version.content,
                    {
                        "locale": room.language,
                        "poi_name": room.poi_name or "este lugar",
                        "poi_description": "",
                        "city_name": "",
                    },
                )
            except PromptRenderingError as error:
                raise CallError(str(error), 503) from error
            model = profile.primary_model
            self.host_id = room.host_id
            self.model_id = model.id
            # config_snapshot_json.call_id is what mobile_billing.py's list_ledger() reads back
            # as usage_call_id (falling back to the session's own public_id otherwise) - this is
            # how the wallet page groups every charge from one call under a single card instead
            # of listing each turn separately.
            voice_session = VoiceSession(
                user_id=self.host_id,
                routing_profile_id=profile.id,
                prompt_version_id=profile.prompt_version_id,
                primary_model_id=model.id,
                active_model_id=model.id,
                status=VoiceSessionStatus.ACTIVE,
                locale=room.language,
                context_type="poi",
                context_public_id=room.poi_public_id,
                config_snapshot_json={"call_id": self.call_id},
                started_at=utc_now(),
            )
            session.add(voice_session)
            await session.commit()
            self.voice_session_id = voice_session.id
        registry = build_provider_registry(self.settings)
        provider = registry.create(model.adapter_code)
        await provider.connect(
            LiveSessionConfig(
                model=model.external_id,
                prompt=prompt,
                locale=room.language,
                audio_format=AudioFormat.PCM16_24KHZ,
                tools=[],
                provider_options=dict(model.runtime_defaults_json or {}),
            )
        )
        logger.info(
            "call_voice_bridge_connected",
            call_id=self.call_id,
            adapter=model.adapter_code,
            model=model.external_id,
        )
        await self.service.mark_ready(self.call_id)
        return provider

    async def _finish(self, status: str) -> None:
        if self.voice_session_id is None:
            return
        async with self.database.sessions() as session:
            voice_session = await session.get(VoiceSession, self.voice_session_id)
            if voice_session is not None and voice_session.ended_at is None:
                voice_session.status = status
                voice_session.ended_at = utc_now()
                await session.commit()

    async def _persist_usage(self, usage: NormalizedUsage) -> None:
        if self.voice_session_id is None or self.model_id is None or self.host_id is None:
            return
        async with self.database.sessions() as session:
            model = await session.get(AIModel, self.model_id)
            if model is None:
                return
            session.add(
                UsageEvent(
                    user_id=self.host_id,
                    voice_session_id=self.voice_session_id,
                    provider_id=model.provider_id,
                    model_id=model.id,
                    dedupe_key=f"{self.trace_id}:{uuid4().hex}",
                    interaction_type="realtime_call",
                    text_input_tokens=usage.text_input_tokens,
                    cached_text_input_tokens=usage.cached_text_input_tokens,
                    text_output_tokens=usage.text_output_tokens,
                    audio_input_tokens=usage.audio_input_tokens,
                    cached_audio_input_tokens=usage.cached_audio_input_tokens,
                    audio_output_tokens=usage.audio_output_tokens,
                    image_input_tokens=usage.image_input_tokens,
                    cached_image_input_tokens=usage.cached_image_input_tokens,
                    audio_input_milliseconds=usage.audio_input_milliseconds,
                    audio_output_milliseconds=usage.audio_output_milliseconds,
                    tool_calls=usage.tool_calls,
                    raw_usage_json=usage.raw,
                    status=UsageStatus.PENDING,
                    trace_id=self.trace_id,
                )
            )
            await session.commit()
        logger.info(
            "call_voice_bridge_usage_recorded",
            call_id=self.call_id,
            voice_session_id=self.voice_session_id,
        )

    async def _consume_commands(self) -> None:
        assert self.provider is not None
        async for command in self.store.commands(self.call_id):
            kind = command.get("type")
            try:
                if kind == "reset":
                    await self.provider.cancel_response()
                elif kind == "audio.chunk":
                    await self.provider.send_audio(base64.b64decode(command["audio"]))
                elif kind == "audio.commit":
                    await self.provider.commit_audio()
                elif kind == "text.submit":
                    author = command.get("author", "")
                    await self.provider.send_text(
                        f"{author}: {command['text']}" if author else command["text"]
                    )
                elif kind == "image.submit":
                    # LiveProvider has no image-input method yet; acknowledge instead of hanging.
                    await self.provider.send_text(
                        f"{command.get('author', 'Alguien')} ha enviado una foto, pero todavia "
                        "no puedo analizar imagenes durante una llamada."
                    )
            except Exception as error:  # noqa: BLE001 - one bad command must not kill the bridge
                logger.warning(
                    "call_voice_bridge_command_failed",
                    call_id=self.call_id,
                    kind=kind,
                    error=str(error),
                )

    async def _pump_provider_events(self) -> None:
        assert self.provider is not None
        async for event in self.provider.events():
            if event.type == ProviderEventType.AUDIO_DELTA:
                await self.store.publish(
                    self.call_id,
                    {
                        "type": "assistant.audio_chunk",
                        "audio": base64.b64encode(event.audio or b"").decode("ascii"),
                    },
                )
            elif event.type == ProviderEventType.TEXT_DELTA:
                self._assistant_text += event.text or ""
                await self.store.publish(
                    self.call_id, {"type": "assistant.transcript_delta", "text": event.text or ""}
                )
            elif event.type == ProviderEventType.TEXT_DONE:
                text = event.text or self._assistant_text
                self._assistant_text = ""
                await self.service.assistant_finished(self.call_id, text)
            elif event.type == ProviderEventType.AUDIO_DONE:
                if self._assistant_text:
                    await self.service.assistant_finished(self.call_id, self._assistant_text)
                    self._assistant_text = ""
            elif event.type == ProviderEventType.INPUT_TRANSCRIPT_DONE:
                if event.text:
                    await self.service.log_user_voice(self.call_id, event.text)
            elif event.type == ProviderEventType.USAGE and event.usage is not None:
                await self._persist_usage(event.usage)
            elif event.type == ProviderEventType.ERROR:
                await self.store.publish(
                    self.call_id, {"type": "call.error", "message": event.text or "assistant_error"}
                )
                if not event.retryable:
                    return

    async def _watch_room_ended(self) -> None:
        while True:
            await asyncio.sleep(5)
            try:
                room = await self.store.get(self.call_id)
            except CallError:
                return
            if room.status == "ended":
                return


async def _ignore_consume(user_id: int) -> None:
    """The bridge only reacts to commands CallService already accepted.

    ensure_host_can_consume() (calls/policy.py) already gated the event before
    it ever reached the Redis command stream this bridge drains, so there is
    nothing left to check here — this just satisfies CallService's `consume`
    parameter.
    """
    return None
