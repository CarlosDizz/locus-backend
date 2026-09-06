"""Bridges a live call room to a real AI voice provider.

The REST/WS layer (api/calls.py) only moves floor/audio/text/image commands
through Redis (locus_v2.calls.store.RoomStore) — it never talks to a model.
One CallVoiceBridge task per active call_id drains that room's command stream
(the same stream a distinct realtime process could drain in a multi-instance
deployment) and drives a locus_v2.voice provider exactly like voice/gateway.py
does for the single-user POI guide, translating provider events back into the
room's own event vocabulary (assistant.audio_chunk / assistant.transcript /
assistant.done) so every connected participant sees the same thing.

Scope note: tool calling (document_poi, plan_poi_visit, find_activities) and
usage/billing persistence are intentionally not wired here yet — the prompt
still renders and the model still answers from its own knowledge, but a call
does not (yet) invoke catalog tools or record a UsageEvent the way voice
sessions do. Image submissions are acknowledged but not analyzed: LiveProvider
has no image-input method today.
"""

import asyncio
import base64
import contextlib
from functools import partial

import structlog
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from locus_v2.ai.enums import PublicationStatus, ServiceKind
from locus_v2.ai.models import AIModel, RoutingProfile
from locus_v2.calls.models import CallError
from locus_v2.calls.service import CallService
from locus_v2.calls.store import RoomStore
from locus_v2.config import Settings
from locus_v2.infrastructure.database.session import Database
from locus_v2.shared.prompting import PromptRenderingError, render_prompt
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

    async def run(self) -> None:
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
        finally:
            with contextlib.suppress(Exception):
                await self.provider.close()

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
