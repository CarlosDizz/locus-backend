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
processes chat_call/realtime_call rows — no worker change needed. Tool calls
(document_poi, plan_poi_visit, find_activities) are billed the same way
voice/gateway.py's _persist_tool_usage() does, against the same priced
tool_model AIModel row — see shared/openai_usage.py.

The routing profile is looked up by experience_code="group_call_guide" (not
"poi_guide", the single-user CP guide's profile) — a call needs its own
prompt: proactive ("who's there?"), scene-vs-stops planning, tool calling.
Nothing here touches the CP's profile/prompt.

Image submissions go to the provider for real when capabilities.image_input is
true (only GeminiLiveProvider today) - otherwise acknowledged with an apology
instead of hanging.

Known gap, next candidate to fix: while a tool call is in flight the model cannot
speak at all (confirmed live 2026-09-06 - see the audio_burst log below), so a
long document_poi/plan_poi_visit run is dead air no matter how the turns are
split. OpenAI's GA `gpt-realtime` model supports real async tool calling
(`async: true` on the tool definition - the model keeps talking while the tool
runs, https://openai.com/index/introducing-gpt-realtime/); our own
voice/providers/openai_realtime.py already declares
`async_function_calling=True` but nothing sets that flag or builds the
non-blocking flow yet. Gemini Live has no equivalent. Switching the group-call
profile's primary model to gpt-realtime and wiring async tools through would
fix this properly, instead of the two-turn "announce, then research" split
below, which only gets the announcement out promptly - the research itself is
still silent.
"""

import asyncio
import base64
import contextlib
from functools import partial
from time import perf_counter
from typing import Any
from uuid import uuid4

import structlog
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from locus_v2.ai.enums import PublicationStatus, ServiceKind
from locus_v2.ai.models import AIModel, AIProvider, RoutingProfile
from locus_v2.billing.models import UsageEvent, UsageStatus
from locus_v2.billing.pricing import NormalizedUsage
from locus_v2.calls.models import CallError
from locus_v2.calls.service import CallService, decode_image
from locus_v2.calls.store import RoomStore
from locus_v2.catalog.models import Poi
from locus_v2.config import Settings
from locus_v2.infrastructure.database.session import Database
from locus_v2.shared.clock import utc_now
from locus_v2.shared.openai_usage import ToolUsage
from locus_v2.shared.prompting import PromptRenderingError, localized_field, render_prompt
from locus_v2.voice.models import VoiceSession, VoiceSessionStatus
from locus_v2.voice.protocol import AudioFormat
from locus_v2.voice.providers.base import (
    LiveProvider,
    LiveSessionConfig,
    ProviderEvent,
    ProviderEventType,
)
from locus_v2.voice.providers.factory import build_provider_registry
from locus_v2.voice.tools import VoiceToolDispatcher

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
        self.locale = "es"
        self.tool_dispatcher = VoiceToolDispatcher(settings)
        self.tool_definitions: list[dict[str, Any]] = []
        self.tool_context: dict[str, Any] = {}
        # Two-turn handshake for the opening research pass - see _on_assistant_turn_done().
        # Confirmed live (2026-09-06): Gemini Live does not stream any audio for a response
        # until the whole turn, tool round-trips included, is done. Letting the model say
        # "give me a moment" and call document_poi in the *same* turn means that line
        # arrives bundled with the narration 45-90s later - useless, the group already sat
        # through the silence it was meant to explain. Splitting the announcement into its
        # own tool-free turn, then nudging research as a second turn, makes it audible when
        # it's supposed to be.
        self._assistant_turn_count = 0
        self._research_kicked_off = False

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
        self.locale = room.language
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
                    RoutingProfile.experience_code == "group_call_guide",
                )
            )
            if profile is None:
                raise CallError("No hay un guia de llamadas publicado", 503)
            poi = await session.scalar(
                select(Poi).options(joinedload(Poi.city)).where(Poi.public_id == room.poi_public_id)
            )
            language = room.language.split("-", 1)[0].lower()
            self.tool_context = {
                "public_id": room.poi_public_id,
                "name": (
                    (localized_field(poi.names_json, room.language, language) or poi.name)
                    if poi is not None
                    else room.poi_name
                )
                or room.poi_name
                or "este lugar",
                "description": (
                    localized_field(poi.short_descriptions_json, room.language, language)
                    or poi.short_description
                    if poi is not None
                    else ""
                ),
                "city_name": poi.city.name if poi is not None and poi.city else "",
                "wikidata_id": poi.wikidata_id if poi is not None else "",
                "wikipedia_title": poi.wikipedia_title if poi is not None else "",
            }
            try:
                prompt = render_prompt(
                    profile.prompt_version.content,
                    {
                        "locale": room.language,
                        "poi_name": self.tool_context["name"],
                        "poi_description": self.tool_context["description"],
                        "city_name": self.tool_context["city_name"],
                    },
                )
            except PromptRenderingError as error:
                raise CallError(str(error), 503) from error
            model = profile.primary_model
            self.host_id = room.host_id
            self.model_id = model.id
            self.tool_definitions = [
                tool
                for tool in (profile.prompt_version.tools_json or [])
                if tool.get("enabled", True)
            ]
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
        tools = [
            {
                "type": "function",
                "name": tool["code"],
                "description": tool["description"],
                "parameters": tool["schema"],
            }
            for tool in self.tool_definitions
        ]
        # Same merge chain voice/configuration.py's _provider() uses for the CP path -
        # this file never applied it, so config_json/runtime_config_json (voice,
        # turn_detection, temperature...) set from the panel's Prompt workshop were
        # silently ignored for calls; confirmed live (2026-09-06) as the "voice stays
        # Kore no matter what I pick in the panel" bug.
        runtime_config = _deep_merge(
            profile.config_json, profile.prompt_version.runtime_config_json
        )
        options = _deep_merge(model.runtime_defaults_json, runtime_config)
        provider_overrides = options.pop("provider_overrides", {})
        options = _deep_merge(options, provider_overrides.get(model.provider.code, {}))
        voice = options.pop("voice", None)
        await provider.connect(
            LiveSessionConfig(
                model=model.external_id,
                prompt=prompt,
                locale=room.language,
                voice=voice,
                audio_format=AudioFormat.PCM16_24KHZ,
                tools=tools,
                provider_options=options,
            )
        )
        logger.info(
            "call_voice_bridge_connected",
            call_id=self.call_id,
            adapter=model.adapter_code,
            model=model.external_id,
            tool_count=len(tools),
        )
        await self.service.mark_ready(self.call_id)
        # Nobody has said anything yet — the bridge has to speak first, same trick V1's
        # RealtimeBridge.handle_session_updated() used (request a response with an empty
        # room log). CALL_GUIDE_PROMPT's opening instruction is "greet and ask if everyone
        # is there"; this line is what actually makes the model say it unprompted.
        await provider.send_text(
            "[instruccion de sistema, no la leas en voz alta: la llamada acaba de "
            "conectar y nadie ha hablado todavia. Saluda y pregunta si ya estan todos, "
            "como indica tu guion. No documentes ni expliques nada todavia.]"
        )
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
                    if self.provider.capabilities.image_input:
                        mime_subtype, image_bytes = decode_image(command["image_data_url"])
                        author = command.get("author", "")
                        caption = f"{author} ha enviado esta foto:" if author else None
                        await self.provider.send_image(
                            image_bytes, f"image/{mime_subtype}", caption=caption
                        )
                    else:
                        await self.provider.send_text(
                            f"{command.get('author', 'Alguien')} ha enviado una foto, pero "
                            "no puedo analizar imagenes con este proveedor."
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
        last_audio_log = 0.0
        async for event in self.provider.events():
            if event.type == ProviderEventType.AUDIO_DELTA:
                # Marks when a burst of speech starts (once per burst, not per chunk, to
                # stay readable) - this is what proved live (2026-09-06) that Gemini Live
                # emits zero audio between a tool call starting and both tools finishing:
                # the two-turn split in _on_assistant_turn_done() exists because of what
                # this log showed. Worth keeping while that area of the design is still
                # moving (e.g. mid-tool filler chatter, see the model notes below).
                now = perf_counter()
                if now - last_audio_log > 1.0:
                    logger.info("call_voice_bridge_audio_burst", call_id=self.call_id)
                last_audio_log = now
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
                await self._on_assistant_turn_done(text)
            elif event.type == ProviderEventType.AUDIO_DONE:
                if self._assistant_text:
                    text = self._assistant_text
                    self._assistant_text = ""
                    await self._on_assistant_turn_done(text)
            elif event.type == ProviderEventType.INPUT_TRANSCRIPT_DONE:
                if event.text:
                    await self.service.log_user_voice(self.call_id, event.text)
            elif event.type == ProviderEventType.USAGE and event.usage is not None:
                await self._persist_usage(event.usage)
            elif event.type == ProviderEventType.TOOL_CALL:
                await self._handle_tool_call(event)
            elif event.type == ProviderEventType.ERROR:
                await self.store.publish(
                    self.call_id, {"type": "call.error", "message": event.text or "assistant_error"}
                )
                if not event.retryable:
                    return

    async def _on_assistant_turn_done(self, text: str) -> None:
        assert self.provider is not None
        await self.service.assistant_finished(self.call_id, text)
        self._assistant_turn_count += 1
        # Turn 1 is the proactive kickoff greeting (_connect()'s send_text). Turn 2 is
        # whatever the model said in response to the group's confirmation - per
        # CALL_GUIDE_PROMPT that should be *only* the "give me a moment" line, no tool
        # call. If it really came back tool-free, nudge research now as its own turn so
        # the announcement's audio has already gone out before the long wait starts.
        # (If the model ignored the instruction and called a tool anyway,
        # _handle_tool_call() already set _research_kicked_off - this is a no-op then.)
        if self._assistant_turn_count == 2 and not self._research_kicked_off:
            self._research_kicked_off = True
            await self.provider.send_text(
                "[instruccion de sistema, no la leas en voz alta: ahora si, documenta a "
                "fondo el lugar y decide el plan de la visita, como tienes indicado]"
            )

    async def _handle_tool_call(self, event: ProviderEvent) -> None:
        """Run a document_poi/plan_poi_visit/find_activities call the model made.

        None of calls' tools set requires_approval (see entrypoints/seed.py's TOOLS) —
        voice/gateway.py's approval round-trip (pending_tools, tool.approval_required)
        doesn't apply here, only the run-and-submit half of it.
        """
        assert self.provider is not None
        self._research_kicked_off = True
        definition = next(
            (tool for tool in self.tool_definitions if tool.get("code") == event.tool_name), None
        )
        if definition is None or not event.tool_call_id:
            logger.warning(
                "call_voice_bridge_unknown_tool", call_id=self.call_id, tool=event.tool_name
            )
            return
        started_at = perf_counter()
        try:
            result = await self.tool_dispatcher.execute(
                definition["handler_code"], event.arguments or {}, self.tool_context, self.locale
            )
        except Exception as error:  # noqa: BLE001 - the call must not die from a bad tool run
            logger.warning(
                "call_voice_bridge_tool_failed",
                call_id=self.call_id,
                tool=event.tool_name,
                error=str(error),
            )
            with contextlib.suppress(Exception):
                await self.provider.submit_tool_result(
                    event.tool_call_id, {"_tool_name": event.tool_name, "error": str(error)}
                )
            return
        usage = self.tool_dispatcher.last_usage
        if usage is not None and usage.billable:
            await self._persist_tool_usage(
                event.tool_name or "unknown", definition["handler_code"], usage
            )
        result["_tool_name"] = event.tool_name
        await self.provider.submit_tool_result(event.tool_call_id, result)
        logger.info(
            "call_voice_bridge_tool_completed",
            call_id=self.call_id,
            tool=event.tool_name,
            elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
        )

    async def _persist_tool_usage(
        self, tool_name: str, handler_code: str, usage: ToolUsage
    ) -> None:
        if self.voice_session_id is None or self.host_id is None:
            return
        async with self.database.sessions() as session:
            model = await session.scalar(
                select(AIModel)
                .join(AIProvider, AIProvider.id == AIModel.provider_id)
                .where(AIProvider.code == "openai", AIModel.external_id == self.settings.tool_model)
            )
            if model is None:
                logger.warning(
                    "call_voice_bridge_tool_usage_unpriced",
                    call_id=self.call_id,
                    tool_model=self.settings.tool_model,
                )
                return
            session.add(
                UsageEvent(
                    user_id=self.host_id,
                    voice_session_id=self.voice_session_id,
                    provider_id=model.provider_id,
                    model_id=model.id,
                    dedupe_key=f"{self.trace_id}:tool:{uuid4().hex}",
                    interaction_type="tool_call",
                    text_input_tokens=usage.text_input_tokens,
                    cached_text_input_tokens=usage.cached_text_input_tokens,
                    text_output_tokens=usage.text_output_tokens,
                    raw_usage_json={"tool": tool_name, "handler": handler_code, **usage.raw},
                    status=UsageStatus.PENDING,
                    trace_id=self.trace_id,
                )
            )
            await session.commit()
        logger.info("call_voice_bridge_tool_usage_recorded", call_id=self.call_id, tool=tool_name)

    async def _watch_room_ended(self) -> None:
        while True:
            await asyncio.sleep(5)
            try:
                room = await self.store.get(self.call_id)
            except CallError:
                return
            if room.status == "ended":
                return


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Same logic as voice/configuration.py's private helper of the same name -
    duplicated rather than imported across modules for a four-line function."""
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


async def _ignore_consume(user_id: int) -> None:
    """The bridge only reacts to commands CallService already accepted.

    ensure_host_can_consume() (calls/policy.py) already gated the event before
    it ever reached the Redis command stream this bridge drains, so there is
    nothing left to check here — this just satisfies CallService's `consume`
    parameter.
    """
    return None
