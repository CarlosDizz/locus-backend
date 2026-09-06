import asyncio
import base64
import traceback
from contextlib import suppress
from time import perf_counter
from uuid import uuid4

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.ai.models import AIModel, AIProvider
from locus_v2.billing.models import UsageEvent, UsageStatus
from locus_v2.config import Settings
from locus_v2.identity.models import User
from locus_v2.infrastructure.database.session import Database
from locus_v2.observability import LocusEventLogger, LogLevel
from locus_v2.shared.clock import utc_now
from locus_v2.voice.configuration import (
    ResolvedProvider,
    ResolvedVoiceConfiguration,
    VoiceConfigurationResolver,
)
from locus_v2.voice.models import VoiceSession, VoiceSessionStatus, VoiceTurn, VoiceTurnRole
from locus_v2.voice.protocol import (
    AudioCommit,
    AudioStart,
    ResponseCancel,
    ServerEvent,
    SessionClose,
    SessionStart,
    TextSend,
    ToolApproval,
    client_event_adapter,
)
from locus_v2.voice.providers.base import LiveProvider, ProviderEvent, ProviderEventType
from locus_v2.voice.providers.registry import ProviderRegistry
from locus_v2.voice.tools import VoiceToolDispatcher

logger = structlog.get_logger()


class VoiceGateway:
    def __init__(
        self,
        websocket: WebSocket,
        session: AsyncSession,
        database: Database,
        settings: Settings,
        registry: ProviderRegistry,
        user: User,
        event_logger: LocusEventLogger | None = None,
    ) -> None:
        self.websocket = websocket
        self.session = session
        self.database = database
        self.settings = settings
        self.registry = registry
        self.user = user
        self.event_logger = event_logger
        self.trace_id = uuid4().hex
        self.sequence = 0
        self.turn_sequence = 0
        self.provider: LiveProvider | None = None
        self.resolved_provider: ResolvedProvider | None = None
        self.configuration: ResolvedVoiceConfiguration | None = None
        self.voice_session: VoiceSession | None = None
        self.pending_tools: dict[str, ProviderEvent] = {}
        self.assistant_text = ""
        self.audio_started = False
        self.tools = VoiceToolDispatcher(settings)
        self.persistence_lock = asyncio.Lock()
        self.started_at = perf_counter()

    async def run(self) -> None:
        final_status = VoiceSessionStatus.COMPLETED
        logger.info(
            "voice_gateway_started",
            trace_id=self.trace_id,
            user_id=self.user.id,
        )
        await self._record_event("info", "voice.gateway.started")
        try:
            request = await self._receive_start()
            self.configuration = await VoiceConfigurationResolver(
                self.session, self.settings
            ).resolve(request)
            logger.info(
                "voice_configuration_resolved",
                trace_id=self.trace_id,
                route=self.configuration.routing_profile_code,
                primary_adapter=self.configuration.primary.adapter_code,
                fallback_adapter=(
                    self.configuration.fallback.adapter_code
                    if self.configuration.fallback
                    else None
                ),
                tool_count=len(self.configuration.snapshot.get("tools", [])),
                elapsed_ms=round((perf_counter() - self.started_at) * 1000, 1),
            )
            await self._record_event(
                "info",
                "voice.configuration.resolved",
                elapsed_ms=(perf_counter() - self.started_at) * 1000,
                context={
                    "route": self.configuration.routing_profile_code,
                    "primary_adapter": self.configuration.primary.adapter_code,
                    "fallback_adapter": (
                        self.configuration.fallback.adapter_code
                        if self.configuration.fallback
                        else None
                    ),
                    "tool_count": len(self.configuration.snapshot.get("tools", [])),
                },
            )
            self.provider, self.resolved_provider = await self._connect_provider(self.configuration)
            await self._create_session()
            await self._send(
                "session.ready",
                {
                    "session_id": self.voice_session.public_id,
                    "provider": self.resolved_provider.provider_code,
                    "model": self.resolved_provider.config.model,
                    "audio_format": request.audio_format,
                    "capabilities": self.provider.capabilities.model_dump(mode="json"),
                },
            )
            await self._bridge()
        except WebSocketDisconnect:
            logger.info("voice_client_disconnected", trace_id=self.trace_id)
        except (ValidationError, ValueError, LookupError) as error:
            final_status = VoiceSessionStatus.FAILED
            await self._record_event(
                "warning",
                "voice.gateway.request_failed",
                message=str(error),
                error_type=type(error).__name__,
            )
            with suppress(Exception):
                await self._send("session.error", {"message": str(error), "retryable": False})
        except Exception as error:
            final_status = VoiceSessionStatus.FAILED
            logger.exception(
                "voice_gateway_failed",
                trace_id=self.trace_id,
                error_type=type(error).__name__,
                elapsed_ms=round((perf_counter() - self.started_at) * 1000, 1),
            )
            await self._record_event(
                "error",
                "voice.gateway.failed",
                message=str(error),
                error_type=type(error).__name__,
                elapsed_ms=(perf_counter() - self.started_at) * 1000,
                context={
                    "traceback": traceback.format_exc(),
                    "provider": (
                        self.resolved_provider.provider_code if self.resolved_provider else None
                    ),
                },
            )
            with suppress(Exception):
                await self._send(
                    "session.error",
                    {"message": str(error), "retryable": False, "code": "gateway_error"},
                )
        finally:
            if self.provider is not None:
                with suppress(Exception):
                    await self.provider.close()
            try:
                await self._finish(final_status)
            except Exception as error:
                logger.exception(
                    "voice_session_finish_failed",
                    trace_id=self.trace_id,
                    error_type=type(error).__name__,
                )
                await self._record_event(
                    "error",
                    "voice.session.finish_failed",
                    message=str(error),
                    error_type=type(error).__name__,
                )
            logger.info(
                "voice_gateway_finished",
                trace_id=self.trace_id,
                status=final_status,
                provider=(self.resolved_provider.provider_code if self.resolved_provider else None),
                elapsed_ms=round((perf_counter() - self.started_at) * 1000, 1),
            )
            await self._record_event(
                "info" if final_status == VoiceSessionStatus.COMPLETED else "warning",
                "voice.gateway.finished",
                elapsed_ms=(perf_counter() - self.started_at) * 1000,
                context={
                    "status": final_status,
                    "provider": (
                        self.resolved_provider.provider_code if self.resolved_provider else None
                    ),
                },
            )

    async def _receive_start(self) -> SessionStart:
        raw = await self.websocket.receive_text()
        event = client_event_adapter.validate_json(raw)
        if not isinstance(event, SessionStart):
            raise ValueError("The first event must be session.start")
        return event

    async def _connect_provider(
        self, configuration: ResolvedVoiceConfiguration
    ) -> tuple[LiveProvider, ResolvedProvider]:
        errors: list[str] = []
        for resolved in (configuration.primary, configuration.fallback):
            if resolved is None:
                continue
            provider: LiveProvider | None = None
            attempt_started_at = perf_counter()
            logger.info(
                "voice_provider_connect_started",
                trace_id=self.trace_id,
                provider=resolved.provider_code,
                adapter=resolved.adapter_code,
                model=resolved.config.model,
            )
            try:
                provider = self.registry.create(resolved.adapter_code)
                if (
                    resolved.config.audio_format
                    not in provider.capabilities.supported_input_formats
                ):
                    raise ValueError(
                        f"{resolved.adapter_code} does not support {resolved.config.audio_format}"
                    )
                await provider.connect(resolved.config)
                logger.info(
                    "voice_provider_connected",
                    trace_id=self.trace_id,
                    provider=resolved.provider_code,
                    adapter=resolved.adapter_code,
                    model=resolved.config.model,
                    elapsed_ms=round((perf_counter() - attempt_started_at) * 1000, 1),
                )
                return provider, resolved
            except Exception as error:
                logger.warning(
                    "voice_provider_connect_failed",
                    trace_id=self.trace_id,
                    provider=resolved.provider_code,
                    adapter=resolved.adapter_code,
                    model=resolved.config.model,
                    error_type=type(error).__name__,
                    elapsed_ms=round((perf_counter() - attempt_started_at) * 1000, 1),
                )
                await self._record_event(
                    "warning",
                    "voice.provider.connect_failed",
                    message=str(error),
                    error_type=type(error).__name__,
                    elapsed_ms=(perf_counter() - attempt_started_at) * 1000,
                    context={
                        "provider": resolved.provider_code,
                        "adapter": resolved.adapter_code,
                        "model": resolved.config.model,
                        "traceback": traceback.format_exc(),
                    },
                )
                errors.append(f"{resolved.adapter_code}: {error}")
                if provider is not None:
                    with suppress(Exception):
                        await provider.close()
        raise RuntimeError("No voice provider could connect: " + "; ".join(errors))

    async def _create_session(self) -> None:
        assert self.configuration is not None
        assert self.resolved_provider is not None
        self.voice_session = VoiceSession(
            user_id=self.user.id,
            routing_profile_id=self.configuration.routing_profile_id,
            prompt_version_id=self.configuration.prompt_version_id,
            primary_model_id=self.configuration.primary.model_id,
            active_model_id=self.resolved_provider.model_id,
            status=VoiceSessionStatus.ACTIVE,
            locale=self.configuration.locale,
            voice=self.resolved_provider.config.voice,
            context_type=self.configuration.context_type,
            context_public_id=self.configuration.context_public_id,
            config_snapshot_json=self.configuration.snapshot,
            started_at=utc_now(),
        )
        self.session.add(self.voice_session)
        await self.session.commit()

    async def _bridge(self) -> None:
        client_task = asyncio.create_task(self._client_loop())
        provider_task = asyncio.create_task(self._provider_loop())
        done, pending = await asyncio.wait(
            {client_task, provider_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in pending:
            with suppress(asyncio.CancelledError):
                await task
        for task in done:
            task.result()

    async def _client_loop(self) -> None:
        assert self.provider is not None
        while True:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                if not self.audio_started:
                    raise ValueError("audio.start is required before binary audio")
                await self.provider.send_audio(message["bytes"])
                continue
            raw = message.get("text")
            if raw is None:
                continue
            event = client_event_adapter.validate_json(raw)
            if isinstance(event, AudioStart):
                expected = self.resolved_provider.config.audio_format
                if event.format != expected:
                    raise ValueError(f"Expected audio format {expected}, got {event.format}")
                self.audio_started = True
                await self._send("audio.ready", {"format": event.format})
            elif isinstance(event, AudioCommit):
                await self.provider.commit_audio()
            elif isinstance(event, TextSend):
                await self._persist_turn(VoiceTurnRole.USER, event.text)
                await self.provider.send_text(event.text)
            elif isinstance(event, ResponseCancel):
                await self.provider.cancel_response()
            elif isinstance(event, ToolApproval):
                await self._handle_approval(event)
            elif isinstance(event, SessionClose):
                await self._send("session.closed")
                return
            elif isinstance(event, SessionStart):
                raise ValueError("session.start can only be sent once")

    async def _provider_loop(self) -> None:
        assert self.provider is not None
        async for event in self.provider.events():
            if event.type == ProviderEventType.READY:
                await self._send("provider.ready")
            elif event.type == ProviderEventType.AUDIO_DELTA:
                await self._send(
                    "audio.delta",
                    {"audio": base64.b64encode(event.audio or b"").decode("ascii")},
                )
            elif event.type == ProviderEventType.TEXT_DELTA:
                self.assistant_text += event.text or ""
                await self._send("transcript.assistant.delta", {"text": event.text or ""})
            elif event.type == ProviderEventType.TEXT_DONE:
                text = event.text or self.assistant_text
                if text:
                    await self._persist_turn(VoiceTurnRole.ASSISTANT, text)
                self.assistant_text = ""
                await self._send("transcript.assistant.done", {"text": text})
            elif event.type == ProviderEventType.INPUT_TRANSCRIPT_DELTA:
                await self._send("transcript.user.delta", {"text": event.text or ""})
            elif event.type == ProviderEventType.INPUT_TRANSCRIPT_DONE:
                await self._persist_turn(VoiceTurnRole.USER, event.text or "")
                await self._send("transcript.user.done", {"text": event.text or ""})
            elif event.type == ProviderEventType.AUDIO_DONE:
                await self._send("audio.done")
            elif event.type == ProviderEventType.TOOL_CALL:
                await self._handle_tool_call(event)
            elif event.type == ProviderEventType.USAGE and event.usage is not None:
                await self._persist_usage(event.usage)
                await self._send("usage.recorded", event.usage.model_dump(mode="json"))
            elif event.type == ProviderEventType.ERROR:
                await self._send(
                    "provider.error",
                    {
                        "message": event.text,
                        "code": event.error_code,
                        "retryable": event.retryable,
                    },
                )
                if not event.retryable:
                    raise RuntimeError(event.text or "Voice provider failed")

    async def _handle_tool_call(self, event: ProviderEvent) -> None:
        definition = self._tool_definition(event.tool_name)
        if definition is None or not event.tool_call_id:
            raise ValueError(f"Provider requested an unknown tool: {event.tool_name}")
        if definition.get("requires_approval"):
            self.pending_tools[event.tool_call_id] = event
            await self._send(
                "tool.approval_required",
                {
                    "call_id": event.tool_call_id,
                    "name": event.tool_name,
                    "arguments": event.arguments or {},
                },
            )
            return
        await self._execute_tool(event, definition)

    async def _handle_approval(self, approval: ToolApproval) -> None:
        event = self.pending_tools.pop(approval.call_id, None)
        if event is None:
            raise ValueError("Unknown or expired tool approval")
        definition = self._tool_definition(event.tool_name)
        if definition is None:
            raise ValueError(f"Provider requested an unknown tool: {event.tool_name}")
        if not approval.approved:
            await self.provider.submit_tool_result(
                approval.call_id,
                {"_tool_name": event.tool_name, "error": "User declined this action"},
            )
            return
        await self._execute_tool(event, definition)

    async def _execute_tool(self, event: ProviderEvent, definition: dict) -> None:
        assert self.provider is not None
        assert self.configuration is not None
        await self._send(
            "tool.started",
            {"call_id": event.tool_call_id, "name": event.tool_name},
        )
        started_at = perf_counter()
        logger.info(
            "voice_gateway_tool_started",
            trace_id=self.trace_id,
            provider=self.resolved_provider.provider_code if self.resolved_provider else None,
            tool=event.tool_name,
            handler=definition["handler_code"],
        )
        try:
            result = await self.tools.execute(
                definition["handler_code"],
                event.arguments or {},
                self.configuration.context,
                self.configuration.locale,
            )
        except Exception as error:
            await self._record_event(
                "error",
                "voice.tool.failed",
                message=str(error),
                error_type=type(error).__name__,
                elapsed_ms=(perf_counter() - started_at) * 1000,
                context={
                    "provider": self.resolved_provider.provider_code
                    if self.resolved_provider
                    else None,
                    "tool": event.tool_name,
                    "handler": definition["handler_code"],
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        await self._persist_tool_usage(event.tool_name, definition["handler_code"])
        result["_tool_name"] = event.tool_name
        await self._persist_turn(
            VoiceTurnRole.TOOL,
            result.get("answer", ""),
            tool_name=event.tool_name,
            payload={"arguments": event.arguments or {}, "result": result},
        )
        await self.provider.submit_tool_result(event.tool_call_id, result)
        logger.info(
            "voice_gateway_tool_submitted",
            trace_id=self.trace_id,
            provider=self.resolved_provider.provider_code if self.resolved_provider else None,
            tool=event.tool_name,
            elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
        )
        await self._record_event(
            "info",
            "voice.tool.completed",
            elapsed_ms=(perf_counter() - started_at) * 1000,
            context={
                "provider": self.resolved_provider.provider_code
                if self.resolved_provider
                else None,
                "tool": event.tool_name,
                "handler": definition["handler_code"],
                "answer_chars": len(result.get("answer", "")),
            },
        )
        await self._send(
            "tool.done",
            {"call_id": event.tool_call_id, "name": event.tool_name},
        )

    def _tool_definition(self, name: str | None) -> dict | None:
        if self.configuration is None:
            return None
        return next(
            (tool for tool in self.configuration.snapshot["tools"] if tool["code"] == name),
            None,
        )

    async def _persist_turn(
        self,
        role: str,
        text: str,
        tool_name: str | None = None,
        payload: dict | None = None,
    ) -> None:
        if self.voice_session is None or not text:
            return
        async with self.persistence_lock:
            self.turn_sequence += 1
            async with self.database.sessions() as session:
                session.add(
                    VoiceTurn(
                        voice_session_id=self.voice_session.id,
                        sequence=self.turn_sequence,
                        role=role,
                        text=text,
                        tool_name=tool_name,
                        tool_payload_json=payload,
                        trace_id=self.trace_id,
                    )
                )
                await session.commit()

    async def _persist_usage(self, usage) -> None:
        if self.voice_session is None or self.resolved_provider is None:
            return
        async with self.database.sessions() as session:
            model = await session.scalar(
                select(AIModel).where(AIModel.id == self.resolved_provider.model_id)
            )
            if model is None:
                return
            session.add(
                UsageEvent(
                    user_id=self.user.id,
                    voice_session_id=self.voice_session.id,
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
            "voice_usage_recorded",
            trace_id=self.trace_id,
            provider=self.resolved_provider.provider_code,
            model=model.external_id,
            text_input_tokens=usage.text_input_tokens,
            text_output_tokens=usage.text_output_tokens,
            audio_input_tokens=usage.audio_input_tokens,
            audio_output_tokens=usage.audio_output_tokens,
        )
        await self._record_event(
            "info",
            "voice.usage.recorded",
            context={
                "provider": self.resolved_provider.provider_code,
                "model": model.external_id,
                "text_input_tokens": usage.text_input_tokens,
                "text_output_tokens": usage.text_output_tokens,
                "audio_input_tokens": usage.audio_input_tokens,
                "audio_output_tokens": usage.audio_output_tokens,
            },
        )

    async def _persist_tool_usage(self, tool_name: str, handler_code: str) -> None:
        """Bill the plain OpenAI call a voice tool just made on its own.

        VoiceToolDispatcher.execute() (voice/tools.py) calls the OpenAI Responses
        API directly for document_poi/plan_poi_visit/find_activities - outside
        the LiveProvider abstraction, so _persist_usage() above never sees it.
        That cost was going completely unbilled until this existed.
        """
        usage = self.tools.last_usage
        if usage is None or not usage.billable or self.voice_session is None:
            return
        async with self.database.sessions() as session:
            model = await session.scalar(
                select(AIModel)
                .join(AIProvider, AIProvider.id == AIModel.provider_id)
                .where(AIProvider.code == "openai", AIModel.external_id == self.settings.tool_model)
            )
            if model is None:
                logger.warning(
                    "voice_tool_usage_unpriced",
                    trace_id=self.trace_id,
                    tool_model=self.settings.tool_model,
                )
                return
            session.add(
                UsageEvent(
                    user_id=self.user.id,
                    voice_session_id=self.voice_session.id,
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
        logger.info(
            "voice_tool_usage_recorded",
            trace_id=self.trace_id,
            tool=tool_name,
            text_input_tokens=usage.text_input_tokens,
            text_output_tokens=usage.text_output_tokens,
        )

    async def _finish(self, status: str) -> None:
        if self.voice_session is None:
            return
        ended_at = utc_now()
        async with self.database.sessions() as session:
            await session.execute(
                update(VoiceSession)
                .where(
                    VoiceSession.id == self.voice_session.id,
                    VoiceSession.ended_at.is_(None),
                )
                .values(status=status, ended_at=ended_at)
            )
            await session.commit()

    async def _send(self, event_type: str, payload: dict | None = None) -> None:
        self.sequence += 1
        await self.websocket.send_json(
            ServerEvent(
                type=event_type,
                sequence=self.sequence,
                trace_id=self.trace_id,
                payload=payload or {},
            ).model_dump(mode="json")
        )

    async def _record_event(
        self,
        level: LogLevel,
        event: str,
        *,
        message: str | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        elapsed_ms: float | int | None = None,
        context: dict | None = None,
    ) -> None:
        if self.event_logger is None:
            return
        await self.event_logger.write(
            level,
            event,
            message=message,
            trace_id=self.trace_id,
            user_id=self.user.id,
            voice_session_id=self.voice_session.id if self.voice_session else None,
            error_type=error_type,
            error_code=error_code,
            elapsed_ms=elapsed_ms,
            context=context,
        )
