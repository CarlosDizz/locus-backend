import base64
import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from locus_v2.billing.pricing import NormalizedUsage
from locus_v2.voice.protocol import AudioFormat
from locus_v2.voice.providers.base import (
    LiveProvider,
    LiveSessionConfig,
    ProviderCapabilities,
    ProviderEvent,
    ProviderEventType,
)


class OpenAIRealtimeProvider(LiveProvider):
    code = "openai_realtime"
    capabilities = ProviderCapabilities(
        full_duplex=True,
        function_calling=True,
        async_function_calling=True,
        input_transcription=True,
        output_transcription=True,
        image_input=True,
        session_resumption=True,
        supported_input_formats=[AudioFormat.PCM16_24KHZ],
    )

    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._manager: Any = None
        self._connection: Any = None
        self._audio_commits_create_response = True

    async def connect(self, config: LiveSessionConfig) -> None:
        if config.audio_format != AudioFormat.PCM16_24KHZ:
            raise ValueError("OpenAI Realtime requires 24 kHz PCM16 audio")
        turn_detection = config.provider_options.get("turn_detection") or {}
        self._audio_commits_create_response = not bool(
            turn_detection.get("type") not in {None, "manual"}
            and turn_detection.get("create_response", True)
        )
        self._manager = self._client.realtime.connect(model=config.model)
        self._connection = await self._manager.__aenter__()
        await self._connection.session.update(session=_openai_session(config))

    async def send_audio(self, chunk: bytes) -> None:
        self._require_connection()
        await self._connection.input_audio_buffer.append(
            audio=base64.b64encode(chunk).decode("ascii")
        )

    async def commit_audio(self) -> None:
        self._require_connection()
        await self._connection.input_audio_buffer.commit()
        if self._audio_commits_create_response:
            await self._connection.response.create()

    async def send_text(self, text: str) -> None:
        self._require_connection()
        await self._connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }
        )
        await self._connection.response.create()

    async def submit_tool_result(self, call_id: str, result: dict) -> None:
        self._require_connection()
        await self._connection.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result, ensure_ascii=False),
            }
        )
        await self._connection.response.create()

    async def cancel_response(self) -> None:
        if self._connection is not None:
            await self._connection.response.cancel()

    async def events(self) -> AsyncIterator[ProviderEvent]:
        self._require_connection()
        async for event in self._connection:
            mapped = _map_openai_event(event.model_dump(mode="json", exclude_none=True))
            if mapped is not None:
                yield mapped

    async def close(self) -> None:
        if self._manager is not None:
            await self._manager.__aexit__(None, None, None)
        self._manager = None
        self._connection = None
        await self._client.close()

    def _require_connection(self) -> None:
        if self._connection is None:
            raise RuntimeError("OpenAI Realtime is not connected")


def _openai_session(config: LiveSessionConfig) -> dict:
    options = dict(config.provider_options)
    turn_detection = _openai_turn_detection(options.pop("turn_detection", None))
    transcription = options.pop("input_audio_transcription", None)
    options.pop("interaction_mode", None)
    options.pop("temperature", None)  # Current Realtime sessions do not expose it.
    audio_input: dict = {
        "format": {"type": "audio/pcm", "rate": 24000},
        "turn_detection": turn_detection,
    }
    if transcription and transcription.get("model"):
        audio_input["transcription"] = {
            "model": transcription["model"],
            "language": config.locale.split("-", 1)[0],
        }
    session = {
        "type": "realtime",
        "model": config.model,
        "instructions": config.prompt,
        "output_modalities": ["audio"],
        "audio": {
            "input": audio_input,
            "output": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "voice": config.voice or "marin",
            },
        },
        "tools": config.tools,
        "tool_choice": "auto",
    }
    for key in ("max_output_tokens", "parallel_tool_calls", "reasoning", "truncation"):
        if key in options:
            session[key] = options[key]
    return session


def _openai_turn_detection(value: dict | None) -> dict | None:
    if not value or value.get("type") in {None, "manual"}:
        return None
    detection_type = value.get("type")
    if detection_type == "provider_native":
        detection_type = "semantic_vad"
    result = {
        "type": detection_type,
        "create_response": value.get("create_response", True),
        "interrupt_response": value.get("interrupt_response", True),
    }
    allowed = (
        ("eagerness",)
        if detection_type == "semantic_vad"
        else ("threshold", "prefix_padding_ms", "silence_duration_ms")
    )
    for key in allowed:
        if key in value:
            result[key] = value[key]
    return result


def _map_openai_event(event: dict) -> ProviderEvent | None:
    event_type = event.get("type", "")
    if event_type == "session.updated":
        return ProviderEvent(type=ProviderEventType.READY)
    if event_type == "conversation.item.input_audio_transcription.delta":
        return ProviderEvent(
            type=ProviderEventType.INPUT_TRANSCRIPT_DELTA, text=event.get("delta", "")
        )
    if event_type == "conversation.item.input_audio_transcription.completed":
        return ProviderEvent(
            type=ProviderEventType.INPUT_TRANSCRIPT_DONE,
            text=event.get("transcript", ""),
        )
    if event_type in {"response.output_text.delta", "response.output_audio_transcript.delta"}:
        return ProviderEvent(type=ProviderEventType.TEXT_DELTA, text=event.get("delta", ""))
    if event_type in {"response.output_text.done", "response.output_audio_transcript.done"}:
        return ProviderEvent(
            type=ProviderEventType.TEXT_DONE,
            text=event.get("text") or event.get("transcript") or "",
        )
    if event_type == "response.output_audio.delta":
        return ProviderEvent(
            type=ProviderEventType.AUDIO_DELTA,
            audio=base64.b64decode(event.get("delta", "")),
        )
    if event_type == "response.output_audio.done":
        return ProviderEvent(type=ProviderEventType.AUDIO_DONE)
    if event_type == "response.function_call_arguments.done":
        try:
            arguments = json.loads(event.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {"raw": event.get("arguments", "")}
        return ProviderEvent(
            type=ProviderEventType.TOOL_CALL,
            tool_call_id=event.get("call_id"),
            tool_name=event.get("name"),
            arguments=arguments,
        )
    if event_type == "response.done":
        response = event.get("response") or {}
        usage = response.get("usage") or {}
        if usage:
            return ProviderEvent(type=ProviderEventType.USAGE, usage=_openai_usage(usage))
    if event_type == "error":
        error = event.get("error") or {}
        return ProviderEvent(
            type=ProviderEventType.ERROR,
            text=error.get("message", "OpenAI Realtime error"),
            error_code=error.get("code") or error.get("type"),
            retryable=error.get("type") in {"server_error", "rate_limit_error"},
        )
    return None


def _openai_usage(usage: dict) -> NormalizedUsage:
    input_details = usage.get("input_token_details") or {}
    output_details = usage.get("output_token_details") or {}
    cached_details = input_details.get("cached_tokens_details") or {}
    audio_input = input_details.get("audio_tokens", 0)
    image_input = input_details.get("image_tokens", 0)
    text_input = input_details.get(
        "text_tokens",
        max(0, usage.get("input_tokens", 0) - audio_input - image_input),
    )
    cached_total = input_details.get("cached_tokens", 0)
    cached_text = cached_details.get("text_tokens", min(cached_total, text_input))
    uncategorized_cached = max(0, cached_total - cached_text)
    cached_audio = cached_details.get("audio_tokens", min(uncategorized_cached, audio_input))
    uncategorized_cached = max(0, uncategorized_cached - cached_audio)
    cached_image = cached_details.get("image_tokens", min(uncategorized_cached, image_input))
    return NormalizedUsage(
        text_input_tokens=max(0, text_input - cached_text),
        cached_text_input_tokens=cached_text,
        text_output_tokens=max(
            0, usage.get("output_tokens", 0) - output_details.get("audio_tokens", 0)
        ),
        audio_input_tokens=max(0, audio_input - cached_audio),
        cached_audio_input_tokens=cached_audio,
        audio_output_tokens=output_details.get("audio_tokens", 0),
        image_input_tokens=max(0, image_input - cached_image),
        cached_image_input_tokens=cached_image,
        raw=usage,
    )
