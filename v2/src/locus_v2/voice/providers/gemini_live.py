import asyncio
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types

from locus_v2.billing.pricing import NormalizedUsage
from locus_v2.voice.protocol import AudioFormat
from locus_v2.voice.providers.base import (
    LiveProvider,
    LiveSessionConfig,
    ProviderCapabilities,
    ProviderEvent,
    ProviderEventType,
)


class GeminiLiveProvider(LiveProvider):
    code = "gemini_live"
    capabilities = ProviderCapabilities(
        full_duplex=True,
        function_calling=True,
        async_function_calling=True,
        input_transcription=True,
        output_transcription=True,
        image_input=True,
        session_resumption=True,
        context_compression=True,
        supported_input_formats=[AudioFormat.PCM16_16KHZ, AudioFormat.PCM16_24KHZ],
    )

    def __init__(self, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._manager: Any = None
        self._session: Any = None
        self._config: LiveSessionConfig | None = None

    async def connect(self, config: LiveSessionConfig) -> None:
        self._config = config
        self._manager = self._client.aio.live.connect(
            model=config.model,
            config=_gemini_config(config),
        )
        self._session = await self._manager.__aenter__()

    async def send_audio(self, chunk: bytes) -> None:
        self._require_session()
        rate = 16000 if self._config.audio_format == AudioFormat.PCM16_16KHZ else 24000
        await self._session.send_realtime_input(
            audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={rate}")
        )

    async def commit_audio(self) -> None:
        self._require_session()
        await self._session.send_realtime_input(audio_stream_end=True)

    async def send_text(self, text: str) -> None:
        self._require_session()
        await self._session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=True,
        )

    async def submit_tool_result(self, call_id: str, result: dict) -> None:
        self._require_session()
        payload = dict(result)
        await self._session.send_tool_response(
            function_responses=types.FunctionResponse(
                id=call_id,
                name=payload.pop("_tool_name", "locus_tool"),
                response=payload,
            )
        )

    async def cancel_response(self) -> None:
        if self._session is not None:
            await self._session.send_realtime_input(activity_start={})

    async def events(self) -> AsyncIterator[ProviderEvent]:
        self._require_session()
        yield ProviderEvent(type=ProviderEventType.READY)
        while self._session is not None:
            async for message in self._session.receive():
                for event in _map_gemini_message(message):
                    yield event
            await asyncio.sleep(0)

    async def close(self) -> None:
        if self._manager is not None:
            await self._manager.__aexit__(None, None, None)
        self._manager = None
        self._session = None
        self._client.close()

    def _require_session(self) -> None:
        if self._session is None or self._config is None:
            raise RuntimeError("Gemini Live is not connected")


def _gemini_config(config: LiveSessionConfig) -> dict:
    options = dict(config.provider_options)
    options.pop("interaction_mode", None)
    turn_detection = options.pop("turn_detection", {})
    options.pop("input_audio_transcription", None)
    live_config: dict = {
        "response_modalities": ["AUDIO"],
        "system_instruction": config.prompt,
        "speech_config": {
            "language_code": config.locale,
            "voice_config": {"prebuilt_voice_config": {"voice_name": config.voice or "Kore"}},
        },
        "tools": [
            {
                "function_declarations": [
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters_json_schema": tool.get("parameters", {}),
                    }
                    for tool in config.tools
                ]
            }
        ]
        if config.tools
        else [],
        "input_audio_transcription": {},
        "output_audio_transcription": {},
        "realtime_input_config": _gemini_turn_detection(turn_detection),
    }
    for key in ("temperature", "top_p", "top_k", "max_output_tokens"):
        if key in options:
            live_config[key] = options[key]
    return live_config


def _gemini_turn_detection(value: dict) -> dict:
    detection_type = value.get("type", "provider_native")
    if detection_type == "manual":
        return {"automatic_activity_detection": {"disabled": True}}
    automatic: dict = {"disabled": False}
    if "prefix_padding_ms" in value:
        automatic["prefix_padding_ms"] = value["prefix_padding_ms"]
    if "silence_duration_ms" in value:
        automatic["silence_duration_ms"] = value["silence_duration_ms"]
    return {"automatic_activity_detection": automatic}


def _map_gemini_message(message: types.LiveServerMessage) -> list[ProviderEvent]:
    events: list[ProviderEvent] = []
    if message.usage_metadata:
        usage = message.usage_metadata
        prompt_text = _modality_tokens(usage.prompt_tokens_details, "TEXT")
        prompt_audio = _modality_tokens(usage.prompt_tokens_details, "AUDIO")
        prompt_image = _modality_tokens(usage.prompt_tokens_details, "IMAGE")
        cached_text = _modality_tokens(usage.cache_tokens_details, "TEXT")
        cached_audio = _modality_tokens(usage.cache_tokens_details, "AUDIO")
        cached_image = _modality_tokens(usage.cache_tokens_details, "IMAGE")
        response_text = _modality_tokens(usage.response_tokens_details, "TEXT")
        response_audio = _modality_tokens(usage.response_tokens_details, "AUDIO")
        if not usage.prompt_tokens_details:
            prompt_text = usage.prompt_token_count or 0
        if not usage.cache_tokens_details:
            cached_text = usage.cached_content_token_count or 0
        if not usage.response_tokens_details:
            response_text = usage.response_token_count or 0
        events.append(
            ProviderEvent(
                type=ProviderEventType.USAGE,
                usage=NormalizedUsage(
                    text_input_tokens=max(0, prompt_text - cached_text),
                    cached_text_input_tokens=cached_text,
                    text_output_tokens=response_text,
                    audio_input_tokens=max(0, prompt_audio - cached_audio),
                    cached_audio_input_tokens=cached_audio,
                    audio_output_tokens=response_audio,
                    image_input_tokens=max(0, prompt_image - cached_image),
                    cached_image_input_tokens=cached_image,
                    raw=usage.model_dump(mode="json", exclude_none=True),
                ),
            )
        )
    content = message.server_content
    if content is not None:
        if content.input_transcription and content.input_transcription.text:
            events.append(
                ProviderEvent(
                    type=ProviderEventType.INPUT_TRANSCRIPT_DONE,
                    text=content.input_transcription.text,
                )
            )
        if content.output_transcription and content.output_transcription.text:
            events.append(
                ProviderEvent(
                    type=ProviderEventType.TEXT_DELTA,
                    text=content.output_transcription.text,
                )
            )
        if content.model_turn:
            for part in content.model_turn.parts or []:
                if part.inline_data and part.inline_data.data:
                    events.append(
                        ProviderEvent(
                            type=ProviderEventType.AUDIO_DELTA,
                            audio=part.inline_data.data,
                        )
                    )
                if part.text:
                    events.append(ProviderEvent(type=ProviderEventType.TEXT_DELTA, text=part.text))
        if content.turn_complete:
            events.extend(
                [
                    ProviderEvent(type=ProviderEventType.TEXT_DONE),
                    ProviderEvent(type=ProviderEventType.AUDIO_DONE),
                ]
            )
    if message.tool_call:
        for call in message.tool_call.function_calls or []:
            events.append(
                ProviderEvent(
                    type=ProviderEventType.TOOL_CALL,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    arguments=dict(call.args or {}),
                )
            )
    if message.go_away:
        events.append(
            ProviderEvent(
                type=ProviderEventType.ERROR,
                text="Gemini Live requested a session reconnect",
                error_code="go_away",
                retryable=True,
            )
        )
    return events


def _modality_tokens(
    details: list[types.ModalityTokenCount] | None,
    modality: str,
) -> int:
    return sum(
        item.token_count or 0
        for item in details or []
        if item.modality is not None and item.modality.value == modality
    )
