"""Two Gemini Live families, two adapters.

They are not one API with a version bump. The 3.x half-cascade models take a
`speech_config.language_code`; the 2.x native-audio ones reject it outright
(`1007 Unsupported language code 'es'`), picking the language up from the
conversation instead. Keeping both behind a single class meant every such
difference had to become a conditional, so each family gets its own adapter and
its own connect config, and `ai_models.adapter_code` decides which one a model
uses.

What is genuinely shared is the wire format of what comes *back* — both speak
the same `LiveServerMessage` — so the response mapping below stays common. If
the two families ever start answering differently, split that too rather than
branching inside it.
"""


import asyncio
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types

from locus_v2.billing.pricing import NormalizedUsage
from locus_v2.voice.locales import speech_locale
from locus_v2.voice.protocol import AudioFormat
from locus_v2.voice.providers.base import (
    LiveProvider,
    LiveSessionConfig,
    ProviderCapabilities,
    ProviderEvent,
    ProviderEventType,
)


class GeminiLive3Provider(LiveProvider):
    code = "gemini_live_3"
    capabilities = ProviderCapabilities(
        full_duplex=True,
        function_calling=True,
        async_function_calling=True,
        input_transcription=True,
        output_transcription=True,
        image_input=True,
        context_seeding=True,
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
            config=_gemini3_config(config),
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

    async def send_image(
        self, image_bytes: bytes, mime_type: str, caption: str | None = None
    ) -> None:
        self._require_session()
        parts = [types.Part(inline_data=types.Blob(data=image_bytes, mime_type=mime_type))]
        if caption:
            parts.insert(0, types.Part(text=caption))
        await self._session.send_client_content(
            turns=types.Content(role="user", parts=parts),
            turn_complete=True,
        )

    async def seed_context(self, entries: list[tuple[str, str]]) -> None:
        self._require_session()
        if not entries:
            return
        await self._session.send_client_content(
            turns=[
                types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=[types.Part(text=text)],
                )
                for role, text in entries
            ],
            # The whole point: append the history to the conversation and stop.
            # turn_complete=True here would make the guide answer the last thing
            # the group said before the drop, all over again.
            turn_complete=False,
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
        """Interrupt the model mid-answer.

        `activity_start` is *explicit* activity control, which the API only
        accepts when automatic voice activity detection is off — send it with
        automatic VAD enabled (our default) and the session is killed outright:
        "1007 Explicit activity control is not supported when automatic activity
        detection is enabled" (measured 2026-09-08; the 2.5 family enforces it
        immediately, 3.x is laxer about it). With automatic VAD there is nothing
        to send: the model yields on its own as soon as the interrupting user's
        audio arrives, which is exactly what a barge-in already is.
        """
        if self._session is None:
            return
        options = self._config.provider_options if self._config else {}
        detection = options.get("turn_detection") or {}
        if detection.get("type") == "manual":
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
            raise RuntimeError("GeminiLive3Provider is not connected")


class GeminiLive2Provider(LiveProvider):
    code = "gemini_live_2"
    capabilities = ProviderCapabilities(
        full_duplex=True,
        function_calling=True,
        async_function_calling=True,
        input_transcription=True,
        output_transcription=True,
        image_input=True,
        context_seeding=True,
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
            config=_gemini2_config(config),
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

    async def send_image(
        self, image_bytes: bytes, mime_type: str, caption: str | None = None
    ) -> None:
        self._require_session()
        parts = [types.Part(inline_data=types.Blob(data=image_bytes, mime_type=mime_type))]
        if caption:
            parts.insert(0, types.Part(text=caption))
        await self._session.send_client_content(
            turns=types.Content(role="user", parts=parts),
            turn_complete=True,
        )

    async def seed_context(self, entries: list[tuple[str, str]]) -> None:
        self._require_session()
        if not entries:
            return
        await self._session.send_client_content(
            turns=[
                types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=[types.Part(text=text)],
                )
                for role, text in entries
            ],
            # The whole point: append the history to the conversation and stop.
            # turn_complete=True here would make the guide answer the last thing
            # the group said before the drop, all over again.
            turn_complete=False,
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
        """Interrupt the model mid-answer.

        `activity_start` is *explicit* activity control, which the API only
        accepts when automatic voice activity detection is off — send it with
        automatic VAD enabled (our default) and the session is killed outright:
        "1007 Explicit activity control is not supported when automatic activity
        detection is enabled" (measured 2026-09-08; the 2.5 family enforces it
        immediately, 3.x is laxer about it). With automatic VAD there is nothing
        to send: the model yields on its own as soon as the interrupting user's
        audio arrives, which is exactly what a barge-in already is.
        """
        if self._session is None:
            return
        options = self._config.provider_options if self._config else {}
        detection = options.get("turn_detection") or {}
        if detection.get("type") == "manual":
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
            raise RuntimeError("GeminiLive2Provider is not connected")


# Live sessions bill the whole prompt on every response, and the prompt carries
# every second of audio said so far — the user's and the guide's. So the cost of
# a turn grows with the call: measured live on 2026-09-10 (voice session 116, 22
# turns over 7 minutes), the audio in context went from 230 tokens on the first
# turn to 7567 on the last, and 81333 audio tokens were billed for a context that
# never held more than 7567. A call's cost grows with the square of its length.
#
# A sliding window caps that. The trigger is set explicitly and low on purpose:
# the documented example passes an empty SlidingWindow(), which inherits the
# model's own default trigger — far above the 11400 tokens this call reached — so
# it would lift the 15-minute cap on audio-only sessions without saving anything.
# 8000/4000 keeps roughly the last two minutes of audio; `system_instruction` is
# not part of the window and survives untouched.
CONTEXT_WINDOW_COMPRESSION = {
    "trigger_tokens": 8000,
    "sliding_window": {"target_tokens": 4000},
}


def _gemini3_config(config: LiveSessionConfig) -> dict:
    options = dict(config.provider_options)
    options.pop("interaction_mode", None)
    turn_detection = options.pop("turn_detection", {})
    options.pop("input_audio_transcription", None)
    live_config: dict = {
        "response_modalities": ["AUDIO"],
        "system_instruction": config.prompt,
        "speech_config": {
            # Con "es" a secas Gemini elige un espanol generico que suena
            # latinoamericano; la region hay que pedirla explicita.
            "language_code": speech_locale(config.locale),
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
        "context_window_compression": dict(CONTEXT_WINDOW_COMPRESSION),
    }
    # context_window_compression is in the list so the trigger can be retuned
    # from the panel's Prompt workshop (runtime_config_json) without a deploy:
    # whether 8000 is too tight is exactly what a live call will tell us.
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "context_window_compression",
    ):
        if key in options:
            live_config[key] = options[key]
    return live_config


def _gemini2_config(config: LiveSessionConfig) -> dict:
    options = dict(config.provider_options)
    options.pop("interaction_mode", None)
    turn_detection = options.pop("turn_detection", {})
    options.pop("input_audio_transcription", None)
    live_config: dict = {
        "response_modalities": ["AUDIO"],
        "system_instruction": config.prompt,
        # No language_code: the native-audio family refuses it outright
        # ("1007 Unsupported language code 'es'") and takes the language from
        # the system instruction and the conversation instead.
        "speech_config": {
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
        "context_window_compression": dict(CONTEXT_WINDOW_COMPRESSION),
    }
    # context_window_compression is in the list so the trigger can be retuned
    # from the panel's Prompt workshop (runtime_config_json) without a deploy:
    # whether 8000 is too tight is exactly what a live call will tell us.
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "context_window_compression",
    ):
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
