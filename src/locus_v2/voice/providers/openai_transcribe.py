"""A transcription-only realtime session, running alongside the conversation.

The live provider that carries the conversation also transcribes what the user
said, but Gemini's transcript is poor and the whole point of the shared log is
that it says what was actually said — `_recap_entries` replays those lines to
rebuild context on every reconnection, so a bad transcript keeps costing after
the turn is over.

This opens a second socket to OpenAI whose only job is speech to text. It earns
that extra moving part by being able to hear names it has been warned about:
`keywords` and `prompt` carry the POI and city names into the session, and
proper nouns are precisely what came back wrong.

The wire protocol is the same one the conversational adapter already speaks —
`input_audio_buffer.append` / `.commit` out, and
`conversation.item.input_audio_transcription.*` back — so the event mapping is
reused from openai_realtime rather than written twice.
"""

import base64
from typing import Any

from openai import AsyncOpenAI

from locus_v2.voice.protocol import AudioFormat
from locus_v2.voice.providers.base import (
    LiveProvider,
    LiveSessionConfig,
    ProviderCapabilities,
)
from locus_v2.voice.providers.openai_realtime import _map_openai_event

# Un tope para que la pista siga orientando: una lista larguisima de nombres
# deja de ser una ayuda y se convierte en ruido.
MAX_KEYWORDS = 60


class OpenAITranscribeProvider(LiveProvider):
    """Speech to text only. Every conversational method is a no-op on purpose."""

    code = "openai_transcribe"
    capabilities = ProviderCapabilities(
        input_transcription=True,
        supported_input_formats=[AudioFormat.PCM16_24KHZ],
    )

    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._manager: Any = None
        self._connection: Any = None

    async def connect(self, config: LiveSessionConfig) -> None:
        if config.audio_format != AudioFormat.PCM16_24KHZ:
            raise ValueError("La transcripcion en vivo requiere PCM16 a 24 kHz")
        self._manager = self._client.realtime.connect(model=config.model)
        self._connection = await self._manager.__aenter__()
        await self._connection.session.update(session=_transcription_session(config))

    async def send_audio(self, chunk: bytes) -> None:
        if self._connection is None:
            return
        await self._connection.input_audio_buffer.append(
            audio=base64.b64encode(chunk).decode("ascii")
        )

    async def commit_audio(self) -> None:
        if self._connection is None:
            return
        await self._connection.input_audio_buffer.commit()

    async def send_text(self, text: str) -> None:
        """Nada que enviar: esta sesion no conversa."""
        return None

    async def submit_tool_result(self, call_id: str, result: dict) -> None:
        """Esta sesion no tiene herramientas."""
        return None

    async def cancel_response(self) -> None:
        """No hay respuesta que cancelar: aqui no se genera nada."""
        return None

    async def events(self):
        if self._connection is None:
            return
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


def _transcription_session(config: LiveSessionConfig) -> dict:
    """Build the session payload.

    `turn_detection` is null on purpose: the room already decides who holds the
    floor and when the turn ends, so letting the provider guess as well would
    cut turns in the middle of a sentence.
    """
    options = dict(config.provider_options)
    transcription: dict[str, Any] = {"model": config.model}

    keywords = options.get("keywords") or []
    if keywords:
        transcription["keywords"] = list(keywords)[:MAX_KEYWORDS]
    prompt = (config.prompt or "").strip()
    if prompt:
        transcription["prompt"] = prompt
    if config.locale:
        transcription["languages"] = [config.locale.split("-", 1)[0].lower()]
    delay = options.get("delay")
    if delay:
        transcription["delay"] = delay

    return {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "turn_detection": None,
                "transcription": transcription,
            }
        },
    }
