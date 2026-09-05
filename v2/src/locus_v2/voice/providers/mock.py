import asyncio
from collections.abc import AsyncIterator

from locus_v2.voice.protocol import AudioFormat
from locus_v2.voice.providers.base import (
    LiveProvider,
    LiveSessionConfig,
    ProviderCapabilities,
    ProviderEvent,
    ProviderEventType,
)


class MockLiveProvider(LiveProvider):
    code = "mock_live"
    capabilities = ProviderCapabilities(
        full_duplex=True,
        function_calling=True,
        input_transcription=True,
        output_transcription=True,
        session_resumption=True,
        supported_input_formats=[AudioFormat.PCM16_16KHZ, AudioFormat.PCM16_24KHZ],
    )

    def __init__(self) -> None:
        self._events: asyncio.Queue[ProviderEvent | None] = asyncio.Queue()
        self._connected = False

    async def connect(self, config: LiveSessionConfig) -> None:
        self._connected = True
        await self._events.put(ProviderEvent(type=ProviderEventType.READY))

    async def send_audio(self, chunk: bytes) -> None:
        if not self._connected:
            raise RuntimeError("Provider is not connected")

    async def commit_audio(self) -> None:
        await self._respond("Audio received by the Locus test provider.")

    async def send_text(self, text: str) -> None:
        await self._respond(f"Locus test response: {text}")

    async def _respond(self, text: str) -> None:
        await self._events.put(ProviderEvent(type=ProviderEventType.TEXT_DELTA, text=text))
        await self._events.put(ProviderEvent(type=ProviderEventType.TEXT_DONE, text=text))
        await self._events.put(ProviderEvent(type=ProviderEventType.AUDIO_DONE))

    async def submit_tool_result(self, call_id: str, result: dict) -> None:
        await self._respond(str(result))

    async def cancel_response(self) -> None:
        return None

    async def events(self) -> AsyncIterator[ProviderEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def close(self) -> None:
        self._connected = False
        await self._events.put(None)
