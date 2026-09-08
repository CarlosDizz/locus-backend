from collections.abc import AsyncIterator

from locus_v2.voice.providers.base import (
    LiveProvider,
    LiveSessionConfig,
    ProviderCapabilities,
    ProviderEvent,
    ProviderEventType,
)


class FutureOpenAILiveProvider(LiveProvider):
    code = "openai_live"
    capabilities = ProviderCapabilities()

    async def connect(self, config: LiveSessionConfig) -> None:
        raise RuntimeError("OpenAI GPT Live is catalogued but not enabled")

    async def send_audio(self, chunk: bytes) -> None:
        raise RuntimeError("OpenAI GPT Live is not enabled")

    async def commit_audio(self) -> None:
        raise RuntimeError("OpenAI GPT Live is not enabled")

    async def send_text(self, text: str) -> None:
        raise RuntimeError("OpenAI GPT Live is not enabled")

    async def submit_tool_result(self, call_id: str, result: dict) -> None:
        raise RuntimeError("OpenAI GPT Live is not enabled")

    async def cancel_response(self) -> None:
        return None

    async def events(self) -> AsyncIterator[ProviderEvent]:
        if False:
            yield ProviderEvent(type=ProviderEventType.READY)  # pragma: no cover

    async def close(self) -> None:
        return None
