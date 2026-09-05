from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import StrEnum

from pydantic import BaseModel, Field

from locus_v2.billing.pricing import NormalizedUsage
from locus_v2.voice.protocol import AudioFormat


class ProviderEventType(StrEnum):
    READY = "ready"
    INPUT_TRANSCRIPT_DELTA = "input_transcript_delta"
    INPUT_TRANSCRIPT_DONE = "input_transcript_done"
    TEXT_DELTA = "text_delta"
    TEXT_DONE = "text_done"
    AUDIO_DELTA = "audio_delta"
    AUDIO_DONE = "audio_done"
    TOOL_CALL = "tool_call"
    USAGE = "usage"
    ERROR = "error"


class ProviderCapabilities(BaseModel):
    full_duplex: bool = False
    function_calling: bool = False
    async_function_calling: bool = False
    input_transcription: bool = False
    output_transcription: bool = False
    image_input: bool = False
    session_resumption: bool = False
    context_compression: bool = False
    supported_input_formats: list[AudioFormat] = Field(default_factory=list)


class ProviderEvent(BaseModel):
    type: ProviderEventType
    text: str | None = None
    audio: bytes | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments: dict | None = None
    usage: NormalizedUsage | None = None
    error_code: str | None = None
    retryable: bool = False


class LiveSessionConfig(BaseModel):
    model: str
    prompt: str
    locale: str
    voice: str | None = None
    audio_format: AudioFormat = AudioFormat.PCM16_24KHZ
    tools: list[dict] = Field(default_factory=list)
    provider_options: dict = Field(default_factory=dict)


class LiveProvider(ABC):
    code: str
    capabilities: ProviderCapabilities

    @abstractmethod
    async def connect(self, config: LiveSessionConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    async def commit_audio(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def send_text(self, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def submit_tool_result(self, call_id: str, result: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    async def cancel_response(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def events(self) -> AsyncIterator[ProviderEvent]:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
