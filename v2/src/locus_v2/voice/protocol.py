from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


class AudioFormat(StrEnum):
    PCM16_16KHZ = "pcm16_16khz"
    PCM16_24KHZ = "pcm16_24khz"


class ClientEventBase(BaseModel):
    event_id: str = Field(min_length=1, max_length=100)


class SessionStart(ClientEventBase):
    type: Literal["session.start"]
    locale: str = Field(min_length=2, max_length=16)
    routing_profile: str = Field(min_length=1, max_length=100)
    context_type: str = Field(default="poi", max_length=40)
    context_id: str | None = Field(default=None, max_length=100)


class AudioStart(ClientEventBase):
    type: Literal["audio.start"]
    format: AudioFormat


class AudioCommit(ClientEventBase):
    type: Literal["audio.commit"]


class ResponseCancel(ClientEventBase):
    type: Literal["response.cancel"]


class TextSend(ClientEventBase):
    type: Literal["text.send"]
    text: str = Field(min_length=1, max_length=8000)


class SessionClose(ClientEventBase):
    type: Literal["session.close"]


ClientEvent = Annotated[
    Union[SessionStart, AudioStart, AudioCommit, ResponseCancel, TextSend, SessionClose],
    Field(discriminator="type"),
]

client_event_adapter = TypeAdapter(ClientEvent)


class ServerEvent(BaseModel):
    type: str
    sequence: int = Field(ge=1)
    trace_id: str
    payload: dict = Field(default_factory=dict)
