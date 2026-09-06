"""The Ionic/V1 map-chat wire contract, independent of legacy app imports."""

from pydantic import BaseModel, Field

from locus_v2.sessions.models import SessionPoi


class ChatSetupRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    user_id: int | None = None
    profile_context: str = ""
    lat: float | None = Field(default=None, ge=-90, le=90)
    lng: float | None = Field(default=None, ge=-180, le=180)


class ChatMessageRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    user_id: int | None = None
    message: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    lat: float | None = Field(default=None, ge=-90, le=90)
    lng: float | None = Field(default=None, ge=-180, le=180)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    pois: list[SessionPoi] = Field(default_factory=list)
    ephemeral_pois: list[SessionPoi] = Field(default_factory=list)
    prompt_preview: str = ""
