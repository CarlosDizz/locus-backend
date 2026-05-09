from typing import Any

from pydantic import BaseModel, Field

from app.schemas.poi import POI


class SessionLocation(BaseModel):
    lat: float | None = None
    lng: float | None = None


class SessionProfile(BaseModel):
    raw_context: str = Field(default="")
    language: str = Field(default="es")
    preferences: dict[str, Any] = Field(default_factory=dict)


class SessionParticipant(BaseModel):
    user_id: int
    display_name: str = Field(default="")
    avatar_url: str = Field(default="")
    joined_at: str = Field(default="")
    last_seen_at: str = Field(default="")
    status: str = Field(default="present")
    active_call: bool = Field(default=False)


class SessionCallLive(BaseModel):
    status: str = Field(default="idle")
    host_user_id: int | None = None
    host_display_name: str = Field(default="")
    started_at: str = Field(default="")
    updated_at: str = Field(default="")


class SessionCallLogEntry(BaseModel):
    id: str
    kind: str
    author: str
    text: str
    timestamp: str
    image_url: str | None = None
    user_id: int | None = None


class SessionState(BaseModel):
    session_id: str
    user_id: int | None = None
    profile: SessionProfile = Field(default_factory=SessionProfile)
    location: SessionLocation = Field(default_factory=SessionLocation)
    active_poi: POI | None = None
    nearby_pois: list[POI] = Field(default_factory=list)
    ephemeral_map_pois: list[POI] = Field(default_factory=list)
    memory: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    participants: list[SessionParticipant] = Field(default_factory=list)
    call_live: SessionCallLive = Field(default_factory=SessionCallLive)
    call_log: list[SessionCallLogEntry] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    session_id: str | None = None
    user_id: int | None = None
    profile_context: str = Field(default="")
    lat: float | None = None
    lng: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionUpdateRequest(BaseModel):
    user_id: int | None = None
    profile_context: str | None = None
    lat: float | None = None
    lng: float | None = None
    active_poi_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session: SessionState


class SessionParticipantTouchRequest(BaseModel):
    active_call: bool = False


class SessionCallStateRequest(BaseModel):
    status: str = Field(default="idle")


class SessionCallLogRequest(BaseModel):
    kind: str
    author: str
    text: str
    image_url: str | None = None
