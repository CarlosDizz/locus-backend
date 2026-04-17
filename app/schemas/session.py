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


class SessionState(BaseModel):
    session_id: str
    user_id: int | None = None
    profile: SessionProfile = Field(default_factory=SessionProfile)
    location: SessionLocation = Field(default_factory=SessionLocation)
    active_poi: POI | None = None
    nearby_pois: list[POI] = Field(default_factory=list)
    memory: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
