"""V1-compatible map session, ported from app/db/models.py::AppSession.

This is the Ionic home screen's "map session": location, chat profile,
conversation memory, active/nearby POIs, and the lightweight group-presence
state (participants, call_live, call_log) used before a real call starts.
Client-generated session_id stays the primary key, same as V1 — the mobile
app builds it locally (`LOCUS-XXXXXXXX`) and uses it as the join key with no
server round-trip needed to mint one.
"""

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import JSON, BigInteger, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from locus_v2.infrastructure.database.base import Base, TimestampMixin


class SessionPoi(BaseModel):
    """Embedded POI snapshot stored inside a session's JSON columns — not a
    live reference to the catalog, matching V1's denormalized shape exactly.
    """

    id: str = ""
    name: str
    lat: float
    lng: float
    poi_type_code: str = ""
    description: str = ""
    summary: str = ""
    source_of_truth: str = "catalog"
    is_ephemeral: bool = False
    google_place_id: str = ""
    context_kind: str = "catalog"


class MapSession(TimestampMixin, Base):
    __tablename__ = "map_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    profile_context: Mapped[str] = mapped_column(Text, default="", nullable=False)
    profile_language: Mapped[str] = mapped_column(String(16), default="es", nullable=False)
    profile_preferences_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    lat: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    lng: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    active_poi_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    nearby_pois_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )
    memory_json: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class SessionProfileView(BaseModel):
    raw_context: str = ""
    language: str = "es"
    preferences: dict[str, Any] = Field(default_factory=dict)


class SessionLocationView(BaseModel):
    lat: float | None = None
    lng: float | None = None


class SessionParticipantState(BaseModel):
    user_id: int
    display_name: str = ""
    avatar_url: str = ""
    joined_at: str = ""
    last_seen_at: str = ""
    status: str = "present"
    active_call: bool = False


class SessionCallLiveState(BaseModel):
    status: str = "idle"
    host_user_id: int | None = None
    host_display_name: str = ""
    started_at: str = ""
    updated_at: str = ""


class SessionCallLogEntryState(BaseModel):
    id: str
    kind: str = "system"
    author: str = "Sistema"
    text: str
    timestamp: str
    image_url: str | None = None
    user_id: int | None = None


class SessionStateView(BaseModel):
    """Wire shape must match V1's SessionState exactly (app/schemas/session.py):
    `profile` and `location` are nested objects, not flattened fields — the
    real Ionic app (home.page.ts, profile.page.ts) reads
    `session.profile.raw_context`/`.preferences` and `session.location.lat`/
    `.lng` directly, and throws if `profile`/`location` are missing. Found by
    running the real app against this API with Playwright (2026-09-06): the
    Profile page crashed with "Cannot read properties of undefined (reading
    'preferences')" because an earlier version of this model flattened these.
    """

    session_id: str
    user_id: int | None = None
    profile: SessionProfileView = Field(default_factory=SessionProfileView)
    location: SessionLocationView = Field(default_factory=SessionLocationView)
    active_poi: SessionPoi | None = None
    nearby_pois: list[SessionPoi] = Field(default_factory=list)
    ephemeral_map_pois: list[SessionPoi] = Field(default_factory=list)
    memory: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    participants: list[SessionParticipantState] = Field(default_factory=list)
    call_live: SessionCallLiveState = Field(default_factory=SessionCallLiveState)
    call_log: list[SessionCallLogEntryState] = Field(default_factory=list)
