from datetime import datetime
from enum import StrEnum

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from locus_v2.infrastructure.database.base import Base, TimestampMixin
from locus_v2.shared.ids import new_public_id


class VoiceSessionStatus(StrEnum):
    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class VoiceTurnRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class VoiceSession(TimestampMixin, Base):
    __tablename__ = "voice_sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(
        String(36), default=new_public_id, unique=True, index=True, nullable=False
    )
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    routing_profile_id: Mapped[int] = mapped_column(
        ForeignKey("ai_routing_profiles.id"), nullable=False
    )
    prompt_version_id: Mapped[int] = mapped_column(
        ForeignKey("prompt_versions.id"), nullable=False
    )
    primary_model_id: Mapped[int] = mapped_column(ForeignKey("ai_models.id"), nullable=False)
    active_model_id: Mapped[int] = mapped_column(ForeignKey("ai_models.id"), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default=VoiceSessionStatus.CREATED, nullable=False
    )
    locale: Mapped[str] = mapped_column(String(16), nullable=False)
    voice: Mapped[str | None] = mapped_column(String(80))
    context_type: Mapped[str] = mapped_column(String(40), default="poi", nullable=False)
    context_public_id: Mapped[str | None] = mapped_column(String(36), index=True)
    config_snapshot_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime())


class VoiceTurn(TimestampMixin, Base):
    __tablename__ = "voice_turns"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(
        String(36), default=new_public_id, unique=True, nullable=False
    )
    voice_session_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("voice_sessions.id", ondelete="CASCADE"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    provider_item_id: Mapped[str | None] = mapped_column(String(200))
    tool_name: Mapped[str | None] = mapped_column(String(120))
    tool_payload_json: Mapped[dict | None] = mapped_column(JSON)
    trace_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
