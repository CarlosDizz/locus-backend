from typing import Any

from sqlalchemy import JSON, BigInteger, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from locus_v2.infrastructure.database.base import Base, TimestampMixin


class LocusLog(TimestampMixin, Base):
    __tablename__ = "locus_logs"
    __table_args__ = (
        Index("ix_locus_logs_service_created", "service", "created_at"),
        Index("ix_locus_logs_level_created", "level", "created_at"),
        Index("ix_locus_logs_trace_created", "trace_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    level: Mapped[str] = mapped_column(String(10), nullable=False)
    service: Mapped[str] = mapped_column(String(40), nullable=False)
    environment: Mapped[str] = mapped_column(String(20), nullable=False)
    event: Mapped[str] = mapped_column(String(120), nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    trace_id: Mapped[str | None] = mapped_column(String(64))
    user_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"))
    voice_session_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("voice_sessions.id")
    )
    error_type: Mapped[str | None] = mapped_column(String(160))
    error_code: Mapped[str | None] = mapped_column(String(100))
    elapsed_ms: Mapped[int | None] = mapped_column(Integer)
    context_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
