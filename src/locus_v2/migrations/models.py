from sqlalchemy import JSON, BigInteger, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from locus_v2.infrastructure.database.base import Base, TimestampMixin


class LegacyAppSession(TimestampMixin, Base):
    __tablename__ = "legacy_app_sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    legacy_session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    legacy_user_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    profile_context: Mapped[str] = mapped_column(Text, default="", nullable=False)
    profile_language: Mapped[str] = mapped_column(String(16), default="es", nullable=False)
    snapshot_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


class DataImportRun(TimestampMixin, Base):
    __tablename__ = "data_import_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    imported_rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    skipped_rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    table_counts_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_log: Mapped[str | None] = mapped_column(Text)
