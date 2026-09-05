from enum import StrEnum

from datetime import datetime
from decimal import Decimal

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from locus_v2.infrastructure.database.base import Base, TimestampMixin
from locus_v2.shared.ids import new_public_id


class UsageStatus(StrEnum):
    PENDING = "pending"
    PRICED = "priced"
    CHARGED = "charged"
    FAILED = "failed"


class LedgerEntryKind(StrEnum):
    CREDIT = "credit"
    CHARGE = "charge"
    REFUND = "refund"
    ADJUSTMENT = "adjustment"


class Wallet(TimestampMixin, Base):
    __tablename__ = "wallets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    legacy_v1_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, index=True)
    public_id: Mapped[str] = mapped_column(String(36), default=new_public_id, unique=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )
    currency: Mapped[str] = mapped_column(String(3), default="EUR", nullable=False)
    balance_cents: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)

    ledger_entries: Mapped[list["LedgerEntry"]] = relationship(back_populates="wallet")


class TopUp(TimestampMixin, Base):
    __tablename__ = "top_ups"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    legacy_v1_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, index=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), index=True)
    wallet_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("wallets.id"), index=True)
    amount_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bonus_cents: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    provider: Mapped[str] = mapped_column(String(64), default="manual", nullable=False)
    provider_reference: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)


class UsageEvent(TimestampMixin, Base):
    __tablename__ = "usage_events"
    __table_args__ = (UniqueConstraint("provider_id", "dedupe_key"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    legacy_v1_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, index=True)
    public_id: Mapped[str] = mapped_column(
        String(36), default=new_public_id, unique=True, nullable=False
    )
    user_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"))
    voice_session_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("voice_sessions.id")
    )
    provider_id: Mapped[int] = mapped_column(ForeignKey("ai_providers.id"), nullable=False)
    model_id: Mapped[int] = mapped_column(ForeignKey("ai_models.id"), nullable=False)
    price_snapshot_id: Mapped[int | None] = mapped_column(
        ForeignKey("provider_price_snapshots.id")
    )
    dedupe_key: Mapped[str] = mapped_column(String(160), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(200), index=True)
    interaction_type: Mapped[str] = mapped_column(String(60), nullable=False)

    text_input_tokens: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    cached_text_input_tokens: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    text_output_tokens: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    audio_input_tokens: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    audio_output_tokens: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    audio_input_milliseconds: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    audio_output_milliseconds: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    provider_cost_microusd: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    provider_cost_eur_cents: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    charged_amount_cents: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    gross_margin_cents: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="EUR", nullable=False)
    margin_multiplier: Mapped[Decimal] = mapped_column(
        Numeric(8, 4), default=Decimal("1"), nullable=False
    )
    raw_usage_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default=UsageStatus.PENDING, nullable=False)
    trace_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)


class LedgerEntry(TimestampMixin, Base):
    __tablename__ = "ledger_entries"
    __table_args__ = (UniqueConstraint("usage_event_id", "kind"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    legacy_v1_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, index=True)
    public_id: Mapped[str] = mapped_column(
        String(36), default=new_public_id, unique=True, nullable=False
    )
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    wallet_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("wallets.id", ondelete="CASCADE"), index=True
    )
    usage_event_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("usage_events.id")
    )
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    amount_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="EUR", nullable=False)
    provider_cost_eur_cents: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    gross_margin_cents: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    balance_after_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    exchange_rate: Mapped[Decimal] = mapped_column(
        Numeric(16, 8), default=Decimal("1"), nullable=False
    )
    margin_multiplier: Mapped[Decimal] = mapped_column(
        Numeric(8, 4), default=Decimal("1"), nullable=False
    )
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    reference_type: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    reference_id: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    trace_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)

    wallet: Mapped[Wallet] = relationship(back_populates="ledger_entries")
