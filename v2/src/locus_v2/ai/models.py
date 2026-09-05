from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from locus_v2.ai.enums import Lifecycle, PublicationStatus, ServiceKind, VoiceMode
from locus_v2.infrastructure.database.base import Base, TimestampMixin


class AIProvider(TimestampMixin, Base):
    __tablename__ = "ai_providers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    models: Mapped[list["AIModel"]] = relationship(back_populates="provider", lazy="selectin")


class AIModel(TimestampMixin, Base):
    __tablename__ = "ai_models"
    __table_args__ = (UniqueConstraint("provider_id", "external_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    provider_id: Mapped[int] = mapped_column(ForeignKey("ai_providers.id"), nullable=False)
    external_id: Mapped[str] = mapped_column(String(160), nullable=False)
    display_name: Mapped[str] = mapped_column(String(160), nullable=False)
    service_kind: Mapped[str] = mapped_column(String(20), nullable=False)
    adapter_code: Mapped[str] = mapped_column(String(80), nullable=False)
    lifecycle: Mapped[str] = mapped_column(String(20), default=Lifecycle.STABLE, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    selectable: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    capabilities_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    runtime_defaults_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    provider: Mapped[AIProvider] = relationship(back_populates="models")

    @property
    def is_voice(self) -> bool:
        return self.service_kind == ServiceKind.VOICE


class AITool(TimestampMixin, Base):
    __tablename__ = "ai_tools"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str] = mapped_column(String(1000), default="", nullable=False)
    handler_code: Mapped[str] = mapped_column(String(100), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    service_kinds_json: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    schema_json: Mapped[dict] = mapped_column(JSON, nullable=False)


class PromptDefinition(TimestampMixin, Base):
    __tablename__ = "prompt_definitions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str] = mapped_column(String(1000), default="", nullable=False)
    service_kind: Mapped[str] = mapped_column(
        String(20), default=ServiceKind.VOICE, nullable=False
    )

    versions: Mapped[list["PromptVersion"]] = relationship(
        back_populates="definition", lazy="selectin"
    )


class PromptVersion(TimestampMixin, Base):
    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("definition_id", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    definition_id: Mapped[int] = mapped_column(
        ForeignKey("prompt_definitions.id"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default=PublicationStatus.DRAFT, nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    variables_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    tools_json: Mapped[list[dict]] = mapped_column(JSON, default=list, nullable=False)
    runtime_config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_by_user_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"))
    published_at: Mapped[datetime | None] = mapped_column(DateTime())

    definition: Mapped[PromptDefinition] = relationship(back_populates="versions")


class RoutingProfile(TimestampMixin, Base):
    __tablename__ = "ai_routing_profiles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    experience_code: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    service_kind: Mapped[str] = mapped_column(String(20), nullable=False)
    environment: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default=PublicationStatus.DRAFT, nullable=False
    )
    voice_mode: Mapped[str] = mapped_column(
        String(30), default=VoiceMode.PUSH_TO_TALK, nullable=False
    )
    primary_model_id: Mapped[int] = mapped_column(ForeignKey("ai_models.id"), nullable=False)
    fallback_model_id: Mapped[int | None] = mapped_column(ForeignKey("ai_models.id"))
    prompt_version_id: Mapped[int] = mapped_column(
        ForeignKey("prompt_versions.id"), nullable=False
    )
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime())

    primary_model: Mapped[AIModel] = relationship(foreign_keys=[primary_model_id], lazy="joined")
    fallback_model: Mapped[AIModel | None] = relationship(
        foreign_keys=[fallback_model_id], lazy="joined"
    )
    prompt_version: Mapped[PromptVersion] = relationship(lazy="joined")


class ProviderPriceSnapshot(TimestampMixin, Base):
    __tablename__ = "provider_price_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    legacy_v1_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, index=True)
    provider_id: Mapped[int] = mapped_column(ForeignKey("ai_providers.id"), nullable=False)
    model_id: Mapped[int] = mapped_column(ForeignKey("ai_models.id"), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    pricing_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    source_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    effective_from: Mapped[datetime] = mapped_column(DateTime(), nullable=False)
    effective_to: Mapped[datetime | None] = mapped_column(DateTime())
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
