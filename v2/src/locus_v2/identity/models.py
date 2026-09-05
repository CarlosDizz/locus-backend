from datetime import datetime
from enum import StrEnum

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from locus_v2.infrastructure.database.base import Base, TimestampMixin
from locus_v2.shared.ids import new_public_id


class UserStatus(StrEnum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    DELETED = "deleted"


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    legacy_v1_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, index=True)
    public_id: Mapped[str] = mapped_column(
        String(36), default=new_public_id, unique=True, nullable=False
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(160), nullable=False)
    avatar_url: Mapped[str | None] = mapped_column(String(1000))
    auth_provider: Mapped[str] = mapped_column(String(40), default="google", nullable=False)
    provider_subject: Mapped[str | None] = mapped_column(String(255), unique=True, index=True)
    locale: Mapped[str] = mapped_column(String(16), default="es-ES", nullable=False)
    status: Mapped[str] = mapped_column(String(20), default=UserStatus.ACTIVE, nullable=False)

    roles: Mapped[list["Role"]] = relationship(
        secondary="user_roles", lazy="selectin", back_populates="users"
    )

    def has_role(self, code: str) -> bool:
        return any(role.code == code for role in self.roles)


class Role(TimestampMixin, Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    is_system: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    users: Mapped[list[User]] = relationship(
        secondary="user_roles", lazy="selectin", back_populates="roles"
    )


class UserRole(Base):
    __tablename__ = "user_roles"
    __table_args__ = (UniqueConstraint("user_id", "role_id"),)

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
    )


class AdminAuditEvent(TimestampMixin, Base):
    __tablename__ = "admin_audit_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor_user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    action: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    resource_type: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(100), nullable=False)
    before_json: Mapped[str | None] = mapped_column(Text)
    after_json: Mapped[str | None] = mapped_column(Text)
    trace_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)


class AdminSession(TimestampMixin, Base):
    __tablename__ = "admin_sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(
        String(36), default=new_public_id, unique=True, nullable=False
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(), index=True, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime())
    user_agent: Mapped[str | None] = mapped_column(String(500))
