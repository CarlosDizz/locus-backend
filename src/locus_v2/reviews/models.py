from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from locus_v2.infrastructure.database.base import Base, TimestampMixin


class CommentStatus:
    """Estado del comentario público sobre el sitio.

    Solo «published» se enseña a otros usuarios. «pending» es lo que queda si la
    moderación no ha podido decidir (por ejemplo, la API no responde): mejor no
    publicar que publicar sin mirar.
    """

    PUBLISHED = "published"
    PENDING = "pending"
    REJECTED = "rejected"
    # Alguien lo ha denunciado: oculto hasta que lo revisemos (norma 1.2 de Apple).
    REPORTED = "reported"


class CallReview(TimestampMixin, Base):
    """Lo que opina una persona de una llamada al colgar.

    Dos partes con destinatarios distintos: el voto y el comentario sobre el
    sitio son públicos (los leen otros viajeros y deciden el tamaño del
    marcador); la nota, las etiquetas y el comentario sobre el guía solo los
    leemos nosotros.
    """

    __tablename__ = "call_reviews"
    __table_args__ = (
        UniqueConstraint("call_id", "user_id"),
        Index("ix_call_reviews_poi_created", "poi_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(String(40), nullable=False)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    poi_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("pois.id", ondelete="CASCADE"), nullable=False
    )
    place_vote: Mapped[bool | None] = mapped_column(Boolean)
    place_comment: Mapped[str | None] = mapped_column(Text)
    place_comment_status: Mapped[str | None] = mapped_column(String(16))
    guide_stars: Mapped[int | None] = mapped_column(SmallInteger)
    guide_tags: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    guide_comment: Mapped[str | None] = mapped_column(Text)
    moderated_at: Mapped[datetime | None] = mapped_column(DateTime())


class ContentReport(TimestampMixin, Base):
    """Una denuncia de algo que publicó otro usuario.

    kind «review» apunta a un CallReview (su comentario público); kind
    «call_image» a una foto compartida en una llamada («CALL-…/<image_id>»).
    Apple exige poder denunciar el contenido de otros usuarios y actuar en
    menos de 24 horas: se revisan en el panel, en Valoraciones.
    """

    __tablename__ = "content_reports"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    reporter_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    target: Mapped[str] = mapped_column(String(120), nullable=False)
    reason: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime())


class UserBlock(TimestampMixin, Base):
    """Quien bloquea deja de ver los comentarios públicos de la persona bloqueada."""

    __tablename__ = "user_blocks"
    __table_args__ = (UniqueConstraint("user_id", "blocked_user_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    blocked_user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
