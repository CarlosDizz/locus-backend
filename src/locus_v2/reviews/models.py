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
