from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from locus_v2.infrastructure.database.base import Base, TimestampMixin


class PushDevice(TimestampMixin, Base):
    """Un móvil que acepta avisos, de una app concreta.

    El token de Apple es de la app que lo pidió: un mismo iPhone con Borea y Locus
    Guide tiene dos, y cada uno se envía con el bundle de su app (apns-topic).
    """

    __tablename__ = "push_devices"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # guide | borea (shared/apps.py)
    app: Mapped[str] = mapped_column(String(20), nullable=False)
    # ios | android
    platform: Mapped[str] = mapped_column(String(10), nullable=False)
    token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(), nullable=False)
    # Apple contestó que el móvil ya no acepta avisos (410): no se le vuelve a enviar.
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime())
