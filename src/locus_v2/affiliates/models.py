from sqlalchemy import BigInteger, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from locus_v2.infrastructure.database.base import Base, TimestampMixin


class ReferralClick(TimestampMixin, Base):
    """Un clic en un enlace de referido (/r/<token>), para saber qué programa rinde."""

    __tablename__ = "referral_clicks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    app: Mapped[str] = mapped_column(String(20), default="guide", nullable=False)
    program: Mapped[str] = mapped_column(String(40), index=True, nullable=False)
    trip_public_id: Mapped[str] = mapped_column(String(36), default="", nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
