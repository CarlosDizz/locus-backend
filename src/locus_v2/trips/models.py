from datetime import date, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from locus_v2.identity.models import User
from locus_v2.infrastructure.database.base import Base, TimestampMixin
from locus_v2.shared.ids import new_public_id


class Trip(TimestampMixin, Base):
    """Un viaje que el usuario prepara con el planificador.

    El plan son datos (TripItem), no texto: el chat los escribe con
    herramientas y la app los pinta. La conversación se guarda aquí para poder
    retomarla con «Editar».
    """

    __tablename__ = "trips"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(String(36), default=new_public_id, unique=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    city: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    country: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    # Hora local de la ciudad del viaje, sin zona: «llego el 5 por la tarde».
    starts_at: Mapped[datetime | None] = mapped_column(DateTime())
    ends_at: Mapped[datetime | None] = mapped_column(DateTime())
    travelers: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    pace: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    conversation_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    # La ciudad en el mapa, para centrarlo aunque ninguna parada tenga sitio.
    center_json: Mapped[dict | None] = mapped_column(JSON)
    # La ruta a pie de cada día, ya pedida a Google (trips/routes.py).
    routes_json: Mapped[dict | None] = mapped_column(JSON)

    # El enlace para unirse al viaje (/trips/join/<código>); se crea al invitar.
    invite_code: Mapped[str | None] = mapped_column(String(16), unique=True)

    owner: Mapped[User] = relationship(foreign_keys=[user_id])
    members: Mapped[list["TripMember"]] = relationship(
        back_populates="trip", cascade="all, delete-orphan", order_by="TripMember.id"
    )
    items: Mapped[list["TripItem"]] = relationship(
        back_populates="trip", cascade="all, delete-orphan", order_by="TripItem.position"
    )


class TripItem(TimestampMixin, Base):
    """Una franja del plan: algo reservado por el usuario o propuesto por Locus."""

    __tablename__ = "trip_items"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(String(36), default=new_public_id, unique=True)
    trip_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("trips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    day: Mapped[date] = mapped_column(Date(), nullable=False)
    time: Mapped[str] = mapped_column(String(5), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    # visit, meal, walk, transport, logistics
    kind: Mapped[str] = mapped_column(String(20), default="visit", nullable=False)
    # booked: lo tiene ya el usuario. proposed: lo propone Locus.
    status: Mapped[str] = mapped_column(String(20), default="proposed", nullable=False)
    note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    travel: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    # El sitio tal como se busca en Google Maps («Coliseo, Roma»), para enlazarlo.
    place: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    # Ese sitio resuelto en Google Places: nombre, place_id y coordenadas.
    map_json: Mapped[dict | None] = mapped_column(JSON)
    price: Mapped[str] = mapped_column(String(40), default="", nullable=False)
    poi_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("pois.id", ondelete="SET NULL")
    )
    booking_json: Mapped[dict | None] = mapped_column(JSON)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    trip: Mapped[Trip] = relationship(back_populates="items")


class TripMember(TimestampMixin, Base):
    """Alguien que va al viaje sin haberlo creado: lo ve entero y opina en el chat.

    El creador (Trip.user_id) no está aquí; es el único que cambia el plan,
    acepta tarjetas o borra el viaje.
    """

    __tablename__ = "trip_members"
    __table_args__ = (UniqueConstraint("trip_id", "user_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trip_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("trips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    trip: Mapped[Trip] = relationship(back_populates="members")
    user: Mapped[User] = relationship()
