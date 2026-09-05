from decimal import Decimal

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    ForeignKey,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from locus_v2.infrastructure.database.base import Base, TimestampMixin
from locus_v2.shared.ids import new_public_id


class City(TimestampMixin, Base):
    __tablename__ = "cities"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    legacy_v1_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, index=True)
    public_id: Mapped[str] = mapped_column(String(36), default=new_public_id, unique=True)
    slug: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    names_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    country_code: Mapped[str] = mapped_column(String(8), default="", nullable=False)
    lat: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    lng: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    source: Mapped[str] = mapped_column(String(64), default="manual", nullable=False)

    pois: Mapped[list["Poi"]] = relationship(back_populates="city", lazy="selectin")


class PoiType(TimestampMixin, Base):
    __tablename__ = "poi_types"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    legacy_v1_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, index=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    description: Mapped[str] = mapped_column(String(500), default="", nullable=False)


class Poi(TimestampMixin, Base):
    __tablename__ = "pois"
    __table_args__ = (UniqueConstraint("city_id", "slug"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    legacy_v1_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, index=True)
    public_id: Mapped[str] = mapped_column(String(36), default=new_public_id, unique=True)
    city_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("cities.id"), index=True)
    poi_type_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("poi_types.id"), index=True
    )
    slug: Mapped[str] = mapped_column(String(160), index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    names_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    lat: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    lng: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    short_description: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    short_descriptions_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    long_description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source_of_truth: Mapped[str] = mapped_column(String(64), default="manual", nullable=False)
    wikidata_id: Mapped[str] = mapped_column(String(64), default="", index=True, nullable=False)
    wikipedia_title: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    google_place_id: Mapped[str] = mapped_column(
        String(128), default="", index=True, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    city: Mapped[City | None] = relationship(back_populates="pois")
