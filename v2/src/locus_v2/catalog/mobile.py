"""Public catalog reads and V1 serialization using strict mobile identifiers."""

from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import raiseload

from locus_v2.affiliates.service import ReferralService
from locus_v2.catalog.bootstrap.normalize import normalize_names, normalize_short_descriptions
from locus_v2.catalog.bootstrap.service import CatalogBootstrapError
from locus_v2.catalog.mobile_bootstrap import MobileCatalogBootstrap
from locus_v2.catalog.mobile_schemas import (
    CityBootstrapFromLocationRequest,
    CityBootstrapFromLocationResponse,
    CityResponse,
    PoiAccessLinksResponse,
    PoiDocumentationResponse,
    PoiResponse,
    PoiTypeResponse,
)
from locus_v2.catalog.models import City, Poi, PoiType
from locus_v2.config import Settings
from locus_v2.shared.mobile_ids import mobile_id, mobile_id_clause


class CatalogNotFoundError(LookupError):
    pass


def display_text(values: dict[str, str], language: str, *, name: bool = False) -> str:
    keys = [language, language.split("-", 1)[0], "es", "en"]
    keys += ["int", "local"] if name else ["local"]
    return next((values[key] for key in keys if values.get(key)), "")


def city_response(city: City, language: str = "es") -> CityResponse:
    names = normalize_names(city.name, city.names_json)
    return CityResponse(
        id=mobile_id(city), slug=city.slug, name=city.name,
        display_name=display_text(names, language, name=True), names=names,
        country_code=city.country_code,
        lat=float(city.lat) if city.lat is not None else None,
        lng=float(city.lng) if city.lng is not None else None,
        source=city.source, created_at=city.created_at,
    )


def poi_response(
    poi: Poi, city: City | None, poi_type: PoiType | None, language: str = "es"
) -> PoiResponse:
    names = normalize_names(poi.name, poi.names_json)
    descriptions = normalize_short_descriptions(
        poi.short_description, poi.short_descriptions_json
    )
    return PoiResponse(
        id=mobile_id(poi),
        city_id=mobile_id(city) if city is not None else None,
        poi_type_id=mobile_id(poi_type) if poi_type is not None else None,
        poi_type_code=poi_type.code if poi_type else None,
        poi_type_name=poi_type.name if poi_type else None,
        slug=poi.slug, name=poi.name,
        display_name=display_text(names, language, name=True), names=names,
        lat=float(poi.lat) if poi.lat is not None else None,
        lng=float(poi.lng) if poi.lng is not None else None,
        short_description=display_text(descriptions, language),
        short_descriptions=descriptions, long_description=poi.long_description,
        source_of_truth=poi.source_of_truth, wikidata_id=poi.wikidata_id,
        wikipedia_title=poi.wikipedia_title, google_place_id=poi.google_place_id,
        is_active=poi.is_active, metadata=poi.metadata_json or {},
        created_at=poi.created_at, updated_at=poi.updated_at,
    )


class MobileCatalogService:
    def __init__(self, session: AsyncSession, settings: Settings, language: str = "es") -> None:
        self.session = session
        self.settings = settings
        self.language = language

    async def list_poi_types(self) -> list[PoiTypeResponse]:
        rows = await self.session.scalars(select(PoiType).order_by(PoiType.name, PoiType.id))
        return [
            PoiTypeResponse(
                id=mobile_id(row), code=row.code, name=row.name, description=row.description
            )
            for row in rows
        ]

    async def list_cities(self, *, q: str = "", limit: int = 100) -> list[CityResponse]:
        stmt = select(City).options(raiseload("*")).order_by(City.name, City.id).limit(limit)
        if q.strip():
            token = f"%{q.strip()}%"
            stmt = stmt.where(or_(City.name.ilike(token), City.slug.ilike(token)))
        return [city_response(row, self.language) for row in await self.session.scalars(stmt)]

    @staticmethod
    def _poi_query() -> Any:
        return (
            select(Poi, City, PoiType)
            .outerjoin(City, Poi.city_id == City.id)
            .outerjoin(PoiType, Poi.poi_type_id == PoiType.id)
            .options(raiseload("*"))
            .where(Poi.is_active.is_(True))
        )

    async def list_pois(
        self, *, city_id: int | None = None, poi_type_code: str | None = None,
        q: str = "", limit: int = 200,
    ) -> list[PoiResponse]:
        stmt = self._poi_query().order_by(Poi.name, Poi.id).limit(limit)
        if city_id is not None:
            stmt = stmt.where(mobile_id_clause(City, city_id))
        if poi_type_code:
            stmt = stmt.where(PoiType.code == poi_type_code)
        if q.strip():
            token = f"%{q.strip()}%"
            stmt = stmt.where(or_(
                Poi.name.ilike(token), Poi.slug.ilike(token), Poi.short_description.ilike(token)
            ))
        rows = (await self.session.execute(stmt)).all()
        return [poi_response(poi, city, poi_type, self.language) for poi, city, poi_type in rows]

    async def get_poi(self, poi_id: int) -> PoiResponse:
        row = (
            await self.session.execute(self._poi_query().where(mobile_id_clause(Poi, poi_id)))
        ).one_or_none()
        if row is None:
            raise CatalogNotFoundError("POI no encontrado")
        return poi_response(*row, language=self.language)

    async def documentation(self, poi_id: int) -> PoiDocumentationResponse:
        poi = await self.get_poi(poi_id)
        # Resolve by ID, never by a potentially duplicated name in another city.
        # This is V1's catalog-only documentation shape, without live enrichment.
        return PoiDocumentationResponse(
            poi=poi, resolved_from_catalog=True,
            documentation={
                "poi_name": poi.name,
                "summary": f"{poi.short_description} {poi.long_description}".strip(),
                "wikidata": None, "facts": {}, "sources": ["catalog"],
                "catalog_poi": {
                    "id": poi.id, "name": poi.name, "wikidata_id": poi.wikidata_id,
                    "wikipedia_title": poi.wikipedia_title,
                },
            },
        )

    async def access_links(self, poi_id: int) -> PoiAccessLinksResponse:
        poi = await self.get_poi(poi_id)
        result = ReferralService(self.settings).poi_access_links(
            poi_id=str(poi.id), poi_name=poi.name,
            poi_type_code=poi.poi_type_code or "", poi_type_name=poi.poi_type_name or "",
            short_description=poi.short_description, long_description=poi.long_description,
            metadata=poi.metadata,
        )
        result["poi_id"] = poi.id
        return PoiAccessLinksResponse.model_validate(result)

    async def bootstrap_from_location(
        self, payload: CityBootstrapFromLocationRequest
    ) -> CityBootstrapFromLocationResponse:
        bootstrap = MobileCatalogBootstrap(self.session, self.settings)
        try:
            try:
                result = await bootstrap.bootstrap_from_location(**payload.model_dump())
            except IntegrityError:
                # Two first requests can both see a missing city. Once the
                # unique-slug winner commits, retry and reuse its seed.
                await self.session.rollback()
                result = await bootstrap.bootstrap_from_location(**payload.model_dump())
        except Exception:
            await self.session.rollback()
            raise
        city = await self.session.scalar(
            select(City).options(raiseload("*")).where(City.id == result.city_id)
        )
        if city is None:
            raise CatalogBootstrapError("No se pudo recuperar la ciudad importada")
        return CityBootstrapFromLocationResponse(
            city=city_response(city, self.language),
            imported_count=result.imported_count, updated_count=result.updated_count,
            skipped_count=0,
            stats={
                "source": result.source,
                "existing_poi_count": bootstrap.existing_poi_count,
                "openai_skipped": True,
            },
            pois=await self.list_pois(city_id=mobile_id(city), limit=payload.limit),
        )
