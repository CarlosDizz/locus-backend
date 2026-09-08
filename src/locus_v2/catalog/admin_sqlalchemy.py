from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.catalog.admin import (
    AdminCityList,
    AdminCitySummary,
    AdminPoiDetail,
    AdminPoiPage,
    AdminPoiSummary,
)
from locus_v2.catalog.models import City, Poi, PoiType


class SQLAlchemyAdminCatalogReader:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def cities(self, *, query: str, limit: int) -> AdminCityList:
        filters = []
        if query:
            pattern = f"%{query}%"
            filters.append(
                or_(
                    City.name.ilike(pattern),
                    City.slug.ilike(pattern),
                    City.country_code.ilike(pattern),
                )
            )
        total = int(
            await self._session.scalar(select(func.count(City.id)).where(*filters)) or 0
        )
        counts = (
            select(
                Poi.city_id.label("city_id"),
                func.count(Poi.id).label("poi_count"),
                func.sum(case((Poi.is_active.is_(True), 1), else_=0)).label(
                    "active_poi_count"
                ),
            )
            .group_by(Poi.city_id)
            .subquery()
        )
        rows = (
            await self._session.execute(
                select(
                    City,
                    func.coalesce(counts.c.poi_count, 0),
                    func.coalesce(counts.c.active_poi_count, 0),
                )
                .outerjoin(counts, counts.c.city_id == City.id)
                .where(*filters)
                .order_by(func.coalesce(counts.c.poi_count, 0).desc(), City.name)
                .limit(limit)
            )
        ).all()
        return AdminCityList(
            total=total,
            items=[
                AdminCitySummary(
                    id=city.id,
                    public_id=city.public_id,
                    name=city.name,
                    names=city.names_json,
                    slug=city.slug,
                    country_code=city.country_code,
                    lat=city.lat,
                    lng=city.lng,
                    source=city.source,
                    poi_count=int(poi_count),
                    active_poi_count=int(active_count),
                )
                for city, poi_count, active_count in rows
            ],
        )

    async def pois(
        self,
        *,
        city_id: int | None,
        query: str,
        active: bool | None,
        limit: int,
        offset: int,
    ) -> AdminPoiPage:
        filters = []
        if city_id is not None:
            filters.append(Poi.city_id == city_id)
        if query:
            pattern = f"%{query}%"
            filters.append(or_(Poi.name.ilike(pattern), Poi.slug.ilike(pattern)))
        if active is not None:
            filters.append(Poi.is_active.is_(active))

        total = int(
            await self._session.scalar(select(func.count(Poi.id)).where(*filters)) or 0
        )
        rows = (
            await self._session.execute(
                select(Poi, PoiType.code, PoiType.name)
                .outerjoin(PoiType, PoiType.id == Poi.poi_type_id)
                .where(*filters)
                .order_by(Poi.is_active.desc(), Poi.name)
                .limit(limit)
                .offset(offset)
            )
        ).all()
        return AdminPoiPage(
            total=total,
            limit=limit,
            offset=offset,
            items=[self._summary(poi, type_code, type_name) for poi, type_code, type_name in rows],
        )

    async def poi_detail(self, poi_id: int) -> AdminPoiDetail | None:
        row = (
            await self._session.execute(
                select(Poi, City.name, PoiType.code, PoiType.name)
                .outerjoin(City, City.id == Poi.city_id)
                .outerjoin(PoiType, PoiType.id == Poi.poi_type_id)
                .where(Poi.id == poi_id)
            )
        ).one_or_none()
        if row is None:
            return None
        poi, city_name, type_code, type_name = row
        return AdminPoiDetail(
            **self._summary(poi, type_code, type_name).model_dump(),
            city_id=poi.city_id,
            city_name=city_name,
            names=poi.names_json,
            short_description=poi.short_description,
            short_descriptions=poi.short_descriptions_json,
            long_description=poi.long_description,
            wikidata_id=poi.wikidata_id,
            wikipedia_title=poi.wikipedia_title,
            google_place_id=poi.google_place_id,
            metadata=poi.metadata_json,
        )

    @staticmethod
    def _summary(poi: Poi, type_code: str | None, type_name: str | None) -> AdminPoiSummary:
        return AdminPoiSummary(
            id=poi.id,
            public_id=poi.public_id,
            name=poi.name,
            slug=poi.slug,
            type_code=type_code,
            type_name=type_name,
            lat=poi.lat,
            lng=poi.lng,
            source_of_truth=poi.source_of_truth,
            is_active=poi.is_active,
        )
