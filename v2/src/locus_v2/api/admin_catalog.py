from typing import Annotated

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.api.admin_auth import require_admin
from locus_v2.catalog.admin import (
    AdminCatalogQueryService,
    AdminCityList,
    AdminPoiDetail,
    AdminPoiPage,
)
from locus_v2.catalog.admin_sqlalchemy import SQLAlchemyAdminCatalogReader
from locus_v2.catalog.admin_write import AdminCatalogWriteService, PoiUpdate, PoiUpdateError
from locus_v2.catalog.bootstrap.dto import BootstrapResult
from locus_v2.catalog.bootstrap.enrichment import CatalogEnrichmentService
from locus_v2.catalog.bootstrap.service import CatalogBootstrapError, CatalogBootstrapService
from locus_v2.catalog.models import PoiType
from locus_v2.config import Settings, get_settings
from locus_v2.identity.models import User
from locus_v2.infrastructure.database.session import get_database, get_session
from locus_v2.observability import LocusEventLogger
from locus_v2.observability.infrastructure import SQLAlchemyEventLogRepository

logger = structlog.get_logger()

router = APIRouter(
    prefix="/admin/v2/catalog",
    tags=["admin-catalog"],
    dependencies=[Depends(require_admin)],
)
SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
AdminDep = Annotated[User, Depends(require_admin)]


def _event_logger(settings: Settings) -> LocusEventLogger:
    return LocusEventLogger(
        SQLAlchemyEventLogRepository(get_database()),
        service="catalog-bootstrap",
        environment=settings.env,
    )


def service(session: AsyncSession) -> AdminCatalogQueryService:
    return AdminCatalogQueryService(SQLAlchemyAdminCatalogReader(session))


class BootstrapFromLocationRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    radius_km: float = Field(default=8.0, gt=0, le=15)
    limit: int = Field(default=60, ge=1, le=150)
    use_ai_candidates: bool = True


@router.get("/cities", response_model=AdminCityList)
async def list_cities(
    session: SessionDep,
    q: str = Query(default="", max_length=160),
    limit: int = Query(default=100, ge=1, le=300),
) -> AdminCityList:
    return await service(session).list_cities(query=q, limit=limit)


@router.get("/pois", response_model=AdminPoiPage)
async def list_pois(
    session: SessionDep,
    city_id: int | None = None,
    q: str = Query(default="", max_length=160),
    active: bool | None = None,
    limit: int = Query(default=100, ge=1, le=250),
    offset: int = Query(default=0, ge=0),
) -> AdminPoiPage:
    return await service(session).list_pois(
        city_id=city_id,
        query=q,
        active=active,
        limit=limit,
        offset=offset,
    )


@router.get("/pois/{poi_id}", response_model=AdminPoiDetail)
async def poi_detail(poi_id: int, session: SessionDep) -> AdminPoiDetail:
    result = await service(session).detail(poi_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="POI not found")
    return result


class PoiTypeOption(BaseModel):
    code: str
    name: str


@router.get("/poi-types", response_model=list[PoiTypeOption])
async def list_poi_types(session: SessionDep) -> list[PoiTypeOption]:
    rows = (await session.scalars(select(PoiType).order_by(PoiType.name))).all()
    return [PoiTypeOption(code=row.code, name=row.name) for row in rows]


class PoiUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=255)
    names: dict[str, str] | None = None
    short_description: str | None = Field(default=None, max_length=500)
    short_descriptions: dict[str, str] | None = None
    long_description: str | None = None
    lat: str | None = None
    lng: str | None = None
    poi_type_code: str | None = None
    is_active: bool | None = None
    wikidata_id: str | None = Field(default=None, max_length=64)
    wikipedia_title: str | None = Field(default=None, max_length=255)
    google_place_id: str | None = Field(default=None, max_length=128)


@router.put("/pois/{poi_id}", response_model=AdminPoiDetail)
async def update_poi(
    poi_id: int, payload: PoiUpdateRequest, session: SessionDep, admin: AdminDep
) -> AdminPoiDetail:
    try:
        await AdminCatalogWriteService(session, admin.id).update_poi(
            poi_id, PoiUpdate(**payload.model_dump())
        )
    except PoiUpdateError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    result = await service(session).detail(poi_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="POI not found")
    return result


async def _run_enrichment(city_id: int, limit: int, settings: Settings) -> None:
    """Fire-and-forget background pass, scheduled after an AI-seeded bootstrap
    response is sent — the async equivalent of V1's `start_pending_enrichment`
    daemon thread. Uses its own DB session and event logger: the request's
    session is gone by the time a `BackgroundTasks` callback runs.
    """
    database = get_database()
    async with database.sessions() as session:
        try:
            enrichment_service = CatalogEnrichmentService(
                session, settings, event_logger=_event_logger(settings)
            )
            await enrichment_service.enrich_city_pending_pois(city_id, limit=limit)
        except Exception:
            logger.exception("catalog_enrichment_failed", city_id=city_id)


@router.post("/bootstrap-from-location", response_model=BootstrapResult)
async def bootstrap_from_location(
    payload: BootstrapFromLocationRequest,
    session: SessionDep,
    settings: SettingsDep,
    background_tasks: BackgroundTasks,
    admin: AdminDep,
) -> BootstrapResult:
    bootstrap_service = CatalogBootstrapService(
        session, settings, event_logger=_event_logger(settings), actor_user_id=admin.id
    )
    try:
        result = await bootstrap_service.bootstrap_from_location(
            lat=payload.lat, lng=payload.lng, radius_km=payload.radius_km, limit=payload.limit,
            use_ai_candidates=payload.use_ai_candidates,
        )
    except CatalogBootstrapError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)
        ) from error
    except Exception as error:
        logger.exception("catalog_bootstrap_unexpected_error", lat=payload.lat, lng=payload.lng)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"El sembrado ha fallado de forma inesperada: {error}",
        ) from error

    if "ai_seed" in result.source:
        background_tasks.add_task(
            _run_enrichment, result.city_id, max(payload.limit, 150), settings
        )
    return result
