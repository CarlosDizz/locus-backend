from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.api.admin_auth import require_admin
from locus_v2.catalog.admin import (
    AdminCatalogQueryService,
    AdminCityList,
    AdminPoiDetail,
    AdminPoiPage,
)
from locus_v2.catalog.admin_sqlalchemy import SQLAlchemyAdminCatalogReader
from locus_v2.infrastructure.database.session import get_session

router = APIRouter(
    prefix="/admin/v2/catalog",
    tags=["admin-catalog"],
    dependencies=[Depends(require_admin)],
)
SessionDep = Annotated[AsyncSession, Depends(get_session)]


def service(session: AsyncSession) -> AdminCatalogQueryService:
    return AdminCatalogQueryService(SQLAlchemyAdminCatalogReader(session))


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
