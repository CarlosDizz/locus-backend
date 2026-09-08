"""V1-compatible public catalog. Mount router in the API entrypoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.api.auth import CurrentUserDep
from locus_v2.catalog.bootstrap.service import CatalogBootstrapError
from locus_v2.catalog.mobile import CatalogNotFoundError, MobileCatalogService
from locus_v2.catalog.mobile_schemas import (
    CityBootstrapFromLocationRequest,
    CityBootstrapFromLocationResponse,
    CityResponse,
    PoiAccessLinksResponse,
    PoiDocumentationResponse,
    PoiResponse,
    PoiTypeResponse,
)
from locus_v2.config import Settings, get_settings
from locus_v2.infrastructure.database.session import get_session

router = APIRouter(prefix="/api/catalog", tags=["catalog"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def catalog_language(
    language: str | None = Query(default=None),
    lang: str | None = Query(default=None),
    accept_language: str | None = Header(default=None),
) -> str:
    explicit = (language or lang or "").strip()
    if explicit:
        return explicit.lower().replace("_", "-")
    candidates: list[tuple[float, str]] = []
    for item in (accept_language or "").split(","):
        tag, *parameters = item.strip().split(";")
        quality = 1.0
        try:
            for parameter in parameters:
                if parameter.strip().startswith("q="):
                    quality = float(parameter.strip()[2:])
        except ValueError:
            continue
        if tag and tag != "*" and 0 < quality <= 1:
            candidates.append((quality, tag.lower().replace("_", "-")))
    return max(candidates, key=lambda item: item[0])[1] if candidates else "es"


def catalog_service(
    session: SessionDep, settings: SettingsDep,
    language: Annotated[str, Depends(catalog_language)],
) -> MobileCatalogService:
    return MobileCatalogService(session, settings, language)


CatalogDep = Annotated[MobileCatalogService, Depends(catalog_service)]


@router.get("/poi-types", response_model=list[PoiTypeResponse])
async def list_poi_types(service: CatalogDep) -> list[PoiTypeResponse]:
    return await service.list_poi_types()


@router.get("/cities", response_model=list[CityResponse])
async def list_cities(
    service: CatalogDep, q: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[CityResponse]:
    return await service.list_cities(q=q, limit=limit)


@router.get("/pois", response_model=list[PoiResponse])
async def list_pois(
    service: CatalogDep, city_id: int | None = Query(default=None),
    poi_type_code: str | None = Query(default=None), q: str = Query(default=""),
    limit: int = Query(default=200, ge=1, le=500),
) -> list[PoiResponse]:
    return await service.list_pois(
        city_id=city_id, poi_type_code=poi_type_code, q=q, limit=limit
    )


@router.get("/pois/{poi_id}", response_model=PoiResponse)
async def get_poi(poi_id: int, service: CatalogDep) -> PoiResponse:
    try:
        return await service.get_poi(poi_id)
    except CatalogNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/pois/{poi_id}/documentation", response_model=PoiDocumentationResponse)
async def get_poi_documentation(poi_id: int, service: CatalogDep) -> PoiDocumentationResponse:
    try:
        return await service.documentation(poi_id)
    except CatalogNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/pois/{poi_id}/access-links", response_model=PoiAccessLinksResponse)
async def get_poi_access_links(poi_id: int, service: CatalogDep) -> PoiAccessLinksResponse:
    try:
        return await service.access_links(poi_id)
    except CatalogNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/cities/bootstrap-from-location", response_model=CityBootstrapFromLocationResponse)
async def bootstrap_city_from_location(
    payload: CityBootstrapFromLocationRequest, service: CatalogDep, current_user: CurrentUserDep,
) -> CityBootstrapFromLocationResponse:
    try:
        return await service.bootstrap_from_location(payload)
    except CatalogBootstrapError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
