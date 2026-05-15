from fastapi import APIRouter, Depends, HTTPException, Query

from app.deps.auth import get_current_user_required
from app.schemas.auth import UserResponse
from app.schemas.catalog import (
    CityBootstrapRequest,
    CityBootstrapFromLocationRequest,
    CityBootstrapFromLocationResponse,
    CityCreateRequest,
    CityPoiImportRequest,
    CityPoiImportResponse,
    PoiAccessLinksResponse,
    CityResponse,
    PoiDocumentationResponse,
    PoiCreateRequest,
    PoiResponse,
    PoiTypeResponse,
    PoiUpdateRequest,
)
from app.services.catalog_service import CatalogError, catalog_service
from app.services.poi_service import poi_service
from app.services.referral_service import referral_service


router = APIRouter(prefix="/api/catalog", tags=["catalog"])


@router.get("/poi-types", response_model=list[PoiTypeResponse])
async def list_poi_types() -> list[PoiTypeResponse]:
    return catalog_service.list_poi_types()


@router.get("/cities", response_model=list[CityResponse])
async def list_cities(
    q: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[CityResponse]:
    return catalog_service.list_cities(q=q, limit=limit)


@router.post("/cities", response_model=CityResponse)
async def create_city(
    payload: CityCreateRequest,
    _current_user: UserResponse = Depends(get_current_user_required),
) -> CityResponse:
    try:
        return catalog_service.create_city(payload)
    except CatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/cities/bootstrap", response_model=CityResponse)
async def bootstrap_city(
    payload: CityBootstrapRequest,
    _current_user: UserResponse = Depends(get_current_user_required),
) -> CityResponse:
    try:
        return catalog_service.bootstrap_city(payload.name, payload.country_code)
    except CatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/cities/bootstrap-from-location", response_model=CityBootstrapFromLocationResponse)
async def bootstrap_city_from_location(
    payload: CityBootstrapFromLocationRequest,
    _current_user: UserResponse = Depends(get_current_user_required),
) -> CityBootstrapFromLocationResponse:
    try:
        city, imported_count, updated_count, skipped_count, stats, pois = catalog_service.bootstrap_city_from_location(
            lat=payload.lat,
            lng=payload.lng,
            radius_km=payload.radius_km,
            limit=payload.limit,
            use_ai_candidates=payload.use_ai_candidates,
        )
    except CatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CityBootstrapFromLocationResponse(
        city=city,
        imported_count=imported_count,
        updated_count=updated_count,
        skipped_count=skipped_count,
        stats=stats,
        pois=pois,
    )


@router.get("/pois", response_model=list[PoiResponse])
async def list_pois(
    city_id: int | None = Query(default=None),
    poi_type_code: str | None = Query(default=None),
    q: str = Query(default=""),
    limit: int = Query(default=200, ge=1, le=500),
) -> list[PoiResponse]:
    return catalog_service.list_pois(city_id=city_id, poi_type_code=poi_type_code, q=q, limit=limit)


@router.get("/pois/{poi_id}", response_model=PoiResponse)
async def get_poi(poi_id: int) -> PoiResponse:
    try:
        return catalog_service.get_poi(poi_id)
    except CatalogError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/pois/{poi_id}/documentation", response_model=PoiDocumentationResponse)
async def get_poi_documentation(poi_id: int) -> PoiDocumentationResponse:
    try:
        poi = catalog_service.get_poi(poi_id)
    except CatalogError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    documentation = poi_service.get_poi_documentation(poi.name)
    return PoiDocumentationResponse(
        poi=poi,
        documentation=documentation,
        resolved_from_catalog=True,
    )


@router.get("/pois/{poi_id}/access-links", response_model=PoiAccessLinksResponse)
async def get_poi_access_links(poi_id: int) -> PoiAccessLinksResponse:
    try:
        poi = catalog_service.get_poi(poi_id)
    except CatalogError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    city_name = ""
    if poi.city_id is not None:
        cities = catalog_service.list_cities(limit=500)
        city_name = next((city.name for city in cities if city.id == poi.city_id), "")
    return PoiAccessLinksResponse(**referral_service.poi_access_links(poi, city_name=city_name))


@router.post("/pois", response_model=PoiResponse)
async def create_poi(
    payload: PoiCreateRequest,
    _current_user: UserResponse = Depends(get_current_user_required),
) -> PoiResponse:
    try:
        return catalog_service.create_poi(payload)
    except CatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/pois/{poi_id}", response_model=PoiResponse)
async def update_poi(
    poi_id: int,
    payload: PoiUpdateRequest,
    _current_user: UserResponse = Depends(get_current_user_required),
) -> PoiResponse:
    try:
        return catalog_service.update_poi(poi_id, payload)
    except CatalogError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/cities/{city_id}/import-pois", response_model=CityPoiImportResponse)
async def import_city_pois(
    city_id: int,
    payload: CityPoiImportRequest,
    _current_user: UserResponse = Depends(get_current_user_required),
) -> CityPoiImportResponse:
    try:
        imported_count, updated_count, skipped_count, stats, pois, city_name = catalog_service.import_city_pois(
            city_id=city_id,
            radius_km=payload.radius_km,
            limit=payload.limit,
            use_ai_candidates=payload.use_ai_candidates,
        )
        if payload.use_ai_candidates:
            catalog_service.start_pending_enrichment(city_id, min(payload.limit, 150))
    except CatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CityPoiImportResponse(
        city_id=city_id,
        city_name=city_name,
        imported_count=imported_count,
        updated_count=updated_count,
        skipped_count=skipped_count,
        stats=stats,
        pois=pois,
    )
