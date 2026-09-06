"""Admin catalog write-side: editing a POI already in the catalog.

Deliberately separate from admin.py/admin_sqlalchemy.py (the read-only
explorer) and from bootstrap/ (which creates POIs from external sources).
This is the missing third piece: correcting or refining a POI that already
exists, from the control panel, with an audit trail.

There is no V1 equivalent to port — V1 had no admin panel at all, catalog
data was edited directly in the database. This is a new V2 capability.
"""

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.catalog.models import Poi, PoiType
from locus_v2.identity.models import AdminAuditEvent


class PoiUpdateError(ValueError):
    pass


@dataclass(frozen=True)
class PoiUpdate:
    name: str | None = None
    names: dict[str, str] | None = None
    short_description: str | None = None
    short_descriptions: dict[str, str] | None = None
    long_description: str | None = None
    lat: str | None = None
    lng: str | None = None
    poi_type_code: str | None = None
    is_active: bool | None = None
    wikidata_id: str | None = None
    wikipedia_title: str | None = None
    google_place_id: str | None = None


class AdminCatalogWriteService:
    def __init__(self, session: AsyncSession, actor_user_id: int) -> None:
        self.session = session
        self.actor_user_id = actor_user_id

    async def update_poi(self, poi_id: int, payload: PoiUpdate) -> Poi:
        poi = await self.session.get(Poi, poi_id)
        if poi is None:
            raise PoiUpdateError("POI not found")

        before = _snapshot(poi)

        if payload.name is not None:
            name = payload.name.strip()
            if not name:
                raise PoiUpdateError("El nombre no puede estar vacío")
            poi.name = name
        if payload.names is not None:
            poi.names_json = {k: v for k, v in payload.names.items() if v.strip()}
        if payload.short_description is not None:
            poi.short_description = payload.short_description.strip()
        if payload.short_descriptions is not None:
            poi.short_descriptions_json = {
                k: v for k, v in payload.short_descriptions.items() if v.strip()
            }
        if payload.long_description is not None:
            poi.long_description = payload.long_description.strip()
        if payload.lat is not None:
            poi.lat = _parse_decimal(payload.lat, "lat")
        if payload.lng is not None:
            poi.lng = _parse_decimal(payload.lng, "lng")
        if payload.poi_type_code is not None:
            poi.poi_type_id = await self._resolve_type(payload.poi_type_code)
        if payload.is_active is not None:
            poi.is_active = payload.is_active
        if payload.wikidata_id is not None:
            poi.wikidata_id = payload.wikidata_id.strip()
        if payload.wikipedia_title is not None:
            poi.wikipedia_title = payload.wikipedia_title.strip()
        if payload.google_place_id is not None:
            poi.google_place_id = payload.google_place_id.strip()

        # source_of_truth reflects who last authored the content; a manual
        # panel edit should say so, same convention bootstrap.py already
        # uses for wikidata/overpass/gpt_seed.
        poi.source_of_truth = "manual"

        self.session.add(
            AdminAuditEvent(
                actor_user_id=self.actor_user_id,
                action="catalog.poi.updated",
                resource_type="poi",
                resource_id=str(poi.id),
                before_json=json.dumps(before, default=str),
                after_json=json.dumps(_snapshot(poi), default=str),
                trace_id=str(uuid4()),
            )
        )
        await self.session.commit()
        await self.session.refresh(poi)
        return poi

    async def _resolve_type(self, code: str) -> int | None:
        code = code.strip()
        if not code:
            return None
        type_row = await self.session.scalar(select(PoiType).where(PoiType.code == code))
        if type_row is None:
            raise PoiUpdateError(f"Tipo de POI desconocido: {code}")
        return type_row.id


def _parse_decimal(value: str, field: str) -> Decimal | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return Decimal(stripped)
    except InvalidOperation as error:
        raise PoiUpdateError(f"Valor de {field} inválido: {value}") from error


def _snapshot(poi: Poi) -> dict[str, object]:
    return {
        "name": poi.name,
        "names": poi.names_json,
        "short_description": poi.short_description,
        "short_descriptions": poi.short_descriptions_json,
        "long_description": poi.long_description,
        "lat": poi.lat,
        "lng": poi.lng,
        "poi_type_id": poi.poi_type_id,
        "is_active": poi.is_active,
        "wikidata_id": poi.wikidata_id,
        "wikipedia_title": poi.wikipedia_title,
        "google_place_id": poi.google_place_id,
        "source_of_truth": poi.source_of_truth,
    }
