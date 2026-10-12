"""La ruta a pie de un día del plan: el camino por las calles y lo que se tarda.

Sale de Google Routes con las paradas del día que tienen su sitio resuelto en
Places, en el orden del plan. Se pide una vez y se guarda en el viaje; solo se
vuelve a pedir si cambian las paradas de ese día.
"""

from datetime import date
from typing import Any

import httpx
import structlog
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.config import Settings
from locus_v2.trips.models import Trip

logger = structlog.get_logger()

# Google admite hasta 25 paradas intermedias andando; con más, no hay ruta.
MAX_STOPS = 27


class RouteLeg(BaseModel):
    duration_s: int
    distance_m: int


class DayRoute(BaseModel):
    date: date
    # La línea del camino, en el formato comprimido de Google (encoded polyline).
    polyline: str
    duration_s: int
    distance_m: int
    # Un tramo por cada par de paradas seguidas: legs[0] va de stops[0] a stops[1].
    legs: list[RouteLeg]
    stops: list[str]


class GoogleRoutesClient:
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"

    def __init__(self, settings: Settings) -> None:
        self._api_key = (
            settings.maps_api_key.get_secret_value().strip()
            if settings.maps_api_key is not None
            else ""
        )
        self._timeout_seconds = settings.maps_timeout_seconds

    async def walking(self, place_ids: list[str]) -> dict[str, Any] | None:
        if not self._api_key or len(place_ids) < 2:
            return None
        body = {
            "origin": {"placeId": place_ids[0]},
            "destination": {"placeId": place_ids[-1]},
            "intermediates": [{"placeId": place_id} for place_id in place_ids[1:-1]],
            "travelMode": "WALK",
            "languageCode": "es",
        }
        headers = {
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": (
                "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline,"
                "routes.legs.duration,routes.legs.distanceMeters"
            ),
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url, json=body, headers=headers, timeout=self._timeout_seconds
                )
            response.raise_for_status()
            routes = response.json().get("routes") or []
        except (httpx.HTTPError, ValueError) as error:
            # Sin ruta el mapa sigue enseñando las paradas; no es un error del usuario.
            logger.warning("trip_route_lookup_failed", error=str(error))
            return None
        return routes[0] if routes else None


def _seconds(value: object) -> int:
    """Google da las duraciones como texto: «498s»."""
    try:
        return int(str(value or "0s").rstrip("s").split(".")[0])
    except ValueError:
        return 0


async def day_route(
    session: AsyncSession, settings: Settings, trip: Trip, day: date
) -> DayRoute | None:
    paradas = sorted(
        (item for item in trip.items if item.day == day and item.map_json),
        key=lambda item: (item.time, item.position),
    )
    if len(paradas) < 2 or len(paradas) > MAX_STOPS:
        return None
    place_ids = [item.map_json["place_id"] for item in paradas]
    clave = "|".join(place_ids)
    guardadas = dict(trip.routes_json or {})
    previa = guardadas.get(day.isoformat())
    if previa and previa.get("key") == clave:
        return DayRoute(**previa["route"])

    route = await GoogleRoutesClient(settings).walking(place_ids)
    if route is None:
        return None
    legs = route.get("legs") or []
    resultado = DayRoute(
        date=day,
        polyline=str((route.get("polyline") or {}).get("encodedPolyline") or ""),
        duration_s=_seconds(route.get("duration")),
        distance_m=int(route.get("distanceMeters") or 0),
        legs=[
            RouteLeg(duration_s=_seconds(leg.get("duration")),
                     distance_m=int(leg.get("distanceMeters") or 0))
            for leg in legs
        ],
        stops=[item.public_id for item in paradas],
    )
    guardadas[day.isoformat()] = {"key": clave, "route": resultado.model_dump(mode="json")}
    # Un diccionario nuevo: SQLAlchemy no ve los cambios dentro de un JSON mutado.
    trip.routes_json = guardadas
    await session.commit()
    return resultado
