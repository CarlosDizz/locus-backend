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
# Un tramo a pie de más de esto no se anda: Barcelona Sants–Poblenou en taxi salía
# como «1 h 27 min a pie» (2026-10-24).
MAX_WALK_S = 40 * 60
# Sube cuando cambia lo que se guarda, para no servir rutas viejas sin los tramos.
_CACHE_VERSION = "v2"


class RouteLeg(BaseModel):
    duration_s: int
    distance_m: int
    # Si ese tramo se hace andando. No: sale de un transporte (taxi, tren, coche) o es
    # demasiado largo para ir a pie. La app lo pinta en discontinua y sin minutos.
    walk: bool = True
    # El camino de ese tramo, para pintar solo lo que se anda.
    polyline: str = ""


class DayRoute(BaseModel):
    date: date
    # La línea del camino, en el formato comprimido de Google (encoded polyline).
    polyline: str
    duration_s: int
    distance_m: int
    # Un tramo por cada par de paradas seguidas: legs[0] va de stops[0] a stops[1].
    legs: list[RouteLeg]
    stops: list[str]
    # Solo lo que se anda de verdad (Borea). duration_s y distance_m siguen siendo el
    # total entero, que es lo que lee la app de Locus Guide.
    walk_duration_s: int = 0
    walk_distance_m: int = 0


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
                "routes.legs.duration,routes.legs.distanceMeters,routes.legs.polyline.encodedPolyline"
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


_TRAVEL_MODES = {"a pie": "WALK", "transporte publico": "TRANSIT",
                 "transporte público": "TRANSIT", "coche": "DRIVE"}


async def travel_time(
    settings: Settings, origin: str, destination: str, mode: str = "a pie"
) -> dict[str, Any] | None:
    """Lo que se tarda de un sitio a otro (la tool «ruta» de las skills de Borea)."""
    client = GoogleRoutesClient(settings)
    if not client._api_key or not origin.strip() or not destination.strip():
        return None
    body = {
        "origin": {"address": origin},
        "destination": {"address": destination},
        "travelMode": _TRAVEL_MODES.get(mode.strip().lower(), "WALK"),
        "languageCode": "es",
    }
    headers = {
        "X-Goog-Api-Key": client._api_key,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters",
    }
    try:
        async with httpx.AsyncClient() as http:
            response = await http.post(
                client.url, json=body, headers=headers, timeout=client._timeout_seconds
            )
        response.raise_for_status()
        routes = response.json().get("routes") or []
    except (httpx.HTTPError, ValueError) as error:
        logger.warning("trip_travel_time_failed", error=str(error))
        return None
    if not routes:
        return None
    return {
        "minutes": round(_seconds(routes[0].get("duration")) / 60),
        "km": round(int(routes[0].get("distanceMeters") or 0) / 1000, 1),
        "mode": mode,
    }


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
    # La clave cambia si cambian los sitios o qué paradas son transporte.
    clave = _CACHE_VERSION + "|" + "|".join(
        f"{item.map_json['place_id']}:{item.kind}" for item in paradas
    )
    guardadas = dict(trip.routes_json or {})
    previa = guardadas.get(day.isoformat())
    if previa and previa.get("key") == clave:
        return DayRoute(**previa["route"])

    route = await GoogleRoutesClient(settings).walking(place_ids)
    if route is None:
        return None
    legs = route.get("legs") or []
    tramos: list[RouteLeg] = []
    for k, leg in enumerate(legs):
        duracion = _seconds(leg.get("duration"))
        # De una parada que es un transporte (el taxi, el tren, recoger el coche) al
        # siguiente sitio se va en ese transporte, no andando.
        sale_de_transporte = k < len(paradas) and paradas[k].kind == "transport"
        tramos.append(RouteLeg(
            duration_s=duracion,
            distance_m=int(leg.get("distanceMeters") or 0),
            walk=not sale_de_transporte and duracion <= MAX_WALK_S,
            polyline=str((leg.get("polyline") or {}).get("encodedPolyline") or ""),
        ))
    resultado = DayRoute(
        date=day,
        polyline=str((route.get("polyline") or {}).get("encodedPolyline") or ""),
        duration_s=_seconds(route.get("duration")),
        distance_m=int(route.get("distanceMeters") or 0),
        legs=tramos,
        stops=[item.public_id for item in paradas],
        walk_duration_s=sum(t.duration_s for t in tramos if t.walk),
        walk_distance_m=sum(t.distance_m for t in tramos if t.walk),
    )
    guardadas[day.isoformat()] = {"key": clave, "route": resultado.model_dump(mode="json")}
    # Un diccionario nuevo: SQLAlchemy no ve los cambios dentro de un JSON mutado.
    trip.routes_json = guardadas
    await session.commit()
    return resultado
