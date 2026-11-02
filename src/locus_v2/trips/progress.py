"""Lo que el planificador está haciendo ahora mismo, para enseñarlo mientras se espera.

Un turno de Borea con búsquedas tarda de 20 s a más de un minuto. En vez de un «generando
respuesta», la app enseña los pasos según pasan: «mirando vuelos Tokio → Okinawa»,
«buscando hoteles en Shibuya»… (idea de Carlos, 2026-11-02). Cada paso es un código y un
detalle; la app pone el texto en el idioma de la persona.

Va en Valkey con caducidad corta y nunca rompe un turno: si falla, no se enseña nada.
"""

import json
from typing import Any

import structlog

from locus_v2.infrastructure.redis import get_redis

logger = structlog.get_logger()

_TTL_SECONDS = 300
_KEY = "planner:progress:{trip}"


def _key(trip_id: str) -> str:
    return _KEY.format(trip=trip_id)


async def start(trip_id: str) -> None:
    await _write(trip_id, {"running": True, "steps": []})


async def finish(trip_id: str) -> None:
    estado = await read(trip_id)
    estado["running"] = False
    await _write(trip_id, estado)


async def step(trip_id: str, kind: str, detail: str = "") -> None:
    estado = await read(trip_id)
    pasos = estado.get("steps") or []
    nuevo = {"kind": kind, "detail": detail.strip()[:80]}
    if pasos and pasos[-1] == nuevo:
        return
    estado["steps"] = [*pasos, nuevo][-20:]
    estado["running"] = True
    await _write(trip_id, estado)


async def read(trip_id: str) -> dict[str, Any]:
    try:
        crudo = await get_redis().get(_key(trip_id))
    except Exception as error:  # noqa: BLE001 - sin progreso, el turno sigue
        logger.warning("planner_progress_unavailable", error=str(error))
        return {"running": False, "steps": []}
    if not crudo:
        return {"running": False, "steps": []}
    try:
        return dict(json.loads(crudo))
    except ValueError:
        return {"running": False, "steps": []}


async def _write(trip_id: str, estado: dict[str, Any]) -> None:
    try:
        await get_redis().set(_key(trip_id), json.dumps(estado, ensure_ascii=False),
                              ex=_TTL_SECONDS)
    except Exception as error:  # noqa: BLE001 - sin progreso, el turno sigue
        logger.warning("planner_progress_unavailable", error=str(error))


_TRANSPORT_STEPS = {
    "vuelo": "flights", "traslado": "transfers", "consigna": "luggage", "coche": "car",
    "tren": "trains", "autobús": "buses",
}


def step_for_tool(name: str, args: dict[str, Any], city: str = "") -> tuple[str, str] | None:
    """El paso que se enseña por una herramienta, o None si no merece la pena enseñarlo."""
    texto = lambda *claves: next(  # noqa: E731 - el primer argumento que traiga algo
        (str(args.get(c)).strip() for c in claves if str(args.get(c) or "").strip()), ""
    )
    if name == "buscar_transporte":
        return _TRANSPORT_STEPS.get(texto("tipo"), "transport"), texto("consulta")
    if name == "buscar_alojamiento":
        return "hotels", texto("ciudad", "zona") or city
    if name in ("buscar_actividades", "find_bookable"):
        return "activities", texto("lugar", "consulta", "query", "place")
    if name == "buscar_sitios":
        return "food", texto("consulta")
    if name == "buscar_esim":
        return "esim", texto("pais")
    if name == "ruta":
        return "route", " → ".join(x for x in (texto("origen"), texto("destino")) if x)
    if name == "buscar_equipamiento":
        return "gear", texto("producto")
    if name == "guardar_clima":
        return "weather", city
    if name == "suggest_items":
        return "cards", ""
    if name in ("add_items", "update_item", "remove_item"):
        return "plan", ""
    return None
