"""Los recordatorios del viaje: dos días antes de salir y cada mañana del viaje.

Para pedir el permiso de avisos con una promesa útil también a quien viaja solo
(hasta ahora solo se avisaba de los cambios de los demás). La hora es la del
destino, aproximada por la longitud de su centro; sin centro, la de España.
Nunca se manda dos veces lo mismo: cada aviso deja una marca en Valkey.
"""

from datetime import date, datetime, timedelta

import structlog
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from locus_v2.config import Settings
from locus_v2.notifications.apns import ApnsSender
from locus_v2.notifications.models import PushDevice
from locus_v2.notifications.service import _sees, _topic
from locus_v2.notifications.texts import text
from locus_v2.trips.models import Trip, TripMember

logger = structlog.get_logger()

PRE_TRIP_HOUR = 18
TODAY_HOUR = 8
_SHOWN_KINDS = ("visit", "meal", "walk", "transport")


def local_now(trip: Trip, now: datetime) -> datetime:
    lng = (trip.center_json or {}).get("lng")
    offset = round(float(lng) / 15) if lng is not None else 1
    return now + timedelta(hours=offset)


def items_of(trip: Trip, day: date) -> list[str]:
    del_dia = sorted(
        (i for i in trip.items if i.day == day and i.kind in _SHOWN_KINDS and not i.lodging),
        key=lambda i: (i.time or "", i.position),
    )
    return [i.title for i in del_dia]


def join_items(titles: list[str], locale: str) -> str:
    titulos = [t[:40] for t in titles[:3]]
    if len(titulos) <= 1:
        return "".join(titulos)
    return ", ".join(titulos[:-1]) + text(locale, "and") + titulos[-1]


def due(trip: Trip, now: datetime) -> tuple[str, date] | None:
    """El recordatorio que toca ahora a este viaje, con su día, o None."""
    if trip.starts_at is None:
        return None
    local = local_now(trip, now)
    hoy = local.date()
    inicio = trip.starts_at.date()
    fin = trip.ends_at.date() if trip.ends_at else inicio
    if local.hour == PRE_TRIP_HOUR and hoy == inicio - timedelta(days=2):
        return "pre_trip", hoy
    if local.hour == TODAY_HOUR and inicio <= hoy <= fin and items_of(trip, hoy):
        return "today", hoy
    return None


async def send_due_reminders(
    session: AsyncSession, settings: Settings, redis: Redis, now: datetime,
    sender: ApnsSender | None = None,
) -> int:
    trips = list(await session.scalars(
        select(Trip)
        .where(Trip.starts_at.is_not(None))
        .where(Trip.starts_at <= now + timedelta(days=3))
        .where(Trip.starts_at >= now - timedelta(days=60))
        .options(
            selectinload(Trip.items), selectinload(Trip.owner),
            selectinload(Trip.members).selectinload(TripMember.user),
        )
    ))
    own_sender = sender is None
    sender = sender or ApnsSender(settings)
    sent = 0
    try:
        for trip in trips:
            toca = due(trip, now)
            if toca is None:
                continue
            kind, dia = toca
            marca = f"reminder:{trip.public_id}:{kind}:{dia.isoformat()}"
            if not await redis.set(marca, "1", nx=True, ex=4 * 24 * 3600):
                continue
            sent += await _send(session, settings, sender, trip, kind, dia)
        await session.commit()
    finally:
        if own_sender:
            await sender.close()
    return sent


async def _send(
    session: AsyncSession, settings: Settings, sender: ApnsSender, trip: Trip, kind: str,
    dia: date,
) -> int:
    personas = {u.id: u for u in [trip.owner, *(m.user for m in trip.members)] if u is not None}
    if not personas:
        return 0
    devices = [
        d for d in await session.scalars(
            select(PushDevice).where(
                PushDevice.user_id.in_(personas), PushDevice.disabled_at.is_(None)
            )
        )
        if d.platform == "ios" and _sees(trip, d.app)
    ]
    enviados = 0
    for device in devices:
        locale = personas[device.user_id].locale
        ciudad = trip.city or text(locale, "trip")
        cuerpo = (
            text(locale, "pre_trip", city=ciudad) if kind == "pre_trip"
            else text(locale, "today", items=join_items(items_of(trip, dia), locale))
        )
        try:
            result = await sender.send(
                device.token, _topic(settings, device.app), ciudad, cuerpo,
                {"trip_id": trip.public_id},
            )
        except Exception as error:  # noqa: BLE001 - un aviso nunca rompe los demás
            logger.warning("reminder_failed", error=str(error), device=device.id)
            continue
        if result.status in ("enviada", "simulada"):
            enviados += 1
        logger.info("push_reminder", kind=kind, status=result.status, trip=trip.public_id)
    return enviados
