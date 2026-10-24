"""Avisos de los viajes compartidos: quién se une y qué cambia en el plan.

Se avisa a todos los que van al viaje menos a quien hizo el cambio, en su
idioma, en los móviles de las apps donde se ve ese viaje. Un aviso que falla no
rompe nada: el cambio ya está guardado y se queda en el log.
"""

from dataclasses import dataclass, field

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.config import Settings
from locus_v2.identity.models import User
from locus_v2.notifications.apns import ApnsSender
from locus_v2.notifications.models import PushDevice
from locus_v2.notifications.texts import text
from locus_v2.shared.apps import ClientApp, visible_in
from locus_v2.shared.clock import utc_now
from locus_v2.trips.models import Trip

logger = structlog.get_logger()

PLATFORMS = ("ios", "android")


@dataclass
class PlanChange:
    # joined | added | changed
    kind: str
    added: list[str] = field(default_factory=list)


def first_name(user: User) -> str:
    return (user.preferred_name or user.display_name or "").split(" ")[0] or "?"


def message(change: PlanChange, actor: User, trip: Trip, locale: str) -> tuple[str, str]:
    """El título (la ciudad) y el texto del aviso, en el idioma de quien lo recibe."""
    name = first_name(actor)
    title = trip.city or text(locale, "trip")
    if change.kind == "joined":
        return title, text(locale, "joined", name=name)
    if change.kind == "added" and change.added:
        item = change.added[0][:60]
        if len(change.added) == 1:
            return title, text(locale, "added_one", name=name, item=item)
        return title, text(
            locale, "added_many", name=name, item=item, more=len(change.added) - 1
        )
    return title, text(locale, "changed", name=name)


def _topic(settings: Settings, app: str) -> str:
    if app == ClientApp.BOREA.value:
        return settings.borea_app_store_bundle_id
    return settings.app_store_bundle_id


async def register_device(
    session: AsyncSession, user: User, app: ClientApp, platform: str, token: str
) -> None:
    now = utc_now()
    device = await session.scalar(select(PushDevice).where(PushDevice.token == token))
    if device is None:
        session.add(PushDevice(
            user_id=user.id, app=app.value, platform=platform, token=token, last_seen_at=now
        ))
    else:
        # El mismo móvil con otra cuenta: los avisos son de quien ha entrado ahora.
        device.user_id = user.id
        device.app = app.value
        device.platform = platform
        device.last_seen_at = now
        device.disabled_at = None
    await session.commit()


async def notify_trip(
    session: AsyncSession, settings: Settings, trip: Trip, actor: User, change: PlanChange,
    sender: ApnsSender | None = None,
) -> int:
    """Avisa a los demás del viaje. Devuelve cuántos avisos se han enviado (o simulado)."""
    people = [trip.owner, *(m.user for m in trip.members)]
    recipients = {u.id: u for u in people if u is not None and u.id != actor.id}
    if not recipients:
        return 0
    devices = list(await session.scalars(
        select(PushDevice).where(
            PushDevice.user_id.in_(recipients), PushDevice.disabled_at.is_(None)
        )
    ))
    devices = [d for d in devices if d.user_id in recipients and _sees(trip, d.app)]
    if not devices:
        return 0
    own_sender = sender is None
    sender = sender or ApnsSender(settings)
    sent = 0
    try:
        for device in devices:
            if device.platform != "ios":
                # Android llegará con FCM; hasta entonces su token se guarda y espera.
                continue
            title, body = message(change, actor, trip, recipients[device.user_id].locale)
            try:
                result = await sender.send(
                    device.token, _topic(settings, device.app), title, body,
                    {"trip_id": trip.public_id},
                )
            except Exception as error:  # noqa: BLE001 - un aviso nunca rompe el cambio
                logger.warning("push_failed", error=str(error), device=device.id)
                continue
            if result.status == "caducado":
                device.disabled_at = utc_now()
            elif result.status in ("enviada", "simulada"):
                sent += 1
            logger.info(
                "push_trip", status=result.status, detail=result.detail, kind=change.kind,
                trip=trip.public_id, device=device.id,
            )
        await session.commit()
    finally:
        if own_sender:
            await sender.close()
    return sent


def _sees(trip: Trip, app: str) -> bool:
    try:
        return visible_in(trip.apps or 1, ClientApp(app))
    except ValueError:
        return False
