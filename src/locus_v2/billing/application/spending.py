"""Los gastos de una persona agrupados como los entiende: por viaje y por llamada.

En Borea cada mensaje cuesta fracciones de céntimo y se cobra exacto
(Wallet.carry_microeur), así que un historial por mensaje enseñaría «0,0042 €»
cien veces. Aquí se suma por viaje del planificador y por llamada con el guía,
con la app de cada gasto para pintar su icono (el saldo es común a las dos).
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.billing.models import UsageEvent, UsageStatus
from locus_v2.config import Settings
from locus_v2.trips.models import Trip

_TITLES = {
    "realtime_call": "Llamada con el guía",
    "chat_call": "Chat del mapa",
    "voice_note": "Nota de voz",
    "tool_call": "Búsquedas",
}


@dataclass(frozen=True)
class SpendingGroup:
    app: str
    title: str
    amount_eur: float
    last_at: datetime
    trip_id: str = ""


def _exact_eur(event: UsageEvent, settings: Settings) -> Decimal:
    """Lo cobrado de verdad: exacto en Borea (si no es gratis), en céntimos en Locus."""
    if event.app == "borea":
        if settings.borea_free:
            return Decimal(0)
        return (
            Decimal(event.provider_cost_microusd) / Decimal(1_000_000)
            * settings.billing_usd_to_eur * settings.billing_margin_multiplier
        )
    return Decimal(event.charged_amount_cents) / 100


async def spending_groups(
    session: AsyncSession, settings: Settings, user_id: int, limit: int = 20
) -> list[SpendingGroup]:
    events = (await session.scalars(
        select(UsageEvent)
        .where(UsageEvent.user_id == user_id, UsageEvent.status == UsageStatus.CHARGED)
        .order_by(UsageEvent.id.desc())
        .limit(500)
    )).all()
    groups: dict[str, dict] = {}
    for event in events:
        trip = str((event.raw_usage_json or {}).get("trip") or "")
        if trip:
            key = f"trip:{trip}"
        elif event.voice_session_id:
            key = f"call:{event.voice_session_id}"
        else:
            key = f"{event.interaction_type}:{event.created_at.date().isoformat()}"
        group = groups.setdefault(key, {
            "app": event.app, "trip": trip, "kind": event.interaction_type,
            "amount": Decimal(0), "last_at": event.created_at,
        })
        group["amount"] += _exact_eur(event, settings)
        group["last_at"] = max(group["last_at"], event.created_at)

    trip_ids = [g["trip"] for g in groups.values() if g["trip"]]
    cities: dict[str, str] = {}
    if trip_ids:
        rows = await session.execute(
            select(Trip.public_id, Trip.city).where(Trip.public_id.in_(trip_ids))
        )
        cities = {public_id: city for public_id, city in rows.all()}

    ordered = sorted(groups.values(), key=lambda g: g["last_at"], reverse=True)[:limit]
    return [
        SpendingGroup(
            app=g["app"],
            title=(
                (f"Viaje a {cities[g['trip']]}" if cities.get(g["trip"]) else "Un viaje")
                if g["trip"] else _TITLES.get(g["kind"], "Consumo")
            ),
            amount_eur=float(g["amount"].quantize(Decimal("0.00001"))),
            last_at=g["last_at"],
            trip_id=g["trip"],
        )
        for g in ordered
    ]
