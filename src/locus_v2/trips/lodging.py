"""De qué alojamiento sale cada día del viaje y a cuál vuelve.

Un día con alojamiento empieza en él y termina en él: sale de donde se durmió la
noche anterior y vuelve a donde se duerme esa noche. El alojamiento se apunta una
vez por estancia (la parada marcada `lodging`, el día que se entra), y vale para
todas las noches hasta el siguiente alojamiento o el último día del viaje.
"""

from datetime import date

from locus_v2.trips.models import Trip, TripItem


def _last_day(trip: Trip) -> date | None:
    if trip.ends_at is not None:
        return trip.ends_at.date()
    return max((item.day for item in trip.items), default=None)


def lodging_for_day(trip: Trip, day: date) -> tuple[TripItem | None, TripItem | None]:
    """(de dónde sale, adónde vuelve) ese día; None donde no hay alojamiento."""
    estancias = sorted(
        (item for item in trip.items if item.lodging),
        key=lambda item: (item.day, item.time, item.position),
    )
    if not estancias:
        return None, None
    ultimo = _last_day(trip)
    # La noche anterior: el último alojamiento en el que se entró antes de hoy.
    sale = next((item for item in reversed(estancias) if item.day < day), None)
    # Esta noche: el último en el que se ha entrado hasta hoy, salvo el último día
    # del viaje, que ya no se duerme.
    vuelve = None
    if ultimo is None or day < ultimo:
        vuelve = next((item for item in reversed(estancias) if item.day <= day), None)
    return sale, vuelve
