"""El planificador escribe el plan con herramientas; aquí se fija qué aceptan.

Lo delicado: los enlaces de reserva (solo GetYourGuide y siempre con nuestro
partner_id, que es de donde sale el dinero), que una propuesta no entre en el
plan hasta que la persona la acepte, y que lo que mande el modelo se valide en
vez de romper el plan.
"""

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from locus_v2.trips.models import Trip, TripItem
from locus_v2.trips.service import (
    TripError,
    TripPlannerService,
    _parse_time,
    trip_view,
)


class _Session:
    async def flush(self) -> None:
        for trip in self.trips:
            for item in trip.items:
                item.public_id = item.public_id or f"item-{id(item)}"

    async def delete(self, obj) -> None:
        pass

    def __init__(self) -> None:
        self.trips: list[Trip] = []


def _servicio(partner_id: str = "CDLANTY") -> tuple[TripPlannerService, _Session]:
    session = _Session()
    settings = SimpleNamespace(getyourguide_partner_id=partner_id)
    return TripPlannerService(session, settings), session  # type: ignore[arg-type]


def _viaje(session: _Session) -> Trip:
    trip = Trip(user_id=1, city="Roma", country="Italia", conversation_json=[], items=[])
    session.trips.append(trip)
    return trip


def _turno():
    from locus_v2.trips.service import _Turn

    return _Turn()


def test_a_getyourguide_link_always_carries_our_partner_id() -> None:
    servicio, _ = _servicio()
    booking = servicio._booking(
        "https://www.getyourguide.es/roma-l33/coliseo-t1234/?utm=x&partner_id=OTRO", "Coliseo"
    )
    assert booking is not None
    assert "partner_id=CDLANTY" in booking["url"]
    assert "OTRO" not in booking["url"]


@pytest.mark.parametrize(
    "url",
    [
        "https://www.tiqets.com/coliseo",  # otro proveedor
        "http://www.getyourguide.es/roma",  # sin https
        "https://getyourguide.es.estafa.com/roma",  # dominio que solo lo parece
    ],
)
def test_other_links_are_dropped(url: str) -> None:
    servicio, _ = _servicio()
    assert servicio._booking(url, "x") is None


async def test_suggestions_stay_as_cards_until_accepted() -> None:
    servicio, session = _servicio()
    trip = _viaje(session)
    turn = _turno()

    await servicio._tool(
        trip,
        "suggest_items",
        {"items": [{
            "date": "2027-10-05", "time": "19:00", "title": "Piazza Navona", "kind": "walk",
        }]},
        turn,
    )

    assert trip.items == []
    assert len(turn.proposals) == 1
    assert turn.proposals[0]["accepted"] is False


async def test_the_users_own_bookings_go_straight_into_the_plan() -> None:
    servicio, session = _servicio()
    trip = _viaje(session)

    await servicio._tool(
        trip,
        "add_items",
        {"items": [{
            "date": "2027-10-06", "time": "11:00", "title": "Coliseo",
            "kind": "visit", "status": "booked",
        }]},
        _turno(),
    )

    assert [(item.title, item.status) for item in trip.items] == [("Coliseo", "booked")]


async def test_trip_details_are_recorded() -> None:
    servicio, session = _servicio()
    trip = _viaje(session)
    await servicio._tool(
        trip, "set_trip_details",
        {"city": "Toledo", "starts_at": "2026-10-17T10:00", "travelers": "Tú y tu pareja"},
        _turno(),
    )
    assert trip.city == "Toledo"
    assert trip.starts_at == datetime(2026, 10, 17, 10, 0)
    assert trip.travelers == "Tú y tu pareja"


async def test_a_bad_date_from_the_model_is_an_error_not_a_crash() -> None:
    servicio, session = _servicio()
    trip = _viaje(session)
    with pytest.raises(TripError):
        await servicio._tool(
            trip, "add_items",
            {"items": [{"date": "el martes", "time": "11:00", "title": "x", "kind": "visit"}]},
            _turno(),
        )


def test_times_are_normalised() -> None:
    assert _parse_time("9:05") == "09:05"
    with pytest.raises(TripError):
        _parse_time("por la tarde")


def test_the_plan_is_grouped_by_day_and_sorted_by_time() -> None:
    trip = Trip(
        public_id="t", city="Roma", country="Italia", travelers="", pace="",
        conversation_json=[], items=[
            TripItem(public_id="b", day=date(2027, 10, 6), time="13:30", title="Comida",
                     kind="meal", status="proposed", note="", travel="", price="", position=1),
            TripItem(public_id="a", day=date(2027, 10, 6), time="11:00", title="Coliseo",
                     kind="visit", status="booked", note="", travel="", price="", position=0),
            TripItem(public_id="c", day=date(2027, 10, 5), time="19:00", title="Navona",
                     kind="walk", status="proposed", note="", travel="", price="", position=2),
        ],
    )
    view = trip_view(trip)
    assert [day.date for day in view.days] == [date(2027, 10, 5), date(2027, 10, 6)]
    assert [item.title for item in view.days[1].items] == ["Coliseo", "Comida"]


async def test_the_place_travels_from_the_card_to_the_plan() -> None:
    """El sitio que sirve para abrirlo en Google Maps no se pierde al aceptar la tarjeta."""
    servicio, session = _servicio()
    trip = _viaje(session)
    turn = _turno()

    await servicio._tool(
        trip,
        "suggest_items",
        {"items": [{
            "date": "2027-10-05", "time": "10:00", "title": "Galería Borghese",
            "kind": "visit", "place": "Galleria Borghese, Roma",
        }]},
        turn,
    )
    assert turn.proposals[0]["place"] == "Galleria Borghese, Roma"

    item = servicio._add_item(trip, {**turn.proposals[0], "status": "proposed"})

    assert item.place == "Galleria Borghese, Roma"
