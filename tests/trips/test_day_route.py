"""La ruta a pie del día se pide a Google una vez y se reutiliza mientras no cambie."""

from datetime import date
from types import SimpleNamespace

import pytest

from locus_v2.trips import routes
from locus_v2.trips.models import Trip, TripItem

DIA = date(2027, 10, 5)


class _Session:
    commits = 0

    async def commit(self) -> None:
        self.commits += 1


def _item(public_id: str, time: str, place_id: str | None, position: int) -> TripItem:
    return TripItem(
        public_id=public_id, day=DIA, time=time, title=public_id, kind="visit",
        status="proposed", position=position,
        map_json={"name": public_id, "place_id": place_id, "lat": 0, "lng": 0}
        if place_id else None,
    )


def _viaje() -> Trip:
    return Trip(
        public_id="t", city="Toledo", country="España", conversation_json=[], routes_json=None,
        items=[
            _item("catedral", "12:00", "P2", 1),
            _item("alcazar", "10:00", "P1", 0),
            _item("comida", "14:00", None, 2),
            _item("sinagoga", "17:00", "P3", 3),
        ],
    )


def _google(monkeypatch) -> list[list[str]]:
    pedidas: list[list[str]] = []

    class _Routes:
        def __init__(self, settings) -> None:
            pass

        async def walking(self, place_ids):
            pedidas.append(place_ids)
            return {
                "duration": "1165s", "distanceMeters": 1329,
                "polyline": {"encodedPolyline": "abc"},
                "legs": [{"duration": "498s", "distanceMeters": 564},
                         {"duration": "668s", "distanceMeters": 766}],
            }

    monkeypatch.setattr(routes, "GoogleRoutesClient", _Routes)
    return pedidas


@pytest.mark.asyncio
async def test_the_route_follows_the_plan_order_and_skips_stops_without_a_place(
    monkeypatch,
) -> None:
    pedidas = _google(monkeypatch)
    trip = _viaje()

    ruta = await routes.day_route(_Session(), SimpleNamespace(), trip, DIA)

    assert pedidas == [["P1", "P2", "P3"]]
    assert ruta is not None
    assert ruta.stops == ["alcazar", "catedral", "sinagoga"]
    assert [leg.duration_s for leg in ruta.legs] == [498, 668]


@pytest.mark.asyncio
async def test_the_same_day_is_not_asked_twice(monkeypatch) -> None:
    pedidas = _google(monkeypatch)
    trip = _viaje()
    session = _Session()

    await routes.day_route(session, SimpleNamespace(), trip, DIA)
    await routes.day_route(session, SimpleNamespace(), trip, DIA)

    assert len(pedidas) == 1


@pytest.mark.asyncio
async def test_changing_the_stops_asks_again(monkeypatch) -> None:
    pedidas = _google(monkeypatch)
    trip = _viaje()
    session = _Session()
    await routes.day_route(session, SimpleNamespace(), trip, DIA)

    trip.items.append(_item("mirador", "19:00", "P4", 4))
    await routes.day_route(session, SimpleNamespace(), trip, DIA)

    assert pedidas[-1] == ["P1", "P2", "P3", "P4"]


@pytest.mark.asyncio
async def test_one_stop_is_not_a_route(monkeypatch) -> None:
    pedidas = _google(monkeypatch)
    trip = _viaje()
    trip.items = [_item("alcazar", "10:00", "P1", 0)]

    assert await routes.day_route(_Session(), SimpleNamespace(), trip, DIA) is None
    assert pedidas == []


@pytest.mark.asyncio
async def test_a_ride_or_a_very_long_leg_is_not_walked(monkeypatch) -> None:
    """Barcelona, 2026-10-24: Sants–Poblenou en taxi salía como «1 h 27 min a pie»."""

    class _Routes:
        def __init__(self, settings) -> None:
            pass

        async def walking(self, place_ids):
            return {
                "duration": "5400s", "distanceMeters": 7000, "polyline": {"encodedPolyline": "x"},
                "legs": [
                    {"duration": "60s", "distanceMeters": 50},  # tren → taxi, en la estación
                    {"duration": "5220s", "distanceMeters": 6400},  # taxi → hotel
                    {"duration": "120s", "distanceMeters": 120},  # hotel → cena
                ],
            }

    monkeypatch.setattr(routes, "GoogleRoutesClient", _Routes)
    taxi = _item("taxi", "20:50", "P2", 1)
    taxi.kind = "transport"
    trip = _viaje()
    trip.items = [
        _item("tren", "14:59", "P1", 0), taxi, _item("hotel", "21:15", "P3", 2),
        _item("cena", "21:45", "P4", 3),
    ]

    ruta = await routes.day_route(_Session(), SimpleNamespace(), trip, DIA)

    assert ruta is not None
    assert [leg.walk for leg in ruta.legs] == [True, False, True]
    assert ruta.walk_duration_s == 180 and ruta.walk_distance_m == 170
    # Locus Guide sigue leyendo el total de siempre.
    assert ruta.duration_s == 5400
