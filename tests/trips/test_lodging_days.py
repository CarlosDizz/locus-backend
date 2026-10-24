"""Un día con alojamiento empieza y termina en él."""

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from locus_v2.trips import routes
from locus_v2.trips.lodging import lodging_for_day
from locus_v2.trips.models import Trip, TripItem


def _item(
    public_id: str, day: int, time: str, place_id: str | None = None, *,
    lodging: bool = False, kind: str = "visit",
) -> TripItem:
    return TripItem(
        public_id=public_id, day=date(2027, 10, day), time=time, title=public_id, kind=kind,
        status="booked", position=0, lodging=lodging,
        map_json={"name": public_id, "place_id": place_id, "lat": 0, "lng": 0}
        if place_id else None,
    )


def _viaje(items: list[TripItem], ends: int = 8) -> Trip:
    return Trip(
        public_id="t", city="Roma", country="Italia", conversation_json=[], routes_json=None,
        starts_at=datetime(2027, 10, 5), ends_at=datetime(2027, 10, ends), items=items,
    )


def _ids(par: tuple[TripItem | None, TripItem | None]) -> tuple[str | None, str | None]:
    return tuple(i.public_id if i else None for i in par)  # type: ignore[return-value]


def test_each_night_of_a_stay_leaves_from_and_returns_to_the_same_hotel() -> None:
    trip = _viaje([
        _item("vuelo", 5, "16:00", kind="transport"),
        _item("hotel", 5, "18:00", "H", lodging=True),
        _item("coliseo", 6, "10:00", "C"),
    ])

    # Llega: no sale de ningún hotel, pero duerme en él.
    assert _ids(lodging_for_day(trip, date(2027, 10, 5))) == (None, "hotel")
    assert _ids(lodging_for_day(trip, date(2027, 10, 6))) == ("hotel", "hotel")
    assert _ids(lodging_for_day(trip, date(2027, 10, 7))) == ("hotel", "hotel")
    # Último día: sale del hotel y ya no vuelve.
    assert _ids(lodging_for_day(trip, date(2027, 10, 8))) == ("hotel", None)


def test_changing_hotel_leaves_from_the_old_one_and_returns_to_the_new_one() -> None:
    trip = _viaje([
        _item("roma", 5, "18:00", "R", lodging=True),
        _item("florencia", 7, "17:00", "F", lodging=True),
    ])

    assert _ids(lodging_for_day(trip, date(2027, 10, 7))) == ("roma", "florencia")
    assert _ids(lodging_for_day(trip, date(2027, 10, 8))) == ("florencia", None)


def test_a_trip_without_lodging_has_no_start_or_end() -> None:
    trip = _viaje([_item("coliseo", 6, "10:00", "C")])

    assert lodging_for_day(trip, date(2027, 10, 6)) == (None, None)


class _Session:
    async def commit(self) -> None:
        pass


def _google(monkeypatch) -> list[list[str]]:
    pedidas: list[list[str]] = []

    class _Routes:
        def __init__(self, settings) -> None:
            pass

        async def walking(self, place_ids):
            pedidas.append(place_ids)
            legs = [{"duration": "600s", "distanceMeters": 800} for _ in place_ids[1:]]
            return {"duration": "1s", "distanceMeters": 1,
                    "polyline": {"encodedPolyline": "abc"}, "legs": legs}

    monkeypatch.setattr(routes, "GoogleRoutesClient", _Routes)
    return pedidas


@pytest.mark.asyncio
async def test_the_day_route_starts_and_ends_at_the_hotel_when_asked(monkeypatch) -> None:
    pedidas = _google(monkeypatch)
    trip = _viaje([
        _item("hotel", 5, "18:00", "H", lodging=True),
        _item("coliseo", 6, "10:00", "C"),
        _item("cena", 6, "21:00", "M"),
    ])

    ruta = await routes.day_route(
        _Session(), SimpleNamespace(), trip, date(2027, 10, 6), with_lodging=True
    )

    assert pedidas == [["H", "C", "M", "H"]]
    assert ruta is not None
    assert ruta.stops == ["hotel", "coliseo", "cena", "hotel"]
    assert ruta.from_lodging and ruta.to_lodging
    assert len(ruta.legs) == 3


@pytest.mark.asyncio
async def test_without_asking_the_route_stays_as_older_apps_expect(monkeypatch) -> None:
    pedidas = _google(monkeypatch)
    trip = _viaje([
        _item("hotel", 5, "18:00", "H", lodging=True),
        _item("coliseo", 6, "10:00", "C"),
        _item("cena", 6, "21:00", "M"),
    ])

    ruta = await routes.day_route(_Session(), SimpleNamespace(), trip, date(2027, 10, 6))

    assert pedidas == [["C", "M"]]
    assert ruta is not None and not ruta.from_lodging and not ruta.to_lodging


@pytest.mark.asyncio
async def test_arrival_day_ending_at_the_hotel_does_not_repeat_it(monkeypatch) -> None:
    pedidas = _google(monkeypatch)
    trip = _viaje([
        _item("estacion", 5, "16:00", "E", kind="transport"),
        _item("hotel", 5, "18:00", "H", lodging=True),
    ])

    ruta = await routes.day_route(
        _Session(), SimpleNamespace(), trip, date(2027, 10, 5), with_lodging=True
    )

    assert pedidas == [["E", "H"]]
    assert ruta is not None and not ruta.from_lodging and not ruta.to_lodging
