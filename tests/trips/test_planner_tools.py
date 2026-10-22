"""El planificador escribe el plan con herramientas; aquí se fija qué aceptan.

Lo delicado: los enlaces de reserva (solo GetYourGuide y siempre con nuestro
partner_id, que es de donde sale el dinero), que una propuesta no entre en el
plan hasta que la persona la acepte, y que lo que mande el modelo se valide en
vez de romper el plan.
"""

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from locus_v2.billing.pricing import NormalizedUsage
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


async def test_the_place_travels_from_the_card_to_the_plan(monkeypatch) -> None:
    """El sitio de Google Maps se resuelve en la tarjeta y pasa al plan al aceptarla."""
    _places(monkeypatch, [
        _resultado("Galleria Borghese", "Piazzale Scipione Borghese, 5, Roma, Italia"),
    ])
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
    assert turn.proposals[0]["map"]["place_id"] == "id-Galleria Borghese"

    item = servicio._add_item(trip, {**turn.proposals[0], "status": "proposed"})

    assert item.place == "Galleria Borghese, Roma"
    assert item.map_json["lat"] == 41.9


async def test_a_place_in_another_city_is_not_linked(monkeypatch) -> None:
    """Mejor sin enlace que mandar a la persona a la Malquerida de otra ciudad."""
    _places(monkeypatch, [_resultado("La Malquerida", "Calle Mayor 3, Madrid")])
    servicio, session = _servicio()
    trip = _viaje(session)

    assert await servicio._locate(trip, "La Malquerida, Roma") is None


async def test_the_same_city_name_in_another_country_is_not_linked(monkeypatch) -> None:
    _places(monkeypatch, [
        _resultado("Toledo Museum of Art", "2445 Monroe St, Toledo, OH, EE. UU."),
    ])
    servicio, session = _servicio()
    trip = Trip(user_id=1, city="Toledo", country="España", conversation_json=[], items=[])

    assert await servicio._locate(trip, "Museo, Toledo") is None


async def test_a_meal_is_only_linked_to_somewhere_to_eat(monkeypatch) -> None:
    _places(monkeypatch, [
        _resultado("CBD Flowers By La Mal Querida", "C. Ciriaco Esteban 6, Roma, Italia"),
        _resultado("La Malquerida", "Via Roma 1, Roma, Italia", tipo="restaurant"),
    ])
    servicio, session = _servicio()
    trip = _viaje(session)

    mapa = await servicio._locate(trip, "La Malquerida, Roma", "meal")

    assert mapa is not None and mapa["name"] == "La Malquerida"


async def test_a_different_place_with_another_name_is_not_linked(monkeypatch) -> None:
    """Google no tenía el Mesón Beatriz y devolvía otro restaurante cercano."""
    _places(monkeypatch, [
        _resultado("Restaurante La Romana", "Concilios de Toledo, Roma, Italia", "restaurant"),
    ])
    servicio, session = _servicio()
    trip = _viaje(session)

    assert await servicio._locate(trip, "Mesón Beatriz, Roma", "meal") is None


async def test_without_a_place_nothing_is_searched(monkeypatch) -> None:
    llamadas = _places(monkeypatch, [])
    servicio, session = _servicio()
    trip = _viaje(session)

    await servicio._tool(
        trip,
        "add_items",
        {"items": [{
            "date": "2027-10-06", "time": "21:00", "title": "Cena por la zona",
            "kind": "meal", "status": "proposed",
        }]},
        _turno(),
    )

    assert llamadas == []
    assert trip.items[0].map_json is None


def _resultado(name: str, address: str, tipo: str = "place") -> dict:
    return {
        "name": name, "description": address, "lat": 41.9, "lng": 12.5,
        "id": f"id-{name}", "google_place_id": f"id-{name}", "poi_type_code": tipo,
    }


def _places(monkeypatch, resultados: list[dict]) -> list[str]:
    llamadas: list[str] = []

    class _Places:
        def __init__(self, settings) -> None:
            pass

        async def search_places(self, query, lat, lng, limit=5, locale=""):
            llamadas.append(query)
            return resultados

    monkeypatch.setattr("locus_v2.trips.service.GooglePlacesClient", _Places)
    return llamadas


async def test_locating_fills_old_stops_once_and_centres_the_city(monkeypatch) -> None:
    """Las paradas de antes de `place` se ponen en el mapa una vez, y la ciudad también."""
    _places(monkeypatch, [
        _resultado("Cervecería Lúpulo", "C. Ángel 9, Roma, Italia", "bar"),
        _resultado("Roma", "Roma, Italia"),
    ])
    servicio, session = _servicio()
    session.commit = _noop
    servicio.settings.openai_api_key = "x"
    trip = _viaje(session)
    trip.center_json = None
    cena = servicio._add_item(trip, {
        "date": "2027-10-05", "time": "20:00", "title": "Cena por la Judería", "kind": "meal",
    })
    cena.public_id = "cena"
    traslado = servicio._add_item(trip, {
        "date": "2027-10-05", "time": "18:00", "title": "Tren", "kind": "transport",
    })
    preguntas: list[list[str]] = []

    async def adivinar(trip, items, locale):
        preguntas.append([item.public_id for item in items])
        return {"cena": "Cervecería Lúpulo, Roma"}, NormalizedUsage(), None

    async def get(user, public_id, **_):
        return trip

    monkeypatch.setattr(servicio, "_guess_places", adivinar)
    monkeypatch.setattr(servicio, "get", get)

    await servicio.locate_missing(SimpleNamespace(id=1), "t")
    await servicio.locate_missing(SimpleNamespace(id=1), "t")

    assert preguntas == [["cena"]]
    assert cena.map_json["name"] == "Cervecería Lúpulo"
    assert traslado.map_json is None
    assert trip.center_json is not None


async def _noop() -> None:
    pass


async def test_a_card_added_by_mistake_can_be_undone(monkeypatch) -> None:
    servicio, session = _servicio()
    session.commit = _noop
    trip = _viaje(session)
    trip.conversation_json = [{"role": "assistant", "text": "", "proposals": [{
        "id": "p1", "date": "2027-10-05", "time": "10:30", "title": "Paseo por la Judería",
        "kind": "walk", "accepted": False,
    }]}]

    async def get(user, public_id, **_):
        return trip

    monkeypatch.setattr(servicio, "get", get)

    await servicio.accept_proposal(SimpleNamespace(id=1), "t", "p1")
    assert len(trip.items) == 1

    await servicio.undo_proposal(SimpleNamespace(id=1), "t", "p1")

    assert trip.items == []
    assert trip.conversation_json[0]["proposals"][0]["accepted"] is False


async def test_a_guest_cannot_change_the_plan_through_the_planner() -> None:
    """Los invitados opinan; el plan solo lo cambia quien organiza el viaje."""
    servicio, session = _servicio()
    trip = _viaje(session)
    turn = _turno()
    turn.readonly = True

    resultado = await servicio._tool(
        trip,
        "add_items",
        {"items": [{"date": "2027-10-06", "time": "11:00", "title": "Coliseo", "kind": "visit"}]},
        turn,
    )

    assert resultado["ok"] is False
    assert trip.items == []


async def test_a_guest_can_still_propose_cards() -> None:
    servicio, session = _servicio()
    trip = _viaje(session)
    turn = _turno()
    turn.readonly = True

    await servicio._tool(
        trip,
        "suggest_items",
        {"items": [{"date": "2027-10-06", "time": "19:00", "title": "Navona", "kind": "walk"}]},
        turn,
    )

    assert len(turn.proposals) == 1


def test_the_viewer_sees_their_role() -> None:
    trip = Trip(
        user_id=1, public_id="t", city="Roma", country="Italia", travelers="", pace="",
        conversation_json=[], items=[],
    )
    assert trip_view(trip, 1).my_role == "owner"
    assert trip_view(trip, 2).my_role == "guest"


def test_planner_language_rule_names_the_language() -> None:
    from locus_v2.trips.service import _language_rule

    assert "en inglés" in _language_rule("en")
    assert "en japonés" in _language_rule("ja-JP")
    assert "en español" in _language_rule("")


async def test_a_city_written_in_another_language_matches_by_distance(monkeypatch) -> None:
    """En francés el viaje es a «Tolède», pero Google escribe la calle en «Toledo»."""
    cerca = {**_resultado("Catedral Primada", "Calle Cardenal Cisneros, 45002 Toledo, Espagne"),
             "lat": 39.857, "lng": -4.024}
    _places(monkeypatch, [cerca])
    servicio, session = _servicio()
    trip = Trip(user_id=1, city="Tolède", country="Espagne", conversation_json=[], items=[],
                center_json={"lat": 39.8628, "lng": -4.0273})

    mapa = await servicio._locate(trip, "Catedral Primada, Toledo")

    assert mapa is not None and mapa["name"] == "Catedral Primada"


async def test_a_far_away_namesake_is_still_rejected_by_distance(monkeypatch) -> None:
    lejos = {**_resultado("Catedral Primada", "Plaza Mayor, Santo Domingo, Espagne"),
             "lat": 18.47, "lng": -69.88}
    _places(monkeypatch, [lejos])
    servicio, session = _servicio()
    trip = Trip(user_id=1, city="Tolède", country="Espagne", conversation_json=[], items=[],
                center_json={"lat": 39.8628, "lng": -4.0273})

    assert await servicio._locate(trip, "Catedral Primada, Toledo") is None


async def test_a_local_name_is_matched_in_spanish_when_the_trip_language_fails(monkeypatch) -> None:
    """En japonés Google llama «エル グレコ美術館» al «Museo del Greco» del planificador."""
    nombres = {"ja": "エル グレコ美術館", "es": "Museo del Greco"}
    pedidos: list[str] = []

    class _Places:
        def __init__(self, settings) -> None:
            pass

        async def search_places(self, query, lat, lng, limit=3, locale="es"):
            pedidos.append(locale)
            return [{**_resultado(nombres.get(locale, "El Greco Museum"),
                                  "P.º del Tránsito, 45002 Toledo"),
                     "lat": 39.856, "lng": -4.028}]

    monkeypatch.setattr("locus_v2.trips.service.GooglePlacesClient", _Places)
    servicio, session = _servicio()
    trip = Trip(user_id=1, city="トレド", country="スペイン", conversation_json=[], items=[],
                center_json={"lat": 39.8628, "lng": -4.0273})

    mapa = await servicio._locate(trip, "Museo del Greco, Toledo", locale="ja")

    assert mapa is not None and mapa["name"] == "Museo del Greco"
    assert pedidos == ["ja", "es"]


def test_planner_does_not_name_the_model_behind_it(monkeypatch) -> None:
    from locus_v2.shared.prompting import IDENTITY_RULE
    from locus_v2.trips import service as trips_service

    monkeypatch.setattr(
        trips_service, "trip_view", lambda trip: SimpleNamespace(model_dump=lambda **_: {})
    )
    servicio, session = _servicio()
    instrucciones = servicio._instructions("Eres Locus.", _viaje(session), locale="zh")

    assert IDENTITY_RULE in instrucciones


def _viaje_en_tren() -> Trip:
    return Trip(
        user_id=1, city="Barcelona", country="España", conversation_json=[],
        items=[
            TripItem(public_id="3f1c9a20-tren", day=date(2026, 10, 20), time="14:29",
                     title="Tren Albacete–Barcelona", kind="transport", position=0),
            TripItem(public_id="77aa0b11-hotel", day=date(2026, 10, 20), time="15:00",
                     title="Check-in", kind="logistics", position=1),
        ],
    )


async def test_a_wrong_item_id_answers_with_the_ids_that_exist() -> None:
    """Con un id que no existe, el modelo recibe las franjas y puede reintentar."""
    servicio, _ = _servicio()

    salida = await servicio._tool(
        _viaje_en_tren(), "update_item", {"item_id": "tren-renfe", "time": "14:59"}, _turno()
    )

    assert salida["ok"] is False
    assert [i["id"] for i in salida["items"]] == ["3f1c9a20-tren", "77aa0b11-hotel"]


async def test_a_unique_id_prefix_still_finds_the_item() -> None:
    servicio, _ = _servicio()
    trip = _viaje_en_tren()

    salida = await servicio._tool(
        trip, "update_item", {"item_id": "3f1c9a20", "time": "14:59"}, _turno()
    )

    assert salida["ok"] is True
    assert trip.items[0].time == "14:59"
