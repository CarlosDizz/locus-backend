"""Borea sobre el backend de Locus: apps, referidos, skills, cobro exacto y calendario.

Lo delicado: que Locus Guide siga igual (solo GetYourGuide, sus viajes, su cobro),
que el código de afiliado lo ponga siempre el backend, que /r/ no sirva para
redirigir a cualquier web, que una tool de skill no se use sin cargar su skill y
que el cobro exacto no pierda fracciones de céntimo.
"""

import asyncio
from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from pydantic import SecretStr

from locus_v2.affiliates.programs import decorate, program_for
from locus_v2.affiliates.tracking import TrackedLink, sign, verify
from locus_v2.billing.application.processor import BillingDecisionEngine
from locus_v2.billing.pricing import NormalizedUsage, PriceCard
from locus_v2.shared.apps import ClientApp, app_bit, client_app, visible_in
from locus_v2.trips.borea_tools import SKILL_TOOLS, run_borea_tool, tools_for_turn
from locus_v2.trips.calendar import trip_ics
from locus_v2.trips.models import Trip, TripItem
from locus_v2.trips.service import TripPlannerService, _Turn, trip_view
from locus_v2.trips.skills import load_skills, parse_skill
from locus_v2.trips.weather import parse_daily


def _settings(**extra):
    base = dict(
        getyourguide_partner_id="CDLANTY", affiliate_links={}, borea_free=False,
        jwt_secret=SecretStr("x" * 40), public_api_base_url="https://api.locusguide.es/api",
    )
    return SimpleNamespace(**{**base, **extra})


# --- apps -----------------------------------------------------------------------


def test_without_header_it_is_locus_guide() -> None:
    assert client_app(None) == ClientApp.GUIDE
    assert client_app("nada") == ClientApp.GUIDE
    assert client_app(" Borea ") == ClientApp.BOREA


def test_a_trip_mask_says_where_it_is_seen() -> None:
    assert visible_in(1, ClientApp.GUIDE) and not visible_in(1, ClientApp.BOREA)
    assert visible_in(2, ClientApp.BOREA) and not visible_in(2, ClientApp.GUIDE)
    assert visible_in(3, ClientApp.GUIDE) and visible_in(3, ClientApp.BOREA)
    # Los viajes de antes de Borea no tienen máscara: son de Locus Guide.
    assert visible_in(0, ClientApp.GUIDE)
    assert app_bit(ClientApp.GUIDE) | app_bit(ClientApp.BOREA) == 3


# --- referidos ------------------------------------------------------------------


def test_booking_gets_our_aid_and_loses_someone_elses() -> None:
    url, program = decorate(
        "https://www.booking.com/hotel/it/x.html?aid=999&checkin=2026-10-05",
        _settings(affiliate_links={"booking": "aid=1234567"}),
    )
    assert program is not None and program.code == "booking"
    query = parse_qs(urlparse(url).query)
    assert query["aid"] == ["1234567"]
    assert query["checkin"] == ["2026-10-05"]


def test_a_network_program_wraps_the_link() -> None:
    url, _ = decorate(
        "https://www.tiqets.com/es/x",
        _settings(affiliate_links={"tiqets": "https://www.awin1.com/cread.php?a=1&ued={url}"}),
    )
    assert url.startswith("https://www.awin1.com/cread.php?a=1&ued=https%3A%2F%2Fwww.tiqets.com")


def test_an_unconfigured_program_is_linked_without_commission() -> None:
    url, program = decorate("https://www.airbnb.es/rooms/1?utm_source=x", _settings())
    assert program is not None and program.code == "airbnb"
    assert url == "https://www.airbnb.es/rooms/1"


def test_unknown_sites_are_not_programs() -> None:
    assert program_for("https://www.ejemplo.com/reserva") is None
    assert program_for("https://booking.com.malo.net/") is None


def test_locus_guide_still_only_accepts_getyourguide() -> None:
    guide = TripPlannerService(None, _settings(), ClientApp.GUIDE)  # type: ignore[arg-type]
    borea = TripPlannerService(None, _settings(), ClientApp.BOREA)  # type: ignore[arg-type]
    booking = "https://www.booking.com/hotel/it/x.html"
    assert guide._booking(booking, "") is None
    assert borea._booking(booking, "")["provider"] == "booking"
    gyg = guide._booking("https://www.getyourguide.es/roma-l33/t1/", "Coliseo")
    assert "partner_id=CDLANTY" in gyg["url"]


def test_a_tracked_link_survives_the_round_trip() -> None:
    settings = _settings()
    link = TrackedLink(
        url="https://www.booking.com/x", program="booking", app="borea", user_id=7, trip_id="t1"
    )
    assert verify(settings, sign(settings, link)) == link


def test_a_tampered_tracked_link_does_not_redirect() -> None:
    settings = _settings()
    token = sign(settings, TrackedLink(
        url="https://www.booking.com/x", program="booking", app="borea", user_id=7, trip_id=""
    ))
    payload, _, mac = token.partition(".")
    other = sign(settings, TrackedLink(
        url="https://malo.net/", program="x", app="borea", user_id=7, trip_id=""
    )).partition(".")[0]
    assert verify(settings, f"{other}.{mac}") is None
    assert verify(_settings(jwt_secret=SecretStr("y" * 40)), token) is None


# --- skills y tools -------------------------------------------------------------


def test_the_four_skills_load_with_their_tools() -> None:
    skills = load_skills()
    assert set(skills) == {"transporte", "hosteleria", "turismo", "equipamiento"}
    for skill in skills.values():
        assert skill.body and skill.description
        assert all(tool in SKILL_TOOLS for tool in skill.tools)


def test_parse_skill_reads_the_header() -> None:
    skill = parse_skill(
        "---\nname: x\ndescription: algo\ntools: [ruta, buscar_sitios]\n---\nCuerpo"
    )
    assert (skill.name, skill.tools, skill.body) == ("x", ("ruta", "buscar_sitios"), "Cuerpo")


def test_skill_tools_appear_only_after_loading_the_skill() -> None:
    names = {t["name"] for t in tools_for_turn([], set())}
    assert "cargar_skill" in names and "buscar_sitios" not in names
    names = {t["name"] for t in tools_for_turn([], {"hosteleria"})}
    assert "buscar_sitios" in names and "ruta" not in names


def _borea_trip() -> Trip:
    return Trip(
        user_id=1, city="Roma", country="Italia", conversation_json=[], items=[],
        apps=2, starts_at=datetime(2026, 10, 5, 18), ends_at=datetime(2026, 10, 9, 12),
        travelers="", pace="",
    )


def _run(name: str, args: dict, turn: _Turn, trip: Trip, settings=None):
    service = TripPlannerService(None, settings or _settings(), ClientApp.BOREA)  # type: ignore[arg-type]
    return asyncio.run(run_borea_tool(service, trip, name, args, turn))


def test_a_skill_tool_needs_its_skill_loaded() -> None:
    turn, trip = _Turn(), _borea_trip()
    assert _run("buscar_equipamiento", {"producto": "paraguas"}, turn, trip)["ok"] is False
    assert _run("cargar_skill", {"nombre": "equipamiento"}, turn, trip)["ok"] is True
    result = _run("buscar_equipamiento", {"producto": "paraguas plegable"}, turn, trip)
    assert result["ok"] and "amazon.es/s?k=paraguas" in result["links"][0]["url"]


def test_recommendations_are_added_capped_and_removed() -> None:
    turn, trip = _Turn(), _borea_trip()
    first = _run("recomendar", {
        "kind": "equipamiento", "title": "Calzado cómodo", "reason": "Empedrado",
        "booking_url": "https://www.amazon.es/s?k=zapatillas",
    }, turn, trip, _settings(affiliate_links={"amazon": "tag=borea-21"}))
    assert first["ok"]
    assert trip.recommendations_json[0]["booking"]["url"].endswith("tag=borea-21")
    for n in range(5):
        _run("recomendar", {"kind": "otro", "title": f"R{n}", "reason": "x"}, turn, trip)
    assert _run("recomendar", {"kind": "otro", "title": "Una más", "reason": "x"}, turn, trip)[
        "ok"
    ] is False
    assert _run("quitar_recomendacion", {"id": first["id"]}, turn, trip)["ok"]
    assert len(trip.recommendations_json) == 5


def test_climate_is_saved_as_ranges() -> None:
    turn, trip = _Turn(), _borea_trip()
    assert _run("guardar_clima", {
        "max_c": 21.4, "min_c": 12, "rain_days": 6, "summary": "Templado", "source": "x",
    }, turn, trip)["ok"]
    assert trip.climate_json["max_c"] == 21


def test_accommodation_search_carries_the_trip_dates() -> None:
    result = _run("buscar_alojamiento", {}, _Turn(), _borea_trip())
    query = parse_qs(urlparse(result["links"][0]["url"]).query)
    assert query["checkin"] == ["2026-10-05"] and query["checkout"] == ["2026-10-09"]


def test_the_view_shows_apps_recommendations_and_tracked_links() -> None:
    trip = _borea_trip()
    trip.public_id = "t1"
    trip.recommendations_json = [{
        "id": "r1", "kind": "entradas", "title": "Vaticano", "reason": "Colas",
        "booking": {"provider": "getyourguide", "title": "x",
                    "url": "https://www.getyourguide.es/a/", "price": ""},
    }]
    view = trip_view(trip, 1, settings=_settings(), app="borea")
    assert view.apps == ["borea"]
    tracked = view.recommendations[0].booking.tracked_url
    assert tracked.startswith("https://api.locusguide.es/api/r/")


# --- cobro exacto ---------------------------------------------------------------


def _engine(**extra) -> BillingDecisionEngine:
    return BillingDecisionEngine(
        usd_to_eur=Decimal("0.87"), margin_multiplier=Decimal("2.20"),
        minimum_realtime_call_charge_cents=3, **extra,
    )


_PRICE = PriceCard(text_input_per_million_usd=Decimal("0.10"),
                   text_output_per_million_usd=Decimal("0.50"))


def test_borea_charges_exact_and_carries_fractions() -> None:
    engine = _engine()
    usage = NormalizedUsage(text_input_tokens=8000, text_output_tokens=300)
    carry, charged = 0, 0
    for _ in range(10):
        decision, carry = engine.decide_exact(
            usage=usage, price=_PRICE, wallet_balance_cents=500, carry_microeur=carry
        )
        charged += decision.charged_amount_cents
    # 10 mensajes de 0,95 milésimas de dólar: ~1,82 céntimos en total, no 10.
    total_microeur = charged * 10_000 + carry
    assert charged == 1
    assert 18_000 < total_microeur < 18_400


def test_borea_free_records_cost_but_charges_nothing() -> None:
    decision, carry = _engine().decide_exact(
        usage=NormalizedUsage(text_input_tokens=10**6), price=_PRICE,
        wallet_balance_cents=500, carry_microeur=7, free=True,
    )
    assert decision.charged_amount_cents == 0 and decision.provider_cost_microusd > 0
    assert carry == 7


def test_locus_guide_still_rounds_each_message_up() -> None:
    decision = _engine().decide(
        usage=NormalizedUsage(text_input_tokens=8000, text_output_tokens=300),
        price=_PRICE, wallet_balance_cents=500,
    )
    assert decision.charged_amount_cents == 1


# --- calendario y tiempo --------------------------------------------------------


def test_the_calendar_keeps_local_times_and_escapes_text() -> None:
    trip = _borea_trip()
    trip.items = [TripItem(
        public_id="i1", day=date(2026, 10, 6), time="11:00", title="Coliseo, Foro; Palatino",
        kind="visit", status="booked", note="Entrada 1", travel="", place="Coliseo, Roma",
        position=0,
    )]
    ics = trip_ics(trip, datetime(2026, 10, 24, 1))
    assert "DTSTART:20261006T110000\r\n" in ics
    assert "SUMMARY:Coliseo\\, Foro\\; Palatino" in ics
    assert ics.startswith("BEGIN:VCALENDAR") and ics.endswith("END:VCALENDAR\r\n")


def test_weatherkit_days_become_icons_and_degrees() -> None:
    data = {"forecastDaily": {"days": [
        {"forecastStart": "2026-10-06T00:00:00Z", "conditionCode": "Clear",
         "temperatureMax": 21.4, "temperatureMin": 12.1, "precipitationChance": 0.05},
        {"forecastStart": "2026-10-07T00:00:00Z", "conditionCode": "Rain",
         "temperatureMax": 17.0, "temperatureMin": 11.0, "precipitationChance": 0.8},
        {"forecastStart": "2026-10-12T00:00:00Z", "conditionCode": "Clear",
         "temperatureMax": 20, "temperatureMin": 10},
    ]}}
    days = parse_daily(data, date(2026, 10, 5), date(2026, 10, 9))
    assert [(d.date.day, d.icon, d.max_c, d.rain_chance) for d in days] == [
        (6, "sun", 21, 5), (7, "rain", 17, 80),
    ]
