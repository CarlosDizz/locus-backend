"""Viajes compartidos en Borea: quién puede cambiar el plan y a quién se avisa.

Lo delicado: que en Borea editen todos sin que Locus Guide cambie (allí solo
quien creó el viaje), que el aviso no le llegue a quien hizo el cambio, que vaya
en el idioma de quien lo recibe y solo a los móviles de las apps donde se ve el
viaje, y que un móvil que Apple da por caducado no se vuelva a intentar.
"""

import re
from datetime import date, datetime
from types import SimpleNamespace

from pydantic import SecretStr

from locus_v2.affiliates.tracking import verify
from locus_v2.api.borea_web import _dates, _language
from locus_v2.identity.models import User
from locus_v2.notifications.apns import SendResult
from locus_v2.notifications.models import PushDevice
from locus_v2.notifications.service import PlanChange, message, notify_trip
from locus_v2.shared.apps import ClientApp
from locus_v2.trips.models import Trip, TripItem, TripMember
from locus_v2.trips.service import (
    TripPlannerService,
    _plan_change,
    _plan_signature,
    _track_text,
    trip_view,
)


def _user(user_id: int, name: str, locale: str = "es-ES") -> User:
    return User(
        id=user_id, display_name=name, preferred_name="", locale=locale, email=f"{user_id}@x"
    )


def _trip(apps: int = 2) -> Trip:
    ana, luis, eva = _user(1, "Ana López"), _user(2, "Luis", "en-GB"), _user(3, "Eva")
    trip = Trip(id=9, public_id="t-1", user_id=1, city="Roma", country="Italia", travelers="",
                pace="", conversation_json=[], items=[],
                apps=apps)
    trip.owner = ana
    trip.members = [TripMember(user_id=2, user=luis), TripMember(user_id=3, user=eva)]
    return trip


class _Session:
    def __init__(self, devices: list[PushDevice]) -> None:
        self.devices = devices
        self.commits = 0

    async def scalars(self, _query):
        return list(self.devices)

    async def commit(self) -> None:
        self.commits += 1


class _Sender:
    def __init__(self, result: str = "enviada") -> None:
        self.sent: list[tuple[str, str, str, str]] = []
        self.result = result

    async def send(self, token, topic, title, body, data=None) -> SendResult:
        self.sent.append((token, topic, title, body))
        return SendResult(self.result)


_SETTINGS = SimpleNamespace(
    getyourguide_partner_id="CDLANTY", affiliate_links={},
    app_store_bundle_id="com.carlos.locusia", borea_app_store_bundle_id="com.carlos.borea"
)


def _device(device_id: int, user_id: int, app: str = "borea", platform: str = "ios") -> PushDevice:
    return PushDevice(id=device_id, user_id=user_id, app=app, platform=platform,
                      token=f"tok-{device_id}")


# --- quién edita ----------------------------------------------------------------


def test_in_borea_everyone_on_the_trip_edits_and_in_locus_only_the_creator() -> None:
    trip = _trip(apps=3)
    luis = trip.members[0].user
    borea = TripPlannerService(None, _SETTINGS, ClientApp.BOREA)  # type: ignore[arg-type]
    guide = TripPlannerService(None, _SETTINGS, ClientApp.GUIDE)  # type: ignore[arg-type]
    assert borea._can_edit(trip, luis)
    assert not guide._can_edit(trip, luis)
    assert guide._can_edit(trip, trip.owner)
    assert trip_view(trip, 2, app="borea").can_edit
    assert not trip_view(trip, 2, app="guide").can_edit
    assert trip_view(trip, 1, app="guide").can_edit


# --- qué ha cambiado ------------------------------------------------------------


def test_a_turn_that_adds_stops_names_them_and_one_that_only_reads_says_nothing() -> None:
    trip = _trip()
    antes = _plan_signature(trip)
    assert _plan_change(antes, trip) is None
    trip.items.append(TripItem(public_id="i-2", day=date(2026, 11, 6), time="21:00",
                               title="Cena en Trastevere", place="", status="proposed",
                               note="", position=1))
    trip.items.append(TripItem(public_id="i-1", day=date(2026, 11, 6), time="10:00",
                               title="Coliseo", place="", status="proposed", note="", position=0))
    cambio = _plan_change(antes, trip)
    assert cambio is not None and cambio.kind == "added"
    assert cambio.added == ["Coliseo", "Cena en Trastevere"]


def test_moving_a_stop_or_the_dates_is_a_change() -> None:
    trip = _trip()
    trip.items.append(TripItem(public_id="i-1", day=date(2026, 11, 6), time="10:00",
                               title="Coliseo", place="", status="proposed", note="", position=0))
    antes = _plan_signature(trip)
    trip.items[0].time = "12:00"
    cambio = _plan_change(antes, trip)
    assert cambio is not None and cambio.kind == "changed"
    antes = _plan_signature(trip)
    trip.starts_at = datetime(2026, 11, 5, 18)
    assert _plan_change(antes, trip) is not None


# --- textos ---------------------------------------------------------------------


def test_the_notice_speaks_the_language_of_whoever_gets_it() -> None:
    trip = _trip()
    ana = trip.owner
    assert message(PlanChange("joined"), ana, trip, "es-ES") == ("Roma", "Ana se ha unido al viaje")
    assert message(PlanChange("added", ["Coliseo"]), ana, trip, "en-GB")[1] == "Ana added “Coliseo”"
    assert message(PlanChange("added", ["Coliseo", "Cena"]), ana, trip, "es")[1] == (
        "Ana ha añadido «Coliseo» y 1 más"
    )
    assert message(PlanChange("changed"), ana, trip, "xx")[1] == "Ana ha cambiado el plan"
    trip.city = ""
    assert message(PlanChange("changed"), ana, trip, "fr")[0] == "Votre voyage"


# --- a quién ---------------------------------------------------------------------


async def test_everyone_but_the_author_is_notified_on_the_apps_that_see_the_trip() -> None:
    trip = _trip(apps=2)  # solo Borea
    devices = [
        _device(1, 1),  # Ana, autora: no
        _device(2, 2),  # Luis en Borea: sí, en inglés
        _device(3, 2, app="guide"),  # Luis en Locus Guide: no ve este viaje
        _device(4, 3, platform="android"),  # Eva en Android: espera a FCM
    ]
    sender = _Sender()
    sent = await notify_trip(
        _Session(devices), _SETTINGS, trip, trip.owner, PlanChange("changed"),  # type: ignore[arg-type]
        sender=sender,  # type: ignore[arg-type]
    )
    assert sent == 1
    assert sender.sent == [("tok-2", "com.carlos.borea", "Roma", "Ana changed the plan")]


async def test_a_phone_apple_says_is_gone_is_not_tried_again() -> None:
    trip = _trip()
    device = _device(2, 2)
    session = _Session([device])
    await notify_trip(session, _SETTINGS, trip, trip.owner, PlanChange("joined"),  # type: ignore[arg-type]
                      sender=_Sender("caducado"))  # type: ignore[arg-type]
    assert device.disabled_at is not None
    assert session.commits == 1


# --- la invitación ----------------------------------------------------------------


def test_the_invite_page_picks_a_known_language_and_writes_the_dates() -> None:
    assert _language("de-DE,de;q=0.9,en;q=0.8") == "de"
    assert _language("ko-KR,en;q=0.5") == "en"
    assert _language(None) == "es"
    trip = _trip()
    trip.starts_at, trip.ends_at = datetime(2026, 11, 5, 18), datetime(2026, 11, 9, 12)
    assert _dates(trip, "es") == "5 nov – 9 nov 2026"
    assert _dates(trip, "en") == "Nov 5 – Nov 9 2026"
    assert _dates(trip, "ja") == "11月5日 – 2026年11月9日"


# --- enlaces en el chat ---------------------------------------------------------------


def test_links_in_borea_chat_get_our_code_and_count_the_click() -> None:
    settings = SimpleNamespace(
        getyourguide_partner_id="CDLANTY", affiliate_links={"booking": "aid=1234567"},
        jwt_secret=SecretStr("x" * 40), public_api_base_url="https://api.locusguide.es/api",
    )
    texto = (
        "Mira [hoteles](https://www.booking.com/searchresults.html?ss=Segovia) o "
        "https://www.getyourguide.es/segovia-l1694/x-t1/. Y la web https://www.catedral.es/."
    )
    nuevo = _track_text(texto, settings, "borea", 7, "t-1")  # type: ignore[arg-type]
    enlaces = [
        e.rstrip(".") for e in re.findall(r"https://api\.locusguide\.es/api/r/[^)\s]+", nuevo)
    ]
    assert len(enlaces) == 2
    assert verify(settings, enlaces[0].rsplit("/", 1)[1]).url.endswith("aid=1234567")  # type: ignore[union-attr,arg-type]
    gyg = verify(settings, enlaces[1].rsplit("/", 1)[1])  # type: ignore[arg-type]
    assert gyg is not None and "partner_id=CDLANTY" in gyg.url and gyg.user_id == 7
    # Lo que no es de un programa se queda igual, y el punto final sigue fuera.
    assert "https://www.catedral.es/." in nuevo
    # En Locus Guide el chat no cambia.
    assert _track_text(texto, settings, "guide", 7, "t-1") == texto  # type: ignore[arg-type]


def test_restaurants_never_carry_a_booking_link() -> None:
    """Sin TheFork: un enlace suyo no es de ningún programa y la tarjeta va sin él."""
    from locus_v2.affiliates.programs import program_for
    from locus_v2.affiliates.search import _is_concrete_page

    assert program_for("https://www.thefork.es/restaurante/da-enzo-al-29-r123") is None
    servicio = TripPlannerService(None, _SETTINGS, ClientApp.BOREA)  # type: ignore[arg-type]
    assert servicio._booking("https://www.thefork.es/restaurante/x-r1", "Reservar") is None
    assert not _is_concrete_page("https://www.tiqets.com/es/blog/algo")
