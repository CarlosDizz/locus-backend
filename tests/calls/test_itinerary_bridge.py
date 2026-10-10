"""El puente en una llamada en itinerario.

Cada ubicación actualiza lo que ve la app («ahora», «próximamente») y, si la
sala está en calma, le pide al guía que cuente lo que aparece. Si alguien está
hablando, el sitio espera: no se corta a nadie para anunciar un edificio.
"""

from types import SimpleNamespace

from locus_v2.calls.bridge import _CallVoiceBridge
from locus_v2.calls.itinerary import ItineraryTracker, Place

COLISEO = Place("coliseo", "Coliseo", 41.8902, 12.4922, tier=3)


class _Store:
    def __init__(self, status: str = "idle") -> None:
        self.status = status
        self.publicado: list[dict] = []

    async def publish(self, call_id: str, event: dict) -> None:
        self.publicado.append(event)

    async def get(self, call_id: str):
        return SimpleNamespace(status=self.status)


class _Provider:
    def __init__(self) -> None:
        self.enviado: list[str] = []

    async def send_text(self, text: str) -> None:
        self.enviado.append(text)


def _puente(status: str = "idle") -> _CallVoiceBridge:
    puente = object.__new__(_CallVoiceBridge)
    puente.call_id = "CALL-TEST"
    puente.store = _Store(status)
    puente.provider = _Provider()
    puente._mode = "itinerary"
    puente._travel_mode = "bus"
    puente._itinerary = ItineraryTracker(mode="bus", places=[COLISEO])
    return puente


CERCA = {"lat": 41.8912, "lng": 12.4908, "t": 100.0, "heading": 125.0, "speed": 5.0}


async def test_the_app_learns_what_is_passing_now() -> None:
    puente = _puente()
    await puente._on_location(CERCA)
    [evento] = puente.store.publicado
    assert evento["type"] == "itinerary.update"
    assert evento["now"]["name"] == "Coliseo"


async def test_with_the_room_quiet_the_guide_is_asked_to_tell_it() -> None:
    puente = _puente()
    await puente._on_location(CERCA)
    assert len(puente.provider.enviado) == 1
    assert "Coliseo" in puente.provider.enviado[0]


async def test_never_while_someone_is_talking_and_the_place_waits() -> None:
    puente = _puente(status="user_speaking")
    await puente._on_location(CERCA)
    assert puente.provider.enviado == []

    puente.store.status = "idle"
    await puente._on_location({**CERCA, "t": 105.0})
    assert len(puente.provider.enviado) == 1


async def test_a_call_that_is_not_an_itinerary_ignores_locations() -> None:
    puente = _puente()
    puente._itinerary = None
    await puente._on_location(CERCA)
    assert puente.store.publicado == [] and puente.provider.enviado == []
