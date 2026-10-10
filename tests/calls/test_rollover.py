"""Pasar la memoria de la llamada de audio a texto, cada pocos turnos.

Gemini guarda cada turno como audio y lo vuelve a cobrar entero en cada turno
siguiente. Cerrar la sesión y abrir otra con la conversación como texto hace
que el guía recuerde lo mismo por mucho menos. Lo que estas pruebas fijan es
cuándo se puede cortar sin que nadie lo note: nunca con alguien hablando (su
audio ya enviado se perdería) ni con una búsqueda a medias, que tiene que
volver a la misma sesión.
"""

import asyncio
from types import SimpleNamespace

from locus_v2.calls.bridge import ROLLOVER_TURNS, _CallVoiceBridge
from locus_v2.voice.providers.base import ProviderEvent, ProviderEventType


class _Room:
    def __init__(self, status: str) -> None:
        self.status = status


class _Store:
    def __init__(self, status: str = "idle") -> None:
        self.status = status

    async def get(self, call_id: str) -> _Room:
        return _Room(self.status)

    async def publish(self, call_id: str, event: dict) -> None:
        pass


class _Service:
    async def log_user_voice(self, call_id: str, text: str) -> None:
        pass


class _Provider:
    capabilities = SimpleNamespace(context_seeding=True)

    def __init__(self, eventos: list[ProviderEvent]) -> None:
        self.eventos = eventos
        self.leidos = 0

    async def seed_context(self, entries: list) -> None:
        pass

    async def events(self):
        for evento in self.eventos:
            self.leidos += 1
            yield evento


def _puente(*, turnos: int = ROLLOVER_TURNS, sala: str = "idle") -> _CallVoiceBridge:
    puente = object.__new__(_CallVoiceBridge)
    puente._mode = "poi"
    puente._itinerary = None
    puente.call_id = "CALL-TEST"
    puente.store = _Store(sala)
    puente.service = _Service()
    puente.transcriber = None
    puente._assistant_text = ""
    puente._user_voice_text = ""
    puente._turns_in_session = turnos
    puente._rolling_over = False
    puente._tool_turn_pending = False
    puente._silence_task = None
    puente._assistant_turn_count = 0
    puente._document_tool_task = None
    puente._research_task = None

    async def turno_terminado(text: str) -> None:
        pass

    puente._on_assistant_turn_done = turno_terminado
    return puente


async def test_rolls_over_after_enough_turns_with_the_room_idle() -> None:
    assert await _puente()._should_roll_over()


async def test_not_before_enough_turns() -> None:
    assert not await _puente(turnos=ROLLOVER_TURNS - 1)._should_roll_over()


async def test_never_while_someone_holds_the_floor() -> None:
    # Su audio ya ha ido a la sesión actual: cortarla ahora lo tiraría.
    assert not await _puente(sala="user_speaking")._should_roll_over()
    assert not await _puente(sala="assistant_speaking")._should_roll_over()


async def test_never_while_a_search_is_running() -> None:
    puente = _puente()
    puente._document_tool_task = asyncio.get_running_loop().create_future()
    assert not await puente._should_roll_over()

    puente._document_tool_task = None
    puente._research_task = asyncio.get_running_loop().create_future()
    assert not await puente._should_roll_over()


async def test_the_session_ends_right_after_the_turn_that_reaches_the_limit() -> None:
    puente = _puente(turnos=ROLLOVER_TURNS - 1)
    eventos = [
        ProviderEvent(type=ProviderEventType.TEXT_DELTA, text="Y así acaba la historia."),
        ProviderEvent(type=ProviderEventType.TEXT_DONE),
        ProviderEvent(type=ProviderEventType.TEXT_DELTA, text="Esto ya no debería leerse."),
    ]
    puente.provider = _Provider(eventos)

    await puente._pump_provider_events()

    assert puente._rolling_over
    assert puente.provider.leidos == 2


async def test_below_the_limit_the_session_carries_on() -> None:
    puente = _puente(turnos=0)
    eventos = [
        ProviderEvent(type=ProviderEventType.TEXT_DELTA, text="Hola."),
        ProviderEvent(type=ProviderEventType.TEXT_DONE),
        ProviderEvent(type=ProviderEventType.TEXT_DELTA, text="Sigo."),
        ProviderEvent(type=ProviderEventType.TEXT_DONE),
    ]
    puente.provider = _Provider(eventos)

    await puente._pump_provider_events()

    assert not puente._rolling_over
    assert puente._turns_in_session == 2


class _Sesion:
    async def close(self) -> None:
        pass


def _bucle(resultados: list[tuple[bool, bool]]) -> tuple[_CallVoiceBridge, list[str]]:
    """Un puente cuyo run() recorre sesiones: (se perdio, fue paso a texto)."""
    puente = object.__new__(_CallVoiceBridge)
    puente._mode = "poi"
    puente._itinerary = None
    puente.call_id = "CALL-TEST"
    puente.provider = None
    puente.voice_session_id = None
    puente._needs_opening_research = False
    puente._document_tool_task = None
    puente._silence_task = None
    puente._heard_from_group = True
    traza: list[str] = []
    pendientes = list(resultados)

    async def conectar(*, resume: bool = False):
        if puente.voice_session_id is None:
            puente.voice_session_id = len(traza) + 100
            traza.append("sesion nueva")
        return _Sesion()

    async def servir() -> bool:
        perdida, paso = pendientes.pop(0)
        puente._rolling_over = paso
        return perdida

    async def terminar(status) -> None:
        if puente.voice_session_id is not None:
            traza.append("sesion cerrada")

    async def sigue() -> bool:
        return bool(pendientes)

    async def no_hace_nada(*args, **kwargs) -> None:
        pass

    puente._connect = conectar
    puente._serve_session = servir
    puente._finish = terminar
    puente._call_still_running = sigue
    puente.service = SimpleNamespace(mark_reconnecting=no_hace_nada)
    return puente, traza


async def test_a_rollover_stays_in_the_same_voice_session(monkeypatch) -> None:
    # El cobro de una llamada se agrupa por sesión de voz, mínimo incluido:
    # abrir una nueva en cada paso a texto cobraba otra vez el mínimo.
    monkeypatch.setattr(asyncio, "sleep", lambda *_: _nada())
    puente, traza = _bucle([(True, True), (True, True), (False, False)])
    await puente.run()
    assert traza == ["sesion nueva", "sesion cerrada"]


async def test_a_real_drop_opens_a_new_voice_session(monkeypatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", lambda *_: _nada())
    puente, traza = _bucle([(True, False), (False, False)])
    await puente.run()
    assert traza == ["sesion nueva", "sesion cerrada", "sesion nueva", "sesion cerrada"]


async def _nada() -> None:
    return None
