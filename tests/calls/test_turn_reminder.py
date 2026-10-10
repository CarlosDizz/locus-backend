"""Explicaciones largas de saque, y un guía que no se queda callado.

Gemini 3.8 contesta en 20-25 segundos por mucho que el prompt de sistema le
pida extenderse; una nota pegada a cada turno lo lleva a explicaciones de un
tirón de 50-70 segundos (medido el 2026-10-10). Lo que se fija aquí: que la nota
se deja tras cada turno del guía sin hacerle hablar, que no se cuela con alguien
ya hablando, que vuelve tras una reconexión, y que si tras una herramienta el
guía se queda mudo el puente le da pie — pero solo si de verdad no ha hablado.
"""

import asyncio
from types import SimpleNamespace

import pytest

from locus_v2.calls import bridge as bridge_module
from locus_v2.calls.bridge import (
    TOOL_SILENCE_PROMPT,
    TURN_REMINDER,
    _CallVoiceBridge,
)
from locus_v2.voice.providers.base import ProviderEvent, ProviderEventType


class _Room:
    def __init__(self, status: str, log: list | None = None) -> None:
        self.status = status
        self.log = log or []
        self.members: dict = {}
        self.host_id = 1


class _Store:
    def __init__(self, status: str = "idle") -> None:
        self.status = status

    async def get(self, call_id: str) -> _Room:
        return _Room(self.status)

    async def publish(self, call_id: str, event: dict) -> None:
        pass


class _Provider:
    capabilities = SimpleNamespace(context_seeding=True)

    def __init__(self, eventos: list[ProviderEvent] | None = None) -> None:
        self.eventos = eventos or []
        self.sembrado: list[list] = []
        self.enviado: list[str] = []

    async def events(self):
        for evento in self.eventos:
            yield evento

    async def seed_context(self, entries: list) -> None:
        self.sembrado.append(entries)

    async def send_text(self, text: str) -> None:
        self.enviado.append(text)


def _puente(eventos: list[ProviderEvent] | None = None, sala: str = "idle") -> _CallVoiceBridge:
    puente = object.__new__(_CallVoiceBridge)
    puente.call_id = "CALL-TEST"
    puente.store = _Store(sala)
    puente.provider = _Provider(eventos)
    puente.transcriber = None
    puente._assistant_text = ""
    puente._user_voice_text = ""
    puente._turns_in_session = 0
    puente._rolling_over = False
    puente._tool_turn_pending = False
    puente._silence_task = None
    puente._last_audio_at = 0.0
    puente._pending_roster_note = None
    puente._roster = None
    puente._document_tool_task = None
    puente._research_task = None

    async def turno_terminado(text: str) -> None:
        pass

    puente._on_assistant_turn_done = turno_terminado
    return puente


def _turno(texto: str) -> list[ProviderEvent]:
    return [
        ProviderEvent(type=ProviderEventType.TEXT_DELTA, text=texto),
        ProviderEvent(type=ProviderEventType.TEXT_DONE),
    ]


async def test_the_note_is_left_after_each_guide_turn() -> None:
    puente = _puente(_turno("La plaza sigue la forma del estadio.") + _turno("Y la fuente."))
    await puente._pump_provider_events()
    assert puente.provider.sembrado == [[("user", TURN_REMINDER)], [("user", TURN_REMINDER)]]
    # Contexto silencioso: no es un mensaje que haga hablar al guía.
    assert puente.provider.enviado == []


async def test_not_while_someone_is_already_speaking() -> None:
    puente = _puente(_turno("La plaza sigue la forma del estadio."), sala="user_speaking")
    await puente._pump_provider_events()
    assert puente.provider.sembrado == []


async def test_a_reconnection_ends_with_the_note() -> None:
    puente = _puente()
    sala = _Room("idle", log=[{"kind": "ai", "text": "Bienvenidos a la plaza."}])
    await puente._reseed(puente.provider, sala)
    assert puente.provider.sembrado[-1][-1] == ("user", TURN_REMINDER)


@pytest.fixture
def sin_espera(monkeypatch):
    monkeypatch.setattr(bridge_module, "TOOL_SILENCE_SECONDS", 0)


async def test_a_silent_guide_after_a_tool_gets_a_cue(sin_espera) -> None:
    puente = _puente(_turno(""))
    puente._tool_turn_pending = True
    await puente._pump_provider_events()
    await puente._silence_task
    assert puente.provider.enviado == [TOOL_SILENCE_PROMPT]


async def test_no_cue_if_the_guide_spoke_after_all(sin_espera) -> None:
    puente = _puente()
    await puente._nudge_if_silent(puente.provider, since=1.0)
    assert puente.provider.enviado == [TOOL_SILENCE_PROMPT]

    puente = _puente()
    puente._last_audio_at = 5.0
    await puente._nudge_if_silent(puente.provider, since=1.0)
    assert puente.provider.enviado == []


async def test_no_cue_if_the_group_took_the_floor(sin_espera) -> None:
    puente = _puente(sala="user_speaking")
    await puente._nudge_if_silent(puente.provider, since=1.0)
    assert puente.provider.enviado == []


async def test_an_empty_turn_without_a_tool_gets_no_cue() -> None:
    puente = _puente(_turno(""))
    await puente._pump_provider_events()
    assert puente._silence_task is None
    await asyncio.sleep(0)
