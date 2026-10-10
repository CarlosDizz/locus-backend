"""Lo que dice el grupo, tal como lo transcribe el propio modelo de voz.

Desde el 2026-10-10 no hay transcriptor externo: el registro de la llamada usa
la transcripción de Gemini. Gemini la manda a trozos a lo largo del turno, y el
puente escribía cada trozo como una línea distinta. Ese registro es lo que lee
el grupo y lo que se le vuelve a dar al guía en cada reconexión, así que tiene
que llegar una línea por turno, entera, y antes de la respuesta del guía.
"""

from types import SimpleNamespace

from locus_v2.calls.bridge import _CallVoiceBridge
from locus_v2.voice.providers.base import ProviderEvent, ProviderEventType


class _Service:
    def __init__(self, registro: list) -> None:
        self.registro = registro

    async def log_user_voice(self, call_id: str, text: str) -> None:
        self.registro.append(("user", text))

    async def assistant_finished(self, call_id: str, text: str) -> None:
        self.registro.append(("ai", text))


class _Room:
    status = "idle"


class _Store:
    async def publish(self, call_id: str, event: dict) -> None:
        pass

    async def get(self, call_id: str) -> _Room:
        return _Room()


class _Provider:
    capabilities = SimpleNamespace(context_seeding=True)

    def __init__(self, eventos: list[ProviderEvent]) -> None:
        self.eventos = eventos

    async def seed_context(self, entries: list) -> None:
        pass

    async def events(self):
        for evento in self.eventos:
            yield evento


def _puente(eventos: list[ProviderEvent], registro: list) -> _CallVoiceBridge:
    puente = object.__new__(_CallVoiceBridge)
    puente._mode = "poi"
    puente._itinerary = None
    puente.call_id = "CALL-TEST"
    puente.service = _Service(registro)
    puente.store = _Store()
    puente.provider = _Provider(eventos)
    puente.transcriber = None
    puente._assistant_text = ""
    puente._user_voice_text = ""
    puente._turns_in_session = 0
    puente._rolling_over = False
    puente._tool_turn_pending = False
    puente._silence_task = None
    puente._assistant_turn_count = 0

    async def turno_terminado(text: str) -> None:
        await puente.service.assistant_finished(puente.call_id, text)

    puente._on_assistant_turn_done = turno_terminado
    return puente


def _oido(texto: str) -> ProviderEvent:
    return ProviderEvent(type=ProviderEventType.INPUT_TRANSCRIPT_DONE, text=texto)


def _dicho(texto: str) -> ProviderEvent:
    return ProviderEvent(type=ProviderEventType.TEXT_DELTA, text=texto)


async def test_the_fragments_of_a_turn_become_one_line_before_the_answer() -> None:
    registro: list = []
    eventos = [
        _oido(" ¿Quién hizo"),
        _oido(" la fuente"),
        _oido(" del centro?"),
        _dicho("La hizo Bernini."),
        ProviderEvent(type=ProviderEventType.AUDIO_DONE),
    ]

    await _puente(eventos, registro)._pump_provider_events()

    assert registro == [
        ("user", "¿Quién hizo la fuente del centro?"),
        ("ai", "La hizo Bernini."),
    ]


async def test_each_turn_gets_its_own_line() -> None:
    registro: list = []
    eventos = [
        _oido("Hola"),
        _dicho("Buenas."),
        ProviderEvent(type=ProviderEventType.AUDIO_DONE),
        _oido("¿Y el obelisco?"),
        _dicho("Es egipcio."),
        ProviderEvent(type=ProviderEventType.AUDIO_DONE),
    ]

    await _puente(eventos, registro)._pump_provider_events()

    assert [linea for linea in registro if linea[0] == "user"] == [
        ("user", "Hola"),
        ("user", "¿Y el obelisco?"),
    ]


async def test_a_turn_with_nothing_heard_writes_no_empty_line() -> None:
    registro: list = []
    eventos = [_dicho("Seguimos."), ProviderEvent(type=ProviderEventType.AUDIO_DONE)]

    await _puente(eventos, registro)._pump_provider_events()

    assert registro == [("ai", "Seguimos.")]
