"""El guía sabe, por debajo, quién está en la llamada y quién habla.

Los mensajes escritos y las fotos ya le llegaban con el nombre del autor, pero la
voz llegaba sin él: oía a alguien y no sabía a quién. Y de quién entraba o salía
no se enteraba nunca. Lo que se fija aquí: el nombre va justo antes del audio,
una vez por turno; las llegadas se avisan; una salida solo cuenta si dura (un
móvil bloqueado no es alguien que se va); y el aviso nunca se cuela con alguien
hablando.
"""

from types import SimpleNamespace

from locus_v2.calls import bridge as bridge_module
from locus_v2.calls.bridge import DEPARTURE_GRACE_SECONDS, _CallVoiceBridge
from locus_v2.calls.models import Member, Room


def _sala(conectados: dict[int, str], status: str = "idle") -> Room:
    return Room(
        call_id="CALL-TEST",
        host_id=1,
        host_session_id="s",
        poi_id=1,
        poi_public_id="p",
        poi_name="Piazza Navona",
        language="es",
        routing_profile="voice.call.local",
        status=status,
        members={
            str(uid): Member(native_id=uid, wire_id=uid, display_name=nombre, connection="c")
            for uid, nombre in conectados.items()
        },
    )


class _Provider:
    capabilities = SimpleNamespace(context_seeding=True)

    def __init__(self) -> None:
        self.orden: list[tuple[str, object]] = []

    async def seed_context(self, entries: list) -> None:
        self.orden.append(("nota", entries))

    async def send_audio(self, pcm: bytes) -> None:
        self.orden.append(("audio", len(pcm)))


class _Store:
    def __init__(self, comandos: list[dict]) -> None:
        self.comandos = comandos

    async def commands(self, call_id: str):
        for comando in self.comandos:
            yield comando


def _puente() -> _CallVoiceBridge:
    puente = object.__new__(_CallVoiceBridge)
    puente.call_id = "CALL-TEST"
    puente.provider = _Provider()
    puente._roster = None
    puente._gone_since = {}
    puente._pending_roster_note = None
    puente._voice_turn = ""
    puente._heard_from_group = False
    puente._silence_task = None
    return puente


def _notas(puente) -> list[str]:
    return [
        texto
        for tipo, entradas in puente.provider.orden
        if tipo == "nota"
        for _, texto in entradas
    ]


async def test_the_speaker_name_goes_right_before_their_audio_once_per_turn() -> None:
    puente = _puente()
    trozo = "AAAA"
    puente.store = _Store(
        [
            {"type": "audio.chunk", "audio": trozo, "turn": "t1", "author": "Marta"},
            {"type": "audio.chunk", "audio": trozo, "turn": "t1", "author": "Marta"},
            {"type": "audio.chunk", "audio": trozo, "turn": "t2", "author": "Carlos"},
        ]
    )

    await puente._consume_commands()

    tipos = [tipo for tipo, _ in puente.provider.orden]
    assert tipos == ["nota", "audio", "audio", "nota", "audio"]
    assert _notas(puente) == ["[habla ahora Marta]", "[habla ahora Carlos]"]


async def test_someone_joining_is_announced_with_the_new_roster() -> None:
    puente = _puente()
    await puente._track_roster(_sala({1: "Carlos"}), now=0)
    await puente._track_roster(_sala({1: "Carlos", 2: "Marta"}), now=2)

    [nota] = _notas(puente)
    assert "acaba de unirse Marta" in nota
    assert "Carlos (quien organiza la llamada), Marta" in nota
    assert "saluda por su nombre" in nota


async def test_a_short_drop_is_not_a_departure() -> None:
    puente = _puente()
    await puente._track_roster(_sala({1: "Carlos", 2: "Marta"}), now=0)
    await puente._track_roster(_sala({1: "Carlos"}), now=2)  # se le bloquea el móvil
    await puente._track_roster(_sala({1: "Carlos", 2: "Marta"}), now=6)  # vuelve

    assert _notas(puente) == []


async def test_a_real_departure_is_announced_after_the_grace_period() -> None:
    puente = _puente()
    await puente._track_roster(_sala({1: "Carlos", 2: "Marta"}), now=0)
    await puente._track_roster(_sala({1: "Carlos"}), now=2)
    assert _notas(puente) == []
    await puente._track_roster(_sala({1: "Carlos"}), now=2 + DEPARTURE_GRACE_SECONDS)

    [nota] = _notas(puente)
    assert "se ha ido Marta" in nota
    assert "saluda" not in nota


async def test_the_note_waits_while_someone_is_speaking() -> None:
    puente = _puente()
    await puente._track_roster(_sala({1: "Carlos"}), now=0)
    await puente._track_roster(_sala({1: "Carlos", 2: "Marta"}, status="user_speaking"), now=2)
    assert _notas(puente) == []
    assert puente._pending_roster_note is not None

    await puente._track_roster(_sala({1: "Carlos", 2: "Marta"}), now=4)
    assert len(_notas(puente)) == 1


def test_names_read_naturally() -> None:
    assert bridge_module._y(["Marta"]) == "Marta"
    assert bridge_module._y(["Marta", "Luis", "Ana"]) == "Marta, Luis y Ana"
