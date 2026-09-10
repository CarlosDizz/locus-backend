"""The traveller profile has to survive an edit of the prompt.

It lives in code rather than in the editable prompt for one reason: a block of
text saying whose profile this is and what to do with it is one careless save
away from being deleted from the panel, and the feature would then keep running
and quietly stop working. These tests pin the two halves of that promise — the
heading travels with the data, and the data is appended rather than substituted.
"""

import asyncio
from dataclasses import dataclass, field

from locus_v2.calls.bridge import TRAVELER_CONTEXT_BLOCK, _CallVoiceBridge
from locus_v2.calls.models import Member, Room


@dataclass
class _User:
    id: int
    preferred_name: str = ""
    profile_context: str = ""


@dataclass
class _FakeSession:
    """Solo hace falta `scalars`: es lo unico que toca _traveler_context."""

    users: list[_User] = field(default_factory=list)

    async def scalars(self, _statement: object) -> list[_User]:
        return [user for user in self.users if user.profile_context]


def _room(members: dict[str, str]) -> Room:
    return Room(
        call_id="CALL-1",
        host_id=7,
        host_session_id="LOCUS-TEST",
        poi_id=1,
        poi_public_id="poi-1",
        poi_name="Recinto Ferial",
        language="es-ES",
        routing_profile="voice.call.local",
        members={
            key: Member(native_id=int(key), wire_id=index + 1, display_name=name)
            for index, (key, name) in enumerate(members.items())
        },
    )


def _context(session: _FakeSession, room: Room) -> str:
    # Sin construir el puente entero: el metodo solo usa la sesion y la sala.
    return asyncio.run(_CallVoiceBridge._traveler_context(None, session, room))


def test_nothing_is_added_when_nobody_filled_it_in() -> None:
    """The common case, and it has to cost nothing.

    A heading announcing a profile that is not there, or an instruction to
    connect with interests nobody stated, is worse than silence: it invites the
    guide to invent an audience. It is also billed on every single turn.
    """
    room = _room(members={"7": "Carlos"})

    assert _context(_FakeSession(users=[]), room) == ""


def test_only_the_travellers_who_wrote_something_appear() -> None:
    room = _room(members={"7": "Carlos", "9": "Ana"})
    session = _FakeSession(
        users=[
            _User(id=7, preferred_name="Carlos", profile_context="fan de One Piece"),
            _User(id=9, preferred_name="Ana"),  # no ha escrito nada
        ]
    )

    rendered = _context(session, room)

    assert "Carlos: fan de One Piece" in rendered
    assert "Ana" not in rendered


def test_the_block_says_whose_profile_it_is() -> None:
    rendered = TRAVELER_CONTEXT_BLOCK.format(travelers="- Carlos: fan de One Piece")

    # Sin encabezado, el modelo recibe dos lineas sueltas sin saber de quien son.
    assert "Perfil de las personas que atienden esta visita" in rendered
    assert "- Carlos: fan de One Piece" in rendered


def test_the_block_tells_the_guide_to_hold_back() -> None:
    # Lo que arruinaria la funcion no es que no la use, es que la use en cada
    # frase: un guia que mete One Piece cada dos parrafos es peor que uno que no
    # sabe nada de ti.
    rendered = TRAVELER_CONTEXT_BLOCK.format(travelers="- Carlos: fan de One Piece")

    assert "con medida" in rendered
    assert "el protagonista sigue siendo el lugar" in rendered


def test_the_prompt_no_longer_carries_the_placeholder() -> None:
    # Si volviera al prompt, editarlo desde el panel volveria a poder borrarlo.
    from locus_v2.entrypoints.seed import CALL_GUIDE_PROMPT

    assert "{traveler_context}" not in CALL_GUIDE_PROMPT


def test_the_prompt_still_renders_without_the_variable() -> None:
    from locus_v2.entrypoints.seed import CALL_GUIDE_PROMPT
    from locus_v2.shared.prompting import render_prompt

    rendered = render_prompt(
        CALL_GUIDE_PROMPT,
        {"locale": "es-ES", "poi_name": "Recinto Ferial", "traveler_context": ""},
    )

    assert "Recinto Ferial" in rendered
