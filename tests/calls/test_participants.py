"""Quien sale de la llamada antes de que acabe sigue pudiendo valorarla.

Al salir, la sala borra al invitado de `members`, y la valoración comprobaba
justo esa lista: en producción (2026-10-10) un invitado que salió con «Salir»
recibió tres 403 al enviar su valoración. `participants` recuerda a todos los
que han entrado.
"""

import pytest

from locus_v2.calls.models import Member, Room
from locus_v2.calls.service import CallService
from locus_v2.identity.models import User


class FakeStore:
    def __init__(self, room: Room) -> None:
        self.room = room

    async def change(self, call_id: str, mutate) -> Room:
        mutate(self.room, [], [])
        return self.room


def build_room() -> Room:
    return Room(
        call_id="CALL-TEST",
        host_id=1,
        host_session_id="LOCUS-TEST",
        poi_id=1,
        poi_public_id="poi-1",
        poi_name="Museo",
        language="es",
        routing_profile="default",
        members={"1": Member(native_id=1, wire_id=1, display_name="Carlos")},
        participants=["1"],
    )


@pytest.mark.asyncio
async def test_guest_who_leaves_stays_a_participant() -> None:
    room = build_room()
    service = object.__new__(CallService)
    service.store = FakeStore(room)
    guest = User(id=2, public_id="usr-2", email="ana@example.com", display_name="Ana")

    await service.join("CALL-TEST", guest)
    await service.leave("CALL-TEST", 2)

    assert "2" not in room.members
    assert room.participants == ["1", "2"]


def test_rooms_saved_before_the_change_still_load() -> None:
    raw = build_room().model_dump()
    raw.pop("participants")

    assert Room.model_validate(raw).participants == []
