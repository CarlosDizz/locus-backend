"""The assistant finishing must not take the floor away from whoever interrupted.

A user who interrupts is granted the floor straight away, but the provider keeps
emitting for a moment and the bridge's `assistant_finished` callback lands after
that. It used to clear the floor unconditionally, so every audio chunk the
interrupting user was already streaming came back as "You do not hold the floor"
and the guide never heard the question. In production this looked like: the
guide stops talking, and then ignores you.
"""

import pytest

from locus_v2.calls.models import Member, Room
from locus_v2.calls.service import CallService


class FakeStore:
    """Applies the mutation to one in-memory room, like the real CAS store would."""

    def __init__(self, room: Room) -> None:
        self.room = room

    async def change(self, call_id: str, mutate) -> Room:
        mutate(self.room, [], [])
        return self.room


def build_room(status: str, speaker_id: int | None) -> Room:
    return Room(
        call_id="CALL-TEST",
        host_id=1,
        host_session_id="LOCUS-TEST",
        poi_id=1,
        poi_public_id="poi-1",
        poi_name="Museo",
        language="es",
        routing_profile="default",
        status=status,
        speaker_id=speaker_id,
        members={"1": Member(native_id=1, wire_id=1, display_name="Carlos")},
    )


def service_for(room: Room) -> CallService:
    service = object.__new__(CallService)
    service.store = FakeStore(room)
    return service


@pytest.mark.asyncio
async def test_interrupting_user_keeps_the_floor() -> None:
    room = build_room("user_speaking", speaker_id=1)
    service = service_for(room)

    await service.assistant_finished("CALL-TEST", "lo que iba diciendo")

    assert room.status == "user_speaking"
    assert room.speaker_id == 1
    assert room.log[-1]["kind"] == "ai"


@pytest.mark.asyncio
async def test_assistant_turn_still_closes_when_nobody_interrupted() -> None:
    room = build_room("assistant_speaking", speaker_id=None)
    service = service_for(room)

    await service.assistant_finished("CALL-TEST", "respuesta completa")

    assert room.status == "idle"
    assert room.speaker_id is None


@pytest.mark.asyncio
async def test_ended_call_is_left_alone() -> None:
    room = build_room("ended", speaker_id=None)
    service = service_for(room)

    await service.assistant_finished("CALL-TEST", "texto")

    assert room.status == "ended"
    assert room.log == []
