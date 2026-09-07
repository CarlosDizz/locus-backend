"""One CAS transaction orders state, bounded commands, and subscriber notifications.

Streams carry accepted input across API/realtime processes. Pub/sub is only a
live fan-out, never the source of truth for authorization or room state.
"""

import json
import re
from collections.abc import Callable

from redis.asyncio import Redis
from redis.exceptions import WatchError

from locus_v2.calls.models import CallError, Room, now


def _observable(room: Room) -> str:
    """Everything a connected client can see, as one comparable value.

    Deliberately built from exactly what api/calls.py sends on a "state" event
    (`room.snapshot()` plus that member's `room.ui()`), so if this is unchanged
    the message a client would receive is byte-identical to the one it already
    has. Comparing the pair rather than just the snapshot matters: `ready` gates
    every control in `ui()` but is not part of the snapshot, and skipping its
    flip would leave the room's buttons disabled forever.
    """
    return json.dumps(
        [
            room.snapshot(),
            {member.native_id: room.ui(member.native_id) for member in room.members.values()},
        ],
        sort_keys=True,
        default=str,
    )


class RoomStore:
    def __init__(self, redis: Redis, namespace: str = "locus:v2:calls") -> None:
        self.redis = redis
        self.namespace = namespace

    def key(self, call_id: str, suffix: str = "room") -> str:
        call_id = call_id.upper()
        if not re.fullmatch(r"CALL-[A-F0-9]{12}", call_id):
            raise CallError("Call not found", 404)
        return f"{self.namespace}:{{{call_id}}}:{suffix}"

    async def create(self, room: Room) -> bool:
        return bool(
            await self.redis.set(
                self.key(room.call_id),
                room.model_dump_json(),
                nx=True,
                ex=7500,
            )
        )

    async def put_image(self, call_id: str, image_id: str, data_url: str, ttl: int) -> None:
        """Park a shared photo outside the room state.

        The payload must never live in the Room: `change()` rewrites and
        re-publishes the whole room on every event (heartbeats included), and
        api/calls.py answers each of those by pushing a full snapshot to every
        member. Measured 2026-09-07 — one 468 KB photo took the room from 5.4 KB
        to 629 KB, so a 3-person call moved ~1.9 MB per heartbeat round and the
        websocket sends stalled with no error. Here the bytes are fetched once,
        by URL, on demand.
        """
        await self.redis.set(self.key(call_id, f"image:{image_id}"), data_url, ex=ttl)

    async def drop_image(self, call_id: str, image_id: str) -> None:
        if re.fullmatch(r"[a-f0-9]{32}", image_id):
            await self.redis.delete(self.key(call_id, f"image:{image_id}"))

    async def get_image(self, call_id: str, image_id: str) -> str | None:
        if not re.fullmatch(r"[a-f0-9]{32}", image_id):
            raise CallError("Image not found", 404)
        raw = await self.redis.get(self.key(call_id, f"image:{image_id}"))
        return str(raw) if raw is not None else None

    async def get(self, call_id: str) -> Room:
        raw = await self.redis.get(self.key(call_id))
        if raw is None:
            raise CallError("Call not found or expired", 404)
        return Room.model_validate_json(raw)

    async def change(
        self,
        call_id: str,
        mutate: Callable[[Room, list[dict], list[dict]], None],
    ) -> Room:
        key = self.key(call_id)
        for _ in range(30):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    if raw is None:
                        raise CallError("Call not found or expired", 404)
                    room = Room.model_validate_json(raw)
                    observable_before = _observable(room)
                    commands = room.expire()
                    events: list[dict] = []
                    mutate(room, commands, events)
                    if commands and await pipe.xlen(self.key(call_id, "commands")) > 128:
                        raise CallError("Realtime input queue is full", 429)
                    ttl = max(60, int(room.expires_at - now()) + 300)
                    pipe.multi()
                    pipe.set(key, room.model_dump_json(), ex=ttl)
                    for command in commands:
                        pipe.xadd(self.key(call_id, "commands"), {"data": json.dumps(command)})
                    if commands:
                        pipe.expire(self.key(call_id, "commands"), ttl)
                    if room.status == "ended":
                        events.append({"type": "call.ended"})
                    # A state hint makes subscribers re-read and re-render the room,
                    # so it is only worth sending when what they can actually see
                    # changed. Most accepted events change nothing visible: an
                    # audio chunk only bumps `audio_bytes`, a heartbeat only bumps
                    # `seen_at`, and neither appears in snapshot() or ui(). Measured
                    # 2026-09-07 on a real 3-member call: 35 audio chunks produced 39
                    # full snapshots *per member*, 341 KB/s of pure re-broadcast, and
                    # it grows with members x event rate x log length.
                    if _observable(room) != observable_before:
                        events.append({"type": "state"})
                    for event in events:
                        pipe.publish(self.key(call_id, "events"), json.dumps(event))
                    await pipe.execute()
                    return room
                except WatchError:
                    continue
        raise CallError("Room is busy; retry", 409)

    async def refresh(self, call_id: str) -> Room:
        return await self.change(call_id, lambda *_: None)

    async def publish(self, call_id: str, event: dict) -> None:
        await self.redis.publish(self.key(call_id, "events"), json.dumps(event))

    async def commands(self, call_id: str):
        key = self.key(call_id, "commands")
        while True:
            batches = await self.redis.xread({key: "0-0"}, count=1, block=1000)
            for _, entries in batches:
                for entry_id, fields in entries:
                    yield json.loads(fields["data"])
                    await self.redis.xdel(key, entry_id)
