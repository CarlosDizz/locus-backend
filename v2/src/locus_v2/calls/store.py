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
                    # A state hint makes reconnecting subscribers read the latest committed room.
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
