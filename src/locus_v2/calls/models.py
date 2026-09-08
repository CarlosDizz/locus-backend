from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from locus_v2.identity.models import User
from locus_v2.shared.mobile_ids import mobile_id


def now() -> float:
    return datetime.now(UTC).timestamp()


def iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


class CallError(ValueError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class CreateCall(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    poi_id: int | str | None = None
    poi_name: str = Field(default="", max_length=300)
    language: str = Field(default="es", min_length=2, max_length=16)


class Member(BaseModel):
    native_id: int
    wire_id: int
    display_name: str
    avatar_url: str = ""
    joined_at: float = Field(default_factory=now)
    reservation_until: float = Field(default_factory=lambda: now() + 600)
    nonce: str = Field(default_factory=lambda: uuid4().hex)
    connection: str | None = None
    seen_at: float = 0

    @classmethod
    def from_user(cls, user: User) -> "Member":
        return cls(
            native_id=user.id,
            wire_id=mobile_id(user),
            display_name=user.display_name,
            avatar_url=user.avatar_url or "",
        )


class Room(BaseModel):
    call_id: str
    host_id: int
    host_session_id: str
    poi_id: int
    poi_public_id: str
    poi_name: str
    language: str
    routing_profile: str
    status: Literal["idle", "user_speaking", "assistant_speaking", "paused", "ended"] = "idle"
    speaker_id: int | None = None
    turn_user_id: int | None = None
    turn_id: str = ""
    audio_bytes: int = 0
    members: dict[str, Member]
    max_members: int = 10
    expires_at: float = Field(default_factory=lambda: now() + 7200)
    host_grace_deadline: float | None = None
    owner: str | None = None
    owner_deadline: float = 0
    ready: bool = False
    log: list[dict] = Field(default_factory=list)

    def member(self, user_id: int) -> Member:
        member = self.members.get(str(user_id))
        if member is None:
            raise CallError("Call membership required", 403)
        return member

    def wire_id(self, native_id: int | None) -> int | None:
        member = self.members.get(str(native_id))
        return member.wire_id if member else None

    def snapshot(self) -> dict:
        host = self.member(self.host_id)
        return {
            "call_id": self.call_id,
            "join_code": self.call_id,
            "host_user_id": host.wire_id,
            "host_display_name": host.display_name,
            "host_session_id": self.host_session_id,
            "poi_id": self.poi_id,
            "poi_name": self.poi_name,
            "language": self.language,
            "status": self.status,
            "speaker_user_id": self.wire_id(self.speaker_id),
            "max_members": self.max_members,
            "member_count": len(self.members),
            "participants": [
                {
                    "user_id": m.wire_id,
                    "display_name": m.display_name,
                    "avatar_url": m.avatar_url,
                    "is_host": m.native_id == self.host_id,
                    "connected": m.connection is not None,
                    "joined_at": iso(m.joined_at),
                }
                for m in sorted(self.members.values(), key=lambda member: member.joined_at)
            ],
            "host_grace_deadline": (
                iso(self.host_grace_deadline) if self.host_grace_deadline else None
            ),
            "expires_at": iso(self.expires_at),
            "log": self.log[-80:],
        }

    def ui(self, user_id: int) -> dict:
        result = dict(
            can_talk=False, can_text=False, can_image=False, can_interrupt=False, reason=""
        )
        member = self.members.get(str(user_id))
        if self.status == "ended":
            result["reason"] = "call_ended"
        elif member is None or not member.connection:
            result["reason"] = "not_connected"
        elif self.status == "paused" or not self.member(self.host_id).connection:
            result["reason"] = "paused_waiting_host"
        elif not self.ready:
            result["reason"] = "provider_connecting"
        elif self.status == "idle":
            result.update(can_talk=True, can_text=True, can_image=True)
        elif self.status == "user_speaking":
            result.update(
                can_talk=self.speaker_id == user_id,
                reason="you_hold_floor" if self.speaker_id == user_id else "another_user_speaking",
            )
        elif self.status == "assistant_speaking":
            result.update(
                can_talk=user_id == self.host_id,
                can_interrupt=user_id == self.host_id,
                reason="host_interrupt_enabled"
                if user_id == self.host_id
                else "assistant_speaking",
            )
        return result

    def append_log(
        self, kind: str, text: str, user_id: int | None = None, image_url: str | None = None
    ) -> None:
        member = self.members.get(str(user_id))
        self.log.append(
            {
                "id": uuid4().hex,
                "kind": kind,
                "author": member.display_name if member else "Locus",
                "text": text,
                "user_id": member.wire_id if member else None,
                "image_url": image_url,
                "timestamp": iso(now()),
            }
        )
        self.log = self.log[-80:]

    def end(self) -> None:
        self.status = "ended"
        self.ready = False
        self.speaker_id = None
        self.host_grace_deadline = None

    def disconnect(self, user_id: int) -> list[dict]:
        member = self.member(user_id)
        member.connection = None
        member.reservation_until = now() + 600
        commands = []
        if user_id == self.host_id:
            self.status = "paused"
            self.host_grace_deadline = now() + 90
            self.speaker_id = None
            self.turn_id = uuid4().hex
            commands.append({"type": "reset"})
        elif self.speaker_id == user_id:
            self.status = "idle"
            self.speaker_id = None
            self.turn_id = uuid4().hex
            commands.append({"type": "reset"})
        return commands

    def expire(self) -> list[dict]:
        if self.status == "ended":
            return []
        timestamp = now()
        commands = []
        for key, member in list(self.members.items()):
            if member.connection and member.seen_at + 20 < timestamp:
                commands.extend(self.disconnect(member.native_id))
            if (
                not member.connection
                and member.native_id != self.host_id
                and member.reservation_until < timestamp
            ):
                self.members.pop(key)
        if (
            self.expires_at <= timestamp
            or (self.host_grace_deadline and self.host_grace_deadline <= timestamp)
            or (self.owner and self.owner_deadline <= timestamp)
        ):
            self.end()
        return commands
