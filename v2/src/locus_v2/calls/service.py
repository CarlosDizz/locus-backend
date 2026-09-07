import base64
import binascii
import secrets
from collections.abc import Awaitable, Callable
from uuid import uuid4

import jwt

from locus_v2.calls.models import CallError, CreateCall, Member, Room, now
from locus_v2.calls.store import RoomStore
from locus_v2.config import Settings
from locus_v2.identity.models import User
from locus_v2.shared.mobile_ids import mobile_id

MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_AUDIO_BYTES = 24000 * 2 * 120
# How many shared photos stay viewable (and stored) in one call. Room.log already
# self-trims to 80 entries, but photo payloads live in their own Redis keys and
# would outlive that window, orphaned, until the room expires.
MAX_RETAINED_PHOTOS = 20


def decode_audio(value: object) -> bytes:
    if not isinstance(value, str) or not value or len(value) > 131072:
        raise CallError("invalid_audio")
    try:
        audio = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise CallError("invalid_audio") from error
    if len(audio) % 2:
        raise CallError("invalid_audio")
    return audio


def decode_image(value: object) -> tuple[str, bytes]:
    if not isinstance(value, str) or len(value) > MAX_IMAGE_BYTES * 4 // 3 + 100:
        raise CallError("invalid_image")
    header, _, data = value.partition(",")
    signatures = {
        "data:image/jpeg;base64": b"\xff\xd8\xff",
        "data:image/png;base64": b"\x89PNG\r\n\x1a\n",
        "data:image/webp;base64": b"RIFF",
    }
    if header not in signatures:
        raise CallError("invalid_image")
    try:
        decoded = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as error:
        raise CallError("invalid_image") from error
    if not decoded.startswith(signatures[header]) or len(decoded) > MAX_IMAGE_BYTES:
        raise CallError("invalid_image")
    if header == "data:image/webp;base64" and decoded[8:12] != b"WEBP":
        raise CallError("invalid_image")
    # Full media type ("image/jpeg"), not the subtype - callers must not re-prefix.
    return header[5:-7], decoded


class CallService:
    def __init__(
        self,
        store: RoomStore,
        settings: Settings,
        consume: Callable[[int], Awaitable[None]],
    ) -> None:
        self.store = store
        self.settings = settings
        self.consume = consume

    def join_token(self, room: Room, user_id: int) -> str:
        return jwt.encode(
            {
                "iss": self.settings.jwt_issuer,
                "aud": "locus-call-room",
                "sub": str(user_id),
                "call_id": room.call_id,
                "nonce": room.member(user_id).nonce,
                "iat": int(now()),
                "exp": min(int(now()) + 600, int(room.expires_at)),
            },
            self.settings.jwt_secret.get_secret_value(),
            algorithm="HS256",
        )

    def verify_token(self, token: str, call_id: str, user_id: int) -> dict:
        try:
            claims = jwt.decode(
                token,
                self.settings.jwt_secret.get_secret_value(),
                algorithms=["HS256"],
                audience="locus-call-room",
                issuer=self.settings.jwt_issuer,
                options={"require": ["exp", "iat", "iss", "aud", "sub", "call_id", "nonce"]},
            )
        except jwt.PyJWTError as error:
            raise CallError("Invalid or expired join token", 403) from error
        if claims["call_id"] != call_id.upper() or claims["sub"] != str(user_id):
            raise CallError("Join token does not match this user and room", 403)
        return claims

    def image_token(self, call_id: str, image_id: str, expires_at: float) -> str:
        return jwt.encode(
            {
                "iss": self.settings.jwt_issuer,
                "aud": "locus-call-image",
                "call_id": call_id.upper(),
                "image_id": image_id,
                "iat": int(now()),
                "exp": int(expires_at),
            },
            self.settings.jwt_secret.get_secret_value(),
            algorithm="HS256",
        )

    def verify_image_token(self, token: str, call_id: str, image_id: str) -> None:
        """Authorize a photo fetch by capability rather than by session.

        The URL is embedded in the room log, so it reaches exactly the people
        who can already see the call transcript — and an <img src> cannot carry
        a bearer header, so the token has to travel in the URL itself.
        """
        try:
            claims = jwt.decode(
                token,
                self.settings.jwt_secret.get_secret_value(),
                algorithms=["HS256"],
                audience="locus-call-image",
                issuer=self.settings.jwt_issuer,
                options={"require": ["exp", "iat", "iss", "aud", "call_id", "image_id"]},
            )
        except jwt.PyJWTError as error:
            raise CallError("Invalid or expired image token", 403) from error
        if claims["call_id"] != call_id.upper() or claims["image_id"] != image_id:
            raise CallError("Image token does not match this room", 403)

    @staticmethod
    def active(room: Room) -> None:
        if room.status == "ended":
            raise CallError("Call ended", 410)

    async def create(self, user: User, request: CreateCall, poi, routing_profile: str) -> Room:
        await self.consume(user.id)
        for _ in range(5):
            room = Room(
                call_id=f"CALL-{secrets.token_hex(6).upper()}",
                host_id=user.id,
                host_session_id=request.session_id.upper(),
                poi_id=mobile_id(poi),
                poi_public_id=poi.public_id,
                poi_name=poi.name,
                language=request.language,
                routing_profile=routing_profile,
                members={str(user.id): Member.from_user(user)},
            )
            if await self.store.create(room):
                return room
        raise CallError("Unable to allocate a call ID", 503)

    async def join(self, call_id: str, user: User) -> Room:
        def change(room, commands, events):
            self.active(room)
            if str(user.id) not in room.members:
                if len(room.members) >= room.max_members:
                    raise CallError("Call is full", 409)
                room.members[str(user.id)] = Member.from_user(user)
            room.member(user.id).reservation_until = now() + 600
            # Token issuance must not resume the host's disconnected call.

        return await self.store.change(call_id, change)

    async def connect(self, call_id: str, user_id: int, token: str, connection: str) -> Room:
        claims = self.verify_token(token, call_id, user_id)

        def change(room, commands, events):
            self.active(room)
            member = room.member(user_id)
            if member.nonce != claims["nonce"]:
                raise CallError("Membership was revoked", 403)
            member.connection = connection
            member.seen_at = now()
            if user_id == room.host_id:
                if room.status == "paused":
                    room.status = "idle"
                room.host_grace_deadline = None

        return await self.store.change(call_id, change)

    async def heartbeat(self, call_id: str, user_id: int, connection: str) -> Room:
        def change(room, commands, events):
            self.connected(room, user_id, connection)
            room.member(user_id).seen_at = now()

        return await self.store.change(call_id, change)

    @staticmethod
    def connected(room: Room, user_id: int, connection: str) -> None:
        CallService.active(room)
        if room.member(user_id).connection != connection:
            raise CallError("Connection was replaced or membership revoked", 403)

    async def disconnect(self, call_id: str, user_id: int, connection: str) -> Room:
        def change(room, commands, events):
            member = room.members.get(str(user_id))
            if room.status != "ended" and member and member.connection == connection:
                commands.extend(room.disconnect(user_id))

        return await self.store.change(call_id, change)

    async def leave(self, call_id: str, user_id: int, *, end: bool = False) -> Room:
        def change(room, commands, events):
            room.member(user_id)
            if end and user_id != room.host_id:
                raise CallError("Only the host can end a call", 403)
            if user_id == room.host_id:
                room.end()
            elif room.status != "ended":
                commands.extend(room.disconnect(user_id))
                room.members.pop(str(user_id))

        return await self.store.change(call_id, change)

    async def mark_ready(self, call_id: str) -> Room:
        """Flip Room.ready once the AI bridge has a live provider connection.

        Room.ui() gates can_talk/can_text/can_image on `ready` (reason
        "provider_connecting" until then) — nothing else in this domain ever
        sets it, so a call's controls stayed disabled forever until the AI
        bridge (calls/bridge.py) called this after connecting.
        """

        def change(room, commands, events):
            if room.status != "ended":
                room.ready = True

        return await self.store.change(call_id, change)

    async def log_user_voice(self, call_id: str, text: str) -> Room:
        """Record what the current speaker actually said, once the provider transcribes it.

        event()'s audio.chunk/audio.commit branches only move floor state and
        forward raw audio to the provider — nothing appended a log entry for a
        voice turn, so the shared log showed the assistant answering out of
        nowhere with no record of the question that prompted it. The bridge
        calls this from ProviderEventType.INPUT_TRANSCRIPT_DONE once Gemini
        hands back the transcript; room.speaker_id (read fresh in this
        transaction) attributes it to whoever was actually holding the floor.
        """

        def change(room, commands, events):
            if room.status != "ended" and text:
                room.append_log("user-voice", text, room.speaker_id)

        return await self.store.change(call_id, change)

    async def assistant_finished(self, call_id: str, text: str) -> Room:
        """Close out the assistant's turn once the live provider stops speaking.

        `event()` moves a room into "assistant_speaking" but has no reason to know
        when the provider is done — that only happens once the AI bridge consuming
        the command stream sees AUDIO_DONE, so it calls back in here.
        """

        def change(room, commands, events):
            if room.status == "ended":
                return
            room.status = "idle"
            room.speaker_id = None
            if text:
                room.append_log("ai", text)  # matches call.page.ts's trackLabel(), not a free label
            events.append({"type": "assistant.done", "text": text})

        return await self.store.change(call_id, change)

    async def event(self, call_id: str, user_id: int, connection: str, event: dict) -> Room:
        kind = event.get("type")
        if kind == "call.leave":
            self.connected(await self.store.get(call_id), user_id, connection)
            return await self.leave(call_id, user_id)
        if kind not in {
            "floor.request",
            "floor.release",
            "audio.chunk",
            "audio.commit",
            "text.submit",
            "image.submit",
        }:
            raise CallError("Unsupported event")
        initial = await self.store.get(call_id)
        self.connected(initial, user_id, connection)
        if kind in {"floor.request", "audio.commit", "text.submit", "image.submit"}:
            await self.consume(initial.host_id)
        audio = decode_audio(event.get("audio")) if kind == "audio.chunk" else b""
        image_id = ""
        image_url = ""
        if kind == "image.submit":
            decode_image(event.get("image_data_url"))
            # Park the payload before the CAS transaction, so what lands in the
            # room (and therefore in every snapshot afterwards) is just a URL.
            image_id = uuid4().hex
            ttl = max(60, int(initial.expires_at - now()) + 300)
            await self.store.put_image(
                call_id, image_id, str(event["image_data_url"]), ttl
            )
            # Absolute on purpose. V1 puts a data: URL here and the Ionic app
            # feeds it straight to <img src> — an absolute http(s) URL works in
            # exactly the same place, so V2 fixes the room-state bloat without
            # the app having to change. That matters: the whole cutover plan is
            # "flip apiBaseUrl, keep the app identical, roll back by flipping
            # it again", and a relative path would have broken that.
            image_url = (
                f"{self.settings.public_api_base_url.rstrip('/')}"
                f"/calls/{call_id.upper()}/images/{image_id}"
                f"?t={self.image_token(call_id, image_id, initial.expires_at)}"
            )
        # Photo payloads are the only unbounded thing a call can accumulate: the
        # log self-trims to 80 entries but their Redis keys would outlive it,
        # orphaned and unreachable, until the room's TTL. Measured 2026-09-07:
        # 20 typical photos already cost 13.2 MB of Redis per call, and at the
        # 2 MB per-image ceiling nothing stopped a single call reaching hundreds
        # of MB. Keeping the last MAX_RETAINED_PHOTOS caps a call at ~12 MB of
        # typical photos (40 MB of worst-case ones) while still being far more
        # than a guided visit needs.
        evicted: list[str] = []
        accepted = False

        def evict_old_photos(room: Room) -> None:
            nonlocal accepted
            # The CAS block can re-run, so recompute both from scratch.
            evicted.clear()
            accepted = True
            photos = [entry for entry in room.log if entry.get("image_url")]
            for entry in photos[: max(0, len(photos) - MAX_RETAINED_PHOTOS)]:
                evicted.append(str(entry["image_url"]).rsplit("/", 1)[-1].split("?", 1)[0])
                entry["image_url"] = None

        text = event.get("text", "")
        if kind == "text.submit" and (
            not isinstance(text, str) or not text.strip() or len(text) > 8000
        ):
            raise CallError("invalid_text")

        def change(room, commands, events):
            self.connected(room, user_id, connection)
            room.member(user_id).seen_at = now()
            ui = room.ui(user_id)
            target = {"target": user_id}
            if kind == "floor.request":
                if not ui["can_talk"] or room.status == "user_speaking":
                    events.append({"type": "floor.denied", "reason": ui["reason"], **target})
                    return
                if room.status == "assistant_speaking":
                    commands.append({"type": "reset"})
                    events.append(
                        {"type": "assistant.interrupted", "by_user_id": room.wire_id(user_id)}
                    )
                room.turn_id = uuid4().hex
                room.turn_user_id = room.speaker_id = user_id
                room.audio_bytes = 0
                room.status = "user_speaking"
                events.append({"type": "floor.granted", **target})
            elif kind in {"floor.release", "audio.chunk", "audio.commit"}:
                if room.status != "user_speaking" or room.speaker_id != user_id:
                    raise CallError("You do not hold the floor", 403)
                if kind == "floor.release":
                    room.status = "idle"
                    room.speaker_id = None
                    room.turn_id = uuid4().hex
                    commands.append({"type": "reset"})
                    events.append({"type": "floor.released", **target})
                elif kind == "audio.chunk":
                    if room.audio_bytes + len(audio) > MAX_AUDIO_BYTES:
                        raise CallError("Audio turn exceeds 120 seconds", 413)
                    room.audio_bytes += len(audio)
                    commands.append({"type": kind, "audio": event["audio"], "turn": room.turn_id})
                    events.append(
                        {
                            "type": "peer_audio.chunk",
                            "audio": event["audio"],
                            "user_id": room.wire_id(user_id),
                            "exclude": user_id,
                        }
                    )
                else:
                    if room.audio_bytes < 4800:
                        raise CallError("Audio turn must contain at least 100 ms", 422)
                    room.status = "assistant_speaking"
                    room.speaker_id = None
                    commands.append({"type": kind, "turn": room.turn_id})
                    events.append({"type": "assistant.started"})
            else:
                capability = "can_text" if kind == "text.submit" else "can_image"
                if not ui[capability]:
                    events.append({"type": "message.rejected", "reason": ui["reason"], **target})
                    return
                room.turn_id = uuid4().hex
                room.turn_user_id = user_id
                room.status = "assistant_speaking"
                command = {
                    "type": kind,
                    "turn": room.turn_id,
                    "author": room.member(user_id).display_name,
                }
                if kind == "text.submit":
                    command["text"] = text.strip()
                    room.append_log("user-text", text.strip(), user_id)
                else:
                    # The bridge fetches the bytes by id; keeping them out of the
                    # command stream too means one 250 KB xadd less per photo.
                    command["image_id"] = image_id
                    room.append_log("user-photo", "Photo", user_id, image_url)
                    evict_old_photos(room)
                commands.append(command)
                events.append({"type": "assistant.started"})

        room = await self.store.change(call_id, change)
        # All of this only after the room commit succeeded: acting on a
        # rolled-back CAS attempt would blank out photos that are still live.
        if kind == "image.submit" and not accepted:
            # The room refused this photo (someone else holds the turn). Its
            # payload was written before the transaction so the bridge could
            # never read a command whose bytes had not landed yet - so a
            # refusal has to take the bytes back out, or a member who keeps
            # tapping the camera while the guide talks fills Redis with
            # payloads nothing will ever reference.
            await self.store.drop_image(call_id, image_id)
        for orphan in evicted:
            await self.store.drop_image(call_id, orphan)
        return room
