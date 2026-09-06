"""V1-compatible map session service, ported from app/services/session_service.py.

Deliberately DB-backed (one row per session_id) rather than in-memory, same
as V1 — this was never the hard part of Capitulo 6. What's still open is the
call-room orchestration and the /ws/calls/{id} <-> /ws/v2/live protocol
bridge (see voice/gateway.py and docs/testing-checklist.md Capitulo 6),
which this file intentionally does not touch.

set_nearby_pois / set_active_poi / set_ephemeral_map_pois exist here (as in
V1) purely as state setters for a future Chat domain to call — no geo
lookup lives in this service, matching V1's exact division of labor.
"""

import secrets
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.identity.models import User
from locus_v2.sessions.models import (
    MapSession,
    SessionCallLiveState,
    SessionCallLogEntryState,
    SessionParticipantState,
    SessionPoi,
    SessionStateView,
)

PARTICIPANTS_KEY = "participants"
CALL_LIVE_KEY = "call_live"
CALL_LOG_KEY = "call_log"
PARTICIPANT_STALE_AFTER_SECONDS = 120
MAX_CALL_LOG_ENTRIES = 120
MAX_MEMORY_ENTRIES = 30
SESSION_ID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


class SessionNotFoundError(RuntimeError):
    pass


def generate_session_id(length: int = 6) -> str:
    return "".join(secrets.choice(SESSION_ID_ALPHABET) for _ in range(length))


class MapSessionService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_session(
        self,
        *,
        session_id: str | None,
        user_id: int | None,
        profile_context: str,
        lat: float | None,
        lng: float | None,
        metadata: dict[str, Any] | None,
    ) -> SessionStateView:
        resolved_id = (session_id or generate_session_id()).upper()
        row = await self.session.get(MapSession, resolved_id)
        if row is None:
            row = MapSession(
                session_id=resolved_id,
                user_id=user_id,
                profile_context=profile_context,
                lat=_as_decimal(lat),
                lng=_as_decimal(lng),
                metadata_json=metadata or {},
            )
            self.session.add(row)
        else:
            if row.user_id is None and user_id is not None:
                row.user_id = user_id
            row.profile_context = profile_context
            if lat is not None:
                row.lat = _as_decimal(lat)
            if lng is not None:
                row.lng = _as_decimal(lng)
            if metadata:
                row.metadata_json = {**(row.metadata_json or {}), **metadata}
        await self.session.commit()
        await self.session.refresh(row)
        return _serialize(row)

    async def get_session(self, session_id: str) -> SessionStateView | None:
        row = await self.session.get(MapSession, session_id.upper())
        return _serialize(row) if row is not None else None

    async def get_or_create(self, session_id: str) -> SessionStateView:
        row = await self._get_or_create(session_id)
        return _serialize(row)

    async def update_session(
        self,
        session_id: str,
        *,
        user_id: int | None,
        profile_context: str | None,
        profile_preferences: dict[str, Any] | None,
        lat: float | None,
        lng: float | None,
        active_poi_name: str | None,
        metadata: dict[str, Any] | None,
    ) -> SessionStateView:
        row = await self._get_or_create(session_id)
        if user_id is not None and row.user_id is None:
            row.user_id = user_id
        if profile_context is not None:
            row.profile_context = profile_context
        if profile_preferences is not None:
            row.profile_preferences_json = profile_preferences
        if lat is not None:
            row.lat = _as_decimal(lat)
        if lng is not None:
            row.lng = _as_decimal(lng)
        if active_poi_name:
            current = dict(row.active_poi_json or {})
            current["name"] = active_poi_name
            if current.get("lat") is None and row.lat is not None:
                current["lat"] = float(row.lat)
            if current.get("lng") is None and row.lng is not None:
                current["lng"] = float(row.lng)
            row.active_poi_json = current
        if metadata:
            row.metadata_json = {**(row.metadata_json or {}), **metadata}
        await self.session.commit()
        await self.session.refresh(row)
        return _serialize(row)

    async def reset_conversation(self, session_id: str) -> SessionStateView:
        row = await self.session.get(MapSession, session_id.upper())
        if row is None:
            raise SessionNotFoundError(session_id)
        row.memory_json = []
        row.active_poi_json = None
        metadata = dict(row.metadata_json or {})
        metadata["ephemeral_map_pois"] = []
        metadata["last_chat_response_id"] = ""
        row.metadata_json = metadata
        await self.session.commit()
        await self.session.refresh(row)
        return _serialize(row)

    async def append_memory(self, session_id: str, role: str, text: str) -> SessionStateView:
        row = await self._get_or_create(session_id)
        clean_text = text.strip()
        if clean_text:
            memory = [*list(row.memory_json or []), {"role": role, "text": clean_text}]
            row.memory_json = memory[-MAX_MEMORY_ENTRIES:]
        await self.session.commit()
        await self.session.refresh(row)
        return _serialize(row)

    async def set_nearby_pois(self, session_id: str, pois: list[SessionPoi]) -> SessionStateView:
        row = await self._get_or_create(session_id)
        row.nearby_pois_json = [poi.model_dump() for poi in pois]
        await self.session.commit()
        await self.session.refresh(row)
        return _serialize(row)

    async def set_active_poi(
        self, session_id: str, poi: SessionPoi | None
    ) -> SessionStateView:
        row = await self._get_or_create(session_id)
        row.active_poi_json = poi.model_dump() if poi else None
        await self.session.commit()
        await self.session.refresh(row)
        return _serialize(row)

    async def set_ephemeral_map_pois(
        self, session_id: str, pois: list[SessionPoi]
    ) -> SessionStateView:
        row = await self._get_or_create(session_id)
        metadata = dict(row.metadata_json or {})
        metadata["ephemeral_map_pois"] = [poi.model_dump() for poi in pois]
        row.metadata_json = metadata
        await self.session.commit()
        await self.session.refresh(row)
        return _serialize(row)

    async def touch_participant(
        self, session_id: str, user: User, *, active_call: bool
    ) -> SessionStateView:
        row = await self._get_or_create(session_id)
        metadata = dict(row.metadata_json or {})
        participants = _prune_participants(list(metadata.get(PARTICIPANTS_KEY) or []))
        now_iso = _now_iso()

        existing = next((item for item in participants if item["user_id"] == user.id), None)
        payload = {
            "display_name": user.display_name,
            "avatar_url": user.avatar_url or "",
            "last_seen_at": now_iso,
            "status": "present",
            "active_call": active_call,
        }
        if existing is None:
            participants.append(
                {"user_id": user.id, "joined_at": now_iso, **payload}
            )
        else:
            existing.update(payload)

        metadata[PARTICIPANTS_KEY] = participants
        if row.user_id is None:
            row.user_id = user.id
        row.metadata_json = metadata
        await self.session.commit()
        await self.session.refresh(row)
        return _serialize(row)

    async def leave_participant(self, session_id: str, user: User) -> SessionStateView:
        row = await self._get_or_create(session_id)
        metadata = dict(row.metadata_json or {})
        participants = _prune_participants(list(metadata.get(PARTICIPANTS_KEY) or []))
        now_iso = _now_iso()
        for item in participants:
            if item["user_id"] == user.id:
                item["status"] = "left"
                item["active_call"] = False
                item["last_seen_at"] = now_iso

        call_live = dict(metadata.get(CALL_LIVE_KEY) or {})
        if call_live.get("host_user_id") == user.id and call_live.get("status") == "live":
            call_live["status"] = "ended"
            call_live["updated_at"] = now_iso
            metadata[CALL_LIVE_KEY] = call_live

        metadata[PARTICIPANTS_KEY] = participants
        row.metadata_json = metadata
        await self.session.commit()
        await self.session.refresh(row)
        return _serialize(row)

    async def set_call_state(
        self, session_id: str, user: User, status: str
    ) -> SessionStateView:
        normalized = status.strip().lower()
        if normalized not in {"idle", "live", "ended"}:
            normalized = "idle"

        row = await self._get_or_create(session_id)
        metadata = dict(row.metadata_json or {})
        participants = _prune_participants(list(metadata.get(PARTICIPANTS_KEY) or []))
        now_iso = _now_iso()
        for item in participants:
            if item["user_id"] == user.id:
                item["active_call"] = normalized == "live"
                item["last_seen_at"] = now_iso

        current = dict(metadata.get(CALL_LIVE_KEY) or {})
        if normalized == "live":
            current = {
                "status": "live",
                "host_user_id": user.id,
                "host_display_name": user.display_name,
                "started_at": str(current.get("started_at") or now_iso),
                "updated_at": now_iso,
            }
        elif current.get("host_user_id") == user.id or current.get("host_user_id") is None:
            current = {
                "status": normalized,
                "host_user_id": user.id if normalized == "idle" else current.get("host_user_id"),
                "host_display_name": (
                    user.display_name
                    if normalized == "idle"
                    else str(current.get("host_display_name") or user.display_name)
                ),
                "started_at": str(current.get("started_at") or ""),
                "updated_at": now_iso,
            }
        else:
            current["updated_at"] = now_iso

        metadata[PARTICIPANTS_KEY] = participants
        metadata[CALL_LIVE_KEY] = current
        row.metadata_json = metadata
        await self.session.commit()
        await self.session.refresh(row)
        return _serialize(row)

    async def append_call_log(
        self,
        session_id: str,
        *,
        user: User,
        kind: str,
        author: str,
        text: str,
        image_url: str | None = None,
    ) -> SessionStateView:
        clean_text = text.strip()
        if not clean_text:
            existing = await self.get_session(session_id)
            return existing or await self.create_session(
                session_id=session_id,
                user_id=user.id,
                profile_context="",
                lat=None,
                lng=None,
                metadata=None,
            )

        row = await self._get_or_create(session_id)
        metadata = dict(row.metadata_json or {})
        raw_log = list(metadata.get(CALL_LOG_KEY) or [])
        raw_log.append(
            {
                "id": f"cl_{secrets.token_hex(8)}",
                "kind": kind.strip() or "system",
                "author": author.strip() or user.display_name or "Sistema",
                "text": clean_text,
                "timestamp": _now_iso(),
                "image_url": image_url,
                "user_id": user.id,
            }
        )
        metadata[CALL_LOG_KEY] = raw_log[-MAX_CALL_LOG_ENTRIES:]
        row.metadata_json = metadata
        await self.session.commit()
        await self.session.refresh(row)
        return _serialize(row)

    async def _get_or_create(self, session_id: str) -> MapSession:
        resolved_id = session_id.upper()
        row = await self.session.get(MapSession, resolved_id)
        if row is None:
            row = MapSession(session_id=resolved_id)
            self.session.add(row)
            await self.session.flush()
        return row


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _as_decimal(value: float | None) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None


def _prune_participants(raw_participants: list[Any]) -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    participants: list[dict[str, Any]] = []
    for item in raw_participants:
        if not isinstance(item, dict):
            continue
        last_seen_raw = str(item.get("last_seen_at") or item.get("joined_at") or "")
        try:
            last_seen = datetime.fromisoformat(last_seen_raw.replace("Z", "+00:00"))
        except ValueError:
            last_seen = now
        is_recent = now - last_seen <= timedelta(seconds=PARTICIPANT_STALE_AFTER_SECONDS)
        status = "present" if is_recent else "stale"
        user_id = int(item.get("user_id") or 0)
        if user_id <= 0:
            continue
        participants.append(
            {
                "user_id": user_id,
                "display_name": str(item.get("display_name") or ""),
                "avatar_url": str(item.get("avatar_url") or ""),
                "joined_at": str(item.get("joined_at") or last_seen_raw or _now_iso()),
                "last_seen_at": last_seen.isoformat(),
                "status": item.get("status") if item.get("status") == "left" else status,
                "active_call": bool(item.get("active_call")) and is_recent,
            }
        )
    return participants


def _coerce_poi(
    raw: dict[str, Any] | None,
    *,
    fallback_lat: float | None = None,
    fallback_lng: float | None = None,
) -> SessionPoi | None:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    lat_raw = raw.get("lat", fallback_lat)
    lng_raw = raw.get("lng", fallback_lng)
    try:
        lat = float(lat_raw) if lat_raw is not None else None
        lng = float(lng_raw) if lng_raw is not None else None
    except (TypeError, ValueError):
        return None
    if lat is None or lng is None:
        return None
    return SessionPoi(
        id=str(raw.get("id") or ""),
        name=name,
        lat=lat,
        lng=lng,
        poi_type_code=str(raw.get("poi_type_code") or ""),
        description=str(raw.get("description") or ""),
        summary=str(raw.get("summary") or ""),
        source_of_truth=str(raw.get("source_of_truth") or "catalog"),
        is_ephemeral=bool(raw.get("is_ephemeral", False)),
        google_place_id=str(raw.get("google_place_id") or ""),
        context_kind=str(raw.get("context_kind") or "catalog"),
    )


def _serialize(row: MapSession) -> SessionStateView:
    fallback_lat = float(row.lat) if row.lat is not None else None
    fallback_lng = float(row.lng) if row.lng is not None else None
    active_poi = _coerce_poi(
        row.active_poi_json, fallback_lat=fallback_lat, fallback_lng=fallback_lng
    )
    nearby_pois = [
        poi
        for poi in (_coerce_poi(item) for item in (row.nearby_pois_json or []))
        if poi is not None
    ]
    metadata = dict(row.metadata_json or {})
    ephemeral_map_pois = [
        poi
        for poi in (_coerce_poi(item) for item in (metadata.get("ephemeral_map_pois") or []))
        if poi is not None
    ]
    raw_participants = _prune_participants(metadata.get(PARTICIPANTS_KEY) or [])
    participants = [SessionParticipantState(**item) for item in raw_participants]
    call_live_raw = metadata.get(CALL_LIVE_KEY) or {}
    call_live = (
        SessionCallLiveState(**call_live_raw)
        if isinstance(call_live_raw, dict)
        else SessionCallLiveState()
    )
    call_log_raw = metadata.get(CALL_LOG_KEY) or []
    call_log = [
        SessionCallLogEntryState(**item)
        for item in (call_log_raw[-MAX_CALL_LOG_ENTRIES:] if isinstance(call_log_raw, list) else [])
        if isinstance(item, dict)
    ]

    return SessionStateView(
        session_id=row.session_id,
        user_id=row.user_id,
        profile_context=row.profile_context or "",
        profile_language=row.profile_language or "es",
        profile_preferences=row.profile_preferences_json or {},
        lat=fallback_lat,
        lng=fallback_lng,
        active_poi=active_poi,
        nearby_pois=nearby_pois,
        ephemeral_map_pois=ephemeral_map_pois,
        memory=row.memory_json or [],
        metadata=metadata,
        participants=participants,
        call_live=call_live,
        call_log=call_log,
    )
