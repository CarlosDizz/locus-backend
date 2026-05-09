from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import secrets

from sqlalchemy import select

from app.db.models import AppSession
from app.schemas.auth import UserResponse
from app.db.session import session_scope
from app.schemas.poi import POI
from app.schemas.session import (
    SessionCallLive,
    SessionCallLogEntry,
    SessionCreateRequest,
    SessionLocation,
    SessionParticipant,
    SessionProfile,
    SessionState,
    SessionUpdateRequest,
)
from app.utils.ids import generate_session_id


class SessionService:
    PARTICIPANTS_METADATA_KEY = "participants"
    CALL_LIVE_METADATA_KEY = "call_live"
    CALL_LOG_METADATA_KEY = "call_log"
    PARTICIPANT_STALE_AFTER_SECONDS = 120
    MAX_CALL_LOG_ENTRIES = 120

    def _now_iso(self) -> str:
        return datetime.now(UTC).isoformat()

    def _prune_participants(self, raw_participants: list[dict]) -> list[dict]:
        now = datetime.now(UTC)
        participants: list[dict] = []
        for item in raw_participants:
            if not isinstance(item, dict):
                continue
            last_seen_raw = str(item.get("last_seen_at") or item.get("joined_at") or "")
            try:
                last_seen = datetime.fromisoformat(last_seen_raw.replace("Z", "+00:00"))
            except ValueError:
                last_seen = now

            is_recent = now - last_seen <= timedelta(seconds=self.PARTICIPANT_STALE_AFTER_SECONDS)
            status = "present" if is_recent else "stale"
            participants.append(
                {
                    "user_id": int(item.get("user_id") or 0),
                    "display_name": str(item.get("display_name") or ""),
                    "avatar_url": str(item.get("avatar_url") or ""),
                    "joined_at": str(item.get("joined_at") or last_seen_raw or self._now_iso()),
                    "last_seen_at": last_seen.isoformat(),
                    "status": status,
                    "active_call": bool(item.get("active_call")) and is_recent,
                }
            )
        return [item for item in participants if item["user_id"] > 0]

    def _parse_participants(self, metadata: dict) -> list[SessionParticipant]:
        raw = metadata.get(self.PARTICIPANTS_METADATA_KEY) or []
        cleaned = self._prune_participants(raw if isinstance(raw, list) else [])
        return [SessionParticipant(**item) for item in cleaned]

    def _parse_call_live(self, metadata: dict) -> SessionCallLive:
        raw = metadata.get(self.CALL_LIVE_METADATA_KEY) or {}
        if not isinstance(raw, dict):
            return SessionCallLive()
        return SessionCallLive(
            status=str(raw.get("status") or "idle"),
            host_user_id=int(raw["host_user_id"]) if raw.get("host_user_id") is not None else None,
            host_display_name=str(raw.get("host_display_name") or ""),
            started_at=str(raw.get("started_at") or ""),
            updated_at=str(raw.get("updated_at") or ""),
        )

    def _parse_call_log(self, metadata: dict) -> list[SessionCallLogEntry]:
        raw = metadata.get(self.CALL_LOG_METADATA_KEY) or []
        if not isinstance(raw, list):
            return []
        entries: list[SessionCallLogEntry] = []
        for item in raw[-self.MAX_CALL_LOG_ENTRIES:]:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(
                    SessionCallLogEntry(
                        id=str(item.get("id") or ""),
                        kind=str(item.get("kind") or "system"),
                        author=str(item.get("author") or "Sistema"),
                        text=str(item.get("text") or ""),
                        timestamp=str(item.get("timestamp") or self._now_iso()),
                        image_url=str(item.get("image_url")) if item.get("image_url") else None,
                        user_id=int(item["user_id"]) if item.get("user_id") is not None else None,
                    )
                )
            except Exception:
                continue
        return entries

    def _lock_row(self, db, session_id: str) -> AppSession | None:
        stmt = select(AppSession).where(AppSession.session_id == session_id.upper()).with_for_update()
        return db.scalar(stmt)

    def _get_or_create_locked_row(self, db, session_id: str) -> AppSession:
        row = self._lock_row(db, session_id)
        if row is None:
            row = AppSession(
                session_id=session_id.upper(),
                profile_context="",
                profile_language="es",
                profile_preferences_json={},
                nearby_pois_json=[],
                memory_json=[],
                metadata_json={},
            )
            db.add(row)
            db.flush()
        return row

    def _serialize(self, row: AppSession) -> SessionState:
        active_poi = POI(**row.active_poi_json) if row.active_poi_json else None
        nearby_pois = [POI(**poi) for poi in (row.nearby_pois_json or [])]
        metadata = row.metadata_json or {}
        ephemeral_map_pois = [POI(**poi) for poi in (metadata.get("ephemeral_map_pois") or [])]
        participants = self._parse_participants(metadata)
        call_live = self._parse_call_live(metadata)
        call_log = self._parse_call_log(metadata)
        return SessionState(
            session_id=row.session_id,
            user_id=row.user_id,
            profile=SessionProfile(
                raw_context=row.profile_context or "",
                language=row.profile_language or "es",
                preferences=row.profile_preferences_json or {},
            ),
            location=SessionLocation(
                lat=float(row.lat) if row.lat is not None else None,
                lng=float(row.lng) if row.lng is not None else None,
            ),
            active_poi=active_poi,
            nearby_pois=nearby_pois,
            ephemeral_map_pois=ephemeral_map_pois,
            memory=row.memory_json or [],
            metadata=metadata,
            participants=participants,
            call_live=call_live,
            call_log=call_log,
        )

    def _get_row(self, session_id: str) -> AppSession | None:
        with session_scope() as db:
            return db.get(AppSession, session_id.upper())

    def create_session(self, data: SessionCreateRequest) -> SessionState:
        session_id = (data.session_id or generate_session_id()).upper()
        with session_scope() as db:
            row = db.get(AppSession, session_id)
            if row is None:
                row = AppSession(
                    session_id=session_id,
                    user_id=data.user_id,
                    profile_context=data.profile_context,
                    lat=Decimal(str(data.lat)) if data.lat is not None else None,
                    lng=Decimal(str(data.lng)) if data.lng is not None else None,
                    metadata_json=data.metadata,
                    profile_preferences_json={},
                    nearby_pois_json=[],
                    memory_json=[],
                )
                db.add(row)
                db.flush()
            else:
                if row.user_id is None and data.user_id is not None:
                    row.user_id = data.user_id
                row.profile_context = data.profile_context
                row.lat = Decimal(str(data.lat)) if data.lat is not None else row.lat
                row.lng = Decimal(str(data.lng)) if data.lng is not None else row.lng
                if data.metadata:
                    row.metadata_json = {**(row.metadata_json or {}), **data.metadata}
            return self._serialize(row)

    def get_or_create(self, session_id: str) -> SessionState:
        session_id = session_id.upper()
        with session_scope() as db:
            row = db.get(AppSession, session_id)
            if row is None:
                row = AppSession(
                    session_id=session_id,
                    profile_context="",
                    profile_language="es",
                    profile_preferences_json={},
                    nearby_pois_json=[],
                    memory_json=[],
                    metadata_json={},
                )
                db.add(row)
                db.flush()
            return self._serialize(row)

    def get_session(self, session_id: str) -> SessionState | None:
        row = self._get_row(session_id)
        return self._serialize(row) if row is not None else None

    def attach_user(self, session_id: str, user_id: int | None) -> SessionState:
        with session_scope() as db:
            row = self._get_or_create_locked_row(db, session_id)
            if row.user_id is None and user_id is not None:
                row.user_id = user_id
            return self._serialize(row)

    def update_session(self, session_id: str, data: SessionUpdateRequest) -> SessionState:
        with session_scope() as db:
            row = self._get_or_create_locked_row(db, session_id)

            if data.user_id is not None and row.user_id is None:
                row.user_id = data.user_id
            if data.profile_context is not None:
                row.profile_context = data.profile_context
            if data.lat is not None:
                row.lat = Decimal(str(data.lat))
            if data.lng is not None:
                row.lng = Decimal(str(data.lng))
            if data.active_poi_name:
                current = row.active_poi_json or {}
                current["name"] = data.active_poi_name
                row.active_poi_json = current
            if data.metadata:
                row.metadata_json = {**(row.metadata_json or {}), **data.metadata}
            return self._serialize(row)

    def append_memory(self, session_id: str, role: str, text: str, limit: int = 30) -> SessionState:
        with session_scope() as db:
            row = self._get_or_create_locked_row(db, session_id)
            memory = list(row.memory_json or [])
            if text.strip():
                memory.append({"role": role, "text": text.strip()})
                row.memory_json = memory[-limit:]
            return self._serialize(row)

    def set_nearby_pois(self, session_id: str, pois: list[POI]) -> SessionState:
        with session_scope() as db:
            row = self._get_or_create_locked_row(db, session_id)
            row.nearby_pois_json = [poi.model_dump() for poi in pois]
            return self._serialize(row)

    def set_ephemeral_map_pois(self, session_id: str, pois: list[POI]) -> SessionState:
        with session_scope() as db:
            row = self._get_or_create_locked_row(db, session_id)
            metadata = dict(row.metadata_json or {})
            metadata["ephemeral_map_pois"] = [poi.model_dump() for poi in pois]
            row.metadata_json = metadata
            return self._serialize(row)

    def set_active_poi(self, session_id: str, poi: POI | None) -> SessionState:
        with session_scope() as db:
            row = self._get_or_create_locked_row(db, session_id)
            row.active_poi_json = poi.model_dump() if poi else None
            return self._serialize(row)

    def touch_participant(self, session_id: str, user: UserResponse, active_call: bool = False) -> SessionState:
        with session_scope() as db:
            row = self._get_or_create_locked_row(db, session_id)
            metadata = dict(row.metadata_json or {})
            participants = self._prune_participants(list(metadata.get(self.PARTICIPANTS_METADATA_KEY) or []))
            now_iso = self._now_iso()

            existing = next((item for item in participants if item["user_id"] == user.id), None)
            if existing is None:
                participants.append(
                    {
                        "user_id": user.id,
                        "display_name": user.display_name,
                        "avatar_url": user.avatar_url,
                        "joined_at": now_iso,
                        "last_seen_at": now_iso,
                        "status": "present",
                        "active_call": active_call,
                    }
                )
            else:
                existing.update(
                    {
                        "display_name": user.display_name,
                        "avatar_url": user.avatar_url,
                        "last_seen_at": now_iso,
                        "status": "present",
                        "active_call": active_call,
                    }
                )

            metadata[self.PARTICIPANTS_METADATA_KEY] = participants
            if row.user_id is None:
                row.user_id = user.id
            row.metadata_json = metadata
            return self._serialize(row)

    def leave_participant(self, session_id: str, user: UserResponse) -> SessionState:
        with session_scope() as db:
            row = self._get_or_create_locked_row(db, session_id)
            metadata = dict(row.metadata_json or {})
            participants = self._prune_participants(list(metadata.get(self.PARTICIPANTS_METADATA_KEY) or []))
            now_iso = self._now_iso()
            for item in participants:
                if item["user_id"] == user.id:
                    item["status"] = "left"
                    item["active_call"] = False
                    item["last_seen_at"] = now_iso

            call_live = dict(metadata.get(self.CALL_LIVE_METADATA_KEY) or {})
            if call_live.get("host_user_id") == user.id and call_live.get("status") == "live":
                call_live["status"] = "ended"
                call_live["updated_at"] = now_iso
                metadata[self.CALL_LIVE_METADATA_KEY] = call_live

            metadata[self.PARTICIPANTS_METADATA_KEY] = participants
            row.metadata_json = metadata
            return self._serialize(row)

    def set_call_state(self, session_id: str, user: UserResponse, status: str) -> SessionState:
        normalized = status.strip().lower()
        if normalized not in {"idle", "live", "ended"}:
            normalized = "idle"

        with session_scope() as db:
            row = self._get_or_create_locked_row(db, session_id)
            metadata = dict(row.metadata_json or {})
            participants = self._prune_participants(list(metadata.get(self.PARTICIPANTS_METADATA_KEY) or []))
            now_iso = self._now_iso()
            for item in participants:
                if item["user_id"] == user.id:
                    item["active_call"] = normalized == "live"
                    item["last_seen_at"] = now_iso

            current = dict(metadata.get(self.CALL_LIVE_METADATA_KEY) or {})
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
                    "host_display_name": user.display_name if normalized == "idle" else str(current.get("host_display_name") or user.display_name),
                    "started_at": str(current.get("started_at") or ""),
                    "updated_at": now_iso,
                }
            else:
                current["updated_at"] = now_iso

            metadata[self.PARTICIPANTS_METADATA_KEY] = participants
            metadata[self.CALL_LIVE_METADATA_KEY] = current
            row.metadata_json = metadata
            return self._serialize(row)

    def append_call_log(
        self,
        session_id: str,
        *,
        user: UserResponse,
        kind: str,
        author: str,
        text: str,
        image_url: str | None = None,
    ) -> SessionState:
        clean_text = text.strip()
        if not clean_text:
            return self.get_or_create(session_id)

        with session_scope() as db:
            row = self._get_or_create_locked_row(db, session_id)
            metadata = dict(row.metadata_json or {})
            raw_log = list(metadata.get(self.CALL_LOG_METADATA_KEY) or [])
            raw_log.append(
                {
                    "id": f"cl_{secrets.token_hex(8)}",
                    "kind": kind.strip() or "system",
                    "author": author.strip() or user.display_name or "Sistema",
                    "text": clean_text,
                    "timestamp": self._now_iso(),
                    "image_url": image_url,
                    "user_id": user.id,
                }
            )
            metadata[self.CALL_LOG_METADATA_KEY] = raw_log[-self.MAX_CALL_LOG_ENTRIES :]
            row.metadata_json = metadata
            return self._serialize(row)

    def reset_conversation(self, session_id: str) -> SessionState:
        with session_scope() as db:
            row = db.get(AppSession, session_id.upper())
            if row is None:
                raise ValueError(f"Session {session_id} not found")
            row.memory_json = []
            row.active_poi_json = None
            metadata = dict(row.metadata_json or {})
            metadata["ephemeral_map_pois"] = []
            metadata["last_chat_response_id"] = ""
            row.metadata_json = metadata
            return self._serialize(row)

    def set_metadata_value(self, session_id: str, key: str, value) -> SessionState:
        with session_scope() as db:
            row = db.get(AppSession, session_id.upper())
            if row is None:
                row = AppSession(
                    session_id=session_id.upper(),
                    profile_context="",
                    profile_language="es",
                    profile_preferences_json={},
                    nearby_pois_json=[],
                    memory_json=[],
                    metadata_json={},
                )
                db.add(row)
                db.flush()
            current = dict(row.metadata_json or {})
            current[key] = value
            row.metadata_json = current
            return self._serialize(row)


session_service = SessionService()
