from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select

from app.db.models import AppSession
from app.db.session import session_scope
from app.schemas.poi import POI
from app.schemas.session import SessionCreateRequest, SessionLocation, SessionProfile, SessionState, SessionUpdateRequest
from app.utils.ids import generate_session_id


class SessionService:
    def _serialize(self, row: AppSession) -> SessionState:
        active_poi = POI(**row.active_poi_json) if row.active_poi_json else None
        nearby_pois = [POI(**poi) for poi in (row.nearby_pois_json or [])]
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
            memory=row.memory_json or [],
            metadata=row.metadata_json or {},
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
                row.user_id = data.user_id or row.user_id
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
            row = db.get(AppSession, session_id.upper())
            if row is None:
                row = AppSession(
                    session_id=session_id.upper(),
                    user_id=user_id,
                    profile_context="",
                    profile_language="es",
                    profile_preferences_json={},
                    nearby_pois_json=[],
                    memory_json=[],
                    metadata_json={},
                )
                db.add(row)
                db.flush()
            else:
                row.user_id = user_id or row.user_id
            return self._serialize(row)

    def update_session(self, session_id: str, data: SessionUpdateRequest) -> SessionState:
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

            if data.user_id is not None:
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
            memory = list(row.memory_json or [])
            if text.strip():
                memory.append({"role": role, "text": text.strip()})
                row.memory_json = memory[-limit:]
            return self._serialize(row)

    def set_nearby_pois(self, session_id: str, pois: list[POI]) -> SessionState:
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
            row.nearby_pois_json = [poi.model_dump() for poi in pois]
            return self._serialize(row)

    def set_active_poi(self, session_id: str, poi: POI | None) -> SessionState:
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
            row.active_poi_json = poi.model_dump() if poi else None
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
