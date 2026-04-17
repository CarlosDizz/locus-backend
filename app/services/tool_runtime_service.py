from __future__ import annotations

import json
from typing import Any

from app.schemas.poi import POI
from app.services.poi_service import poi_service
from app.services.session_service import session_service


class ToolRuntimeService:
    def execute(self, session_id: str, tool_name: str, arguments: dict[str, Any]) -> str:
        handlers = {
            "get_session_profile": self._get_session_profile,
            "set_active_poi": self._set_active_poi,
            "get_nearby_pois": self._get_nearby_pois,
            "get_poi_summary": self._get_poi_summary,
            "resolve_poi_facts": self._resolve_poi_facts,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return json.dumps({"ok": False, "error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)
        result = handler(session_id, arguments)
        return json.dumps(result, ensure_ascii=False)

    def _get_session_profile(self, session_id: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        session = session_service.get_or_create(session_id)
        return {
            "ok": True,
            "profile_context": session.profile.raw_context,
            "preferences": session.profile.preferences,
            "active_poi": session.active_poi.model_dump() if session.active_poi else None,
        }

    def _set_active_poi(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session = session_service.get_or_create(session_id)
        poi_name = arguments.get("poi_name", "").strip()
        if not poi_name:
            return {"ok": False, "error": "poi_name is required"}

        selected = next((poi for poi in session.nearby_pois if poi.name.lower() == poi_name.lower()), None)
        if selected is None:
            selected = POI(name=poi_name, lat=0.0, lng=0.0, description="", summary="")

        selected = poi_service.enrich_poi(selected)
        session_service.set_active_poi(session_id, selected)
        return {"ok": True, "active_poi": selected.model_dump()}

    def _get_nearby_pois(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session = session_service.get_or_create(session_id)
        lat = arguments.get("lat", session.location.lat)
        lng = arguments.get("lng", session.location.lng)
        query = (arguments.get("query") or "lugares turísticos").strip()

        pois = poi_service.search_nearby_pois(query=query, lat=lat, lng=lng, limit=5)
        session_service.set_nearby_pois(session_id, pois)
        return {
            "ok": True,
            "query": query,
            "pois": [poi.model_dump() for poi in pois],
        }

    def _get_poi_summary(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        poi_name = arguments.get("poi_name", "").strip()
        if not poi_name:
            return {"ok": False, "error": "poi_name is required"}

        documentation = poi_service.get_poi_documentation(poi_name)
        summary = documentation["summary"]
        session = session_service.get_or_create(session_id)
        if session.active_poi and session.active_poi.name.lower() == poi_name.lower():
            active_poi = session.active_poi.model_copy()
            active_poi.summary = summary
            session_service.set_active_poi(session_id, active_poi)
        return {"ok": True, "poi_name": poi_name, "summary": summary, "documentation": documentation}

    def _resolve_poi_facts(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        poi_name = arguments.get("poi_name", "").strip()
        question = (arguments.get("question") or "").strip()
        if not poi_name:
            return {"ok": False, "error": "poi_name is required"}

        documentation = poi_service.get_poi_documentation(poi_name)
        return {
            "ok": True,
            "poi_name": poi_name,
            "question": question,
            "facts_summary": documentation["summary"],
            "facts": documentation["facts"],
            "wikidata": documentation["wikidata"],
            "sources": documentation["sources"],
            "source_policy": "Usa hechos documentales de Wikidata y resumen narrativo prudente; si falta un dato concreto, no debe inventarse.",
        }


tool_runtime_service = ToolRuntimeService()
