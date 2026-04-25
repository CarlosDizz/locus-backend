from __future__ import annotations

import json
from typing import Any

from app.schemas.poi import POI
from app.services.poi_service import poi_service
from app.services.session_service import session_service


class ToolRuntimeService:
    TOOL_CANDIDATES_METADATA_KEY = "tool_candidate_pois"

    def execute(self, session_id: str, tool_name: str, arguments: dict[str, Any]) -> str:
        handlers = {
            "get_session_profile": self._get_session_profile,
            "set_active_poi": self._set_active_poi,
            "get_nearby_pois": self._get_nearby_pois,
            "search_tourism_candidates": self._search_tourism_candidates,
            "search_contextual_recommendations": self._search_contextual_recommendations,
            "identify_map_landmark": self._identify_map_landmark,
            "mark_pois_on_map": self._mark_pois_on_map,
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
        visible_names = ", ".join(poi.name for poi in session.nearby_pois[:4])
        ephemeral_names = ", ".join(poi.name for poi in session.ephemeral_map_pois[:3])
        return {
            "ok": True,
            "profile_context": session.profile.raw_context,
            "preferences": session.profile.preferences,
            "nearby_pois": [poi.model_dump() for poi in session.nearby_pois[:10]],
            "ephemeral_map_pois": [poi.model_dump() for poi in session.ephemeral_map_pois[:6]],
            "location_hint": (
                f"Zona actual con geolocalizacion activa y lugares visibles como {visible_names}. Recomendaciones efimeras marcadas en el mapa: {ephemeral_names}."
                if session.nearby_pois and session.ephemeral_map_pois
                else (
                    f"Zona actual con geolocalizacion activa y lugares visibles como {visible_names}."
                    if session.nearby_pois
                    else (
                        f"Zona actual con geolocalizacion activa y recomendaciones efimeras ya marcadas en el mapa: {ephemeral_names}."
                        if session.ephemeral_map_pois
                        else ("Zona actual con geolocalizacion activa." if session.location.lat is not None and session.location.lng is not None else "")
                    )
                )
            ),
            "map_hint": (
                "Salvo que el usuario nombre Google Maps, Apple Maps u otra app, cuando diga 'el mapa' se refiere al mapa de Locus."
                if session.nearby_pois or session.ephemeral_map_pois
                else ""
            ),
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
        if poi_service._is_generic_query(query):
            session_service.set_nearby_pois(session_id, pois)
            map_action = "base_map_refreshed"
        else:
            self._store_tool_candidates(session_id, pois)
            map_action = "none_specific_search_did_not_overwrite_base_map"
        return {
            "ok": True,
            "query": query,
            "map_action": map_action,
            "pois": [poi.model_dump() for poi in pois],
        }

    def _context_kind_for_need(self, need: str) -> str:
        lowered = need.lower()
        if any(token in lowered for token in ["restaurante", "comer", "cenar", "bar", "cerveza", "cafe", "café", "desayuno", "pizza", "pasta"]):
            return "hospitality"
        return "service"

    def _build_landmark_query(self, reference_text: str, near_poi_name: str) -> str:
        reference = reference_text.strip()
        nearby = near_poi_name.strip()
        if reference and nearby:
            return f"{reference} {nearby}".strip()
        return reference or nearby

    def _store_tool_candidates(self, session_id: str, pois: list[POI]) -> None:
        session_service.set_metadata_value(
            session_id,
            self.TOOL_CANDIDATES_METADATA_KEY,
            [poi.model_dump() for poi in pois],
        )

    def _load_tool_candidates(self, session_id: str) -> list[POI]:
        session = session_service.get_or_create(session_id)
        raw = session.metadata.get(self.TOOL_CANDIDATES_METADATA_KEY) or []
        return [POI(**item) for item in raw]

    def _resolve_markable_pois(self, session_id: str, poi_names: list[str]) -> list[POI]:
        session = session_service.get_or_create(session_id)
        candidates = [
            *self._load_tool_candidates(session_id),
            *session.ephemeral_map_pois,
            *session.nearby_pois,
        ]
        resolved: list[POI] = []
        wanted = {name.strip().lower() for name in poi_names if name.strip()}
        seen: set[str] = set()
        for poi in candidates:
            key = (poi.google_place_id or poi.id or poi.name).lower()
            if poi.name.lower() in wanted and key not in seen:
                resolved.append(poi.model_copy(update={"is_ephemeral": True}))
                seen.add(key)
        return resolved

    def _merge_ephemeral_pois(self, existing: list[POI], new: list[POI]) -> list[POI]:
        merged: list[POI] = []
        seen: set[str] = set()
        for poi in [*existing, *new]:
            key = (poi.google_place_id or poi.id or poi.name).lower()
            if key in seen:
                continue
            merged.append(poi.model_copy(update={"is_ephemeral": True}))
            seen.add(key)
        return merged

    def _search_tourism_candidates(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session = session_service.get_or_create(session_id)
        query = (arguments.get("query") or "").strip()
        near_poi_name = (arguments.get("near_poi_name") or "").strip()
        lat = arguments.get("lat", session.location.lat)
        lng = arguments.get("lng", session.location.lng)
        limit = max(1, min(int(arguments.get("limit", 5) or 5), 8))
        if not query:
            return {"ok": False, "error": "query is required"}

        search_query = self._build_landmark_query(query, near_poi_name)
        pois = poi_service.search_tourism_candidates(query=search_query, lat=lat, lng=lng, limit=limit)
        self._store_tool_candidates(session_id, pois)
        return {
            "ok": True,
            "query": query,
            "search_query": search_query,
            "near_poi_name": near_poi_name,
            "persistence_policy": "candidate_only_do_not_persist_automatically",
            "map_action": "none_call_mark_pois_on_map_if_you_want_them_visible",
            "pois": [poi.model_dump() for poi in pois],
        }

    def _search_contextual_recommendations(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session = session_service.get_or_create(session_id)
        need = (arguments.get("need") or "").strip()
        lat = arguments.get("lat", session.location.lat)
        lng = arguments.get("lng", session.location.lng)
        limit = max(1, min(int(arguments.get("limit", 5) or 5), 8))
        if not need:
            return {"ok": False, "error": "need is required"}

        context_kind = self._context_kind_for_need(need)
        pois = [
            poi.model_copy(update={"context_kind": context_kind})
            for poi in poi_service.search_contextual_places(query=need, lat=lat, lng=lng, limit=limit)
        ]
        self._store_tool_candidates(session_id, pois)
        return {
            "ok": True,
            "need": need,
            "context_kind": context_kind,
            "persistence_policy": "ephemeral_only_do_not_persist",
            "map_action": "none_call_mark_pois_on_map_if_you_want_them_visible",
            "pois": [poi.model_dump() for poi in pois],
        }

    def _identify_map_landmark(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session = session_service.get_or_create(session_id)
        reference_text = (arguments.get("reference_text") or "").strip()
        near_poi_name = (arguments.get("near_poi_name") or "").strip()
        if not near_poi_name and session.active_poi is not None:
            near_poi_name = session.active_poi.name
        lat = arguments.get("lat", session.location.lat)
        lng = arguments.get("lng", session.location.lng)
        limit = max(1, min(int(arguments.get("limit", 5) or 5), 8))
        if not reference_text:
            return {"ok": False, "error": "reference_text is required"}

        search_query = self._build_landmark_query(reference_text, near_poi_name)
        pois = poi_service.search_tourism_candidates(query=search_query, lat=lat, lng=lng, limit=limit)
        self._store_tool_candidates(session_id, pois)
        return {
            "ok": True,
            "reference_text": reference_text,
            "near_poi_name": near_poi_name,
            "search_query": search_query,
            "persistence_policy": "candidate_only_do_not_persist_automatically",
            "map_action": "none_call_mark_pois_on_map_if_you_want_them_visible",
            "pois": [poi.model_dump() for poi in pois],
        }

    def _mark_pois_on_map(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        poi_names = arguments.get("poi_names") or []
        replace_existing = bool(arguments.get("replace_existing", False))
        reason = (arguments.get("reason") or "").strip()
        if not isinstance(poi_names, list) or not poi_names:
            return {"ok": False, "error": "poi_names is required"}

        selected = self._resolve_markable_pois(session_id, [str(name) for name in poi_names])
        if not selected:
            return {"ok": False, "error": "No matching POIs available to mark on the map"}

        session = session_service.get_or_create(session_id)
        marked = selected if replace_existing else self._merge_ephemeral_pois(session.ephemeral_map_pois, selected)
        session_service.set_ephemeral_map_pois(session_id, marked)
        return {
            "ok": True,
            "reason": reason,
            "replace_existing": replace_existing,
            "marked_pois": [poi.model_dump() for poi in marked],
            "map_action": "ephemeral_pois_marked_on_locus_map",
            "persistence_policy": "ephemeral_only_do_not_persist",
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
