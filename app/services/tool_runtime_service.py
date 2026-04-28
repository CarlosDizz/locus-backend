from __future__ import annotations

import json
from typing import Any

from sqlalchemy import or_, select

from app.db.models import City, Poi
from app.db.session import session_scope
from app.schemas.poi import POI
from app.schemas.catalog import CityCreateRequest, PoiCreateRequest
from app.services.catalog_service import CatalogError, catalog_service
from app.services.poi_service import poi_service
from app.services.session_service import session_service
from app.utils.text import clean_text, slugify


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
            "promote_poi_to_catalog": self._promote_poi_to_catalog,
            "get_poi_summary": self._get_poi_summary,
            "resolve_poi_facts": self._resolve_poi_facts,
            "search_wikipedia": self._search_wikipedia,
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

    def _resolve_single_candidate(self, session_id: str, poi_name: str) -> POI | None:
        session = session_service.get_or_create(session_id)
        wanted = clean_text(poi_name).lower()
        if not wanted:
            return None

        for poi in [
            *self._load_tool_candidates(session_id),
            *session.ephemeral_map_pois,
            *session.nearby_pois,
        ]:
            if clean_text(poi.name).lower() == wanted:
                return poi
        return None

    def _resolve_session_city(self, session_id: str) -> dict[str, Any] | None:
        session = session_service.get_or_create(session_id)
        if session.location.lat is None or session.location.lng is None:
            return None

        lat = float(session.location.lat)
        lng = float(session.location.lng)

        with session_scope() as db:
            cities = list(db.scalars(select(City).where(City.lat.is_not(None), City.lng.is_not(None))).all())

        def distance_km(city: City) -> float:
            return poi_service._distance_km(lat, lng, float(city.lat), float(city.lng))  # type: ignore[arg-type]

        if cities:
            nearest = min(cities, key=distance_km)
            if distance_km(nearest) <= 20:
                return {
                    "id": nearest.id,
                    "name": nearest.name,
                    "country_code": nearest.country_code,
                }

        try:
            city, _imported_count, _updated_count, _skipped_count, _stats, _pois = catalog_service.bootstrap_city_from_location(
                lat=lat,
                lng=lng,
                radius_km=8,
                limit=40,
                use_ai_candidates=False,
            )
            return {
                "id": city.id,
                "name": city.name,
                "country_code": city.country_code,
            }
        except CatalogError:
            slug = slugify(f"zona-{lat:.3f}-{lng:.3f}")
            fallback = catalog_service.create_city(
                CityCreateRequest(
                    name="Zona actual",
                    slug=slug,
                    country_code="",
                    lat=lat,
                    lng=lng,
                    source="session_fallback",
                )
            )
            return {
                "id": fallback.id,
                "name": fallback.name,
                "country_code": fallback.country_code,
            }

    def _candidate_looks_catalog_worthy(self, poi: POI) -> tuple[bool, str]:
        name = clean_text(poi.name).lower()
        description = clean_text(poi.description or poi.summary).lower()
        combined = f"{name} {description}".strip()

        if not name:
            return False, "missing_name"
        if poi.lat == 0.0 and poi.lng == 0.0:
            return False, "missing_coords"
        if poi_service._is_hospitality_or_service_label(combined):
            return False, "service_or_hospitality"

        positive_terms = [
            "teatro", "teatre", "theatre", "theater", "circo", "museo", "museum",
            "catedral", "cathedral", "iglesia", "church", "plaza", "square",
            "palacio", "palace", "castillo", "castle", "puerta", "gate",
            "monumento", "monument", "alcázar", "alcazar", "foro", "anfiteatro",
            "arqueológico", "arqueologico", "archaeological",
        ]
        if any(term in combined for term in positive_terms):
            return True, "landmark_keyword"

        if poi.source_of_truth in {"catalog", "wikidata", "google_places"} and len(name) >= 6:
            return True, "trusted_source_candidate"

        return False, "not_distinctive_enough"

    def _infer_catalog_type_code(self, poi: POI) -> str:
        combined = clean_text(f"{poi.name} {poi.description} {poi.summary}").lower()
        if any(token in combined for token in ["teatro", "theatre", "theater", "palacio", "palace", "edificio"]):
            return "building"
        if any(token in combined for token in ["museo", "museum", "galería", "galeria", "gallery"]):
            return "museum"
        if any(token in combined for token in ["catedral", "iglesia", "church", "basílica", "basilica", "sinagoga", "synagogue"]):
            return "church"
        if any(token in combined for token in ["plaza", "square"]):
            return "square"
        if any(token in combined for token in ["castillo", "castle", "monumento", "monument", "puerta", "gate", "alcázar", "alcazar"]):
            return "monument"
        if any(token in combined for token in ["anfiteatro", "foro", "arqueológico", "arqueologico", "archaeological", "circo romano"]):
            return "archaeological_site"
        return "building"

    def _catalog_row_to_runtime_poi(self, row: Poi) -> POI:
        return POI(
            id=str(row.id),
            name=row.name,
            lat=float(row.lat) if row.lat is not None else 0.0,
            lng=float(row.lng) if row.lng is not None else 0.0,
            poi_type_code="",
            description=row.short_description or row.long_description or "",
            summary=row.long_description or row.short_description or "",
            source_of_truth="catalog",
            is_ephemeral=False,
            google_place_id=row.google_place_id or "",
            context_kind="catalog",
        )

    def _upsert_session_catalog_poi(self, session_id: str, runtime_poi: POI) -> None:
        session = session_service.get_or_create(session_id)
        merged: list[POI] = []
        seen: set[str] = set()
        for poi in [runtime_poi, *session.nearby_pois]:
            key = slugify(poi.name)
            if key in seen:
                continue
            merged.append(poi.model_copy(update={"is_ephemeral": False, "context_kind": "catalog"}))
            seen.add(key)
        session_service.set_nearby_pois(session_id, merged[:12])

        filtered_ephemeral = [
            poi for poi in session.ephemeral_map_pois
            if slugify(poi.name) != slugify(runtime_poi.name)
        ]
        session_service.set_ephemeral_map_pois(session_id, filtered_ephemeral)
        session_service.set_active_poi(session_id, runtime_poi)

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

    def _promote_poi_to_catalog(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        poi_name = clean_text(arguments.get("poi_name") or "")
        reason = clean_text(arguments.get("reason") or "")
        if not poi_name:
            return {"ok": False, "error": "poi_name is required"}

        session = session_service.get_or_create(session_id)
        candidate = self._resolve_single_candidate(session_id, poi_name)

        with session_scope() as db:
            existing = db.scalar(
                select(Poi).where(
                    Poi.is_active.is_(True),
                    or_(Poi.name.ilike(poi_name), Poi.slug == slugify(poi_name)),
                )
            )
            if existing is not None:
                runtime_poi = self._catalog_row_to_runtime_poi(existing)
                self._upsert_session_catalog_poi(session_id, runtime_poi)
                return {
                    "ok": True,
                    "poi_name": existing.name,
                    "status": "already_in_catalog",
                    "catalog_poi": runtime_poi.model_dump(),
                    "map_action": "catalog_poi_available_in_base_map",
                    "persistence_policy": "catalog_fixed_poi",
                }

        if candidate is None:
            return {
                "ok": False,
                "poi_name": poi_name,
                "error": "candidate_not_found",
                "message": "No tengo un candidato resoluble con ese nombre en el contexto actual.",
            }

        worthy, decision_reason = self._candidate_looks_catalog_worthy(candidate)
        if not worthy:
            return {
                "ok": False,
                "poi_name": candidate.name,
                "error": "not_catalog_worthy",
                "reason": decision_reason,
                "message": "No cumple criterios suficientes para subirlo como POI fijo del catálogo.",
            }

        city = self._resolve_session_city(session_id)
        if city is None:
            return {
                "ok": False,
                "poi_name": candidate.name,
                "error": "city_not_resolved",
                "message": "No he podido asociar el lugar a una ciudad del catálogo.",
            }

        try:
            created = catalog_service.create_poi(
                PoiCreateRequest(
                    city_id=int(city["id"]),
                    poi_type_code=self._infer_catalog_type_code(candidate),
                    slug=slugify(candidate.name),
                    name=candidate.name,
                    lat=float(candidate.lat),
                    lng=float(candidate.lng),
                    short_description=clean_text(candidate.description or candidate.summary)[:500],
                    long_description="",
                    source_of_truth="manual_curated",
                    google_place_id=candidate.google_place_id,
                    metadata={
                        "promoted_from_chat": True,
                        "promotion_reason": reason,
                        "session_id": session_id,
                        "source_context_kind": candidate.context_kind,
                        "promoted_original_source": candidate.source_of_truth,
                    },
                )
            )
        except CatalogError as exc:
            return {
                "ok": False,
                "poi_name": candidate.name,
                "error": "catalog_create_failed",
                "message": str(exc),
            }

        runtime_poi = POI(
            id=str(created.id),
            name=created.name,
            lat=float(created.lat) if created.lat is not None else float(candidate.lat),
            lng=float(created.lng) if created.lng is not None else float(candidate.lng),
            poi_type_code=created.poi_type_code or self._infer_catalog_type_code(candidate),
            description=created.short_description or candidate.description,
            summary=created.long_description or created.short_description or candidate.summary,
            source_of_truth="catalog",
            is_ephemeral=False,
            google_place_id=created.google_place_id or candidate.google_place_id,
            context_kind="catalog",
        )
        self._upsert_session_catalog_poi(session_id, runtime_poi)

        return {
            "ok": True,
            "poi_name": runtime_poi.name,
            "status": "promoted_to_catalog",
            "catalog_poi": runtime_poi.model_dump(),
            "city_id": city["id"],
            "city_name": city["name"],
            "map_action": "catalog_poi_added_and_base_map_refreshed",
            "persistence_policy": "catalog_fixed_poi",
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


    def _search_wikipedia(self, _session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from app.clients.wikipedia_client import WikipediaClient
        query = (arguments.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "query is required"}
        summary = WikipediaClient().get_summary(query, sentences=5)
        if not summary:
            return {"ok": False, "error": f"No Wikipedia article found for: {query}"}
        return {"ok": True, "query": query, "summary": summary}


tool_runtime_service = ToolRuntimeService()
