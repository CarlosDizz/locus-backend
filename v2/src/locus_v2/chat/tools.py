"""Tool handlers for the map chat, ported from V1 `tool_runtime_service.py`.

V1 had 14 handlers; this is 7, because several of V1's existed only to hand
the model state that V2 already injects straight into the prompt
(`get_session_profile`), or were three near-identical wrappers around the
same geo search with different filters (`get_nearby_pois`,
`search_tourism_candidates`, `identify_map_landmark` — all now
`search_map_places`).

Two handler codes are delegated to `voice.tools.VoiceToolDispatcher` rather
than reimplemented: `catalog.document_poi` and `affiliates.find_activities`
already exist there, are already billed through `shared/openai_usage.py`,
and behave identically whether a voice or a text guide asked for them.

Candidate bookkeeping: searches do NOT touch the map. They stash results in
session metadata, and only `mark_pois_on_map` makes anything visible — same
two-step V1 used, so the model can look things up while answering without
every lookup redrawing the user's map.
"""

import json
from typing import Any

import structlog
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.catalog.bootstrap.service import CatalogBootstrapError, CatalogBootstrapService
from locus_v2.catalog.models import City, Poi, PoiType
from locus_v2.config import Settings
from locus_v2.places.service import PlaceSearchService, distance_km, looks_like_service
from locus_v2.sessions.application.service import MapSessionService
from locus_v2.sessions.models import SessionPoi, SessionStateView
from locus_v2.shared.openai_usage import ToolUsage
from locus_v2.shared.text import clean_text, slugify
from locus_v2.voice.tools import VoiceToolDispatcher

logger = structlog.get_logger()

CANDIDATES_METADATA_KEY = "tool_candidate_pois"
CITY_MATCH_RADIUS_KM = 20.0
MAX_NEARBY_POIS = 12


class ChatToolError(RuntimeError):
    pass


class ChatToolDispatcher:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        *,
        session_id: str,
        locale: str,
    ) -> None:
        self.session = session
        self.settings = settings
        self.session_id = session_id
        self.locale = locale
        self.sessions = MapSessionService(session)
        self.places = PlaceSearchService(session, settings)
        # Real spend made by delegated handlers, outside this turn's own
        # provider call. chat/service.py reads it after every execute() and
        # folds it into the turn's UsageEvent, or the tool call is free money
        # out the door (the same leak fixed in calls/bridge.py on 2026-09-06).
        self.last_usage: ToolUsage | None = None

    async def execute(self, handler_code: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.last_usage = None
        handlers = {
            "map.search_places": self._search_places,
            "map.search_services": self._search_services,
            "map.mark_pois": self._mark_pois,
            "map.set_active_poi": self._set_active_poi,
            "catalog.promote_poi": self._promote_poi,
        }
        handler = handlers.get(handler_code)
        if handler is not None:
            return await handler(arguments)
        if handler_code in {"catalog.document_poi", "affiliates.find_activities"}:
            return await self._delegate_to_voice_tools(handler_code, arguments)
        return {"ok": False, "error": f"Unknown tool handler: {handler_code}"}

    async def _delegate_to_voice_tools(
        self, handler_code: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        state = await self.sessions.get_or_create(self.session_id)
        active = state.active_poi
        context = {
            "name": arguments.get("poi_name") or (active.name if active else ""),
            "description": (active.description or active.summary) if active else "",
            "city_name": str(state.metadata.get("city_name") or ""),
            "wikidata_id": "",
            "wikipedia_title": "",
        }
        dispatcher = VoiceToolDispatcher(self.settings)
        result = await dispatcher.execute(handler_code, arguments, context, self.locale)
        self.last_usage = dispatcher.last_usage
        return {"ok": True, **result}

    # ---- map search -----------------------------------------------------

    async def _search_places(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = clean_text(str(arguments.get("query") or ""))
        if not query:
            return {"ok": False, "error": "query is required"}
        near = clean_text(str(arguments.get("near_poi_name") or ""))
        state = await self.sessions.get_or_create(self.session_id)
        lat, lng = _coords(arguments, state)
        limit = _limit(arguments)

        search_query = f"{query} {near}".strip() if near else query
        pois = await self.places.search_landmarks(
            query=search_query, lat=lat, lng=lng, locale=self.locale, limit=limit
        )
        await self._store_candidates(pois)
        return {
            "ok": True,
            "query": query,
            "search_query": search_query,
            "map_action": "nothing_shown_yet_call_mark_pois_on_map_to_show_them",
            "pois": [poi.model_dump() for poi in pois],
        }

    async def _search_services(self, arguments: dict[str, Any]) -> dict[str, Any]:
        need = clean_text(str(arguments.get("need") or ""))
        if not need:
            return {"ok": False, "error": "need is required"}
        state = await self.sessions.get_or_create(self.session_id)
        lat, lng = _coords(arguments, state)
        pois = await self.places.search_services(
            query=need, lat=lat, lng=lng, locale=self.locale, limit=_limit(arguments)
        )
        await self._store_candidates(pois)
        if not pois and not self.places.places.enabled:
            return {
                "ok": False,
                "need": need,
                "error": "places_lookup_not_configured",
                "message": (
                    "No hay busqueda de sitios en vivo configurada en este entorno; "
                    "responde solo con lo que haya en el catalogo."
                ),
            }
        return {
            "ok": True,
            "need": need,
            "persistence_policy": "ephemeral_only_never_saved_to_catalog",
            "map_action": "nothing_shown_yet_call_mark_pois_on_map_to_show_them",
            "pois": [poi.model_dump() for poi in pois],
        }

    # ---- map state ------------------------------------------------------

    async def _mark_pois(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_names = arguments.get("poi_names") or []
        if not isinstance(raw_names, list) or not raw_names:
            return {"ok": False, "error": "poi_names is required"}
        replace = bool(arguments.get("replace_existing", False))

        state = await self.sessions.get_or_create(self.session_id)
        wanted = {clean_text(str(name)).lower() for name in raw_names if str(name).strip()}
        pool = [*await self._load_candidates(), *state.ephemeral_map_pois, *state.nearby_pois]

        selected: list[SessionPoi] = []
        seen: set[str] = set()
        for poi in pool:
            key = (poi.google_place_id or poi.id or poi.name).lower()
            if poi.name.lower() in wanted and key not in seen:
                seen.add(key)
                selected.append(poi.model_copy(update={"is_ephemeral": True}))
        if not selected:
            return {
                "ok": False,
                "error": "no_matching_pois",
                "message": (
                    "Ninguno de esos nombres esta entre los resultados de busqueda de "
                    "este turno. Busca primero y usa los nombres exactos que devuelva."
                ),
            }

        marked = selected if replace else _merge_ephemeral(state.ephemeral_map_pois, selected)
        await self.sessions.set_ephemeral_map_pois(self.session_id, marked)
        return {
            "ok": True,
            "replace_existing": replace,
            "map_action": "pois_now_visible_on_the_locus_map",
            "marked_pois": [poi.model_dump() for poi in marked],
        }

    async def _set_active_poi(self, arguments: dict[str, Any]) -> dict[str, Any]:
        poi_name = clean_text(str(arguments.get("poi_name") or ""))
        if not poi_name:
            return {"ok": False, "error": "poi_name is required"}
        resolved = await self._resolve_candidate(poi_name)
        if resolved is None:
            return {
                "ok": False,
                "error": "poi_not_found",
                "message": "Ese lugar no esta en el contexto actual del mapa.",
            }
        await self.sessions.set_active_poi(self.session_id, resolved)
        return {"ok": True, "active_poi": resolved.model_dump()}

    # ---- catalog promotion ----------------------------------------------

    async def _promote_poi(self, arguments: dict[str, Any]) -> dict[str, Any]:
        poi_name = clean_text(str(arguments.get("poi_name") or ""))
        if not poi_name:
            return {"ok": False, "error": "poi_name is required"}

        existing = await self.session.scalar(
            select(Poi).where(
                Poi.is_active.is_(True),
                or_(Poi.name.ilike(poi_name), Poi.slug == slugify(poi_name)),
            )
        )
        if existing is not None:
            runtime = _catalog_row_to_session_poi(existing)
            await self._adopt_catalog_poi(runtime)
            return {
                "ok": True,
                "poi_name": existing.name,
                "status": "already_in_catalog",
                "catalog_poi": runtime.model_dump(),
            }

        candidate = await self._resolve_candidate(poi_name)
        if candidate is None:
            return {
                "ok": False,
                "error": "candidate_not_found",
                "message": "No tengo ese lugar entre los resultados de busqueda actuales.",
            }
        # The only guard kept from V1's worthiness check: this writes a real
        # row into the shared catalog, so a place with no coordinates or an
        # obvious restaurant/bar must never land there. V1 also required the
        # name to contain one of ~35 Spanish landmark keywords, which silently
        # refused legitimate non-Spanish landmarks - dropped, the model plus
        # this guard decide.
        if candidate.lat == 0.0 and candidate.lng == 0.0:
            return {"ok": False, "poi_name": candidate.name, "error": "missing_coordinates"}
        if looks_like_service(f"{candidate.name} {candidate.description} {candidate.summary}"):
            return {
                "ok": False,
                "poi_name": candidate.name,
                "error": "not_a_landmark",
                "message": (
                    "Los restaurantes, bares y servicios no entran en el catalogo; "
                    "quedan solo como marca temporal en el mapa."
                ),
            }

        city = await self._resolve_city(candidate)
        if city is None:
            return {
                "ok": False,
                "poi_name": candidate.name,
                "error": "city_not_resolved",
                "message": "No he podido asociar el lugar a ninguna ciudad del catalogo.",
            }
        type_id = await self._resolve_poi_type(str(arguments.get("poi_type_code") or ""))

        # names_json is what catalog/mobile.py hands the app to localize with;
        # leaving it empty would make the new POI the only one on the map with
        # no localized label at all.
        language = self.locale.split("-", 1)[0].lower() or "es"
        short_description = clean_text(candidate.description or candidate.summary)[:500]
        row = Poi(
            city_id=city.id,
            poi_type_id=type_id,
            slug=slugify(candidate.name),
            name=candidate.name,
            names_json={language: candidate.name, "local": candidate.name},
            lat=candidate.lat,
            lng=candidate.lng,
            short_description=short_description,
            short_descriptions_json=(
                {language: short_description} if short_description else {}
            ),
            source_of_truth="chat_promoted",
            google_place_id=candidate.google_place_id,
            metadata_json={
                "promoted_from_chat": True,
                "promotion_reason": clean_text(str(arguments.get("reason") or "")),
                "session_id": self.session_id,
                "promoted_original_source": candidate.source_of_truth,
            },
        )
        self.session.add(row)
        await self.session.commit()
        await self.session.refresh(row)
        logger.info(
            "chat_poi_promoted",
            session_id=self.session_id,
            poi_id=row.id,
            poi_name=row.name,
            city=city.name,
        )

        runtime = _catalog_row_to_session_poi(row)
        await self._adopt_catalog_poi(runtime)
        return {
            "ok": True,
            "poi_name": runtime.name,
            "status": "promoted_to_catalog",
            "city_name": city.name,
            "catalog_poi": runtime.model_dump(),
        }

    async def _resolve_city(self, candidate: SessionPoi) -> City | None:
        cities = list(
            (
                await self.session.scalars(
                    select(City).where(City.lat.is_not(None), City.lng.is_not(None))
                )
            )
            .unique()
            .all()
        )
        located = [(city, _km_from(candidate, city)) for city in cities]
        placed = [item for item in located if item[1] is not None]
        if placed:
            nearest, nearest_km = min(placed, key=lambda item: item[1] or 0.0)
            if (nearest_km or 0.0) <= CITY_MATCH_RADIUS_KM:
                return nearest
        try:
            # No AI candidates: this is a synchronous step inside a live chat
            # turn, and all it needs is the city row to hang the POI off.
            bootstrap = CatalogBootstrapService(self.session, self.settings)
            result = await bootstrap.bootstrap_from_location(
                lat=candidate.lat, lng=candidate.lng, radius_km=8, limit=1,
                use_ai_candidates=False,
            )
        except CatalogBootstrapError as error:
            logger.warning("chat_promote_city_bootstrap_failed", error=str(error))
            return None
        return await self.session.get(City, result.city_id)

    async def _resolve_poi_type(self, code: str) -> int | None:
        normalized = clean_text(code).lower()
        if not normalized:
            return None
        row = await self.session.scalar(
            select(PoiType).where(PoiType.code == normalized)
        )
        return row.id if row is not None else None

    async def _adopt_catalog_poi(self, poi: SessionPoi) -> None:
        """Make a freshly-promoted POI a fixed pin and drop its ephemeral twin."""
        state = await self.sessions.get_or_create(self.session_id)
        merged: list[SessionPoi] = []
        seen: set[str] = set()
        for item in [poi, *state.nearby_pois]:
            key = slugify(item.name)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(
                item.model_copy(update={"is_ephemeral": False, "context_kind": "catalog"})
            )
        await self.sessions.set_nearby_pois(self.session_id, merged[:MAX_NEARBY_POIS])
        await self.sessions.set_ephemeral_map_pois(
            self.session_id,
            [item for item in state.ephemeral_map_pois if slugify(item.name) != slugify(poi.name)],
        )
        await self.sessions.set_active_poi(self.session_id, poi)

    # ---- candidate bookkeeping ------------------------------------------

    async def _store_candidates(self, pois: list[SessionPoi]) -> None:
        state = await self.sessions.get_or_create(self.session_id)
        await self.sessions.update_session(
            self.session_id,
            user_id=None, profile_context=None, profile_preferences=None,
            lat=None, lng=None, active_poi_name=None,
            metadata={
                **state.metadata,
                CANDIDATES_METADATA_KEY: [poi.model_dump() for poi in pois],
            },
        )

    async def _load_candidates(self) -> list[SessionPoi]:
        state = await self.sessions.get_or_create(self.session_id)
        raw = state.metadata.get(CANDIDATES_METADATA_KEY) or []
        candidates: list[SessionPoi] = []
        for item in raw:
            try:
                candidates.append(SessionPoi(**item))
            except (TypeError, ValueError):
                continue
        return candidates

    async def _resolve_candidate(self, poi_name: str) -> SessionPoi | None:
        wanted = clean_text(poi_name).lower()
        state = await self.sessions.get_or_create(self.session_id)
        for poi in [
            *await self._load_candidates(),
            *state.ephemeral_map_pois,
            *state.nearby_pois,
        ]:
            if clean_text(poi.name).lower() == wanted:
                return poi
        return None


def _coords(
    arguments: dict[str, Any], state: SessionStateView
) -> tuple[float | None, float | None]:
    lat = arguments.get("lat", state.location.lat)
    lng = arguments.get("lng", state.location.lng)
    try:
        return (
            float(lat) if lat is not None else None,
            float(lng) if lng is not None else None,
        )
    except (TypeError, ValueError):
        return state.location.lat, state.location.lng


def _limit(arguments: dict[str, Any]) -> int:
    try:
        return max(1, min(int(arguments.get("limit") or 5), 8))
    except (TypeError, ValueError):
        return 5


def _merge_ephemeral(existing: list[SessionPoi], new: list[SessionPoi]) -> list[SessionPoi]:
    merged: list[SessionPoi] = []
    seen: set[str] = set()
    for poi in [*existing, *new]:
        key = (poi.google_place_id or poi.id or poi.name).lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(poi.model_copy(update={"is_ephemeral": True}))
    return merged


def _km_from(candidate: SessionPoi, city: City) -> float | None:
    if city.lat is None or city.lng is None:
        return None
    return distance_km(candidate.lat, candidate.lng, float(city.lat), float(city.lng))


def _catalog_row_to_session_poi(row: Poi) -> SessionPoi:
    return SessionPoi(
        id=str(row.id),
        name=row.name,
        lat=float(row.lat) if row.lat is not None else 0.0,
        lng=float(row.lng) if row.lng is not None else 0.0,
        description=row.short_description or row.long_description or "",
        summary=row.long_description or row.short_description or "",
        source_of_truth="catalog",
        is_ephemeral=False,
        google_place_id=row.google_place_id or "",
        context_kind="catalog",
    )


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
