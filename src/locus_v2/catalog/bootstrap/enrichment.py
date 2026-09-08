"""Background enrichment worker, ported from V1
`CatalogService.enrich_city_pending_pois` / `_resolve_ai_candidate` /
`_resolve_candidate_with_overpass` / `start_pending_enrichment`.

V1 fires this as a daemon thread inside the same process right after a
bootstrap that used AI candidates. V2 has no such thread pool in its async
web process, so the equivalent here is a FastAPI `BackgroundTasks` callback
scheduled from the bootstrap endpoint (`api/admin_catalog.py`) — same
architectural role (fire-and-forget, same process, best-effort), just
`asyncio`-native instead of a Python thread.

Every POI created by the AI-seed path (`service.py::_upsert_ai_seed_candidates`)
lands with `import_status` in `pending_wikidata` (no coordinates at all) or
`seeded_gpt_coords` (provisional AI coordinates). This worker tries, once per
run, to resolve each such POI against real Wikidata data (and Overpass as a
fallback), replacing the provisional data with authoritative coordinates and
marking it `resolved` — or `unresolved` / `rate_limited_retry` when it can't.
"""

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from time import perf_counter
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.catalog.bootstrap.city_entity import resolve_city_entity_id
from locus_v2.catalog.bootstrap.overpass_client import OverpassClient
from locus_v2.catalog.bootstrap.overpass_queries import (
    build_overpass_name_query,
    normalize_overpass_element,
)
from locus_v2.catalog.bootstrap.poi_scoring import (
    build_ai_search_terms,
    distance_km,
    extract_entity_coords,
    score_ai_entity_candidate,
    score_wikidata_resolution,
)
from locus_v2.catalog.bootstrap.wikidata_client import WikidataClient, WikidataRateLimitError
from locus_v2.catalog.models import City, Poi
from locus_v2.config import Settings
from locus_v2.observability import LocusEventLogger
from locus_v2.shared.text import clean_text

logger = structlog.get_logger()

PENDING_STATUSES = {"pending_wikidata", "retry_wikidata", "rate_limited_retry"}
THROTTLE_SECONDS = 0.15


@dataclass(frozen=True)
class EnrichmentSummary:
    city_id: int
    eligible: int
    processed: int
    resolved: int
    overpass_resolved: int
    unresolved: int
    rate_limited: bool


class CatalogEnrichmentService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        *,
        event_logger: LocusEventLogger | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.wikidata = WikidataClient(settings)
        self.overpass = OverpassClient(settings)
        self.event_logger = event_logger

    async def enrich_city_pending_pois(self, city_id: int, limit: int = 150) -> EnrichmentSummary:
        started_at = perf_counter()
        logger.info("catalog_enrichment_started", city_id=city_id, limit=limit)
        city = await self.session.get(City, city_id)
        if city is None:
            logger.warning("catalog_enrichment_city_missing", city_id=city_id)
            return EnrichmentSummary(city_id, 0, 0, 0, 0, 0, False)

        city_entity_id = await resolve_city_entity_id(self.wikidata, city)
        pending_pois = (
            await self.session.scalars(
                select(Poi).where(Poi.city_id == city.id).order_by(Poi.id.asc())
            )
        ).all()

        eligible = processed = resolved_count = overpass_count = unresolved_count = 0
        rate_limited = False
        for poi in pending_pois:
            metadata = dict(poi.metadata_json or {})
            status = metadata.get("import_status", "")
            if status not in PENDING_STATUSES:
                continue
            eligible += 1
            candidate = _build_pending_candidate(poi)
            metadata["resolution_attempts"] = int(metadata.get("resolution_attempts", 0)) + 1

            resolved, reason = await self._resolve_ai_candidate(city, city_entity_id, candidate)
            if resolved is not None:
                if resolved["lat"] is not None:
                    poi.lat = Decimal(str(resolved["lat"]))
                if resolved["lng"] is not None:
                    poi.lng = Decimal(str(resolved["lng"]))
                poi.wikidata_id = resolved["wikidata_id"] or poi.wikidata_id
                poi.wikipedia_title = resolved["wikipedia_title"] or poi.wikipedia_title
                if resolved["description"]:
                    poi.short_description = resolved["description"]
                poi.source_of_truth = "wikidata"
                metadata.update(
                    import_status="resolved", import_tier="featured",
                    catalog_source=resolved["source"], resolution_reason="wikidata",
                    resolution_score=resolved.get("resolution_score"),
                    distance_km=resolved.get("distance_km"),
                )
                resolved_count += 1
            else:
                overpass_resolved = None
                if reason != "wikidata_rate_limited":
                    overpass_resolved = await self._resolve_candidate_with_overpass(city, candidate)
                if overpass_resolved is not None:
                    if overpass_resolved["lat"] is not None:
                        poi.lat = Decimal(str(overpass_resolved["lat"]))
                    if overpass_resolved["lng"] is not None:
                        poi.lng = Decimal(str(overpass_resolved["lng"]))
                    poi.wikipedia_title = (
                        overpass_resolved["wikipedia_title"] or poi.wikipedia_title
                    )
                    poi.source_of_truth = "overpass"
                    metadata.update(
                        import_status="resolved", import_tier="map", catalog_source="overpass",
                        source_id=overpass_resolved.get("source_id", ""),
                        resolution_reason="overpass_fallback",
                        distance_km=overpass_resolved.get("distance_km"),
                    )
                    overpass_count += 1
                elif reason == "wikidata_rate_limited":
                    metadata["import_status"] = "rate_limited_retry"
                    metadata["resolution_reason"] = reason
                    poi.metadata_json = metadata
                    await self.session.flush()
                    rate_limited = True
                    break
                else:
                    metadata["import_status"] = "unresolved"
                    metadata["resolution_reason"] = reason
                    unresolved_count += 1
            poi.metadata_json = metadata
            processed += 1
            await self.session.flush()
            if processed >= limit:
                break
            await asyncio.sleep(THROTTLE_SECONDS)

        await self.session.commit()
        elapsed_ms = round((perf_counter() - started_at) * 1000, 1)
        logger.info(
            "catalog_enrichment_done", city_id=city.id, city=city.name, eligible=eligible,
            processed=processed, resolved=resolved_count, overpass_resolved=overpass_count,
            unresolved=unresolved_count, rate_limited=rate_limited, elapsed_ms=elapsed_ms,
        )
        if self.event_logger is not None:
            await self.event_logger.write(
                "info", "catalog.enrichment.done", elapsed_ms=elapsed_ms,
                context={
                    "city_id": city.id, "city": city.name, "eligible": eligible,
                    "processed": processed, "resolved": resolved_count,
                    "overpass_resolved": overpass_count, "unresolved": unresolved_count,
                    "rate_limited": rate_limited,
                },
            )
        return EnrichmentSummary(
            city_id=city.id, eligible=eligible, processed=processed,
            resolved=resolved_count, overpass_resolved=overpass_count,
            unresolved=unresolved_count, rate_limited=rate_limited,
        )

    async def _resolve_ai_candidate(
        self, city: City, city_entity_id: str | None, candidate: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        search_terms = build_ai_search_terms(
            city_name=city.name, country_code=city.country_code, candidate=candidate
        )
        prelim: dict[str, dict[str, Any]] = {}
        search_failed = False
        for term in search_terms:
            try:
                search_results = await self.wikidata.search_entities(term, limit=5)
            except WikidataRateLimitError:
                return None, "wikidata_rate_limited"
            for result in search_results:
                result_id = result.get("id", "")
                if not result_id:
                    continue
                score = score_wikidata_resolution_for(city, candidate, result)
                current = prelim.get(result_id)
                if current is None or score > current["score"]:
                    prelim[result_id] = {"result": result, "score": score}

        if not prelim:
            return None, "search_no_match"

        ranked = sorted(prelim.values(), key=lambda item: item["score"], reverse=True)[:3]
        entity_ids = [item["result"]["id"] for item in ranked if item.get("result", {}).get("id")]
        try:
            entity_lookup = await self.wikidata.get_entities(entity_ids)
        except WikidataRateLimitError:
            return None, "wikidata_rate_limited"

        best_score = -9999
        best_reason = "low_confidence"
        best_distance: float | None = None
        best_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
        for item in ranked:
            result = item["result"]
            entity = entity_lookup.get(result["id"])
            if not entity:
                search_failed = True
                continue
            score, dist, reason = score_ai_entity_candidate(
                city_name=city.name, country_code=city.country_code,
                city_lat=float(city.lat) if city.lat is not None else None,
                city_lng=float(city.lng) if city.lng is not None else None,
                city_entity_id=city_entity_id, candidate=candidate, result=result, entity=entity,
            )
            if score > best_score:
                best_score, best_reason, best_distance = score, reason, dist
                best_pair = (result, entity)

        if best_pair is None or best_reason != "resolved":
            if search_failed and best_pair is None:
                return None, "wikidata_rate_limited"
            return None, best_reason

        best_result, entity = best_pair
        lat, lng = extract_entity_coords(entity)
        sitelinks = entity.get("sitelinks", {})
        wikipedia_title = ""
        if "eswiki" in sitelinks:
            wikipedia_title = sitelinks["eswiki"].get("title", "")
        elif "enwiki" in sitelinks:
            wikipedia_title = sitelinks["enwiki"].get("title", "")
        description = (
            entity.get("descriptions", {}).get("es", {}).get("value")
            or best_result.get("description", "")
            or ""
        )
        return (
            {
                "source": "wikidata_ai",
                "wikidata_id": best_result["id"],
                "lat": lat,
                "lng": lng,
                "description": clean_text(description),
                "wikipedia_title": wikipedia_title,
                "resolution_score": best_score,
                "distance_km": round(best_distance, 2) if best_distance is not None else None,
            },
            "resolved",
        )

    async def _resolve_candidate_with_overpass(
        self, city: City, candidate: dict[str, Any]
    ) -> dict[str, Any] | None:
        if city.lat is None or city.lng is None:
            return None
        query = build_overpass_name_query(
            lat=float(city.lat), lng=float(city.lng),
            name=candidate["name"], aliases=candidate.get("aliases", []),
        )
        if not query:
            return None
        try:
            elements = await self.overpass.query(query)
        except Exception:
            return None
        normalized = [
            item for item in (normalize_overpass_element(el) for el in elements) if item
        ]
        if not normalized:
            return None
        city_lat, city_lng = float(city.lat), float(city.lng)
        normalized.sort(
            key=lambda item: (
                0
                if clean_text(item["name"]).lower() == clean_text(candidate["name"]).lower()
                else 1,
                distance_km(city_lat, city_lng, item["lat"], item["lng"]) or 9999,
            )
        )
        best = normalized[0]
        dist = distance_km(city_lat, city_lng, best["lat"], best["lng"]) or 0
        return {
            "source_id": best["source_id"],
            "lat": best["lat"],
            "lng": best["lng"],
            "wikipedia_title": best["wikipedia_title"],
            "distance_km": round(dist, 2),
        }


def _build_pending_candidate(poi: Poi) -> dict[str, Any]:
    metadata = dict(poi.metadata_json or {})
    aliases = metadata.get("candidate_aliases") or []
    return {
        "name": poi.name,
        "poi_type_code": clean_text(metadata.get("candidate_type_code") or "") or "building",
        "aliases": [clean_text(alias) for alias in aliases if clean_text(alias)],
        "short_description": clean_text(
            metadata.get("seed_short_description") or poi.short_description
        ),
        "lat": metadata.get("seed_lat"),
        "lng": metadata.get("seed_lng"),
    }


def score_wikidata_resolution_for(
    city: City, candidate: dict[str, Any], result: dict[str, Any]
) -> int:
    return score_wikidata_resolution(
        city_name=city.name, country_code=city.country_code,
        candidate_name=candidate["name"], result=result, aliases=candidate.get("aliases", []),
    )
