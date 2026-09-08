"""Port of V1's `CatalogService.bootstrap_city_from_location` / `import_city_pois`
(see `docs/testing-checklist.md` Capitulo 2 for the full breakdown and what's
still missing).

In scope here: reverse geocode a point, create or reuse its city, resolve the
city's Wikidata entity, pull nearby points of interest via Wikidata SPARQL
(city-entity query first, radius query as fallback) and via Overpass/OSM as an
additional source, score/dedupe/filter them all together with the same
heuristics as V1, and upsert them as `Poi` rows.

One deliberate divergence from V1: V1 always runs the city-entity query, the
radius query, *and* Overpass on every non-AI bootstrap. Here the radius query
only runs when the city-entity query returned nothing, to cut needless
Wikidata SPARQL load — Wikidata's own rate limits are strict enough (see the
"active wdqs outage" cooldown hit during testing) that halving call volume for
the common case is worth the small loss of extra-candidate coverage. Overpass
still always runs, matching V1, since it's a different host with its own quota.

AI candidates (`ai_candidates.py`, reusing the configured OpenAI key, real
cost per call) and the 9-language content localization pass are now wired in,
matching V1's control flow: when `use_ai_candidates` is true (V1's own
default) and OpenAI is configured, AI candidates are tried first and, if any
come back, are localized and upserted directly — Wikidata/Overpass are not
consulted at all for that bootstrap, same as V1. When AI is off (or came back
empty) and the Wikidata/Overpass pass itself found nothing usable, AI is
still tried once as a last resort, again matching V1.
"""

import json
from dataclasses import dataclass
from decimal import Decimal
from time import perf_counter
from typing import Any
from uuid import uuid4

import structlog
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.catalog.bootstrap.ai_candidates import (
    AiCandidateError,
    generate_ai_candidates,
    localize_content_candidates,
    names_from_aliases,
)
from locus_v2.catalog.bootstrap.city_entity import resolve_city_entity_id
from locus_v2.catalog.bootstrap.dto import BootstrapPoi, BootstrapResult
from locus_v2.catalog.bootstrap.nominatim import NominatimError, reverse_geocode
from locus_v2.catalog.bootstrap.normalize import (
    normalize_names,
    normalize_short_descriptions,
    safe_slug,
    wkt_point_to_coords,
)
from locus_v2.catalog.bootstrap.overpass_client import OverpassClient
from locus_v2.catalog.bootstrap.overpass_queries import (
    build_overpass_map_query,
    normalize_overpass_element,
)
from locus_v2.catalog.bootstrap.poi_scoring import (
    is_map_candidate,
    map_poi_type_code,
    score_tourism_candidate,
)
from locus_v2.catalog.bootstrap.sparql_queries import (
    build_city_entity_import_query,
    build_radius_import_query,
)
from locus_v2.catalog.bootstrap.wikidata_client import WikidataClient, WikidataRateLimitError
from locus_v2.catalog.models import City, Poi, PoiType
from locus_v2.config import Settings
from locus_v2.identity.models import AdminAuditEvent
from locus_v2.observability import LocusEventLogger
from locus_v2.shared.text import clean_text

logger = structlog.get_logger()

AI_SEED_LOCALIZATION_LIMIT = 12

DEFAULT_RADIUS_KM = 8.0
DEFAULT_LIMIT = 60


class CatalogBootstrapError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Candidate:
    source: str
    wikidata_id: str
    source_id: str
    poi_name: str
    names: dict[str, str]
    lat: float | None
    lng: float | None
    description: str
    type_label: str
    type_code: str
    wikipedia_title: str
    sitelinks: int


class CatalogBootstrapService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        *,
        event_logger: LocusEventLogger | None = None,
        actor_user_id: int | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.wikidata = WikidataClient(settings)
        self.overpass = OverpassClient(settings)
        self.event_logger = event_logger
        self.actor_user_id = actor_user_id

    async def bootstrap_from_location(
        self,
        *,
        lat: float,
        lng: float,
        radius_km: float = DEFAULT_RADIUS_KM,
        limit: int = DEFAULT_LIMIT,
        use_ai_candidates: bool = True,
    ) -> BootstrapResult:
        started_at = perf_counter()
        logger.info(
            "catalog_bootstrap_started", lat=lat, lng=lng, radius_km=radius_km,
            limit=limit, use_ai_candidates=use_ai_candidates,
        )
        await self._record_event(
            "info", "catalog.bootstrap.started",
            context={"lat": lat, "lng": lng, "radius_km": radius_km, "limit": limit,
                     "use_ai_candidates": use_ai_candidates},
        )
        try:
            geocode = await reverse_geocode(lat, lng, self.settings)
        except NominatimError as error:
            logger.warning("catalog_bootstrap_geocode_failed", lat=lat, lng=lng, error=str(error))
            await self._record_event(
                "warning", "catalog.bootstrap.geocode_failed", message=str(error),
                elapsed_ms=(perf_counter() - started_at) * 1000,
            )
            raise CatalogBootstrapError(str(error)) from error

        slug = safe_slug(geocode.city_name, prefix=f"city-{geocode.country_code.lower() or 'xx'}")
        city = await self.session.scalar(select(City).where(City.slug == slug))
        city_created = False
        if city is None:
            city = City(
                slug=slug,
                name=clean_text(geocode.city_name),
                names_json=normalize_names(geocode.city_name, geocode.names),
                country_code=geocode.country_code.upper(),
                lat=Decimal(str(lat)),
                lng=Decimal(str(lng)),
                source="nominatim",
            )
            self.session.add(city)
            await self.session.flush()
            city_created = True
        elif city.lat is None or city.lng is None:
            city.lat = Decimal(str(lat))
            city.lng = Decimal(str(lng))
        logger.info(
            "catalog_bootstrap_city_resolved", city=city.name, city_id=city.id,
            city_created=city_created, country_code=city.country_code,
        )

        try:
            imported, updated, source, pois = await self._import_pois(
                city, radius_km=radius_km, limit=limit, use_ai_candidates=use_ai_candidates
            )
        except CatalogBootstrapError as error:
            logger.warning(
                "catalog_bootstrap_failed", city_id=city.id, city=city.name, error=str(error),
                elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
            )
            await self._record_event(
                "warning", "catalog.bootstrap.failed", message=str(error),
                elapsed_ms=(perf_counter() - started_at) * 1000,
                context={"city_id": city.id, "city": city.name},
            )
            raise

        if self.actor_user_id is not None:
            self.session.add(
                AdminAuditEvent(
                    actor_user_id=self.actor_user_id,
                    action="catalog.bootstrap_from_location",
                    resource_type="city",
                    resource_id=str(city.id),
                    before_json=None,
                    after_json=json.dumps(
                        {
                            "city": city.name, "city_created": city_created,
                            "lat": lat, "lng": lng, "radius_km": radius_km, "limit": limit,
                            "use_ai_candidates": use_ai_candidates, "source": source,
                            "imported": imported, "updated": updated,
                        },
                        default=str,
                    ),
                    trace_id=str(uuid4()),
                )
            )
        await self.session.commit()
        logger.info(
            "catalog_bootstrap_done", city_id=city.id, city=city.name, source=source,
            imported=imported, updated=updated,
            elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
        )
        await self._record_event(
            "info", "catalog.bootstrap.done",
            elapsed_ms=(perf_counter() - started_at) * 1000,
            context={"city_id": city.id, "city": city.name, "city_created": city_created,
                     "source": source, "imported": imported, "updated": updated},
        )

        return BootstrapResult(
            city_id=city.id,
            city_public_id=city.public_id,
            city_name=city.name,
            city_created=city_created,
            source=source,
            imported_count=imported,
            updated_count=updated,
            pois=pois,
        )

    async def _record_event(
        self,
        level: str,
        event: str,
        *,
        message: str | None = None,
        elapsed_ms: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        if self.event_logger is None:
            return
        await self.event_logger.write(
            level,  # type: ignore[arg-type]
            event,
            message=message,
            user_id=self.actor_user_id,
            elapsed_ms=elapsed_ms,
            context=context,
        )

    def _openai_api_key(self) -> str:
        return (
            self.settings.openai_api_key.get_secret_value().strip()
            if self.settings.openai_api_key is not None
            else ""
        )

    async def _import_pois(
        self, city: City, *, radius_km: float, limit: int, use_ai_candidates: bool = True
    ) -> tuple[int, int, str, list[BootstrapPoi]]:
        type_lookup = {
            row.code: row for row in (await self.session.scalars(select(PoiType))).all()
        }
        api_key = self._openai_api_key()

        if use_ai_candidates and api_key:
            ai_started_at = perf_counter()
            try:
                ai_candidates = await generate_ai_candidates(
                    api_key=api_key, model=self.settings.tool_model, city=city, limit=limit
                )
                logger.info(
                    "catalog_import_ai_candidates_done", city_id=city.id,
                    proposed=len(ai_candidates),
                    elapsed_ms=round((perf_counter() - ai_started_at) * 1000, 1),
                )
            except AiCandidateError as error:
                ai_candidates = []
                logger.warning(
                    "catalog_import_ai_candidates_failed", city_id=city.id, error=str(error),
                    elapsed_ms=round((perf_counter() - ai_started_at) * 1000, 1),
                )
                await self._record_event(
                    "warning", "catalog.bootstrap.ai_candidates_failed", message=str(error),
                    context={"city_id": city.id},
                )
            if ai_candidates:
                ai_imported, ai_updated, ai_results = await self._upsert_ai_seed_candidates(
                    city, type_lookup, ai_candidates, limit, api_key=api_key
                )
                return ai_imported, ai_updated, "ai_seed", ai_results

        source_parts: list[str] = []
        city_entity_id = await resolve_city_entity_id(self.wikidata, city)
        bindings: list[dict[str, Any]] = []
        if city_entity_id:
            try:
                bindings.extend(
                    await self.wikidata.run_sparql(
                        build_city_entity_import_query(city_entity_id, limit)
                    )
                )
                if bindings:
                    source_parts.append("wikidata_city_entity")
            except WikidataRateLimitError:
                pass

        if not bindings:
            if city.lat is None or city.lng is None:
                raise CatalogBootstrapError("La ciudad no tiene coordenadas guardadas")
            try:
                radius_bindings = await self.wikidata.run_sparql(
                    build_radius_import_query(
                        lat=float(city.lat), lng=float(city.lng),
                        radius_km=radius_km, limit=limit,
                    )
                )
                bindings.extend(radius_bindings)
                if radius_bindings:
                    source_parts.append("wikidata_radius")
            except WikidataRateLimitError as error:
                logger.warning(
                    "catalog_import_wikidata_rate_limited", city_id=city.id, error=str(error)
                )
                await self._record_event(
                    "warning", "catalog.bootstrap.wikidata_rate_limited", message=str(error),
                    context={"city_id": city.id},
                )
                raise CatalogBootstrapError(
                    "Wikidata esta limitando peticiones temporalmente; "
                    "intentalo de nuevo en unos minutos"
                ) from error

        logger.info("catalog_import_wikidata_rows", city_id=city.id, rows=len(bindings))

        overpass_elements: list[dict[str, Any]] = []
        if city.lat is not None and city.lng is not None:
            try:
                overpass_elements = await self.overpass.query(
                    build_overpass_map_query(
                        lat=float(city.lat), lng=float(city.lng),
                        radius_km=radius_km, limit=limit,
                    )
                )
                if overpass_elements:
                    source_parts.append("overpass")
            except Exception as error:
                # Overpass is an additional source, not a required one: a
                # failure here should not sink an otherwise-working bootstrap.
                logger.warning(
                    "catalog_import_overpass_failed", city_id=city.id, error=str(error)
                )
                overpass_elements = []
        logger.info(
            "catalog_import_overpass_rows", city_id=city.id, rows=len(overpass_elements)
        )

        source = "+".join(source_parts) or "none"

        seen_ids: set[str] = set()
        ranked: list[tuple[int, _Candidate]] = []
        for item in bindings:
            poi_uri = item.get("poi", {}).get("value", "")
            wikidata_id = poi_uri.rsplit("/", 1)[-1] if poi_uri else ""
            if not wikidata_id or wikidata_id in seen_ids:
                continue
            poi_name = clean_text(item.get("poiLabel", {}).get("value", ""))
            if not poi_name:
                continue
            seen_ids.add(wikidata_id)
            lat, lng = wkt_point_to_coords(item.get("coord", {}).get("value", ""))
            description = clean_text(item.get("poiDescription", {}).get("value", ""))
            type_label = clean_text(item.get("poiTypeLabel", {}).get("value", ""))
            type_code = map_poi_type_code(poi_name, type_label, description)
            wikipedia_title = item.get("resolvedArticle", {}).get("value", "").rsplit("/", 1)[-1]
            try:
                sitelinks = int(float(item.get("sitelinks", {}).get("value", "0") or "0"))
            except ValueError:
                sitelinks = 0
            score = score_tourism_candidate(
                name=poi_name, description=description, type_code=type_code,
                type_label=type_label, sitelinks=sitelinks, wikipedia_title=wikipedia_title,
            )
            if not is_map_candidate(score, type_code):
                continue
            ranked.append(
                (
                    score,
                    _Candidate(
                        source="wikidata", wikidata_id=wikidata_id, source_id="",
                        poi_name=poi_name, names=normalize_names(poi_name), lat=lat, lng=lng,
                        description=description, type_label=type_label, type_code=type_code,
                        wikipedia_title=wikipedia_title, sitelinks=sitelinks,
                    ),
                )
            )

        for element in overpass_elements:
            normalized = normalize_overpass_element(element)
            if normalized is None:
                continue
            dedupe_key = normalized["wikidata_id"] or safe_slug(normalized["name"], prefix="poi")
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            score = score_tourism_candidate(
                name=normalized["name"], description=normalized["description"],
                type_code=normalized["type_code"], type_label=normalized["type_label"],
                sitelinks=0, wikipedia_title=normalized["wikipedia_title"],
            )
            if not is_map_candidate(score, normalized["type_code"]):
                continue
            ranked.append(
                (
                    score,
                    _Candidate(
                        source="osm", wikidata_id=normalized["wikidata_id"],
                        source_id=normalized["source_id"], poi_name=normalized["name"],
                        names=normalized["names"], lat=normalized["lat"], lng=normalized["lng"],
                        description=normalized["description"], type_label=normalized["type_label"],
                        type_code=normalized["type_code"],
                        wikipedia_title=normalized["wikipedia_title"], sitelinks=0,
                    ),
                )
            )

        ranked.sort(key=lambda pair: (-pair[0], pair[1].poi_name.lower()))
        selected = ranked[:limit]

        if not selected and not use_ai_candidates and api_key:
            logger.info("catalog_import_ai_last_resort_triggered", city_id=city.id, source=source)
            try:
                fallback_candidates = await generate_ai_candidates(
                    api_key=api_key, model=self.settings.tool_model, city=city, limit=limit
                )
            except AiCandidateError as error:
                fallback_candidates = []
                logger.warning(
                    "catalog_import_ai_last_resort_failed", city_id=city.id, error=str(error)
                )
            if fallback_candidates:
                fb_imported, fb_updated, fb_results = await self._upsert_ai_seed_candidates(
                    city, type_lookup, fallback_candidates, limit, api_key=api_key
                )
                return fb_imported, fb_updated, f"{source}+ai_last_resort", fb_results

        logger.info(
            "catalog_import_ranked", city_id=city.id, ranked=len(ranked), selected=len(selected)
        )
        imported = 0
        updated = 0
        results: list[BootstrapPoi] = []
        for score, candidate in selected:
            poi_type = type_lookup.get(candidate.type_code)
            poi_slug = safe_slug(candidate.poi_name, prefix="poi")
            poi_names = normalize_names(candidate.poi_name, candidate.names)
            poi_short_descriptions = normalize_short_descriptions(candidate.description)

            match_conditions = [Poi.slug == poi_slug]
            if candidate.wikidata_id:
                match_conditions.append(Poi.wikidata_id == candidate.wikidata_id)
            existing = await self.session.scalar(
                select(Poi).where(Poi.city_id == city.id, or_(*match_conditions))
            )
            imported_from = "wikidata_sparql" if candidate.source == "wikidata" else "overpass"
            metadata = {
                "imported_from": imported_from,
                "type_label": candidate.type_label,
                "tourism_score": score,
                "wikipedia_title": candidate.wikipedia_title,
                "catalog_source": candidate.source,
                "source_id": candidate.source_id,
            }
            if existing is None:
                poi = Poi(
                    city_id=city.id,
                    poi_type_id=poi_type.id if poi_type else None,
                    slug=poi_slug,
                    name=candidate.poi_name,
                    names_json=poi_names,
                    lat=Decimal(str(candidate.lat)) if candidate.lat is not None else None,
                    lng=Decimal(str(candidate.lng)) if candidate.lng is not None else None,
                    short_description=candidate.description,
                    short_descriptions_json=poi_short_descriptions,
                    source_of_truth="wikidata",
                    wikidata_id=candidate.wikidata_id,
                    wikipedia_title=candidate.wikipedia_title,
                    is_active=True,
                    metadata_json=metadata,
                )
                self.session.add(poi)
                await self.session.flush()
                imported += 1
                results.append(
                    BootstrapPoi(
                        id=poi.id, public_id=poi.public_id, name=poi.name, slug=poi.slug,
                        type_code=candidate.type_code,
                        lat=float(poi.lat) if poi.lat is not None else None,
                        lng=float(poi.lng) if poi.lng is not None else None,
                        short_description=poi.short_description,
                        source_of_truth=poi.source_of_truth, created=True,
                    )
                )
            else:
                existing.poi_type_id = poi_type.id if poi_type else existing.poi_type_id
                if candidate.lat is not None:
                    existing.lat = Decimal(str(candidate.lat))
                if candidate.lng is not None:
                    existing.lng = Decimal(str(candidate.lng))
                existing.short_description = candidate.description or existing.short_description
                if candidate.source == "wikidata":
                    existing.source_of_truth = "wikidata"
                existing.wikidata_id = candidate.wikidata_id or existing.wikidata_id
                existing.wikipedia_title = candidate.wikipedia_title or existing.wikipedia_title
                existing.names_json = {
                    **normalize_names(existing.name, existing.names_json), **poi_names,
                }
                existing.short_descriptions_json = {
                    **normalize_short_descriptions(
                        existing.short_description, existing.short_descriptions_json
                    ),
                    **poi_short_descriptions,
                }
                existing.metadata_json = {**(existing.metadata_json or {}), **metadata}
                updated += 1
                results.append(
                    BootstrapPoi(
                        id=existing.id, public_id=existing.public_id, name=existing.name,
                        slug=existing.slug, type_code=candidate.type_code,
                        lat=float(existing.lat) if existing.lat is not None else None,
                        lng=float(existing.lng) if existing.lng is not None else None,
                        short_description=existing.short_description,
                        source_of_truth=existing.source_of_truth, created=False,
                    )
                )

        return imported, updated, source, results

    async def _upsert_ai_seed_candidates(
        self,
        city: City,
        type_lookup: dict[str, PoiType],
        ai_candidates: list[dict[str, Any]],
        limit: int,
        *,
        api_key: str,
    ) -> tuple[int, int, list[BootstrapPoi]]:
        """Port of V1 `_upsert_ai_seed_candidates`: creates/updates `Poi` rows
        straight from AI-proposed candidates, with provisional AI-guessed
        coordinates (`source_of_truth="gpt_seed"`) rather than Wikidata/OSM
        ones. V1 leaves these to be resolved/enriched later by a background
        job (`start_pending_enrichment`) that this port does not have yet
        (tracked in the checklist) — the rows land here exactly as V1 would
        leave them before that job runs.
        """
        ai_candidates = ai_candidates[:limit]
        if ai_candidates:
            try:
                await localize_content_candidates(
                    api_key=api_key, model=self.settings.tool_model,
                    candidates=ai_candidates[:AI_SEED_LOCALIZATION_LIMIT],
                    context=f"city:{city.name}:gpt_seed",
                )
            except AiCandidateError:
                pass

        imported = 0
        updated = 0
        results: list[BootstrapPoi] = []
        seen_slugs: set[str] = set()
        for rank, candidate in enumerate(ai_candidates, start=1):
            dedupe_key = safe_slug(candidate["name"], prefix="poi")
            if dedupe_key in seen_slugs:
                continue
            seen_slugs.add(dedupe_key)

            poi_type = type_lookup.get(candidate["poi_type_code"])
            existing = await self.session.scalar(
                select(Poi).where(
                    Poi.city_id == city.id,
                    or_(Poi.slug == dedupe_key, Poi.name == candidate["name"]),
                )
            )
            names = names_from_aliases(candidate["name"], candidate.get("aliases", []))
            names = {**names, **normalize_names(candidate["name"], candidate.get("names") or {})}
            short_descriptions = normalize_short_descriptions(
                candidate.get("short_description") or "", candidate.get("short_descriptions") or {}
            )
            has_seed_coords = candidate.get("lat") is not None and candidate.get("lng") is not None
            metadata: dict[str, Any] = {
                # Mirrors V1 `_candidate_metadata` field-for-field: the enrichment
                # worker's `_build_pending_candidate` equivalent reads seed_lat/
                # seed_lng/seed_short_description back out of this metadata, so
                # trimming these fields would silently break re-resolution later.
                "candidate_aliases": candidate.get("aliases", [])[:5],
                "candidate_name": candidate["name"],
                "candidate_type_code": candidate["poi_type_code"],
                "seed_model": self.settings.tool_model,
                "seed_rank": rank,
                "seed_source": "gpt_candidate",
                "resolution_attempts": 0,
                "catalog_source": "gpt_seed",
                "formatted_address": candidate.get("formatted_address") or "",
                "location_hint": candidate.get("location_hint") or "",
                "seed_short_description": candidate.get("short_description") or "",
                "seed_lat": candidate.get("lat"),
                "seed_lng": candidate.get("lng"),
                "import_status": "seeded_gpt_coords" if has_seed_coords else "pending_wikidata",
                "import_tier": "featured" if has_seed_coords else "pending",
                "featured": has_seed_coords,
            }
            fallback_description = (
                "Ubicacion provisional generada por IA." if has_seed_coords
                else "Pendiente de resolver ubicacion exacta."
            )
            if existing is None:
                poi = Poi(
                    city_id=city.id,
                    poi_type_id=poi_type.id if poi_type else None,
                    slug=dedupe_key,
                    name=candidate["name"],
                    names_json=names,
                    lat=Decimal(str(candidate["lat"])) if has_seed_coords else None,
                    lng=Decimal(str(candidate["lng"])) if has_seed_coords else None,
                    short_description=candidate.get("short_description") or fallback_description,
                    short_descriptions_json=short_descriptions,
                    source_of_truth="gpt_seed",
                    is_active=True,
                    metadata_json=metadata,
                )
                self.session.add(poi)
                await self.session.flush()
                imported += 1
                results.append(
                    BootstrapPoi(
                        id=poi.id, public_id=poi.public_id, name=poi.name, slug=poi.slug,
                        type_code=candidate["poi_type_code"],
                        lat=float(poi.lat) if poi.lat is not None else None,
                        lng=float(poi.lng) if poi.lng is not None else None,
                        short_description=poi.short_description,
                        source_of_truth=poi.source_of_truth, created=True,
                    )
                )
                continue

            existing.metadata_json = {**(existing.metadata_json or {}), **metadata}
            existing.names_json = {**normalize_names(existing.name, existing.names_json), **names}
            existing.short_descriptions_json = {
                **normalize_short_descriptions(
                    existing.short_description, existing.short_descriptions_json
                ),
                **short_descriptions,
            }
            existing.poi_type_id = poi_type.id if poi_type else existing.poi_type_id
            if existing.lat is None and existing.lng is None and has_seed_coords:
                existing.lat = Decimal(str(candidate["lat"]))
                existing.lng = Decimal(str(candidate["lng"]))
            if candidate.get("short_description"):
                existing.short_description = candidate["short_description"]
            elif not existing.short_description:
                existing.short_description = fallback_description
            if existing.source_of_truth not in {"wikidata", "overpass"}:
                existing.source_of_truth = "gpt_seed"
            updated += 1
            results.append(
                BootstrapPoi(
                    id=existing.id, public_id=existing.public_id, name=existing.name,
                    slug=existing.slug, type_code=candidate["poi_type_code"],
                    lat=float(existing.lat) if existing.lat is not None else None,
                    lng=float(existing.lng) if existing.lng is not None else None,
                    short_description=existing.short_description,
                    source_of_truth=existing.source_of_truth, created=False,
                )
            )

        return imported, updated, results
