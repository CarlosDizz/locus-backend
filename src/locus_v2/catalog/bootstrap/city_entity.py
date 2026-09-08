"""Shared by `service.py` (bootstrap) and `enrichment.py` (background
resolution): finds the Wikidata entity that represents a given `City` row.
"""

from locus_v2.catalog.bootstrap.poi_scoring import score_city_candidate
from locus_v2.catalog.bootstrap.wikidata_client import WikidataClient, WikidataRateLimitError
from locus_v2.catalog.models import City


async def resolve_city_entity_id(wikidata: WikidataClient, city: City) -> str | None:
    searches: list[dict[str, object]] = []
    for term in (city.name, f"{city.name} city", city.slug.replace("-", " ")):
        try:
            searches.extend(await wikidata.search_entities(term, limit=8))
        except WikidataRateLimitError:
            return None
    best_id: str | None = None
    best_score = -9999
    seen_ids: set[str] = set()
    for candidate in searches:
        candidate_id = candidate.get("id", "")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        score = score_city_candidate(city.name, candidate, city.country_code)
        if score > best_score:
            best_id, best_score = candidate_id, score
    return best_id
