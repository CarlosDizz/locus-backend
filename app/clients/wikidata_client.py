from __future__ import annotations

from typing import Any

import requests

from app.config import settings
from app.utils.logging import get_logger


logger = get_logger(__name__)


class WikidataClient:
    def __init__(self) -> None:
        self.base_url = settings.wikidata_base_url.rstrip("/")
        self.api_url = f"{self.base_url}/w/api.php"
        self.sparql_url = settings.wikidata_sparql_url
        self.headers = {
            "User-Agent": f"{settings.app_name}/{settings.app_build} (Locus backend prototype)"
        }

    def search_entity(self, query: str, limit: int = 1) -> dict[str, Any] | None:
        if not query:
            return None
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": settings.wikidata_language,
            "type": "item",
            "limit": limit,
            "search": query,
        }
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            results = response.json().get("search", [])
        except Exception as exc:
            logger.warning("Wikidata search failed: %s", exc)
            return None
        return results[0] if results else None

    def get_entity(self, entity_id: str) -> dict[str, Any]:
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": entity_id,
            "languages": settings.wikidata_language,
            "props": "labels|descriptions|claims|sitelinks",
        }
        response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.json().get("entities", {}).get(entity_id, {})

    def get_entity_labels(self, entity_ids: list[str]) -> dict[str, str]:
        if not entity_ids:
            return {}
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": "|".join(sorted(set(entity_ids))),
            "languages": settings.wikidata_language,
            "props": "labels|descriptions",
        }
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            entities = response.json().get("entities", {})
        except Exception as exc:
            logger.warning("Wikidata label lookup failed: %s", exc)
            return {}

        labels: dict[str, str] = {}
        for entity_id, entity in entities.items():
            labels[entity_id] = (
                entity.get("labels", {})
                .get(settings.wikidata_language, {})
                .get("value", "")
            )
        return labels

    def run_sparql(self, query: str) -> list[dict[str, Any]]:
        headers = {
            **self.headers,
            "Accept": "application/sparql-results+json",
        }
        response = requests.get(
            self.sparql_url,
            params={"query": query, "format": "json"},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("results", {}).get("bindings", [])
