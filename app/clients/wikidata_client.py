from __future__ import annotations

import time
from typing import Any

import requests
from requests import HTTPError, RequestException

from app.config import settings
from app.utils.logging import get_logger


logger = get_logger(__name__)


class WikidataRateLimitError(RequestException):
    pass


class WikidataClient:
    RATE_LIMIT_COOLDOWN_SECONDS = 90
    MIN_REQUEST_INTERVAL_SECONDS = 0.35

    def __init__(self) -> None:
        self.base_url = settings.wikidata_base_url.rstrip("/")
        self.api_url = f"{self.base_url}/w/api.php"
        self.sparql_url = settings.wikidata_sparql_url
        self.headers = {
            "User-Agent": f"{settings.app_name}/{settings.app_build} (Locus backend prototype)"
        }
        self._rate_limited_until = 0.0
        self._last_request_at = 0.0
        self._search_cache: dict[tuple[str, int, str], list[dict[str, Any]]] = {}

    def is_rate_limited(self) -> bool:
        return time.monotonic() < self._rate_limited_until

    def _raise_if_rate_limited(self) -> None:
        if not self.is_rate_limited():
            return
        retry_in = max(1, int(self._rate_limited_until - time.monotonic()))
        raise WikidataRateLimitError(f"Wikidata cooldown active; retry in {retry_in}s")

    def _record_rate_limit(self, response: requests.Response | None = None) -> None:
        retry_after = 0
        if response is not None:
            try:
                retry_after = int(response.headers.get("Retry-After", "0"))
            except ValueError:
                retry_after = 0
        cooldown = max(retry_after, self.RATE_LIMIT_COOLDOWN_SECONDS)
        self._rate_limited_until = time.monotonic() + cooldown
        logger.warning("Wikidata rate limited; entering cooldown for %ss", cooldown)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(self.MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        self._last_request_at = time.monotonic()

    def _request_get(self, url: str, **kwargs: Any) -> requests.Response:
        self._raise_if_rate_limited()
        self._throttle()
        response = requests.get(url, **kwargs)
        if response.status_code == 429:
            self._record_rate_limit(response)
            raise WikidataRateLimitError("Wikidata returned HTTP 429")
        try:
            response.raise_for_status()
        except HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                self._record_rate_limit(exc.response)
                raise WikidataRateLimitError("Wikidata returned HTTP 429") from exc
            raise
        return response

    def search_entities(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not query:
            return []
        cache_key = (query.strip().lower(), int(limit), settings.wikidata_language)
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": settings.wikidata_language,
            "type": "item",
            "limit": limit,
            "search": query,
        }
        try:
            response = self._request_get(self.api_url, params=params, headers=self.headers, timeout=10)
            results = response.json().get("search", [])
        except WikidataRateLimitError:
            raise
        except Exception as exc:
            logger.warning("Wikidata search failed: %s", exc)
            return []
        self._search_cache[cache_key] = results
        return results

    def search_entity(self, query: str, limit: int = 1) -> dict[str, Any] | None:
        results = self.search_entities(query, limit=limit)
        return results[0] if results else None

    def get_entity(self, entity_id: str) -> dict[str, Any]:
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": entity_id,
            "languages": settings.wikidata_language,
            "props": "labels|descriptions|claims|sitelinks",
        }
        response = self._request_get(self.api_url, params=params, headers=self.headers, timeout=10)
        return response.json().get("entities", {}).get(entity_id, {})

    def get_entities(self, entity_ids: list[str]) -> dict[str, Any]:
        if not entity_ids:
            return {}
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": "|".join(sorted(set(entity_ids))),
            "languages": settings.wikidata_language,
            "props": "labels|descriptions|claims|sitelinks",
        }
        response = self._request_get(self.api_url, params=params, headers=self.headers, timeout=10)
        return response.json().get("entities", {})

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
            response = self._request_get(self.api_url, params=params, headers=self.headers, timeout=10)
            entities = response.json().get("entities", {})
        except WikidataRateLimitError:
            raise
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

    def run_sparql(self, query: str, timeout: int = 12) -> list[dict[str, Any]]:
        headers = {
            **self.headers,
            "Accept": "application/sparql-results+json",
        }
        response = self._request_get(
            self.sparql_url,
            params={"query": query, "format": "json"},
            headers=headers,
            timeout=timeout,
        )
        return response.json().get("results", {}).get("bindings", [])
