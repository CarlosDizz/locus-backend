"""Async port of V1 `app/clients/wikidata_client.py`.

Public, unauthenticated API — no credentials needed. Ported to `httpx.AsyncClient`
instead of V1's synchronous `requests` because V2's whole stack is async; a sync
HTTP call here would block the event loop for every other request being served.
Same rate-limit cooldown behaviour as V1 (Wikidata returns 429 under load).
"""

import asyncio
import time
from typing import Any

import httpx

from locus_v2.config import Settings


class WikidataRateLimitError(RuntimeError):
    pass


class WikidataClient:
    RATE_LIMIT_COOLDOWN_SECONDS = 90
    MIN_REQUEST_INTERVAL_SECONDS = 0.35

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.wikidata_base_url.rstrip("/")
        self.api_url = f"{self.base_url}/w/api.php"
        self.sparql_url = settings.wikidata_sparql_url
        self.language = settings.wikidata_language
        self.headers = {"User-Agent": "LocusV2/0.1 (Locus backend)"}
        self._rate_limited_until = 0.0
        self._last_request_at = 0.0
        self._search_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}

    def is_rate_limited(self) -> bool:
        return time.monotonic() < self._rate_limited_until

    def _raise_if_rate_limited(self) -> None:
        if not self.is_rate_limited():
            return
        retry_in = max(1, int(self._rate_limited_until - time.monotonic()))
        raise WikidataRateLimitError(f"Wikidata cooldown active; retry in {retry_in}s")

    def _record_rate_limit(self, response: httpx.Response | None) -> None:
        retry_after = 0
        if response is not None:
            try:
                retry_after = int(response.headers.get("Retry-After", "0"))
            except ValueError:
                retry_after = 0
        cooldown = max(retry_after, self.RATE_LIMIT_COOLDOWN_SECONDS)
        self._rate_limited_until = time.monotonic() + cooldown

    async def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.MIN_REQUEST_INTERVAL_SECONDS:
            await asyncio.sleep(self.MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        self._last_request_at = time.monotonic()

    async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        self._raise_if_rate_limited()
        await self._throttle()
        async with httpx.AsyncClient() as client:
            response = await client.get(url, **kwargs)
        if response.status_code == 429:
            self._record_rate_limit(response)
            raise WikidataRateLimitError("Wikidata returned HTTP 429")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 429:
                self._record_rate_limit(error.response)
                raise WikidataRateLimitError("Wikidata returned HTTP 429") from error
            raise
        return response

    async def search_entities(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not query:
            return []
        cache_key = (query.strip().lower(), int(limit))
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": self.language,
            "type": "item",
            "limit": limit,
            "search": query,
        }
        try:
            response = await self._get(
                self.api_url, params=params, headers=self.headers, timeout=10
            )
            results: list[dict[str, Any]] = response.json().get("search", [])
        except WikidataRateLimitError:
            raise
        except Exception:
            return []
        self._search_cache[cache_key] = results
        return results

    async def get_entities(self, entity_ids: list[str]) -> dict[str, Any]:
        if not entity_ids:
            return {}
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": "|".join(sorted(set(entity_ids))),
            "languages": self.language,
            "props": "labels|descriptions|claims|sitelinks",
        }
        response = await self._get(self.api_url, params=params, headers=self.headers, timeout=10)
        entities: dict[str, Any] = response.json().get("entities", {})
        return entities

    async def run_sparql(
        self, query: str, timeout_seconds: int = 15
    ) -> list[dict[str, Any]]:
        headers = {**self.headers, "Accept": "application/sparql-results+json"}
        response = await self._get(
            self.sparql_url,
            params={"query": query, "format": "json"},
            headers=headers,
            timeout=timeout_seconds,
        )
        result: list[dict[str, Any]] = response.json().get("results", {}).get("bindings", [])
        return result
