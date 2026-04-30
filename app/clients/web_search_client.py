from __future__ import annotations

from typing import Any

import requests

from app.config import settings
from app.utils.logging import get_logger


logger = get_logger(__name__)


class WebSearchClient:
    def __init__(self) -> None:
        self.provider = settings.web_search_provider.lower().strip()
        self.base_url = settings.web_search_base_url.rstrip("/")

    def is_enabled(self) -> bool:
        return bool(settings.web_search_api_key) and self.provider == "brave"

    def search_web(
        self,
        query: str,
        *,
        preferred_domains: list[str] | None = None,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        cleaned_query = query.strip()
        if not cleaned_query or not self.is_enabled():
            return []

        collected: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        normalized_domains = [
            domain.strip().lower()
            for domain in (preferred_domains or [])
            if isinstance(domain, str) and domain.strip()
        ]

        for domain in normalized_domains[:3]:
            scoped_query = f"site:{domain} {cleaned_query}"
            self._append_results(
                collected,
                seen_urls,
                self._search_brave(scoped_query, max_results=max_results),
                max_results=max_results,
            )
            if len(collected) >= max_results:
                return collected[:max_results]

        self._append_results(
            collected,
            seen_urls,
            self._search_brave(cleaned_query, max_results=max_results),
            max_results=max_results,
        )
        return collected[:max_results]

    def _append_results(
        self,
        collected: list[dict[str, Any]],
        seen_urls: set[str],
        items: list[dict[str, Any]],
        *,
        max_results: int,
    ) -> None:
        for item in items:
            url = str(item.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            collected.append(item)
            if len(collected) >= max_results:
                return

    def _search_brave(self, query: str, *, max_results: int) -> list[dict[str, Any]]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": settings.web_search_api_key,
        }
        params = {
            "q": query,
            "count": min(max(max_results, 1), 10),
            "country": settings.web_search_country,
            "search_lang": settings.web_search_language,
        }

        try:
            response = requests.get(
                f"{self.base_url}/web/search",
                headers=headers,
                params=params,
                timeout=settings.web_search_timeout_seconds,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Web search failed for query '%s': %s", query, exc)
            return []

        payload = response.json()
        web_results = payload.get("web", {}).get("results", []) or []
        normalized: list[dict[str, Any]] = []
        for item in web_results:
            url = str(item.get("url", "")).strip()
            title = str(item.get("title", "")).strip()
            description = str(item.get("description", "")).strip()
            if not url or not title:
                continue
            normalized.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": description,
                    "source": self._classify_source(url),
                }
            )
        return normalized

    def _classify_source(self, url: str) -> str:
        lowered = url.lower()
        if any(token in lowered for token in [".gob.", ".gov.", ".edu.", "ayto", "turismo", "museo", "theatre", "teatro"]):
            return "official_or_institutional"
        if "wikipedia.org" in lowered or "wikidata.org" in lowered:
            return "wiki"
        return "web"


web_search_client = WebSearchClient()
