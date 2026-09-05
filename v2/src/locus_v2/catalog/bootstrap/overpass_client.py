"""Async port of V1 `app/clients/overpass_client.py`. Public OSM API, no key."""

from typing import Any

import httpx

from locus_v2.config import Settings


class OverpassClient:
    def __init__(self, settings: Settings) -> None:
        self.api_url = settings.overpass_api_url
        self.timeout_seconds = settings.overpass_timeout_seconds
        self.headers = {"User-Agent": "LocusV2/0.1 (Locus backend)"}

    async def query(self, query: str) -> list[dict[str, Any]]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_url,
                content=query.encode("utf-8"),
                headers={**self.headers, "Content-Type": "text/plain; charset=utf-8"},
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        elements: list[dict[str, Any]] = response.json().get("elements", [])
        return elements
