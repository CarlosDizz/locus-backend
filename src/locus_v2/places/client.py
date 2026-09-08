"""Async port of V1 `app/clients/maps_client.py` (Google Places Text Search)."""

from typing import Any

import httpx
import structlog

from locus_v2.config import Settings
from locus_v2.shared.text import clean_text

logger = structlog.get_logger()

_RESTAURANT_TOKENS = ("carbonara", "pizza", "pasta", "ristorante", "comer", "restaurante")
_BAR_TOKENS = ("ipa", "cerveza", "beer", "bar", "pub")
_CAFE_TOKENS = ("cafe", "café", "coffee")


class GooglePlacesClient:
    base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    def __init__(self, settings: Settings) -> None:
        self._api_key = (
            settings.maps_api_key.get_secret_value().strip()
            if settings.maps_api_key is not None
            else ""
        )
        self._timeout_seconds = settings.maps_timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    async def search_places(
        self,
        query: str,
        lat: float | None,
        lng: float | None,
        limit: int = 5,
        locale: str = "",
    ) -> list[dict[str, Any]]:
        if not self.enabled or not query.strip():
            return []

        params: dict[str, str | int] = {"query": query, "key": self._api_key}
        if locale:
            # Without this Google answers in the place's own language, so a
            # Spanish user got "Trajan's Column" and, worse, a POI promoted to
            # the catalog was stored under that name for everyone.
            params["language"] = locale.split("-", 1)[0].lower()
        if lat is not None and lng is not None:
            params["location"] = f"{lat},{lng}"
            params["radius"] = 2500

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.base_url, params=params, timeout=self._timeout_seconds
                )
            response.raise_for_status()
            results = response.json().get("results", [])[:limit]
        except (httpx.HTTPError, ValueError) as error:
            # Same posture as V1: a failed places lookup degrades the answer,
            # it never fails the user's turn.
            logger.warning("google_places_lookup_failed", query=query, error=str(error))
            return []

        normalized: list[dict[str, Any]] = []
        for item in results:
            geometry = (item.get("geometry") or {}).get("location") or {}
            if "lat" not in geometry or "lng" not in geometry:
                continue
            place_id = str(item.get("place_id") or "")
            normalized.append(
                {
                    "id": place_id,
                    "name": str(item.get("name") or ""),
                    "lat": float(geometry["lat"]),
                    "lng": float(geometry["lng"]),
                    "description": str(item.get("formatted_address") or ""),
                    "summary": "",
                    "poi_type_code": _place_type(item.get("types") or [], query),
                    "source_of_truth": "google_places",
                    "is_ephemeral": True,
                    "google_place_id": place_id,
                    "context_kind": "hospitality",
                }
            )
        return normalized


def _place_type(raw_types: list[str], query: str) -> str:
    lowered = clean_text(query).lower()
    if "restaurant" in raw_types or any(token in lowered for token in _RESTAURANT_TOKENS):
        return "restaurant"
    if (
        "bar" in raw_types
        or "night_club" in raw_types
        or any(token in lowered for token in _BAR_TOKENS)
    ):
        return "bar"
    if "cafe" in raw_types or any(token in lowered for token in _CAFE_TOKENS):
        return "cafe"
    return "place"
