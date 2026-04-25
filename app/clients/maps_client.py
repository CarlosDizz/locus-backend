from app.config import settings
from app.utils.logging import get_logger
from app.utils.text import clean_text
import requests


logger = get_logger(__name__)


class MapsClient:
    base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    def is_enabled(self) -> bool:
        return bool(settings.maps_api_key)

    def search_places(self, query: str, lat: float | None, lng: float | None, limit: int = 5) -> list[dict]:
        if not self.is_enabled() or not query:
            return []

        params: dict[str, str | int] = {
            "query": query,
            "key": settings.maps_api_key,
        }
        if lat is not None and lng is not None:
            params["location"] = f"{lat},{lng}"
            params["radius"] = 2500

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get("results", [])[:limit]
        except Exception as exc:
            logger.warning("Maps lookup failed: %s", exc)
            return []

        normalized = []
        for item in results:
            geometry = item.get("geometry", {}).get("location", {})
            if "lat" not in geometry or "lng" not in geometry:
                continue
            raw_types = item.get("types", []) or []
            type_code = self._map_place_type(raw_types, query)
            normalized.append(
                {
                    "id": item.get("place_id", ""),
                    "name": item.get("name", ""),
                    "lat": geometry["lat"],
                    "lng": geometry["lng"],
                    "description": item.get("formatted_address", ""),
                    "poi_type_code": type_code,
                    "source_of_truth": "google_places",
                    "is_ephemeral": True,
                    "google_place_id": item.get("place_id", ""),
                    "context_kind": "hospitality",
                }
            )
        return normalized

    def _map_place_type(self, raw_types: list[str], query: str) -> str:
        lowered = clean_text(query).lower()
        if "restaurant" in raw_types or any(token in lowered for token in ["carbonara", "pizza", "pasta", "ristorante", "comer", "restaurante"]):
            return "restaurant"
        if "bar" in raw_types or "night_club" in raw_types or any(token in lowered for token in ["ipa", "cerveza", "beer", "bar", "pub"]):
            return "bar"
        if "cafe" in raw_types or any(token in lowered for token in ["cafe", "café", "coffee"]):
            return "cafe"
        return "place"
