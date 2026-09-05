"""Async port of V1's Nominatim reverse-geocoding step
(`CatalogService._resolve_city_name_from_coords`). Public, unauthenticated API.
"""

from dataclasses import dataclass

import httpx

from locus_v2.config import Settings
from locus_v2.shared.text import clean_text


class NominatimError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReverseGeocodeResult:
    city_name: str
    country_code: str
    names: dict[str, str]


async def reverse_geocode(lat: float, lng: float, settings: Settings) -> ReverseGeocodeResult:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.nominatim_base_url}/reverse",
                params={
                    "format": "jsonv2",
                    "lat": lat,
                    "lon": lng,
                    "zoom": 10,
                    "addressdetails": 1,
                    "namedetails": 1,
                },
                headers={"Accept-Language": "es,en,ja", "User-Agent": "LocusV2/0.1"},
                timeout=12,
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as error:
        raise NominatimError(
            f"No se pudo resolver la ciudad desde la ubicacion: {error}"
        ) from error

    address = payload.get("address", {}) or {}
    country_code = str(address.get("country_code") or "").upper()
    if country_code == "JP":
        display_name = clean_text(payload.get("display_name", ""))
        city_name = (
            address.get("city")
            or address.get("state")
            or address.get("province")
            or address.get("municipality")
            or address.get("county")
            or address.get("town")
            or address.get("village")
            or ""
        )
        if clean_text(city_name).endswith("区") and (
            "tokyo" in display_name.lower() or "東京都" in display_name or "東京" in display_name
        ):
            city_name = "Tokyo"
        if clean_text(city_name).endswith("区") and address.get("state"):
            city_name = address.get("state") or city_name
    else:
        city_name = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or address.get("county")
            or ""
        )

    city_name = clean_text(city_name)
    if country_code == "JP" and city_name in {"東京都", "東京"}:
        city_name = "Tokyo"
    names = _names_from_payload(payload, city_name, country_code)
    if not city_name:
        raise NominatimError("No se pudo deducir una ciudad valida desde la ubicacion")
    return ReverseGeocodeResult(city_name=city_name, country_code=country_code, names=names)


def _names_from_payload(
    payload: dict[str, object], fallback_name: str, country_code: str
) -> dict[str, str]:
    raw_namedetails = payload.get("namedetails")
    namedetails = raw_namedetails if isinstance(raw_namedetails, dict) else {}
    names: dict[str, str] = {}
    clean_name = clean_text(fallback_name)
    if clean_name:
        names["local"] = clean_name
    for source_key, target_key in [
        ("name:es", "es"),
        ("name:en", "en"),
        ("name:ja", "ja"),
        ("int_name", "int"),
        ("name", "local"),
    ]:
        value = namedetails.get(source_key)
        if value:
            names[target_key] = clean_text(str(value))
    if country_code == "JP" and fallback_name in {"Tokyo", "Tokio", "東京都", "東京"}:
        names.update(
            {"local": "東京都", "ja": "東京都", "en": "Tokyo", "es": "Tokio", "int": "Tokyo"}
        )
    return names
