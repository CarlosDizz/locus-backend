"""Overpass (OpenStreetMap) query + result normalization, ported from V1
`CatalogService._build_overpass_map_query` / `_normalize_overpass_element`.
"""

from typing import Any

from locus_v2.catalog.bootstrap.normalize import normalize_names
from locus_v2.catalog.bootstrap.poi_scoring import map_overpass_type_code
from locus_v2.shared.text import clean_text


def build_overpass_map_query(*, lat: float, lng: float, radius_km: float, limit: int) -> str:
    safe_limit = max(40, min(int(limit) * 5, 320))
    safe_radius_m = int(min(max(radius_km, 1.5), 12.0) * 1000)
    around = f"around:{safe_radius_m},{lat},{lng}"
    return f"""
[out:json][timeout:20];
(
  nwr({around})["tourism"~"^(attraction|museum|gallery|artwork)$"]["name"];
  nwr({around})["historic"~"^(monument|memorial|castle|archaeological_site|ruins)$"]["name"];
  nwr({around})["historic"~"^(yes|roman_road|citywalls|fort|aqueduct)$"]["name"];
  nwr({around})["amenity"="place_of_worship"]["name"]["wikidata"];
  nwr({around})["amenity"="place_of_worship"]["name"]["heritage"];
  nwr({around})["building"~"^(cathedral|church|synagogue|chapel|monastery)$"]["name"]["wikidata"];
  nwr({around})["building"~"^(cathedral|church|synagogue|chapel|monastery)$"]["name"]["wikipedia"];
  nwr({around})["building"~"^(palace|public|yes|historic)$"]["name"]["wikidata"];
  nwr({around})["building"~"^(palace|public|yes|historic)$"]["name"]["heritage"];
  nwr({around})["place"="square"]["name"]["wikidata"];
  nwr({around})["place"="square"]["name"]["wikipedia"];
  nwr({around})["heritage"]["name"];
);
out center {safe_limit};
"""


def build_overpass_name_query(
    *, lat: float, lng: float, name: str, aliases: list[str]
) -> str:
    radius_m = 12000
    terms = [clean_text(name), *[clean_text(alias) for alias in aliases[:3]]]
    clauses = []
    for term in terms:
        if not term:
            continue
        escaped = term.replace('"', '\\"')
        clauses.append(f'nwr(around:{radius_m},{lat},{lng})["name"="{escaped}"];')
    if not clauses:
        return ""
    body = "\n  ".join(clauses)
    return f"""
[out:json][timeout:20];
(
  {body}
);
out center 10;
"""


def normalize_overpass_element(element: dict[str, Any]) -> dict[str, Any] | None:
    tags = element.get("tags", {}) or {}
    name = clean_text(str(tags.get("name", "")))
    if not name:
        return None
    lat = element.get("lat")
    lng = element.get("lon")
    center = element.get("center", {}) or {}
    if lat is None:
        lat = center.get("lat")
    if lng is None:
        lng = center.get("lon")
    if lat is None or lng is None:
        return None
    wikipedia_title = clean_text(str(tags.get("wikipedia", "")))
    if ":" in wikipedia_title:
        wikipedia_title = wikipedia_title.split(":", 1)[1]
    description = clean_text(
        str(
            tags.get("description")
            or tags.get("historic")
            or tags.get("tourism")
            or tags.get("amenity")
            or ""
        )
    )
    type_code = map_overpass_type_code(tags)
    wikidata_id = clean_text(str(tags.get("wikidata", "")))
    names = normalize_names(name)
    for source_key, target_key in [
        ("name:es", "es"), ("name:en", "en"), ("name:ja", "ja"), ("int_name", "int"),
    ]:
        value = clean_text(str(tags.get(source_key, "")))
        if value:
            names[target_key] = value
    return {
        "source_id": f"osm:{element.get('type', '')}:{element.get('id', '')}",
        "name": name,
        "names": names,
        "lat": float(lat),
        "lng": float(lng),
        "description": description,
        "type_code": type_code,
        "type_label": clean_text(
            " ".join(
                str(tags.get(key, ""))
                for key in ("tourism", "historic", "amenity", "building", "place")
            )
        ),
        "wikipedia_title": wikipedia_title,
        "wikidata_id": wikidata_id,
    }
