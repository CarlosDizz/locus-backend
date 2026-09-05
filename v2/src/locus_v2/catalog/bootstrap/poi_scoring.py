"""Ported from V1 `CatalogService`'s type-mapping and scoring heuristics.

These dictionaries encode real tuning against real cities (which terms mean a
place is or isn't tourism-relevant, which Wikidata/OSM type maps to which
Locus `poi_types.code`). Kept as plain data + pure functions here rather than
re-derived, since the values themselves are the asset.
"""

import math
import re
from typing import Any

from locus_v2.shared.text import clean_text

POI_TYPE_MAP: dict[str, str] = {
    "tourist attraction": "monument",
    "atraccion turistica": "monument",
    "museum": "museum",
    "museo": "museum",
    "art museum": "museum",
    "church building": "church",
    "iglesia": "church",
    "church": "church",
    "cathedral": "church",
    "catedral": "church",
    "basilica": "church",
    "basilica menor": "church",
    "square": "square",
    "plaza": "square",
    "monument": "monument",
    "monumento": "monument",
    "palace": "building",
    "palacio": "building",
    "alcazar": "monument",
    "alcázar": "monument",
    "synagogue": "church",
    "sinagoga": "church",
    "monasterio": "church",
    "monastery": "church",
    "convento": "church",
    "teatro": "building",
    "archaeological site": "archaeological_site",
    "yacimiento arqueologico": "archaeological_site",
    "yacimiento arqueológico": "archaeological_site",
    "anfiteatro romano": "archaeological_site",
    "anfiteatro": "archaeological_site",
    "foro romano": "archaeological_site",
    "bridge": "building",
    "building": "building",
}

TOURISM_TYPE_SCORES: dict[str, int] = {
    "archaeological_site": 120,
    "monument": 115,
    "museum": 110,
    "church": 95,
    "square": 85,
    "building": 55,
}

TOURISM_POSITIVE_TERMS: dict[str, int] = {
    "catedral": 60,
    "cathedral": 60,
    "museo": 55,
    "museum": 55,
    "plaza": 40,
    "square": 40,
    "palacio": 40,
    "palace": 40,
    "castillo": 55,
    "castle": 55,
    "puerta": 35,
    "gate": 35,
    "iglesia": 35,
    "church": 35,
    "basílica": 45,
    "basilica": 45,
    "monumento": 50,
    "monument": 50,
    "anfiteatro": 70,
    "teatro": 55,
    "foro": 60,
    "arqueológico": 65,
    "arqueologico": 65,
    "archaeological": 65,
    "histórico": 25,
    "historico": 25,
    "histórico-artístico": 35,
    "patrimonio": 35,
    "tourist": 20,
    "turístico": 20,
    "turistico": 20,
}

TOURISM_NEGATIVE_TERMS: dict[str, int] = {
    "justicia": -140,
    "judicial": -140,
    "tribunal": -140,
    "juzgado": -140,
    "court": -140,
    "administrativo": -120,
    "administrative": -120,
    "gobierno": -100,
    "government": -100,
    "oficina": -100,
    "office": -100,
    "hospital": -120,
    "estación": -90,
    "estacion": -90,
    "station": -90,
    "aeropuerto": -120,
    "airport": -120,
    "universidad": -60,
    "university": -60,
    "colegio": -60,
    "school": -60,
    "prisón": -160,
    "prision": -160,
    "prison": -160,
    "complejo judicial": -180,
    "fiscalía": -140,
    "fiscalia": -140,
    "sede": -45,
    "campus": -60,
    "policía": -140,
    "policia": -140,
    "police": -140,
    "barrio": -180,
    "neighborhood": -180,
    "district": -150,
    "estadio": -130,
    "stadium": -130,
    "fútbol": -120,
    "futbol": -120,
    "football": -120,
}

OVERPASS_TAG_TYPE_MAP: dict[str, str] = {
    "tourism:museum": "museum",
    "tourism:gallery": "museum",
    "tourism:attraction": "monument",
    "historic:monument": "monument",
    "historic:memorial": "monument",
    "historic:castle": "monument",
    "historic:archaeological_site": "archaeological_site",
    "amenity:place_of_worship": "church",
    "building:cathedral": "church",
    "building:church": "church",
    "building:synagogue": "church",
    "place:square": "square",
}

COUNTRY_TERMS: dict[str, list[str]] = {
    "IT": ["italy", "italia"],
    "ES": ["spain", "españa", "castilla-la mancha", "castilla la mancha"],
    "FR": ["france", "francia"],
    "GB": ["united kingdom", "england", "reino unido"],
    "US": ["united states", "usa", "estados unidos"],
}

EXPECTED_TYPE_TERMS: dict[str, dict[str, set[str]]] = {
    "church": {
        "positive": {
            "church", "iglesia", "cathedral", "catedral", "basilica", "basílica",
            "synagogue", "sinagoga", "mosque", "mezquita", "monastery", "monasterio",
            "convent", "convento", "temple", "templo", "parish", "parroquia",
            "ermita", "ermitaño",
        },
        "negative": {
            "library", "biblioteca", "museum", "museo", "school", "colegio", "hospital",
            "stadium", "estadio", "office", "oficina", "hotel", "restaurant",
        },
    },
    "museum": {
        "positive": {
            "museum", "museo", "gallery", "galería", "exhibition", "colección", "collection",
            "visitor centre", "centro de interpretación",
        },
        "negative": {
            "church", "iglesia", "cathedral", "catedral", "synagogue", "sinagoga",
            "bridge", "puente", "gate", "puerta", "square", "plaza",
        },
    },
    "monument": {
        "positive": {
            "monument", "monumento", "gate", "puerta", "bridge", "puente", "castle",
            "castillo", "fortress", "fortaleza", "tower", "torre", "wall", "muralla",
            "alcázar", "alcazar",
        },
        "negative": {
            "library", "biblioteca", "hospital", "school", "colegio", "stadium", "estadio",
        },
    },
    "square": {
        "positive": {"square", "plaza", "piazza", "place"},
        "negative": {"church", "iglesia", "museum", "museo", "hospital", "biblioteca"},
    },
    "building": {
        "positive": {
            "building", "edificio", "palace", "palacio", "hospital", "college", "colegio",
            "castle", "castillo", "alcázar", "alcazar",
        },
        "negative": set(),
    },
    "archaeological_site": {
        "positive": {
            "archaeological", "arqueológico", "arqueologico", "ruins", "ruinas",
            "roman", "romano", "circus", "circo", "thermae", "termas", "forum", "foro",
            "site", "yacimiento",
        },
        "negative": {"library", "biblioteca", "office", "oficina", "hospital"},
    },
}


def map_poi_type_code(name: str, type_label: str, fallback_description: str) -> str:
    label = (type_label or "").strip().lower()
    description = (fallback_description or "").strip().lower()
    title = (name or "").strip().lower()
    combined = " ".join(part for part in [title, label, description] if part)
    for token, code in POI_TYPE_MAP.items():
        if token in combined:
            return code
    return "building"


def map_overpass_type_code(tags: dict[str, Any]) -> str:
    for key in ("tourism", "historic", "amenity", "building", "place"):
        value = clean_text(str(tags.get(key, ""))).lower()
        if not value:
            continue
        mapped = OVERPASS_TAG_TYPE_MAP.get(f"{key}:{value}")
        if mapped:
            return mapped
    return map_poi_type_code(
        clean_text(str(tags.get("name", ""))),
        "",
        " ".join(
            [
                clean_text(str(tags.get("tourism", ""))),
                clean_text(str(tags.get("historic", ""))),
                clean_text(str(tags.get("amenity", ""))),
                clean_text(str(tags.get("building", ""))),
            ]
        ),
    )


def score_tourism_candidate(
    *,
    name: str,
    description: str,
    type_code: str,
    type_label: str,
    sitelinks: int,
    wikipedia_title: str,
) -> int:
    text = " ".join(
        [clean_text(name).lower(), clean_text(description).lower(), clean_text(type_label).lower()]
    ).strip()
    score = TOURISM_TYPE_SCORES.get(type_code, 0)
    score += min(max(sitelinks, 0), 120)

    if wikipedia_title:
        score += 15
    if len(clean_text(description)) > 18:
        score += 10

    for token, value in TOURISM_POSITIVE_TERMS.items():
        if token in text:
            score += value
    for token, value in TOURISM_NEGATIVE_TERMS.items():
        if token in text:
            score += value

    if "lista de" in text or "list of" in text:
        score -= 120
    if "edificio" in text and type_code == "building":
        score -= 20
    return score


def is_map_candidate(score: int, type_code: str) -> bool:
    threshold = 35 if type_code != "building" else 45
    return score >= threshold


def is_featured_candidate(score: int, type_code: str) -> bool:
    threshold = 55 if type_code != "building" else 75
    return score >= threshold


def score_city_candidate(query: str, candidate: dict[str, Any], country_code: str = "") -> int:
    normalized_query = clean_text(query).lower()
    label = clean_text(candidate.get("label", "")).lower()
    description = clean_text(candidate.get("description", "")).lower()
    text = f"{label} {description}".strip()
    score = 0

    if label == normalized_query:
        score += 120
    elif normalized_query and normalized_query in label:
        score += 40

    city_terms = ["capital", "city", "ciudad", "comuna", "municipio", "municipality"]
    if any(token in description for token in city_terms):
        score += 80
    sports_terms = ["football", "fútbol", "soccer", "club", "team", "f.c."]
    if any(token in description for token in sports_terms):
        score -= 200
    if any(
        token in text
        for token in [
            "magazine", "newspaper", "journal", "publication", "periodical", "website",
            "media company", "television", "radio", "publisher",
            "revista", "periódico", "periodico", "publicación", "publicacion",
        ]
    ):
        score -= 260
    if any(token in text for token in ["roman empire", "película", "album", "song", "novel"]):
        score -= 80
    country_terms = COUNTRY_TERMS.get((country_code or "").upper(), [])
    if country_terms and any(term in description for term in country_terms):
        score += 20
    return score


def distance_km(
    lat_a: float | None, lng_a: float | None, lat_b: float | None, lng_b: float | None
) -> float | None:
    if None in {lat_a, lng_a, lat_b, lng_b}:
        return None
    lat1, lng1 = math.radians(float(lat_a)), math.radians(float(lng_a))  # type: ignore[arg-type]
    lat2, lng2 = math.radians(float(lat_b)), math.radians(float(lng_b))  # type: ignore[arg-type]
    d_lat = lat2 - lat1
    d_lng = lng2 - lng1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 6371.0 * c


def extract_entity_coords(entity: dict[str, Any]) -> tuple[float | None, float | None]:
    coords_claim = entity.get("claims", {}).get("P625") or []
    if not coords_claim:
        return None, None
    value = coords_claim[0].get("mainsnak", {}).get("datavalue", {}).get("value", {})
    return value.get("latitude"), value.get("longitude")


def extract_claim_entity_ids(entity: dict[str, Any], property_id: str) -> set[str]:
    values: set[str] = set()
    for claim in entity.get("claims", {}).get(property_id, []) or []:
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        entity_id = value.get("id")
        if entity_id:
            values.add(entity_id)
    return values


def score_expected_type_compatibility(
    candidate_type_code: str, label: str, description: str
) -> int:
    rules = EXPECTED_TYPE_TERMS.get(candidate_type_code, {})
    if not rules:
        return 0
    text = f"{clean_text(label).lower()} {clean_text(description).lower()}".strip()
    score = 0
    if any(term in text for term in rules.get("positive", set())):
        score += 90
    if any(term in text for term in rules.get("negative", set())):
        score -= 180
    return score


def score_wikidata_resolution(
    *, city_name: str, country_code: str, candidate_name: str,
    result: dict[str, Any], aliases: list[str] | None = None,
) -> int:
    label = clean_text(result.get("label", "")).lower()
    description = clean_text(result.get("description", "")).lower()
    target = clean_text(candidate_name).lower()
    alias_tokens = [clean_text(alias).lower() for alias in aliases or [] if clean_text(alias)]
    score = 0
    if label == target:
        score += 120
    elif target and target in label:
        score += 60
    elif label and label in target:
        score += 45
    if any(label == alias for alias in alias_tokens):
        score += 90
    elif any(alias and alias in label for alias in alias_tokens):
        score += 55
    target_words = {token for token in re.split(r"\W+", target) if len(token) > 2}
    label_words = {token for token in re.split(r"\W+", label) if len(token) > 2}
    score += len(target_words & label_words) * 12
    if clean_text(city_name).lower() in description:
        score += 80
    for token in COUNTRY_TERMS.get((country_code or "").upper(), []):
        if token in description:
            score += 20
    match = result.get("match") or {}
    if isinstance(match, dict):
        if match.get("type") == "label":
            score += 10
        if match.get("type") == "alias":
            score += 18
    negative_terms = [
        "district", "barrio", "neighborhood", "stadium", "football", "hospital", "court",
    ]
    if any(token in description for token in negative_terms):
        score -= 120
    return score


def score_ai_entity_candidate(
    *, city_name: str, country_code: str, city_lat: float | None, city_lng: float | None,
    city_entity_id: str | None, candidate: dict[str, Any],
    result: dict[str, Any], entity: dict[str, Any],
) -> tuple[int, float | None, str]:
    description = (
        entity.get("descriptions", {}).get("es", {}).get("value")
        or result.get("description", "")
        or ""
    )
    label = clean_text(result.get("label", ""))
    lat, lng = extract_entity_coords(entity)
    if lat is None or lng is None:
        return -9999, None, "missing_coordinates"

    score = score_wikidata_resolution(
        city_name=city_name, country_code=country_code, candidate_name=candidate["name"],
        result=result, aliases=candidate.get("aliases", []),
    )
    dist = distance_km(city_lat, city_lng, lat, lng)
    if dist is not None:
        if dist <= 2:
            score += 120
        elif dist <= 5:
            score += 95
        elif dist <= 12:
            score += 65
        elif dist <= 25:
            score += 35
        elif dist <= 60:
            score += 10
        else:
            score -= 140

    context_ids = (
        extract_claim_entity_ids(entity, "P131")
        | extract_claim_entity_ids(entity, "P276")
        | extract_claim_entity_ids(entity, "P361")
    )
    if city_entity_id and city_entity_id in context_ids:
        score += 130

    score += score_expected_type_compatibility(candidate["poi_type_code"], label, description)

    guessed_type = map_poi_type_code(label, "", description)
    if guessed_type == candidate["poi_type_code"]:
        score += 30

    sitelinks = entity.get("sitelinks", {})
    score += min(len(sitelinks), 40)
    negative_terms = [
        "district", "barrio", "neighborhood", "stadium", "football", "hospital", "court",
    ]
    if any(token in clean_text(description).lower() for token in negative_terms):
        score -= 140
    if dist is not None and dist > 80:
        return score, dist, "too_far"
    if score < 95:
        return score, dist, "low_confidence"
    return score, dist, "resolved"


def build_ai_search_terms(
    *, city_name: str, country_code: str, candidate: dict[str, Any]
) -> list[str]:
    clean_city = clean_text(city_name)
    terms = [
        candidate["name"],
        f'{candidate["name"]} {clean_city}',
        f'{candidate["name"]} de {clean_city}',
        f'{candidate["name"]} {country_code}'.strip(),
        *candidate.get("aliases", []),
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = clean_text(term)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped
