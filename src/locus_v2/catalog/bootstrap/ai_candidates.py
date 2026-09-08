"""AI candidate generation and multi-language localization, ported from V1
`CatalogService._generate_ai_candidates` / `_localize_content_candidates`.

Spends real OpenAI credit on every call (reuses the same `LOCUS_OPENAI_API_KEY`
already configured, no new credentials). Gated by `use_ai_candidates` in
`CatalogBootstrapService` — off by default there is not how V1 behaves
(V1 defaults to `True`), so callers should surface this as an explicit choice.
"""

import json
from typing import Any

from locus_v2.catalog.bootstrap.ai_client import AiClientError, create_structured_response
from locus_v2.catalog.bootstrap.normalize import normalize_names, normalize_short_descriptions
from locus_v2.catalog.models import City
from locus_v2.shared.text import clean_text

CONTENT_LANGUAGES = ["es", "en", "fr", "it", "de", "pt", "zh", "ja", "ar"]
CONTENT_LOCALIZATION_CHUNK_SIZE = 4
CONTENT_LOCALIZATION_MAX_OUTPUT_TOKENS = 12000
POI_TYPE_CODES = ("monument", "museum", "church", "square", "building", "archaeological_site")


class AiCandidateError(RuntimeError):
    pass


def _build_candidate_instructions(limit: int) -> str:
    poi_types = ", ".join(
        f"{code} ({label})"
        for code, label in [
            ("monument", "monumento o hito historico"),
            ("museum", "museo o galeria"),
            ("church", "catedral, iglesia, sinagoga, monasterio o templo"),
            ("square", "plaza o espacio civico emblematico"),
            ("building", "edificio visitable relevante"),
            ("archaeological_site", "yacimiento o ruina arqueologica"),
        ]
    )
    return (
        "Devuelve candidatos de puntos de interes turisticos para una ciudad en JSON puro. "
        "No uses markdown, no expliques nada y no rellenes la lista si la ciudad "
        "tiene pocos sitios claros. "
        f"El limite superior es {limit}, pero puedes devolver menos. "
        "Prioriza lugares famosos, visitables o claramente reconocibles por viajeros. "
        "Incluye nombres alternativos utiles para resolver el sitio en Wikidata y para UI. "
        "Si conoces con suficiente confianza una ubicacion utilizable, devuelve lat y lng; "
        "si no estas seguro, devuelve null y no inventes. "
        "Incluye short_description con una sola frase breve y neutra sobre por que es relevante. "
        "Evita barrios, conjuntos demasiado genericos, juzgados, hospitales, oficinas, estaciones, "
        "estadios y relleno dudoso. "
        f"Usa solo estos poi_type_code: {poi_types}. "
        'Devuelve este shape exacto: {"city":"nombre","items":[{"name":"...",'
        '"poi_type_code":"...","aliases":["..."],"formatted_address":"... o null",'
        '"location_hint":"... o null","short_description":"...",'
        '"lat":0.0 o null,"lng":0.0 o null}]}.'
    )


def _candidate_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "poi_type_code": {"type": "string", "enum": list(POI_TYPE_CODES)},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "formatted_address": {"type": ["string", "null"]},
            "location_hint": {"type": ["string", "null"]},
            "short_description": {"type": "string"},
            "lat": {"type": ["number", "null"]},
            "lng": {"type": ["number", "null"]},
        },
        "required": [
            "name", "poi_type_code", "aliases", "formatted_address",
            "location_hint", "short_description", "lat", "lng",
        ],
    }


def _normalize_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


async def generate_ai_candidates(
    *, api_key: str, model: str, city: City, limit: int
) -> list[dict[str, Any]]:
    input_items = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        f"Ciudad: {city.name}\n"
                        f"Pais: {city.country_code or '(sin dato)'}\n"
                        f"Limite maximo: {limit}\n"
                        "Devuelve solo el JSON."
                    ),
                }
            ],
        }
    ]
    try:
        payload = await create_structured_response(
            api_key=api_key,
            model=model,
            instructions=_build_candidate_instructions(limit),
            input_items=input_items,
            json_schema_name="city_poi_candidates",
            json_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "city": {"type": "string"},
                    "items": {"type": "array", "items": _candidate_item_schema()},
                },
                "required": ["city", "items"],
            },
            max_output_tokens=min(12000, max(2000, 400 + (limit * 90))),
        )
    except AiClientError as error:
        raise AiCandidateError(f"No se pudieron generar candidatos con OpenAI: {error}") from error

    candidates: list[dict[str, Any]] = []
    for item in payload.get("items", [])[:limit]:
        name = clean_text(item.get("name", ""))
        poi_type_code = clean_text(item.get("poi_type_code", "")).lower()
        if not name or poi_type_code not in POI_TYPE_CODES:
            continue
        aliases = [clean_text(alias) for alias in item.get("aliases", []) if clean_text(alias)]
        lat = _normalize_optional_float(item.get("lat"))
        lng = _normalize_optional_float(item.get("lng"))
        if lat is not None and not (-90 <= lat <= 90):
            lat = None
        if lng is not None and not (-180 <= lng <= 180):
            lng = None
        if (lat is None) != (lng is None):
            lat, lng = None, None
        candidates.append(
            {
                "name": name,
                "poi_type_code": poi_type_code,
                "aliases": aliases[:5],
                "formatted_address": clean_text(str(item.get("formatted_address") or ""))[:255],
                "location_hint": clean_text(str(item.get("location_hint") or ""))[:255],
                "short_description": clean_text(str(item.get("short_description") or ""))[:500],
                "lat": lat,
                "lng": lng,
            }
        )
    return candidates


async def localize_content_candidates(
    *, api_key: str, model: str, candidates: list[dict[str, Any]], context: str
) -> list[dict[str, Any]]:
    """Fills `candidate["names"]` / `candidate["short_descriptions"]` in place
    (also returned) for every language in `CONTENT_LANGUAGES`. Best-effort: on
    any failure the candidates are returned with only their original (Spanish)
    content, same as V1.
    """
    pending = [c for c in candidates if _needs_localization(c)]
    if not pending:
        return candidates
    if len(pending) > CONTENT_LOCALIZATION_CHUNK_SIZE:
        for offset in range(0, len(pending), CONTENT_LOCALIZATION_CHUNK_SIZE):
            await localize_content_candidates(
                api_key=api_key, model=model,
                candidates=pending[offset : offset + CONTENT_LOCALIZATION_CHUNK_SIZE],
                context=f"{context}:chunk_{offset // CONTENT_LOCALIZATION_CHUNK_SIZE + 1}",
            )
        return candidates

    items = [
        {
            "key": str(index),
            "name": candidate.get("name") or "",
            "aliases": candidate.get("aliases") or [],
            "short_description": candidate.get("short_description") or "",
            "type": candidate.get("poi_type_code") or "",
            "context": context,
        }
        for index, candidate in enumerate(pending)
    ]
    language_schema = {language: {"type": "string"} for language in CONTENT_LANGUAGES}

    try:
        payload = await create_structured_response(
            api_key=api_key,
            model=model,
            instructions=(
                "Eres un traductor de catalogo turistico. Devuelve JSON puro. "
                "Traduce o translitera nombres de POIs cuando sea util para una interfaz "
                "turistica, "
                "preservando el nombre local en local/ja/zh/ar si aplica. "
                "Traduce short_description en una sola frase natural por idioma. "
                "No inventes datos nuevos ni anadas claims: solo traduce el significado dado."
            ),
            input_items=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {"languages": CONTENT_LANGUAGES, "items": items}, ensure_ascii=False
                            ),
                        }
                    ],
                }
            ],
            json_schema_name="localized_catalog_items",
            json_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "key": {"type": "string"},
                                "names": {
                                    "type": "object", "additionalProperties": False,
                                    "properties": language_schema, "required": CONTENT_LANGUAGES,
                                },
                                "short_descriptions": {
                                    "type": "object", "additionalProperties": False,
                                    "properties": language_schema, "required": CONTENT_LANGUAGES,
                                },
                            },
                            "required": ["key", "names", "short_descriptions"],
                        },
                    }
                },
                "required": ["items"],
            },
            max_output_tokens=min(
                CONTENT_LOCALIZATION_MAX_OUTPUT_TOKENS, max(1800, len(items) * 950)
            ),
        )
    except AiClientError:
        return candidates

    by_key = {
        str(item.get("key")): item
        for item in payload.get("items", [])
        if isinstance(item, dict)
    }
    for index, candidate in enumerate(pending):
        translated = by_key.get(str(index)) or {}
        candidate["names"] = {
            **normalize_names(candidate.get("name") or "", candidate.get("names") or {}),
            **normalize_names(candidate.get("name") or "", translated.get("names") or {}),
        }
        candidate["short_descriptions"] = {
            **normalize_short_descriptions(
                candidate.get("short_description") or "", candidate.get("short_descriptions") or {}
            ),
            **normalize_short_descriptions(
                candidate.get("short_description") or "", translated.get("short_descriptions") or {}
            ),
        }
    return candidates


def _needs_localization(candidate: dict[str, Any]) -> bool:
    names = normalize_names(candidate.get("name") or "", candidate.get("names") or {})
    descriptions = normalize_short_descriptions(
        candidate.get("short_description") or "", candidate.get("short_descriptions") or {}
    )
    return any(not names.get(lang) for lang in CONTENT_LANGUAGES) or any(
        not descriptions.get(lang) for lang in CONTENT_LANGUAGES
    )


def names_from_aliases(name: str, aliases: list[str] | None = None) -> dict[str, str]:
    names = normalize_names(name)
    for alias in aliases or []:
        clean_alias = clean_text(alias)
        if not clean_alias:
            continue
        if _is_latinish_text(clean_alias):
            names.setdefault("en", clean_alias)
            names.setdefault("es", clean_alias)
            break
    return names


def _is_latinish_text(value: str) -> bool:
    cleaned = clean_text(value)
    if not cleaned:
        return False
    letters = [char for char in cleaned if char.isalpha()]
    ascii_count = sum(1 for char in letters if ord(char) < 128)
    return bool(letters) and ascii_count >= max(1, len(letters) // 2)
