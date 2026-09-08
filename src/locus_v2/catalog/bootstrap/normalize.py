"""Small normalization helpers ported from V1 `CatalogService`, used to keep
`names_json`/`short_descriptions_json` shaped the same way regardless of
which pipeline (bootstrap, admin edit, V1 import) wrote them.
"""

import hashlib
import re

from locus_v2.shared.text import clean_text, slugify


def safe_slug(value: str, *, prefix: str = "item") -> str:
    slug = slugify(value)
    if slug:
        return slug
    normalized = clean_text(value)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def normalize_names(name: str, names: dict[str, str] | None = None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in (names or {}).items():
        clean_key = clean_text(str(key)).lower()
        clean_value = clean_text(str(value))
        if clean_key and clean_value:
            normalized[clean_key] = clean_value
    clean_name = clean_text(name)
    if clean_name:
        normalized.setdefault("local", clean_name)
    return normalized


def normalize_short_descriptions(
    description: str, descriptions: dict[str, str] | None = None
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in (descriptions or {}).items():
        clean_key = clean_text(str(key)).lower()
        clean_value = clean_text(str(value))
        if clean_key and clean_value:
            normalized[clean_key] = clean_value[:500]
    clean_description = clean_text(description)
    if clean_description:
        normalized.setdefault("local", clean_description[:500])
    return normalized


def wkt_point_to_coords(wkt: str) -> tuple[float | None, float | None]:
    match = re.match(r"Point\(([-0-9.]+)\s+([-0-9.]+)\)", wkt or "")
    if not match:
        return None, None
    lng = float(match.group(1))
    lat = float(match.group(2))
    return lat, lng
