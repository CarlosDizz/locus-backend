"""Buscar enlaces reales de reserva en los programas de referidos de un tipo.

Para actividades sigue la búsqueda de GetYourGuide de Locus (service.py). Para lo
demás, el modelo de herramientas busca en la web limitado a los dominios de los
programas de ese tipo y se quedan solo páginas concretas, decoradas con nuestro
código. Algunos tipos no necesitan buscar: un alojamiento en Booking o un
producto en Amazon se resuelven mejor con su propia búsqueda (con fechas, en la
tienda del país), que no se rompe nunca.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlencode, urlparse

import structlog

from locus_v2.affiliates.programs import decorate, program_for, programs_of
from locus_v2.config import Settings
from locus_v2.shared.openai_client import build_openai_client
from locus_v2.shared.openai_usage import ToolUsage, usage_from_openai_response
from locus_v2.shared.text import clean_text

logger = structlog.get_logger()

# La tienda de Amazon según el idioma de quien planifica.
_AMAZON_STORE = {
    "es": "amazon.es", "de": "amazon.de", "fr": "amazon.fr", "it": "amazon.it",
    "en": "amazon.co.uk", "pt": "amazon.es",
}


@dataclass(frozen=True)
class FoundLink:
    title: str
    url: str
    program: str

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "provider": self.program}


def booking_search(settings: Settings, city: str, checkin: date | None, checkout: date | None,
                   adults: int = 2) -> FoundLink:
    """Alojamiento en Booking para esas fechas: su búsqueda, con nuestro código."""
    params: dict[str, Any] = {"ss": city, "group_adults": max(1, adults)}
    if checkin and checkout and checkout > checkin:
        params["checkin"] = checkin.isoformat()
        params["checkout"] = checkout.isoformat()
    url, _ = decorate("https://www.booking.com/searchresults.html?" + urlencode(params), settings)
    return FoundLink(title=f"Alojamientos en {city} en Booking", url=url, program="booking")


def amazon_search(settings: Settings, product: str, locale: str = "es") -> FoundLink:
    store = _AMAZON_STORE.get(locale.split("-", 1)[0].lower(), "amazon.es")
    url, _ = decorate(f"https://www.{store}/s?" + urlencode({"k": product}), settings)
    return FoundLink(title=f"{product} en Amazon", url=url, program="amazon")


def _is_concrete_page(url: str) -> bool:
    """Una página de producto o de ruta, no la portada ni una búsqueda."""
    path = urlparse(url).path.strip("/")
    if not path or path.count("/") < 1 and len(path) < 12:
        return False
    return not any(part in path.lower() for part in ("search", "busqueda", "/s/", "results"))


async def web_program_links(
    settings: Settings, kind: str, query: str, city: str = "", max_results: int = 3
) -> tuple[list[FoundLink], ToolUsage]:
    """Páginas concretas de los programas de un tipo, encontradas en la web."""
    programs = programs_of(kind)
    domains = [d for p in programs for d in p.domains]
    if settings.openai_api_key is None or not domains or not query.strip():
        return [], ToolUsage()
    client = build_openai_client(settings)
    try:
        response = await client.responses.create(
            model=settings.tool_model,
            instructions=(
                "Busca páginas concretas donde reservar lo que se pide (un trayecto, una "
                "mesa, un coche, una actividad). No devuelvas portadas, categorías ni "
                "artículos. Si no aparece nada claro, no inventes."
            ),
            input=f"{query}\nCiudad: {city or '(sin ciudad)'}",
            tools=[{"type": "web_search", "filters": {"allowed_domains": domains}}],
            max_output_tokens=1500,
            reasoning={"effort": "low"},
            include=["web_search_call.action.sources"],
        )
    except Exception as error:  # noqa: BLE001 - sin enlaces el plan sigue
        logger.warning("program_websearch_failed", kind=kind, error=str(error))
        return [], ToolUsage()
    finally:
        await client.close()

    usage = usage_from_openai_response(response)
    links: list[FoundLink] = []
    seen: set[str] = set()
    for source in _sources(response):
        url = clean_text(source.get("url", ""))
        program = program_for(url)
        if program is None or program.kind != kind or not _is_concrete_page(url):
            continue
        final, _ = decorate(url, settings)
        if final in seen:
            continue
        seen.add(final)
        links.append(FoundLink(
            title=clean_text(source.get("title", "")) or program.name, url=final,
            program=program.code,
        ))
        if len(links) >= max(1, min(max_results, 5)):
            break
    return links, usage


def _sources(response: Any) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "web_search_call":
            for source in getattr(getattr(item, "action", None), "sources", None) or []:
                found.append({"url": str(getattr(source, "url", "") or ""),
                              "title": str(getattr(source, "title", "") or "")})
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", None) or []:
                for note in getattr(content, "annotations", None) or []:
                    if getattr(note, "type", None) == "url_citation":
                        found.append({"url": str(getattr(note, "url", "") or ""),
                                      "title": str(getattr(note, "title", "") or "")})
    return found
