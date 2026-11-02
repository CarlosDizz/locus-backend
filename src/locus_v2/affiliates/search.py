"""Buscar enlaces reales de reserva en los programas de referidos de un tipo.

Para actividades sigue la búsqueda de GetYourGuide de Locus (service.py). Para lo
demás, el modelo de herramientas busca en la web limitado a los dominios de los
programas de ese tipo y se quedan solo páginas concretas, decoradas con nuestro
código. Algunos tipos no necesitan buscar: un alojamiento en Booking o un
producto en Amazon se resuelven mejor con su propia búsqueda (con fechas, en la
tienda del país), que no se rompe nunca.
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlencode, urlparse

import structlog

from locus_v2.affiliates.programs import decorate, is_configured, program_for, programs_of
from locus_v2.config import Settings
from locus_v2.shared.openai_client import build_openai_client
from locus_v2.shared.openai_usage import ToolUsage, usage_from_openai_response
from locus_v2.shared.text import clean_text

logger = structlog.get_logger()

# La tienda de Amazon según el idioma de quien planifica.
# Nuestro ID de Amazon Afiliados (borea0b-21, 2026-10-24) solo cobra en amazon.es, que
# envía a toda la UE: todos los idiomas van ahí. Si un día hay ID en otra tienda, se
# añade aquí su idioma (y su ID en LOCUS_AFFILIATE_LINKS, que hoy vale para todas).
_AMAZON_STORE: dict[str, str] = {}


@dataclass(frozen=True)
class FoundLink:
    title: str
    url: str
    program: str

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "provider": self.program}


def booking_search(settings: Settings, city: str, checkin: date | None, checkout: date | None,
                   adults: int = 2, rooms: int = 1) -> FoundLink:
    """Alojamiento en Booking para esas fechas: su búsqueda, con nuestro código."""
    params: dict[str, Any] = {
        "ss": city, "group_adults": max(1, adults), "no_rooms": max(1, rooms),
    }
    if checkin and checkout and checkout > checkin:
        params["checkin"] = checkin.isoformat()
        params["checkout"] = checkout.isoformat()
    url, _ = decorate("https://www.booking.com/searchresults.html?" + urlencode(params), settings)
    return FoundLink(title=f"Alojamientos en {city} en Booking", url=url, program="booking")


def amazon_search(settings: Settings, product: str, locale: str = "es") -> FoundLink:
    store = _AMAZON_STORE.get(locale.split("-", 1)[0].lower(), "amazon.es")
    url, _ = decorate(f"https://www.{store}/s?" + urlencode({"k": product}), settings)
    return FoundLink(title=f"{product} en Amazon", url=url, program="amazon")


# Subdominios de ayuda, ficheros o imágenes: nunca son la página de un producto
# (res.klook.com servía PDF con la guía de un pase, 2026-11-02).
_NOT_PRODUCT_HOSTS = ("res.", "cdn", "static.", "support.", "help.", "mobilecontent.", "media.")


def _is_concrete_page(url: str) -> bool:
    """Una página de producto o de ruta, no la portada ni una búsqueda."""
    parts = urlparse(url)
    host = (parts.hostname or "").lower()
    path = parts.path.strip("/")
    if host.startswith(_NOT_PRODUCT_HOSTS) or path.lower().endswith(".pdf"):
        return False
    if not path or path.count("/") < 1 and len(path) < 12:
        return False
    excluidas = (
        "search", "busqueda", "/s/", "results", "blog", "restaurantes/", "restaurants/",
        "category/", "landing/",
    )
    return not any(part in f"{path.lower()}/" for part in excluidas)


def _page_key(url: str) -> str:
    """La misma página en otro idioma cuenta una vez (klook.com/en-US/… y /en-GB/…)."""
    parts = urlparse(url)
    path = re.sub(r"^/[a-z]{2}(-[a-zA-Z]{2})?(?=/)", "", parts.path.rstrip("/"))
    return f"{(parts.hostname or '').removeprefix('www.')}{path}"


async def web_program_links(
    settings: Settings, kind: str, query: str, city: str = "", max_results: int = 3
) -> tuple[list[FoundLink], ToolUsage]:
    """Páginas concretas de los programas de un tipo, encontradas en la web."""
    programs = programs_of(kind)
    domains = [d for p in programs for d in p.domains]
    if settings.openai_api_key is None or not domains or not query.strip():
        return [], ToolUsage()
    paid = [d for p in programs if is_configured(settings, p.code) for d in p.domains]
    if kind == "transport" and paid and len(paid) < len(domains):
        # Omio y Trainline copan los resultados de trenes: el tren bala Tokio-Osaka solo
        # devolvía Omio aunque Klook lo vende. A la vez, una búsqueda solo en los que
        # pagan; lo suyo va primero. Solo en trenes: es una búsqueda web más que pagar.
        (de_pago, uso_pago), (todos, uso) = await asyncio.gather(
            _search(settings, kind, query, city, paid),
            _search(settings, kind, query, city, domains),
        )
        found, usage = de_pago + todos, uso_pago + uso
    else:
        found, usage = await _search(settings, kind, query, city, domains)

    links: list[FoundLink] = []
    seen: set[str] = set()
    for link in found:
        key = _page_key(link.url)
        if key in seen:
            continue
        seen.add(key)
        final, _ = decorate(link.url, settings)
        links.append(FoundLink(title=link.title, url=final, program=link.program))
    # Primero los programas donde tenemos código: a la persona le sirven igual y son
    # los que sostienen la app (Omio antes que Trainline, por ejemplo).
    links.sort(key=lambda link: not is_configured(settings, link.program))
    return links[: max(1, min(max_results, 5))], usage


async def _search(
    settings: Settings, kind: str, query: str, city: str, domains: list[str]
) -> tuple[list[FoundLink], ToolUsage]:
    """Lo que la búsqueda web limitada a esos dominios encuentra, sin decorar."""
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

    found: list[FoundLink] = []
    for source in _sources(response):
        url = clean_text(source.get("url", ""))
        program = program_for(url)
        if program is None or not program.sells(kind) or not _is_concrete_page(url):
            continue
        found.append(FoundLink(
            title=clean_text(source.get("title", "")) or program.name, url=url,
            program=program.code,
        ))
    return found, usage_from_openai_response(response)


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
