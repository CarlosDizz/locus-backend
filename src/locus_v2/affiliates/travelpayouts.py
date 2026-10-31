"""El enlace de afiliado de un programa de Travelpayouts, pedido a su API al hacer clic.

Travelpayouts agrupa decenas de programas (Klook, Airalo, Welcome Pickups…) y su
API de enlaces convierte cualquier URL de un programa aprobado en la suya
(tp.media/r?campaign_id=…). Se pide en /r/, cuando alguien pulsa, y no al proponer:
casi ningún enlace propuesto se pulsa, y así un programa que aprueben mañana
funciona sin tocar nada. Si la API falla o tarda, se va al enlace normal: la
persona llega igual a la página, solo que sin comisión.
"""

import time

import httpx
import structlog

from locus_v2.affiliates.programs import program_for
from locus_v2.config import Settings

logger = structlog.get_logger()

API_URL = "https://api.travelpayouts.com/links/v1/create"
TIMEOUT_SECONDS = 3.0
# Cada programa da siempre el mismo campaign_id y p: un día de caché sobra.
_CACHE_SECONDS = 24 * 3600
_CACHE_MAX = 2000
_cache: dict[str, tuple[float, str]] = {}


def _wants(settings: Settings, url: str) -> bool:
    program = program_for(url)
    return (
        settings.travelpayouts_token is not None
        and program is not None
        and program.network == "travelpayouts"
    )


async def partner_url(settings: Settings, url: str) -> str | None:
    """El enlace de afiliado de esa URL, o None si no es de un programa o falla."""
    if not _wants(settings, url):
        return None
    ahora = time.monotonic()
    guardado = _cache.get(url)
    if guardado is not None and guardado[0] > ahora:
        return guardado[1]
    assert settings.travelpayouts_token is not None
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                API_URL,
                headers={"X-Access-Token": settings.travelpayouts_token.get_secret_value()},
                json={
                    "trs": settings.travelpayouts_trs,
                    "marker": settings.travelpayouts_marker,
                    "shorten": False,
                    "links": [{"url": url}],
                },
            )
        response.raise_for_status()
        enlace = (response.json().get("result") or {}).get("links", [{}])[0]
    except (httpx.HTTPError, ValueError, IndexError, AttributeError) as error:
        logger.warning("travelpayouts_link_failed", url=url, error=str(error))
        return None
    partner = str(enlace.get("partner_url") or "")
    if enlace.get("code") != "success" or not partner.startswith("https://"):
        logger.warning("travelpayouts_link_refused", url=url, code=enlace.get("code"))
        return None
    if len(_cache) >= _CACHE_MAX:
        _cache.clear()
    _cache[url] = (ahora + _CACHE_SECONDS, partner)
    return partner
