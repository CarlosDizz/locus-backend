"""Los programas de referidos y cómo se convierte un enlace en uno nuestro.

El modelo nunca pone el código de afiliado: propone un enlace real y el backend
lo decora si el dominio es de un programa. Así no se pierde una comisión porque
el modelo copie mal el enlace, y añadir un programa no obliga a tocar las skills.

Cada programa se activa en la configuración (LOCUS_AFFILIATE_LINKS) con una de
dos formas, porque no todos funcionan igual:

- «param=valor»: se añade al enlace como parámetro (Booking: «aid=1234567»,
  Amazon: «tag=borea-21», Civitatis: «aid=123»).
- una plantilla con {url}: el enlace va envuelto en el de una red de afiliados,
  como Awin («https://www.awin1.com/cread.php?awinmid=1&awinaffid=2&ued={url}»).

Un programa sin configurar sigue sirviendo: el enlace se da tal cual, sin
comisión, porque a la persona le sirve igual (lo que pasa hoy con Airbnb).
"""

from dataclasses import dataclass
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

from locus_v2.config import Settings

# Lo que el modelo puede querer reservar; cada tool de skill busca en un tipo.
KINDS = (
    "activity", "stay", "transport", "car", "gear",
    "transfer", "esim", "luggage", "flight", "claims",
)


@dataclass(frozen=True)
class AffiliateProgram:
    code: str
    name: str
    kind: str
    # Dominios del programa, sin «www.»; vale cualquier subdominio.
    domains: tuple[str, ...]
    # «travelpayouts»: el enlace de afiliado lo da su API al hacer clic
    # (affiliates/travelpayouts.py), sin configurar nada por programa.
    network: str = ""
    # Otros tipos en los que también vende: Klook y KKday venden el JR Pass y los
    # billetes de Shinkansen, no solo actividades.
    also: tuple[str, ...] = ()

    def sells(self, kind: str) -> bool:
        return kind == self.kind or kind in self.also


PROGRAMS: tuple[AffiliateProgram, ...] = (
    AffiliateProgram(
        "getyourguide", "GetYourGuide", "activity", ("getyourguide.com", "getyourguide.es")
    ),
    AffiliateProgram("civitatis", "Civitatis", "activity", ("civitatis.com",)),
    AffiliateProgram("tiqets", "Tiqets", "activity", ("tiqets.com",)),
    AffiliateProgram("booking", "Booking.com", "stay", ("booking.com",)),
    AffiliateProgram("vrbo", "Vrbo", "stay", ("vrbo.com",)),
    # Sin programa de afiliados público: se enlaza sin comisión (2026-10-24).
    AffiliateProgram("airbnb", "Airbnb", "stay", ("airbnb.com", "airbnb.es")),
    AffiliateProgram("trainline", "Trainline", "transport", ("thetrainline.com", "trainline.com")),
    AffiliateProgram("omio", "Omio", "transport", ("omio.com", "omio.es")),
    # Sin TheFork (2026-10-24): los restaurantes se recomiendan por las valoraciones de
    # Google y sin enlace de reserva; nadie paga por aparecer.
    AffiliateProgram("discovercars", "DiscoverCars", "car", ("discovercars.com",)),
    AffiliateProgram("rentalcars", "Rentalcars", "car", ("rentalcars.com",)),
    AffiliateProgram(
        "amazon", "Amazon", "gear",
        ("amazon.es", "amazon.com", "amazon.de", "amazon.fr", "amazon.it", "amazon.co.uk"),
    ),
    # Aprobados en Travelpayouts (2026-10-31), comprobados uno a uno contra su API de
    # enlaces. Booking, GetYourGuide, Omio, Viator o DiscoverCars siguen bloqueados
    # ahí por falta de tráfico; GigSky no convierte su dominio y no está.
    *(
        AffiliateProgram(code, name, kind, domains, network="travelpayouts", also=tuple(also))
        for code, name, kind, domains, *also in (
            ("klook", "Klook", "activity", ("klook.com",), "transport"),
            ("wegotrip", "WeGoTrip", "activity", ("wegotrip.com",)),
            ("gocity", "Go City", "activity", ("gocity.com",)),
            ("kkday", "KKday", "activity", ("kkday.com",), "transport"),
            ("intui", "intui.travel", "activity", ("intui.travel",)),
            ("welcomepickups", "Welcome Pickups", "transfer", ("welcomepickups.com",)),
            ("kiwitaxi", "Kiwitaxi", "transfer", ("kiwitaxi.com",)),
            ("gettransfer", "GetTransfer", "transfer", ("gettransfer.com",)),
            ("localrent", "Localrent", "car", ("localrent.com",)),
            ("getrentacar", "GetRentacar", "car", ("getrentacar.com",)),
            ("economybookings", "Economybookings", "car", ("economybookings.com",)),
            ("qeeq", "QEEQ", "car", ("qeeq.com",)),
            ("autoeurope", "Auto Europe", "car", ("autoeurope.eu", "autoeurope.co.uk")),
            ("bikesbooking", "BikesBooking", "car", ("bikesbooking.com",)),
            ("airalo", "Airalo", "esim", ("airalo.com",)),
            ("yesim", "Yesim", "esim", ("yesim.app",)),
            ("saily", "Saily", "esim", ("saily.com",)),
            ("drimsim", "Drimsim", "esim", ("drimsim.com",)),
            ("radicalstorage", "Radical Storage", "luggage", ("radicalstorage.com",)),
            ("aviasales", "Aviasales", "flight", ("aviasales.com",)),
            ("kiwi", "Kiwi.com", "flight", ("kiwi.com",)),
            ("airhelp", "AirHelp", "claims", ("airhelp.com",)),
            ("compensair", "Compensair", "claims", ("compensair.com",)),
        )
    ),
)

# Parámetros de seguimiento ajenos que se quitan al decorar un enlace.
_FOREIGN_TRACKING = {
    "partner_id", "utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term",
    "aid", "tag", "affid", "cmp", "visitor-id", "msockid",
}


def program_for(url: str) -> AffiliateProgram | None:
    host = (urlparse(url.strip()).hostname or "").lower()
    for program in PROGRAMS:
        if any(host == d or host.endswith("." + d) for d in program.domains):
            return program
    return None


def programs_of(kind: str) -> list[AffiliateProgram]:
    return [p for p in PROGRAMS if p.sells(kind)]


def _program(code: str) -> AffiliateProgram | None:
    return next((p for p in PROGRAMS if p.code == code), None)


def program_config(settings: Settings, code: str) -> str:
    """Cómo se marca un enlace de ese programa; vacío si no está configurado."""
    configured = (getattr(settings, "affiliate_links", None) or {}).get(code, "")
    if configured:
        return configured.strip()
    # GetYourGuide ya tenía su propia variable desde Locus.
    if code == "getyourguide" and getattr(settings, "getyourguide_partner_id", ""):
        return f"partner_id={settings.getyourguide_partner_id}"
    return ""


def is_configured(settings: Settings, code: str) -> bool:
    program = _program(code)
    if program is not None and program.network == "travelpayouts":
        return getattr(settings, "travelpayouts_token", None) is not None
    return bool(program_config(settings, code))


def decorate(url: str, settings: Settings) -> tuple[str, AffiliateProgram | None]:
    """El enlace con nuestro código si es de un programa configurado.

    Devuelve el enlace (decorado o tal cual) y su programa, o el enlace sin tocar
    y None si no es de ningún programa conocido. Solo se aceptan enlaces https.
    """
    clean = url.strip()
    parts = urlparse(clean)
    program = program_for(clean)
    if program is None or parts.scheme != "https":
        return clean, program
    config = program_config(settings, program.code)
    query = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _FOREIGN_TRACKING
    ]
    plain = urlunparse(parts._replace(query=urlencode(query)))
    if not config:
        return plain, program
    if "{url}" in config:
        return config.replace("{url}", quote(plain, safe="")), program
    key, _, value = config.partition("=")
    query.append((key.strip(), value.strip()))
    if program.code == "getyourguide":
        query.append(("utm_medium", "travel_agent"))
    return urlunparse(parts._replace(query=urlencode(query))), program
