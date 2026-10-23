"""Las apps del grupo Locus que hablan con este backend.

Locus Guide y Borea comparten base de datos, cuentas y saldo; lo que cambia por
app es lo que se ve y cómo habla el planificador. Cada petición dice de qué app
viene con la cabecera X-Locus-App; sin ella es Locus Guide, que es lo que manda
la app ya publicada.

Los viajes guardan en qué apps se ven como máscara de bits (1 Locus Guide,
2 Borea, 3 las dos), para que una tercera app sea el 4 sin tocar lo que hay.
"""

from enum import StrEnum


class ClientApp(StrEnum):
    GUIDE = "guide"
    BOREA = "borea"


APP_BITS: dict[ClientApp, int] = {ClientApp.GUIDE: 1, ClientApp.BOREA: 2}


def client_app(value: str | None) -> ClientApp:
    """La app de una cabecera X-Locus-App; lo desconocido o vacío es Locus Guide."""
    try:
        return ClientApp((value or "").strip().lower())
    except ValueError:
        return ClientApp.GUIDE


def app_bit(app: ClientApp) -> int:
    return APP_BITS[app]


def visible_in(apps_mask: int, app: ClientApp) -> bool:
    return bool((apps_mask or APP_BITS[ClientApp.GUIDE]) & APP_BITS[app])
