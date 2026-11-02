"""Las herramientas del planificador de Borea.

Borea parte de las herramientas del planificador de Locus (apuntar el viaje,
añadir, proponer, cambiar y quitar franjas) y añade las suyas:

- siempre: cargar_skill, recomendar y quitar_recomendacion («Para este viaje»),
  guardar_clima y buscar_alojamiento;
- al cargar una skill, las que esa skill declara en su línea `tools:` (ruta,
  buscar_transporte, buscar_sitios, buscar_actividades,
  buscar_equipamiento). Una tool por tipo de cosa, no por plataforma: en qué
  programas se busca lo decide el backend (affiliates/programs.py).

Los enlaces que devuelven ya van con nuestro código; el modelo solo los copia en
booking_url, y aun así el backend los vuelve a decorar al apuntarlos.
"""

import asyncio
import copy
import json
import time
from datetime import date
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from locus_v2.affiliates.programs import is_configured
from locus_v2.affiliates.search import amazon_search, booking_search, web_program_links
from locus_v2.affiliates.service import ReferralService
from locus_v2.places.client import GooglePlacesClient
from locus_v2.places.service import distance_km
from locus_v2.trips.routes import travel_time
from locus_v2.trips.skills import load_skills

if TYPE_CHECKING:
    from locus_v2.trips.models import Trip
    from locus_v2.trips.service import TripPlannerService, _Turn

MAX_RECOMMENDATIONS = 6
RECOMMENDATION_KINDS = ("equipamiento", "entradas", "ritmo", "tiempo", "documentos", "otro")


def _function(name: str, description: str, properties: dict[str, Any],
              required: list[str]) -> dict[str, Any]:
    return {
        "type": "function", "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


BASE_TOOLS: list[dict[str, Any]] = [
    _function(
        "cargar_skill",
        "Carga el criterio de un área (transporte, hostelería, turismo, equipamiento) y "
        "activa sus herramientas. Hazlo antes de trabajar en esa área.",
        {"nombre": {"type": "string", "enum": sorted(load_skills())}},
        ["nombre"],
    ),
    _function(
        "recomendar",
        "Apunta una recomendación en «Para este viaje», que la persona ve al final del plan: "
        "lo que conviene llevar, reservar con tiempo, el ritmo o el tiempo. Una frase de "
        "motivo, concreta para este viaje.",
        {
            "kind": {"type": "string", "enum": list(RECOMMENDATION_KINDS)},
            "title": {
                "type": "string", "description": "Corto: «Calzado cómodo para el empedrado».",
            },
            "reason": {
                "type": "string", "description": "Por qué en este viaje, en una o dos frases.",
            },
            "booking_url": {
                "type": "string",
                "description": "Solo un enlace devuelto por una herramienta de búsqueda.",
            },
            "booking_title": {"type": "string"},
        },
        ["kind", "title", "reason"],
    ),
    _function(
        "quitar_recomendacion",
        "Quita una recomendación de «Para este viaje» (por su id).",
        {"id": {"type": "string"}},
        ["id"],
    ),
    _function(
        "guardar_clima",
        "Apunta el tiempo que suele hacer en esas fechas, buscado en la web, cuando el viaje "
        "aún no tiene previsión (queda a más de diez días). Rangos, no un dato exacto.",
        {
            "max_c": {"type": "number", "description": "Máxima media en °C."},
            "min_c": {"type": "number", "description": "Mínima media en °C."},
            "rain_days": {"type": "number", "description": "Días de lluvia de media en el mes."},
            "summary": {"type": "string", "description": "Una frase: «Templado, alguna lluvia»."},
            "source": {"type": "string", "description": "La web de donde sale."},
        },
        ["max_c", "min_c", "summary"],
    ),
    _function(
        "buscar_alojamiento",
        "Hoteles reales de la ciudad del viaje en Google, con valoración, reseñas, nivel de "
        "precio, distancia al centro y su enlace de Booking para las fechas del viaje. Úsalo "
        "solo si la persona aún no tiene dónde dormir.",
        {
            "ciudad": {
                "type": "string",
                "description": (
                    "Ciudad donde dormirá. Obligatoria si el viaje pasa por varias "
                    "(«Hué»), y siempre que no sea la ciudad del viaje."
                ),
            },
            "entrada": {
                "type": "string",
                "description": (
                    "Noche de llegada a ese alojamiento, AAAA-MM-DD, si no es la del viaje."
                ),
            },
            "salida": {
                "type": "string",
                "description": "Día que lo deja, AAAA-MM-DD, si no es el final del viaje.",
            },
            "personas": {
                "type": "integer",
                "description": "Cuántos adultos duermen (quien viaja solo, 1).",
            },
            "habitaciones": {"type": "integer", "description": "Cuántas habitaciones, si se sabe."},
            "zona": {"type": "string", "description": "Barrio o zona, si se ha elegido."},
            "tipo": {
                "type": "string",
                "description": "Lo que busca en pocas palabras: «hotel con parking», «hostal».",
            },
        },
        [],
    ),
]

def _entero(valor: Any, defecto: int) -> int:
    try:
        return max(1, min(int(valor), 30))
    except (TypeError, ValueError):
        return defecto


def _fecha(valor: Any) -> date | None:
    try:
        return date.fromisoformat(str(valor or "").strip()[:10])
    except ValueError:
        return None


_NOTHING_BOOKABLE = (
    "No hay ninguna actividad reservable concreta para esto. No pongas tarjeta con enlace de "
    "reserva ni digas que hay una visita. Si existe algo real (la visita gratuita de la "
    "oficina de turismo, un paseo por libre), propónlo sin enlace y di de dónde lo sacas."
)


def only_concrete(resultado: dict[str, Any]) -> dict[str, Any]:
    """Sin búsquedas genéricas: en Borea solo se ofrece lo que tiene su página.

    Cuando no había un tour concreto, GetYourGuide devolvía una búsqueda con la frase
    entera; para Albacete («Visita guiada de Albacete centro histórico») mandaba a
    Breslavia, y el planificador la presentaba como una actividad (2026-10-24).
    """
    if resultado.get("provider") != "getyourguide_fallback_search":
        return resultado
    return {"ok": True, "links": [], "policy": _NOTHING_BOOKABLE}


SKILL_TOOLS: dict[str, dict[str, Any]] = {
    "ruta": _function(
        "ruta",
        "Cuánto se tarda de un sitio a otro, comprobado con Google.",
        {
            "origen": {"type": "string", "description": "Sitio y ciudad: «Coliseo, Roma»."},
            "destino": {"type": "string"},
            "modo": {"type": "string", "enum": ["a pie", "transporte público", "coche"]},
        },
        ["origen", "destino"],
    ),
    "buscar_transporte": _function(
        "buscar_transporte",
        "Enlaces reales para reservar un tren, un autobús, un vuelo, un coche de alquiler, "
        "el traslado del aeropuerto al alojamiento o una consigna de equipaje.",
        {
            "consulta": {
                "type": "string",
                "description": "«Tren Madrid Toledo el domingo», «vuelo Tokio Okinawa», "
                               "«traslado aeropuerto Narita a Shinjuku», «consigna Lisboa».",
            },
            "tipo": {
                "type": "string",
                "enum": ["tren", "autobús", "vuelo", "coche", "traslado", "consigna"],
            },
        },
        ["consulta"],
    ),
    "buscar_esim": _function(
        "buscar_esim",
        "Una eSIM de datos para el país del viaje, con su enlace, para no pagar roaming.",
        {"pais": {"type": "string", "description": "El país en inglés: «Japan», «Vietnam»."}},
        ["pais"],
    ),
    "buscar_sitios": _function(
        "buscar_sitios",
        "Restaurantes, bares o cafés en Google con su valoración y número de reseñas, para "
        "elegir con criterio y huir de las turistadas.",
        {"consulta": {"type": "string", "description": "«Trattoria romana en Monti»."}},
        ["consulta"],
    ),
    "buscar_actividades": _function(
        "buscar_actividades",
        "Entradas, visitas guiadas, tours y excursiones reservables, con enlaces reales.",
        {
            "consulta": {"type": "string", "description": "«Museos Vaticanos sin colas»."},
            "lugar": {"type": "string", "description": "Sitio concreto, si lo hay."},
        },
        ["consulta"],
    ),
    "buscar_equipamiento": _function(
        "buscar_equipamiento",
        "Un enlace para comprar algo que conviene llevar a este viaje (un paraguas plegable, "
        "un calzado cómodo, un adaptador de enchufe).",
        {"producto": {
            "type": "string",
            "description": "El producto en 2 a 4 palabras, sin marca ni destino: «chaqueta "
                           "impermeable plegable», «adaptador enchufe Reino Unido».",
        }},
        ["producto"],
    ),
}

# Qué programas de referidos (affiliates/programs.py) mira cada tipo de buscar_transporte.
_TRANSPORT_KINDS = {
    "tren": "transport", "autobús": "transport", "vuelo": "flight", "coche": "car",
    "traslado": "transfer", "consigna": "luggage",
}

BOREA_TOOL_NAMES = frozenset(
    [t["name"] for t in BASE_TOOLS] + list(SKILL_TOOLS)
)


def tools_for_turn(base: list[dict[str, Any]], loaded_skills: set[str]) -> list[dict[str, Any]]:
    """Las tools de este momento del turno: las de base más las de las skills cargadas."""
    skills = load_skills()
    names: list[str] = []
    for name in sorted(loaded_skills):
        for tool in skills[name].tools if name in skills else ():
            if tool in SKILL_TOOLS and tool not in names:
                names.append(tool)
    return [*base, *BASE_TOOLS, *(SKILL_TOOLS[n] for n in names)]


# Búsquedas que se guardan un día: el resultado de «hoteles con encanto en Hué» o de
# la eSIM de Japón no cambia de un mensaje a otro, y cada búsqueda son 5-35 s.
_CACHEABLE = frozenset({
    "buscar_actividades", "buscar_transporte", "buscar_esim", "buscar_alojamiento",
    "buscar_equipamiento",
})
_CACHE_SECONDS = 24 * 3600
_CACHE_MAX = 3000
_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _cache_key(trip: "Trip", name: str, args: dict[str, Any], turn: "_Turn") -> str:
    fechas = (
        [str(trip.starts_at), str(trip.ends_at)] if name == "buscar_alojamiento" else []
    )
    return json.dumps(
        [name, args, trip.city, turn.locale, *fechas], sort_keys=True, ensure_ascii=False,
        default=str,
    )


async def run_borea_tool(
    service: "TripPlannerService", trip: "Trip", name: str, args: dict[str, Any], turn: "_Turn"
) -> dict[str, Any]:
    if name not in _CACHEABLE:
        return await _run_borea_tool(service, trip, name, args, turn)
    clave = _cache_key(trip, name, args, turn)
    ahora = time.monotonic()
    guardado = _cache.get(clave)
    if guardado is not None and guardado[0] > ahora:
        return copy.deepcopy(guardado[1])
    resultado = await _run_borea_tool(service, trip, name, args, turn)
    if resultado.get("ok") and (resultado.get("links") or resultado.get("hotels")):
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[clave] = (ahora + _CACHE_SECONDS, copy.deepcopy(resultado))
    return resultado


async def _run_borea_tool(
    service: "TripPlannerService", trip: "Trip", name: str, args: dict[str, Any], turn: "_Turn"
) -> dict[str, Any]:
    settings = service.settings
    if name == "cargar_skill":
        skill = load_skills().get(str(args.get("nombre") or ""))
        if skill is None:
            return {"ok": False, "error": "No existe esa skill.", "skills": sorted(load_skills())}
        turn.skills.add(skill.name)
        return {
            "ok": True, "instrucciones": skill.body, "herramientas": list(skill.tools),
            # Las skills están en español: tras cargarlas, a un usuario francés le
            # llegó la respuesta en español (banco, 2026-11-02).
            "idioma": "Estas instrucciones son para ti; a la persona le sigues "
                      "contestando en su idioma.",
        }

    if name in SKILL_TOOLS and not any(
        name in load_skills()[s].tools for s in turn.skills if s in load_skills()
    ):
        return {"ok": False, "error": "Carga antes la skill que trae esta herramienta."}

    if name == "recomendar":
        recomendaciones = list(trip.recommendations_json or [])
        titulo = str(args.get("title") or "").strip()[:120]
        if not titulo:
            return {"ok": False, "error": "Falta el título."}
        recomendaciones = [r for r in recomendaciones if r.get("title") != titulo]
        if len(recomendaciones) >= MAX_RECOMMENDATIONS:
            return {"ok": False, "error": f"Ya hay {MAX_RECOMMENDATIONS}; quita una antes."}
        kind = str(args.get("kind") or "otro")
        booking = (
            service._booking(str(args["booking_url"]), str(args.get("booking_title") or ""))
            if args.get("booking_url") else None
        )
        nueva = {
            "id": uuid4().hex[:10],
            "kind": kind if kind in RECOMMENDATION_KINDS else "otro",
            "title": titulo,
            "reason": str(args.get("reason") or "").strip()[:400],
            "booking": booking,
        }
        # Una lista nueva: SQLAlchemy no ve los cambios dentro de un JSON mutado.
        trip.recommendations_json = [*recomendaciones, nueva]
        return {"ok": True, "id": nueva["id"]}

    if name == "quitar_recomendacion":
        antes = list(trip.recommendations_json or [])
        despues = [r for r in antes if r.get("id") != str(args.get("id") or "")]
        trip.recommendations_json = despues
        return {"ok": len(despues) < len(antes)}

    if name == "guardar_clima":
        try:
            trip.climate_json = {
                "max_c": round(float(args["max_c"])),
                "min_c": round(float(args["min_c"])),
                "rain_days": round(float(args.get("rain_days") or 0)),
                "summary": str(args.get("summary") or "")[:160],
                "source": str(args.get("source") or "")[:300],
            }
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "error": "Faltan las temperaturas."}
        return {"ok": True}

    if name == "buscar_alojamiento":
        if not trip.city:
            return {"ok": False, "error": "Apunta antes la ciudad del viaje."}
        zona = str(args.get("zona") or "").strip()
        tipo = str(args.get("tipo") or "").strip() or "hotel"
        # En un viaje por varias ciudades trip.city es «Hanoï, Hué, Hội An…» entera, y
        # Google devolvía lo que pillaba de cualquiera de ellas (2026-10-30, Vietnam).
        ciudad = str(args.get("ciudad") or "").strip() or trip.city
        # Las fechas de esa estancia: con las del viaje entero, Booking buscaba en Hanói
        # del 12 de febrero al 2 de marzo para una etapa de cuatro noches.
        entrada = _fecha(args.get("entrada")) or (trip.starts_at.date() if trip.starts_at else None)
        salida = _fecha(args.get("salida")) or (trip.ends_at.date() if trip.ends_at else None)
        # El centro del viaje solo orienta la búsqueda si es en esa misma ciudad.
        centro = (trip.center_json or {}) if ciudad == trip.city else {}
        consulta = " ".join(x for x in (tipo, zona, ciudad) if x)
        sitios = await GooglePlacesClient(settings).search_places(
            consulta, centro.get("lat"), centro.get("lng"), limit=8, locale=turn.locale,
            place_type="lodging",
        )
        hoteles = []
        for s in sitios:
            if s.get("types") and "lodging" not in s["types"]:
                continue
            # Cada hotel con su búsqueda en Booking por su nombre: abre ese hotel con las
            # fechas del viaje, no una lista de la ciudad.
            # Quien viaja solo veía el hotel para 2 adultos (prueba en local, 2026-11-02).
            reserva = booking_search(
                settings, f"{s['name']}, {ciudad}", entrada, salida,
                adults=_entero(args.get("personas"), 2), rooms=_entero(args.get("habitaciones"), 1),
            )
            distancia = (
                round(distance_km(centro["lat"], centro["lng"], s["lat"], s["lng"]), 1)
                if centro.get("lat") is not None and s.get("lat") is not None else None
            )
            hoteles.append({
                "name": s["name"], "address": s["description"], "rating": s.get("rating"),
                "reviews": s.get("user_ratings_total"), "price_level": s.get("price_level"),
                "km_to_center": distancia, "booking_url": reserva.url,
            })
        if not hoteles:
            return {"ok": False, "error": "No encuentro hoteles; prueba con otra zona o tipo."}
        return {
            "ok": True, "hotels": hoteles,
            "how": (
                "Elige dos o tres que encajen con lo que ha pedido y enséñalos como tarjetas "
                "con propose_items: kind logistics, lodging true, el hotel en place y su "
                "booking_url. Di la valoración y lo cerca que queda del centro. El precio "
                "por noche se ve en Booking: no lo inventes (price_level, de 1 a 4, suele "
                "venir vacío en hoteles)."
            ),
        }

    if name == "ruta":
        resultado = await travel_time(
            settings, str(args.get("origen") or ""), str(args.get("destino") or ""),
            str(args.get("modo") or "a pie"),
        )
        return {"ok": True, **resultado} if resultado else {
            "ok": False, "error": "No he podido calcular el trayecto; búscalo en la web.",
        }

    if name == "buscar_transporte":
        kind = _TRANSPORT_KINDS.get(str(args.get("tipo") or ""), "transport")
        enlaces, usage = await web_program_links(
            settings, kind, str(args.get("consulta") or ""), trip.city
        )
        turn.referral_usages.append(usage)
        return {"ok": True, "links": [e.to_dict() for e in enlaces]}

    if name == "buscar_esim":
        pais = str(args.get("pais") or "").strip()
        enlaces, usage = await web_program_links(
            settings, "esim", f"eSIM de datos para viajar a {pais}", pais, max_results=2
        )
        turn.referral_usages.append(usage)
        if not enlaces:
            return {"ok": False, "error": "No encuentro una eSIM concreta para ese país."}
        return {"ok": True, "links": [e.to_dict() for e in enlaces]}

    if name == "buscar_sitios":
        centro = trip.center_json or {}
        sitios = await GooglePlacesClient(settings).search_places(
            str(args.get("consulta") or ""), centro.get("lat"), centro.get("lng"),
            limit=6, locale=turn.locale,
        )
        return {"ok": True, "places": [
            {
                "name": s["name"], "address": s["description"], "rating": s.get("rating"),
                "reviews": s.get("user_ratings_total"), "price_level": s.get("price_level"),
                "open_now": s.get("open_now"),
            }
            for s in sitios
        ]}

    if name == "buscar_actividades":
        consulta = str(args.get("consulta") or "")
        # GetYourGuide y los demás programas a la vez: en fila sumaban 25-35 s.
        gyg = ReferralService(settings).activity_referrals(
            session_id="", query=consulta, poi_name=str(args.get("lugar") or ""),
            city_name=trip.city,
        )
        mas_programas = is_configured(settings, "civitatis") or is_configured(settings, "tiqets")
        if mas_programas:
            resultado, (otros, usage_otros) = await asyncio.gather(
                gyg, web_program_links(settings, "activity", consulta, trip.city, max_results=3)
            )
        else:
            resultado, otros, usage_otros = await gyg, [], None
        usage = resultado.pop("_usage", None)
        if usage is not None:
            turn.referral_usages.append(usage)
        if usage_otros is not None:
            turn.referral_usages.append(usage_otros)
        otros = [e.to_dict() for e in otros if e.program != "getyourguide"]
        gyg_vacio = (
            not resultado.get("ok") or resultado.get("provider") == "getyourguide_fallback_search"
        )
        if otros and gyg_vacio:
            # La búsqueda genérica de GetYourGuide no vale, pero Klook o Tiqets sí
            # tienen la entrada: antes se mandaba igualmente «no hay nada reservable» y
            # el planificador lo obedecía (Universal Studios Japan, 2026-11-02).
            return {"ok": True, "links": otros}
        resultado = only_concrete(resultado)
        resultado.setdefault("links", []).extend(otros)
        return resultado

    if name == "buscar_equipamiento":
        enlace = amazon_search(settings, str(args.get("producto") or ""), turn.locale)
        return {"ok": True, "links": [enlace.to_dict()]}

    return {"ok": False, "error": f"Herramienta desconocida: {name}"}
