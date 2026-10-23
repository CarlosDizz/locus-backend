"""Las herramientas del planificador de Borea.

Borea parte de las herramientas del planificador de Locus (apuntar el viaje,
añadir, proponer, cambiar y quitar franjas) y añade las suyas:

- siempre: cargar_skill, recomendar y quitar_recomendacion («Para este viaje»),
  guardar_clima y buscar_alojamiento;
- al cargar una skill, las que esa skill declara en su línea `tools:` (ruta,
  buscar_transporte, buscar_sitios, reservar_mesa, buscar_actividades,
  buscar_equipamiento). Una tool por tipo de cosa, no por plataforma: en qué
  programas se busca lo decide el backend (affiliates/programs.py).

Los enlaces que devuelven ya van con nuestro código; el modelo solo los copia en
booking_url, y aun así el backend los vuelve a decorar al apuntarlos.
"""

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from locus_v2.affiliates.programs import is_configured
from locus_v2.affiliates.search import amazon_search, booking_search, web_program_links
from locus_v2.affiliates.service import ReferralService
from locus_v2.places.client import GooglePlacesClient
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
        "Un enlace para buscar alojamiento en la ciudad y las fechas del viaje. Úsalo solo si "
        "la persona aún no tiene dónde dormir.",
        {"zona": {"type": "string", "description": "Barrio o zona, si se ha elegido."}},
        [],
    ),
]

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
        "Enlaces reales para reservar un tren, un autobús, un vuelo o un coche de alquiler.",
        {
            "consulta": {"type": "string", "description": "«Tren Madrid Toledo el domingo»."},
            "tipo": {"type": "string", "enum": ["tren", "autobús", "vuelo", "coche"]},
        },
        ["consulta"],
    ),
    "buscar_sitios": _function(
        "buscar_sitios",
        "Restaurantes, bares o cafés en Google con su valoración y número de reseñas, para "
        "elegir con criterio y huir de las turistadas.",
        {"consulta": {"type": "string", "description": "«Trattoria romana en Monti»."}},
        ["consulta"],
    ),
    "reservar_mesa": _function(
        "reservar_mesa",
        "Un enlace para reservar mesa en un restaurante concreto, si se puede reservar online.",
        {"restaurante": {"type": "string"}},
        ["restaurante"],
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
        {"producto": {"type": "string", "description": "El tipo de producto, sin marca."}},
        ["producto"],
    ),
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


async def run_borea_tool(
    service: "TripPlannerService", trip: "Trip", name: str, args: dict[str, Any], turn: "_Turn"
) -> dict[str, Any]:
    settings = service.settings
    if name == "cargar_skill":
        skill = load_skills().get(str(args.get("nombre") or ""))
        if skill is None:
            return {"ok": False, "error": "No existe esa skill.", "skills": sorted(load_skills())}
        turn.skills.add(skill.name)
        return {"ok": True, "instrucciones": skill.body, "herramientas": list(skill.tools)}

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
        ciudad = ", ".join(x for x in (str(args.get("zona") or ""), trip.city) if x)
        if not ciudad:
            return {"ok": False, "error": "Apunta antes la ciudad del viaje."}
        enlace = booking_search(
            settings, ciudad,
            trip.starts_at.date() if trip.starts_at else None,
            trip.ends_at.date() if trip.ends_at else None,
        )
        return {"ok": True, "links": [enlace.to_dict()]}

    if name == "ruta":
        resultado = await travel_time(
            settings, str(args.get("origen") or ""), str(args.get("destino") or ""),
            str(args.get("modo") or "a pie"),
        )
        return {"ok": True, **resultado} if resultado else {
            "ok": False, "error": "No he podido calcular el trayecto; búscalo en la web.",
        }

    if name == "buscar_transporte":
        kind = "car" if str(args.get("tipo") or "") == "coche" else "transport"
        enlaces, usage = await web_program_links(
            settings, kind, str(args.get("consulta") or ""), trip.city
        )
        turn.referral_usages.append(usage)
        return {"ok": True, "links": [e.to_dict() for e in enlaces]}

    if name == "reservar_mesa":
        enlaces, usage = await web_program_links(
            settings, "table", f"Reservar mesa en {args.get('restaurante') or ''}", trip.city
        )
        turn.referral_usages.append(usage)
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
        resultado = await ReferralService(settings).activity_referrals(
            session_id="", query=str(args.get("consulta") or ""),
            poi_name=str(args.get("lugar") or ""), city_name=trip.city,
        )
        usage = resultado.pop("_usage", None)
        if usage is not None:
            turn.referral_usages.append(usage)
        if is_configured(settings, "civitatis") or is_configured(settings, "tiqets"):
            otros, usage = await web_program_links(
                settings, "activity", str(args.get("consulta") or ""), trip.city, max_results=2
            )
            turn.referral_usages.append(usage)
            resultado.setdefault("links", []).extend(
                e.to_dict() for e in otros if e.program != "getyourguide"
            )
        return resultado

    if name == "buscar_equipamiento":
        enlace = amazon_search(settings, str(args.get("producto") or ""), turn.locale)
        return {"ok": True, "links": [enlace.to_dict()]}

    return {"ok": False, "error": f"Herramienta desconocida: {name}"}
